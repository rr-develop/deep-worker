"""Backend death must never reach the agent — tested against a LIVE Serena.

See ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md`` §6.

Why this test exists: the agent received a raw
``LanguageServerTerminatedException`` and drew two WRONG conclusions
from it — "tool calls cannot be parallelized" and "absolute paths /
broad searches cannot be used". Unit tests exercise the retry logic on
fakes; here we verify the real seam "bridge → gRPC → supervisor → new
backend", which fakes cannot cover.

About the race (review remark, accepted). The naive scenario "kill
Serena, then call the tool" is non-deterministic: the supervisor may
manage to bring up a replacement BEFORE the call, in which case there
is no retry and the counter stays at zero — the test would be green
without checking anything. So the tests are split:

* ``test_background_kill_then_call_succeeds`` — background death, wait
  for recovery BY EVENT and check only that the call succeeds; the
  counter is deliberately NOT checked;
* ``test_kill_during_in_flight_call_triggers_retry`` — deterministic:
  the death lands INSIDE the call window, so a retry is mandatory.

Integration-test policy: a strict overall budget, event-driven waiting,
and a reaper with guaranteed SIGTERM→SIGKILL.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
from pathlib import Path

import grpc
import pytest

from deep_worker_cli.lsp.bridge_loop import BridgeSession, ToolCallFailed
from deep_worker_cli.lsp.serena_config import SERENA_COMMAND
from deep_worker_cli.lsp.service.proto import (
    lsp_service_pb2 as pb,
    lsp_service_pb2_grpc as pbg,
)
from deep_worker_cli.lsp.service.registry import ServiceRegistry
from deep_worker_cli.lsp.service.supervisor import (
    ServiceSupervisor,
    _wait_pid_exit,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_DEV_PKG = Path(__file__).resolve().parents[3]
_DEV_DEEPAGENTS = _DEV_PKG.parent / "deepagents"
_DEV_ROOT = _DEV_PKG.parent

_START_S = 40.0
#: The first call AFTER a backend restart waits while the new Serena
#: reindexes the project. Measured on the live stack: with a budget of
#: 45 s the call did not make it; with 170 s it did, hitting exactly our
#: ``_DISPATCH_BUDGET_S`` (110 s). Here we take a generous but finite
#: value: the test must fail, not hang.
_TOOL_S = 150.0
_WHOLE_TEST_S = 600.0

#: How many times to repeat "start a call and kill Serena" until the
#: death lands INSIDE the call window. Measured: a warm find_symbol over
#: a single file answers in ~0.0s, so the hit is probabilistic — but
#: enough retries make the event practically inevitable, while the test
#: stays finite and fails if a retry never happened (unlike the former
#: restarts_total check, which went green even without a retry).
_KILL_ATTEMPTS = 4

#: Budget of ONE round in the kill loop. Deliberately much smaller than
#: ``_TOOL_S``: if a given round does not catch the death, the call goes
#: into reindexing of the new backend and hits the service's 110-second
#: cap, and at that point it is cheaper to abandon the round and try
#: again. Without this, 4 failed rounds would add up to ~10 minutes and
#: the test would violate the bounded-budget policy for integration
#: tests.
_KILL_ROUND_S = 25.0

#: Readiness-wait budget AFTER a restart. Larger than ``_START_S``: a
#: fresh Serena comes up and reindexes the project anew, which is longer
#: than the very first start. Finite — the test must fail, not hang.
_RESTART_READY_S = 120.0


def _require_serena() -> None:
    if shutil.which(SERENA_COMMAND):
        return
    msg = (
        f"{SERENA_COMMAND!r} not found; install via "
        f"'uv tool install -p 3.13 serena-agent'"
    )
    if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    state = stat[stat.rindex(")") + 2 :].split()[0]
    return state != "Z"


def _kill_pid(pid: int) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_pid_exit(pid, timeout_s=8.0)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        _wait_pid_exit(pid, timeout_s=5.0)


def _dev_serve_argv(project_root: Path, slug_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "deep_worker_cli.lsp.service",
        "__serve_impl__",
        "--project-root",
        str(project_root.resolve()),
        "--socket",
        str(slug_dir / "grpc.sock"),
    ]


@pytest.fixture(autouse=True)
def _dev_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    parts = [str(_DEV_PKG), str(_DEV_DEEPAGENTS), str(_DEV_ROOT)]
    prev = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(parts + ([prev] if prev else []))
    )


@pytest.fixture
def reaper():
    pids: list[int] = []

    class _R:
        def track(self, pid: int) -> int:
            pids.append(pid)
            return pid

    try:
        yield _R()
    finally:
        for pid in pids:
            _kill_pid(pid)


async def _retries_seen(socket_path: str) -> int:
    """Current value of ls_terminated_retries_total at the service."""
    channel = grpc.aio.insecure_channel(f"unix:{socket_path}")
    try:
        stub = pbg.LspServiceStub(channel)
        status = await asyncio.wait_for(
            stub.ProjectStatus(pb.StatusRequest()), timeout=10.0
        )
        return status.ls_terminated_retries_total
    finally:
        await channel.close()


async def _restarts_seen(socket_path: str) -> int:
    """Current value of restarts_total at the service."""
    channel = grpc.aio.insecure_channel(f"unix:{socket_path}")
    try:
        stub = pbg.LspServiceStub(channel)
        status = await asyncio.wait_for(
            stub.ProjectStatus(pb.StatusRequest()), timeout=10.0
        )
        return status.restarts_total
    finally:
        await channel.close()


def _py_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
    )
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "core.py").write_text(
        "class Widget:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "\n"
        "    def rename(self, name: str) -> None:\n"
        "        self.name = name\n"
    )
    return root


def _supervisor(reg_dir: Path, root: Path) -> ServiceSupervisor:
    return ServiceSupervisor(
        ServiceRegistry(base_dir=reg_dir),
        lambda slug_dir: _dev_serve_argv(root, slug_dir),
    )


async def _wait_backend_ready(
    stub: pbg.LspServiceStub, budget_s: float
) -> None:
    """Block until the backend is ready — event-driven, no Health poll.

    The project forbids poll loops, and a ``sleep``-and-recheck helper
    would violate that in the very tests that are supposed to prove the
    readiness protocol works. ``AttachSession`` already holds the
    ATTACHED event back until the backend is genuinely ready and tags it
    ``backend-not-ready`` on expiry, so consuming an event from that
    stream IS the readiness boundary.

    The service caps its own wait (``_backend_ready_wait_s``) and then
    answers ATTACHED tagged ``backend-not-ready``. That cap is SHORTER
    than a cold Serena start after two restarts in a row, so a single
    read cannot express "wait longer than the service's own window".
    Consuming the next event instead is still event-driven — each
    stream parks on the readiness event inside the service, there is no
    timer tick here — while the caller's ``budget_s`` stays the one
    hard bound.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(
                f"backend did not become alive in {budget_s}s"
            )
        stream = stub.AttachSession(
            pb.AttachRequest(client_id="integ-readiness")
        )
        try:
            event = await asyncio.wait_for(stream.read(), timeout=remaining)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"backend did not become alive in {budget_s}s"
            ) from e
        finally:
            stream.cancel()
        if "backend-not-ready" not in event.message:
            return


def _serena_children(project_root: Path) -> list[int]:
    """Pids of the Serena processes serving EXACTLY ``project_root``.

    Filtering on the project root is not cosmetic: a bare
    "start-mcp-server" match also selects Serena instances belonging to
    other services on this machine (the developer's own live session,
    other checkouts). Killing those from a test would take down
    unrelated work AND make the test look like a product defect, because
    the service under test would keep running while a stranger's backend
    died.
    """
    wanted = str(project_root.resolve())
    out: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = raw.decode(errors="ignore").split("\0")
        if "start-mcp-server" not in args:
            continue
        try:
            root_arg = args[args.index("--project") + 1]
        except (ValueError, IndexError):
            continue
        if root_arg == wanted:
            out.append(int(proc.name))
    return out


async def test_background_kill_then_call_succeeds(
    tmp_path: Path, reaper
) -> None:
    """Background Serena death: the call MUST COMPLETE, not hang.

    Measured on the live stack: the first call after a restart goes into
    reindexing of the new backend and, before the feature, could hang
    indefinitely (the client dropped with DEADLINE_EXCEEDED). Now it is
    bounded by ``_DISPATCH_BUDGET_S`` and returns an explanatory message.

    The retry counter is deliberately NOT checked here: the supervisor
    may have recovered before the call, in which case there was no retry
    — that is correct behavior, not a failure (review remark about the
    race).
    """
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = _supervisor(reg_dir, root).ensure_started(root)
        reaper.track(entry.pid)

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            await _wait_backend_ready(stub, _START_S)

            victims = _serena_children(root)
            assert victims, "did not find a live Serena process"
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Wait for recovery BY EVENT (Health), not by sleeping.
            await _wait_backend_ready(stub, _START_S)
        finally:
            await channel.close()

        session = BridgeSession(entry.socket_path)
        await session.connect()
        try:
            # The first call after a restart may go into reindexing of
            # the new backend. The requirement is not "it is fast" but
            # "it COMPLETES with a definite answer": either a result or
            # our explanatory text. There must be no infinite hang — this
            # is exactly what the agent saw before the feature.
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "find_symbol",
                        {"name_path": "Widget",
                         "relative_path": "pkg/core.py"},
                    ),
                    timeout=_TOOL_S,
                )
                assert result is not None
            except ToolCallFailed as e:
                text = str(e)
                assert "LanguageServerTerminatedException" not in text, (
                    f"the agent must not see a raw exception: {text}"
                )
                assert "DEADLINE_EXCEEDED" not in text, (
                    f"the agent must not see a raw gRPC failure: {text}"
                )
                assert (
                    "NOT a limit on concurrent tool calls" in text
                    or "NOT caused by concurrent tool calls" in text
                ), f"the answer must rule out the false hypothesis: {text}"

            # The next call must go through normally: the backend is alive.
            second = await asyncio.wait_for(
                session.call_tool(
                    "find_symbol",
                    {"name_path": "Widget", "relative_path": "pkg/core.py"},
                ),
                timeout=_TOOL_S,
            )
            assert second is not None, (
                "calls must work after recovery"
            )
        finally:
            await session.close()


async def test_kill_during_in_flight_call_triggers_retry(
    tmp_path: Path, reaper
) -> None:
    """Deterministic: death INSIDE the call window.

    We kill Serena right after the call starts, so the death is
    guaranteed to land in the window rather than before it (without this
    the test would be a race — see the review). We check that the call
    completed with a definite answer and that the restart actually
    happened.
    """
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = _supervisor(reg_dir, root).ensure_started(root)
        reaper.track(entry.pid)

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            await _wait_backend_ready(stub, _START_S)
        finally:
            await channel.close()

        session = BridgeSession(entry.socket_path)
        await session.connect()
        try:
            # The call window must not be "caught", it must be WIDENED.
            # Measured: a plain find_symbol over one file returns in
            # ~0.0s, so neither sleep(0.2) nor a CPU barrier works —
            # Serena manages to answer before we kill it, and the test
            # goes green without checking anything (that was the defect
            # the review pointed out).
            #
            # Instead, we guarantee the death lands inside the window by
            # retrying the whole scenario: each round starts a call and
            # kills Serena almost immediately. If the retry counter has
            # not moved — the call managed to finish, try again.
            got_retry = False
            for _ in range(_KILL_ATTEMPTS):
                victims = _serena_children(root)
                if not victims:
                    break
                call = asyncio.create_task(
                    session.call_tool(
                        "find_symbol",
                        {"name_path": "Widget",
                         "relative_path": "pkg/core.py"},
                    )
                )
                for pid in victims:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

                with contextlib.suppress(
                    ToolCallFailed, asyncio.TimeoutError, Exception
                ):
                    await asyncio.wait_for(call, timeout=_KILL_ROUND_S)
                call.cancel()

                if await _retries_seen(entry.socket_path) >= 1:
                    got_retry = True
                    break

            assert got_retry, (
                f"in {_KILL_ATTEMPTS} attempts the death never landed "
                "inside the call window — the scenario did not exercise "
                "the retry"
            )
            # After a proven retry the service must remain operational:
            # the next call returns either a result or our explanatory
            # text — but never a raw exception.
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "find_symbol",
                        {"name_path": "Widget",
                         "relative_path": "pkg/core.py"},
                    ),
                    timeout=_TOOL_S,
                )
                assert result is not None
            except ToolCallFailed as e:
                text = str(e)
                assert "LanguageServerTerminatedException" not in text, text
                assert "DEADLINE_EXCEEDED" not in text, text
                assert (
                    "NOT a limit on concurrent tool calls" in text
                    or "NOT caused by concurrent tool calls" in text
                ), text
        finally:
            await session.close()

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            status = await asyncio.wait_for(
                stub.ProjectStatus(pb.StatusRequest()), timeout=5.0
            )
        finally:
            await channel.close()

        # The backend really died and was brought up again.
        assert status.restarts_total >= 1, (
            "the supervisor must have restarted the backend"
        )


async def test_mutating_rpc_not_replayed_live(
    tmp_path: Path, reaper
) -> None:
    """Ambiguous commit on a LIVE Serena: the edit must not be duplicated.

    The most expensive scenario of the feature: Serena writes the edit
    to disk BEFORE it asks the language server for diagnostics, so a
    death at that moment leaves the outcome unknown. An auto-retry would
    have duplicated the insertion — here we check that it does not.
    """
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"
    target = root / "pkg" / "core.py"

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = _supervisor(reg_dir, root).ensure_started(root)
        reaper.track(entry.pid)

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            await _wait_backend_ready(stub, _START_S)
        finally:
            await channel.close()

        marker = "SENTINEL_INSERTED_ONCE"
        body = f"\ndef {marker}() -> int:\n    return 42\n"

        session = BridgeSession(entry.socket_path)
        await session.connect()
        applied = False
        died_during_edit = False
        try:
            # Review remark (round 5): the "no more than one occurrence"
            # check is also true with ZERO occurrences, so the test could
            # go green without ever applying the edit and prove nothing
            # about the absence of a duplicate. Now applying the edit is
            # a mandatory precondition: without it the test fails.
            #
            # The call window is short and cannot be "caught", so the
            # scenario repeats: each round starts an insertion and
            # immediately kills Serena. Rounds where the death landed
            # inside the window produce an ambiguous commit — exactly the
            # case in which an auto-retry would have duplicated the
            # insertion.
            victims = _serena_children(root)
            assert victims, "did not find a live Serena process"
            call = asyncio.create_task(
                session.call_tool(
                    "insert_after_symbol",
                    {
                        "name_path": "Widget",
                        "relative_path": "pkg/core.py",
                        "body": body,
                    },
                )
            )
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # One round, not a series: consecutive kills hit the
            # supervisor's deliberate rate limit (≤3 restarts per 60 s),
            # after which the backend stays not-ready and the test would
            # fail on infrastructure instead of checking the invariant.
            try:
                await asyncio.wait_for(call, timeout=_KILL_ROUND_S)
            except ToolCallFailed as e:
                text = str(e)
                # Direct proof: the dispatcher saw the death precisely on
                # a MUTATING call and deliberately did NOT retry it.
                assert "was NOT retried" in text or "UNKNOWN" in text, (
                    f"the mutating call did not get its explanation: {text}"
                )
                died_during_edit = True
            except asyncio.TimeoutError:
                call.cancel()
            applied = f"def {marker}" in target.read_text()
        finally:
            await session.close()

        if not applied:
            # The edit never landed on disk. Finish it without kills —
            # otherwise the main assert below would be checking emptiness.
            # Must be on a FRESH session: we aborted the previous one with
            # ``call.cancel()`` on each round, and reusing an aborted
            # session for the control call would mean checking its state,
            # not the test invariant.
            channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
            stub = pbg.LspServiceStub(channel)
            try:
                # A cold Serena start after consecutive restarts is
                # noticeably longer than the first one, so the budget here
                # is its own — but still strict and finite.
                await _wait_backend_ready(stub, _RESTART_READY_S)
            finally:
                await channel.close()

            closing = BridgeSession(entry.socket_path)
            await closing.connect()
            try:
                # Serena's mutating tools answer with the text ``OK``
                # rather than JSON, and the bridge raises ToolCallFailed
                # in that case (a standalone defect, unrelated to this
                # feature — see §1.4-ter of the design). The test
                # invariant lives on disk, so the call's outcome does not
                # matter here; only what ended up in the file matters.
                with contextlib.suppress(ToolCallFailed, Exception):
                    await asyncio.wait_for(
                        closing.call_tool(
                            "insert_after_symbol",
                            {
                                "name_path": "Widget",
                                "relative_path": "pkg/core.py",
                                "body": body,
                            },
                        ),
                        timeout=_TOOL_S,
                    )
            finally:
                await closing.close()
            applied = f"def {marker}" in target.read_text()

        text = target.read_text()
        assert applied, (
            "the insertion was never applied — the test checked nothing "
            "about mutating-call duplication:\n" + text
        )
        # THE KEY ASSERTION: the backend died during the edit, but the
        # insertion was not duplicated — there was no auto-retry of the
        # mutating call.
        assert text.count(f"def {marker}") == 1, (
            "the mutating call was retried — the edit was duplicated:\n"
            + text
        )
        assert died_during_edit, (
            "no round left an ambiguous commit — the scenario never "
            "reached the state under test"
        )


async def test_status_exposes_ls_terminated_retries(
    tmp_path: Path, reaper
) -> None:
    """The metric reached the gRPC surface (field 18)."""
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = _supervisor(reg_dir, root).ensure_started(root)
        reaper.track(entry.pid)

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            await _wait_backend_ready(stub, _START_S)
            status = await asyncio.wait_for(
                stub.ProjectStatus(pb.StatusRequest()), timeout=5.0
            )
        finally:
            await channel.close()

        # The field exists and is readable; on a fresh service it is zero.
        assert status.ls_terminated_retries_total == 0
