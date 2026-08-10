"""Kotlin: the language server's death must not reach the agent.

A coverage gap acknowledged after a field report. The other integration
tests for this feature run on a synthetic PYTHON project, where the
language server starts instantly and dies predictably. Yet the whole
failure class came precisely from Kotlin: an upstream crash of
``workspace/diagnostic/refresh`` (see ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md``
§1.2), slow re-indexing after restart and, as a consequence, calls that
land in the window "server already dead, replacement still indexing".

The field report asserted three things: narrow calls (``get_symbols_overview``
on a single file) stopped working; the agent sees ``ClosedResourceError``;
the service does not come up. The first two are pinned here as a contract,
the third via the requirement that a call ALWAYS completes with a defined
answer in finite time.

Why the project is synthetic and not a real host repository: the test
must be reproducible and must not depend on someone else's working
directory that could be changed or in use. The language server here is a
REAL Kotlin LS (the same ``intellij-server``), so exactly the seam that
broke is exercised. Gradle files are deliberately NOT created: with them,
Kotlin LS spawns Gradle daemons of several gigabytes that outlive the
test and degrade the whole machine — observed in practice.

The integration-test policy is followed: a hard overall budget, event-driven
readiness waiting, a reaper with guaranteed SIGTERM→SIGKILL, and selection
of Serena processes by the EXACT ``--project`` (so foreign ones are not
touched).
"""
from __future__ import annotations

import asyncio
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
from deep_worker_cli.lsp.service.supervisor import ServiceSupervisor, _wait_pid_exit

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_DEV_PKG = Path(__file__).resolve().parents[3]
_DEV_DEEPAGENTS = _DEV_PKG.parent / "deepagents"
_DEV_ROOT = _DEV_PKG.parent

#: Directory of the statically installed Kotlin LS. Serena does not
#: download it itself, so without it the test is skipped, not failed.
_KOTLIN_LS_ROOT = Path.home() / ".serena/language_servers/static/KotlinLanguageServer"

#: Cold start of Kotlin LS on a minimal project — measured at 12.6 s.
#: Taken with margin: a JVM on a loaded machine starts noticeably slower.
_START_S = 180.0

#: One call. The first call after death goes into re-indexing, but the
#: service bounds it with its own ``_DISPATCH_BUDGET_S`` (110 s) and must
#: return a defined answer, so the client budget is a bit larger.
_TOOL_S = 150.0

_WHOLE_TEST_S = 900.0

#: Texts the agent must NEVER see. ``ClosedResourceError``
#: — from the field report: the anyio exception of a closed stream,
#: meaning the request went into dead transport. ``LanguageServerTerminatedException``
#: — the raw solidlsp exception from which the agent drew wrong conclusions.
_FORBIDDEN_IN_AGENT_TEXT = (
    "ClosedResourceError",
    "LanguageServerTerminatedException",
    "BrokenResourceError",
    "DEADLINE_EXCEEDED",
)

_KOTLIN_SOURCE = """package demo

class Widget(val name: String) {
    fun rename(newName: String): Widget = Widget(newName)
}

fun buildWidget(): Widget = Widget("default")
"""

_REL_PATH = "src/main/kotlin/demo/Widget.kt"


def _require_kotlin_stack() -> None:
    if not shutil.which(SERENA_COMMAND):
        msg = f"{SERENA_COMMAND!r} not found"
        if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    if not _kotlin_ls_path():
        msg = f"Kotlin language server not installed under {_KOTLIN_LS_ROOT}"
        if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") == "1":
            pytest.fail(msg)
        pytest.skip(msg)


def _kotlin_ls_path() -> Path | None:
    """Path to ``intellij-server``; the version in the directory name is not pinned."""
    if not _KOTLIN_LS_ROOT.is_dir():
        return None
    for version_dir in sorted(_KOTLIN_LS_ROOT.iterdir(), reverse=True):
        candidate = version_dir / "bin" / "intellij-server"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _kotlin_project(tmp_path: Path) -> Path:
    """Minimal REAL Kotlin project (no Gradle — see the docstring)."""
    root = tmp_path / "ktproj"
    src = root / "src/main/kotlin/demo"
    src.mkdir(parents=True)
    (src / "Widget.kt").write_text(_KOTLIN_SOURCE)

    serena_dir = root / ".serena"
    serena_dir.mkdir()
    config = serena_dir / "project.yml"
    config.write_text(
        "project_name: kt_e2e\n"
        "languages:\n"
        "- kotlin\n"
        "language_backend: lsp\n"
        "language_server_settings:\n"
        "  kotlin:\n"
        f'    ls_path: "{_kotlin_ls_path()}"\n'
    )
    # The project config is not public: keep permissions narrow (permissions policy).
    config.chmod(0o600)
    return root


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat[stat.rindex(")") + 2 :].split()[0] != "Z"


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


def _descendants(pid: int) -> list[int]:
    """All descendants of ``pid`` (not just direct children), via ``/proc``.

    Parentage is what matters, not cmdline: Kotlin LS is launched as
    ``.../intellij-server --stdio`` and does NOT contain ``--project``, so
    it cannot be linked to the test project by command-line text. Under it
    lives a JVM of nearly a gigabyte — if it is not killed, it outlives the
    test and degrades the machine (observed in practice).
    """
    children: dict[int, list[int]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text()
            ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(proc.name))

    out: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def _serena_children(project_root: Path) -> list[int]:
    """Pids of Serena serving exactly ``project_root``.

    The filter by exact ``--project`` is not cosmetic: without it the test
    would kill the Serena processes of other services on this machine
    (the developer's live sessions) and it would look like a product defect.
    """
    wanted = str(project_root.resolve())
    out: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="ignore").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "start-mcp-server" not in args:
            continue
        try:
            if args[args.index("--project") + 1] == wanted:
                out.append(int(proc.name))
        except (ValueError, IndexError):
            continue
    return out


#: How long we wait for the ``sent`` signal from the service. Generous:
#: this includes Serena's cold re-indexing before the call itself.
_BARRIER_WAIT_S = 180.0

#: How long the service holds an attempt, waiting for ``release``. Bounded
#: so a hung test cannot suspend the service forever.
_BARRIER_HOLD_S = 60.0


async def _await_file(path: Path, timeout_s: float) -> bool:
    """Event-driven wait for a file to appear (inotify), no polling.

    Polling is forbidden by project policy; here it would also blur
    exactly the timing precision the barrier exists for.
    """
    import ctypes
    import struct

    if path.exists():
        return True

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    try:
        wd = libc.inotify_add_watch(fd, str(path.parent).encode(), 0x100 | 0x80)
        if wd < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch failed")
        loop = asyncio.get_running_loop()
        seen: asyncio.Future[bool] = loop.create_future()

        def _on_readable() -> None:
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                return
            offset = 0
            while offset + 16 <= len(data):
                _wd, _mask, _cookie, length = struct.unpack_from("iIII", data, offset)
                name = data[offset + 16 : offset + 16 + length].split(b"\0", 1)[0]
                offset += 16 + length
                if name.decode(errors="ignore") == path.name and not seen.done():
                    seen.set_result(True)

        loop.add_reader(fd, _on_readable)
        try:
            if path.exists():
                return True
            return await asyncio.wait_for(seen, timeout=timeout_s)
        except asyncio.TimeoutError:
            return False
        finally:
            loop.remove_reader(fd)
    finally:
        os.close(fd)


def _barrier_serve_argv(
    project_root: Path, slug_dir: Path, barrier_dir: Path
) -> list[str]:
    """argv of the instrumented service (test-only).

    Production code is unchanged: the barrier lives in a separate test
    module that the production supervisor never launches.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "_kotlin_barrier_serve.py"),
        "--barrier-dir",
        str(barrier_dir),
        "--barrier-hold-s",
        str(_BARRIER_HOLD_S),
        "__serve_impl__",
        "--project-root",
        str(project_root.resolve()),
        "--socket",
        str(slug_dir / "grpc.sock"),
    ]


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
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts + ([prev] if prev else [])))


@pytest.fixture
def reaper():
    """Guaranteed cleanup: SIGTERM→SIGKILL, whatever happens.

    Besides our own processes we also kill Serena of the test project: it
    outlives the service, and the language-server JVM lives on with it.
    """
    pids: list[int] = []
    roots: list[Path] = []
    grandchildren: set[int] = set()

    class _R:
        def track(self, pid: int) -> int:
            pids.append(pid)
            return pid

        def track_project(self, root: Path) -> Path:
            roots.append(root)
            return root

        def note_descendants(self, pid: int) -> list[int]:
            """Remember Serena's descendants BEFORE killing it.

            After the parent's SIGKILL the relationship is lost: Kotlin LS
            and its JVM move to PPID=1 and are no longer tied to the
            project by cmdline. So the snapshot is taken in advance.
            """
            found = _descendants(pid)
            grandchildren.update(found)
            return found

    try:
        yield _R()
    finally:
        for pid in pids:
            _kill_pid(pid)
        for root in roots:
            for pid in _serena_children(root):
                grandchildren.update(_descendants(pid))
                _kill_pid(pid)
        for pid in sorted(grandchildren):
            _kill_pid(pid)
        # Final check, not an assumption: the test regularly SIGKILLs the
        # parent, and an orphaned Kotlin LS with its JVM must be removed.
        # Assert AFTER the guaranteed killing above.
        survivors = [pid for pid in sorted(grandchildren) if _pid_alive(pid)]
        assert not survivors, (
            f"Kotlin LS/JVM processes alive after teardown: {survivors}"
        )


async def _wait_backend_ready(stub: pbg.LspServiceStub, budget_s: float) -> None:
    """Readiness by event, without polling Health.

    The service bounds its own waiting window and then emits ATTACHED
    marked ``backend-not-ready``; for a cold Kotlin LS one window is not
    enough, so we read the next event until the overall hard deadline.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"kotlin backend not ready in {budget_s}s")
        stream = stub.AttachSession(pb.AttachRequest(client_id="integ-kotlin"))
        try:
            event = await asyncio.wait_for(stream.read(), timeout=remaining)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"kotlin backend not ready in {budget_s}s") from e
        finally:
            stream.cancel()
        if "backend-not-ready" not in event.message:
            return


def _assert_agent_safe(text: str) -> None:
    for forbidden in _FORBIDDEN_IN_AGENT_TEXT:
        assert forbidden not in text, (
            f"agent saw a raw infrastructure exception {forbidden!r}: {text}"
        )


async def _narrow_call(session: BridgeSession) -> object:
    """Narrow call on a single file — what "stopped working"."""
    return await asyncio.wait_for(
        session.call_tool("get_symbols_overview", {"relative_path": _REL_PATH}),
        timeout=_TOOL_S,
    )


async def test_narrow_calls_survive_kotlin_ls_death(tmp_path: Path, reaper) -> None:
    """Narrow calls must work before AND after Kotlin LS death.

    Exactly the field-report scenario. Its three assertions are checked:
    the narrow call works; after the language server's death it keeps
    working; the agent sees neither ``ClosedResourceError`` nor a raw
    ``LanguageServerTerminatedException``.
    """
    _require_kotlin_stack()
    root = _kotlin_project(tmp_path)
    reaper.track_project(root)
    reg_dir = tmp_path / "reg"

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = ServiceSupervisor(
            ServiceRegistry(base_dir=reg_dir),
            lambda slug_dir: _dev_serve_argv(root, slug_dir),
        ).ensure_started(root)
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
            first = await _narrow_call(session)
            assert first is not None, "narrow call did not return a result"
            assert "Widget" in str(first), f"unexpected answer: {first}"

            victims = _serena_children(root)
            assert victims, "no live Serena process found for the test project"
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Narrow call right after death. It is either completed on the
            # recovered backend, or answers with our explanatory text —
            # but NEVER with a raw exception and never hangs: the agent
            # already saw both, and it is precisely from them that it drew
            # false conclusions about concurrency and paths.
            try:
                after_death = await _narrow_call(session)
                assert after_death is not None
                _assert_agent_safe(str(after_death))
            except ToolCallFailed as e:
                text = str(e)
                _assert_agent_safe(text)
                assert (
                    "NOT caused by concurrent tool calls" in text
                    or "NOT a limit on concurrent tool calls" in text
                ), f"answer must disprove the false concurrency hypothesis: {text}"

            # And after that the service must be normally operational.
            recovered = await _narrow_call(session)
            assert recovered is not None, (
                "after recovery narrow calls must work"
            )
            assert "Widget" in str(recovered), f"unexpected answer: {recovered}"
            _assert_agent_safe(str(recovered))
        finally:
            await session.close()

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            status = await asyncio.wait_for(
                stub.ProjectStatus(pb.StatusRequest()), timeout=15.0
            )
        finally:
            await channel.close()

        # The death really happened and was handled: either the
        # supervisor started a replacement, or the dispatcher retried.
        assert status.restarts_total >= 1 or status.ls_terminated_retries_total >= 1, (
            "the language server was killed, but neither a restart nor a retry was recorded"
        )


async def test_narrow_call_in_flight_survives_serena_kill(
    tmp_path: Path, reaper
) -> None:
    """Serena's death INSIDE the call window must fall into a retry.

    Apart from the "killed-between-calls" scenario above: if Serena is
    killed while a call is already sent and waiting for an answer, exactly
    the two narrow mechanisms of the design's §1.4-bis must fire — the
    pidfd-watcher of child death in ``_OwnedStdioSession`` and the
    ``_call_or_death`` race in ``SerenaBackend``. A mutation check
    "remove either of them" stays green without this scenario — meaning
    the test does not pin the fix it was written for.

    The death moment is set by a barrier ON THE SERVICE SIDE, not by a
    guess about the client's scheduling order: the service marks ``sent``
    only after writing the request into Serena's MCP session, and only
    then does the test send SIGKILL. The client-side approach
    "``ensure_future`` + one ``sleep(0)``" does not prove this — the call
    first enters a nested ``wait_for`` and in the first quantum only
    manages to create an inner task, so the kill could arrive BEFORE the
    send while the test still went green.

    We require the call to complete with a correct answer in finite
    time — the result of a retry on the recovered backend, or an
    explanatory text. Neither ``ClosedResourceError`` nor a raw
    ``LanguageServerTerminatedException`` can reach the agent.

    Mutation boundary K5 (barrier signal earlier than the actual send).
    This test could not kill it, and that is a conscious boundary, not an
    oversight. The barrier holds the request BEFORE the pipe write, and
    before releasing the hold the test waits for Serena's actual death via
    ``_wait_pid_exit`` — so the death window is guaranteed to occur no
    matter when the signal is raised. Because of this an "early signal"
    stops changing the observed behavior: measurements show an identical
    picture — two attempts ~1.5 s apart, ``ls_terminated_retries_total >= 1``
    in both cases. Distinguishing the mutant would require returning to a
    non-guaranteed death window, which would make the test flaky — a price
    this scenario is not worth. The barrier itself is not decorative:
    without it SIGKILL landed on the warm-up call, the observed call never
    happened, and the test was still green.
    """
    _require_kotlin_stack()
    root = _kotlin_project(tmp_path)
    reaper.track_project(root)
    reg_dir = tmp_path / "reg"
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    barrier_dir.chmod(0o700)

    async with asyncio.timeout(_WHOLE_TEST_S):
        entry = ServiceSupervisor(
            ServiceRegistry(base_dir=reg_dir),
            lambda slug_dir: _barrier_serve_argv(root, slug_dir, barrier_dir),
        ).ensure_started(root)
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
            # Warm run: guarantee Serena has finished indexing and the
            # second call really reaches the LS quickly (otherwise the
            # death would land in re-indexing rather than in the
            # LS request — a different scenario).
            first = await _narrow_call(session)
            assert first is not None

            # The barrier removes guesses about scheduling order: the
            # ``sent`` signal is set by the SERVICE itself, and only AFTER
            # returning from the real pipe write — i.e. the request has
            # already left the process and lies in Serena's pipe. We kill
            # strictly after that, so "the call was in flight" is a fact,
            # not a coincidence. Holding the request BEFORE the write
            # would be a mistake: then SIGKILL would hit an unsent
            # request, and the scenario would degrade into
            # request-after-death.
            #
            # Until ``release`` is created, the attempt is held on the
            # service side ALREADY AFTER the send: otherwise a fast
            # overview would get its answer before SIGKILL and the test
            # would go green without exercising the death path at all.
            sent_flag = barrier_dir / "sent"
            release_flag = barrier_dir / "release"

            # Arm the barrier only now: the warm-up call above must not
            # consume it, otherwise SIGKILL would hit the warm-up, the
            # observed call would never happen, and the test would still
            # go green.
            (barrier_dir / "arm").write_text("1")

            call = asyncio.ensure_future(_narrow_call(session))
            appeared = await _await_file(sent_flag, _BARRIER_WAIT_S)
            assert appeared, (
                "service did not mark the request as sent to Serena — "
                "the barrier did not fire, in-flight not proven"
            )

            victims = _serena_children(root)
            assert victims, "no live Serena process found"
            for pid in victims:
                reaper.note_descendants(pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # ``os.kill`` only SENDS the signal. The request itself went
            # out even earlier — before ``sent`` appeared — so the hold
            # affects not the send but only the delivery of the result:
            # while it is held, the wrapper does not give back the answer.
            # Releasing it right after ``os.kill`` would let a fast
            # overview return before the process actually died: the death
            # window would collapse and the death path would stay
            # untested. So we wait for the actual death.
            for pid in victims:
                assert _wait_pid_exit(pid, timeout_s=30.0), (
                    f"Serena {pid} did not die after SIGKILL — the death "
                    "window never opened, the in-flight scenario was not reproduced"
                )

            # Release the hold: from here the call must complete on its
            # own — via a retry on the recovered backend or an
            # explanatory text.
            release_flag.write_text("1")

            # Require exactly finiteness with a defined outcome. Without
            # the death watcher and the ``_call_or_death`` race the call
            # would hang for the whole ``_DISPATCH_BUDGET_S``, which is
            # caught here by the hard local timeout.
            try:
                result = await asyncio.wait_for(call, timeout=_TOOL_S)
                _assert_agent_safe(str(result))
            except ToolCallFailed as e:
                text = str(e)
                _assert_agent_safe(text)
                assert (
                    "NOT caused by concurrent tool calls" in text
                    or "NOT a limit on concurrent tool calls" in text
                ), f"answer must disprove the false hypothesis: {text}"
        finally:
            call.cancel()
            await session.close()

        channel = grpc.aio.insecure_channel(f"unix:{entry.socket_path}")
        stub = pbg.LspServiceStub(channel)
        try:
            status = await asyncio.wait_for(
                stub.ProjectStatus(pb.StatusRequest()), timeout=15.0
            )
        finally:
            await channel.close()

        # Exactly a retry — not a restart between calls: the death
        # happened while the request was already in flight, so the
        # dispatcher must have seen the terminated error and retried on a
        # new backend.
        assert status.ls_terminated_retries_total >= 1, (
            "in-flight kill did not lead to a retry: either the child-death "
            "watcher or the call_or_death race did not fire"
        )
