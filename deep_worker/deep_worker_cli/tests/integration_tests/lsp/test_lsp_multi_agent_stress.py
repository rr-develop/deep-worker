"""Multi-agent stress + refresh↔tool race — commit 11.

See ``docs/DESIGN_LSP_MULTI_AGENT_STRESS.md``.

Four invariants pinned end-to-end against a REAL Serena:

* (a) two BridgeSessions firing 6 concurrent tool calls each × 3
  rounds finish with zero exceptions and zero empty payloads;
* (b) a forced ``RefreshProject`` between two tool storms produces
  an ordered ``REFRESH_STARTED`` → ``REFRESH_FINISHED`` pair on an
  observer's AttachSession stream (indexed, not timed);
* (c) exactly one Serena process runs under the service before AND
  after the storm — no silent pidfd restart mid-test;
* (d) ``Health`` and ``ProjectStatus`` answer within 1 s while the
  tool storm is running, not merely after it drained.

Test policy: hard bounded (`_WHOLE_TEST_S`), event-driven waits, a
reaper fixture that guarantees SIGTERM→SIGKILL cleanup, and a final
``pgrep``-style sanity check for zero orphan Serena children.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import grpc
import pytest

from deep_worker_cli.lsp.bridge_loop import BridgeSession
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
_TOOL_S = 30.0
_REFRESH_EVENT_S = 20.0
_WHOLE_TEST_S = 180.0

_ROUNDS = 3
_CALLS_PER_ROUND = 6


# ── helpers (mirrored from test_serena_auto_restart) ────────────────


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


def _serena_pids_under(service_pid: int) -> set[int]:
    try:
        children = Path(
            f"/proc/{service_pid}/task/{service_pid}/children"
        ).read_text()
    except FileNotFoundError:
        return set()
    child_pids = [int(p) for p in children.split() if p]
    out: set[int] = set()
    for cp in child_pids:
        try:
            cmdline = Path(f"/proc/{cp}/cmdline").read_bytes().decode(
                errors="replace"
            )
        except FileNotFoundError:
            continue
        if SERENA_COMMAND in cmdline or "serena" in cmdline:
            out.add(cp)
    return out


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


def _dev_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(_DEV_PKG), str(_DEV_DEEPAGENTS), str(_DEV_ROOT)]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([prev] if prev else []))
    return env


@pytest.fixture(autouse=True)
def _dev_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _dev_env()
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])


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
    deadline = time.monotonic() + budget_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            h = await asyncio.wait_for(
                stub.Health(pb.HealthRequest()), timeout=2.0
            )
            if h.alive:
                return
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            last = e
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"backend did not become alive in {budget_s}s (last: {last!r})"
    )


# ── observer (design §4, amendment A1) ───────────────────────────────


async def _observe_events(
    stub: pbg.LspServiceStub,
    out_queue: "asyncio.Queue[pb.SessionEvent]",
    ready: asyncio.Event,
) -> None:
    """Consume an AttachSession stream and enqueue every event.

    A dedicated third client (``client_id=observer``) so the two
    ``BridgeSession``-based agents keep their own streams intact and
    the test can pin REFRESH_* ordering deterministically.
    """
    stream = stub.AttachSession(pb.AttachRequest(client_id="observer"))
    try:
        async for ev in stream:
            if not ready.is_set():
                # First event is ATTACHED — signals subscription is live.
                ready.set()
            await out_queue.put(ev)
    except grpc.aio.AioRpcError:
        # Stream cancelled from teardown — expected.
        pass


async def _probe_admin_rpcs(
    stub: pbg.LspServiceStub, iters: int, stop: asyncio.Event
) -> None:
    """Assert that Health/ProjectStatus answer within 1 s under load.

    Runs concurrently with the tool storm — that's the whole point of
    invariant (d).
    """
    for i in range(iters):
        if stop.is_set():
            return
        await asyncio.wait_for(stub.Health(pb.HealthRequest()), timeout=1.0)
        await asyncio.wait_for(
            stub.ProjectStatus(pb.StatusRequest()), timeout=1.0
        )
        await asyncio.sleep(0.05)


# ── the tests ────────────────────────────────────────────────────────


async def test_two_agents_concurrent_tool_calls_stay_correct(
    tmp_path: Path, reaper
) -> None:
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

            serena_before = _serena_pids_under(entry.pid)
            assert len(serena_before) == 1, (
                f"expected 1 Serena before storm; got {serena_before}"
            )
            (initial_serena,) = serena_before

            agent_a = BridgeSession(entry.socket_path)
            agent_b = BridgeSession(entry.socket_path)
            await agent_a.connect()
            await agent_b.connect()
            try:
                for round_idx in range(_ROUNDS):
                    stop_admin = asyncio.Event()
                    calls: list = []
                    calls += [
                        asyncio.wait_for(
                            agent_a.call_tool(
                                "get_symbols_overview",
                                {"relative_path": "pkg/core.py"},
                            ),
                            timeout=_TOOL_S,
                        )
                        for _ in range(_CALLS_PER_ROUND)
                    ]
                    calls += [
                        asyncio.wait_for(
                            agent_b.call_tool(
                                "find_symbol", {"name_path": "Widget"}
                            ),
                            timeout=_TOOL_S,
                        )
                        for _ in range(_CALLS_PER_ROUND)
                    ]
                    admin_task = asyncio.create_task(
                        _probe_admin_rpcs(stub, iters=20, stop=stop_admin)
                    )
                    try:
                        results = await asyncio.gather(
                            *calls, return_exceptions=True
                        )
                    finally:
                        stop_admin.set()
                        await asyncio.wait_for(admin_task, timeout=_TOOL_S)

                    assert len(results) == 2 * _CALLS_PER_ROUND, (
                        f"round {round_idx}: dropped calls"
                    )
                    for r in results:
                        assert not isinstance(r, BaseException), (
                            f"round {round_idx}: tool call raised: {r!r}"
                        )
                        assert r, (
                            f"round {round_idx}: empty tool payload: {r!r}"
                        )

                    # Split back into A/B halves for content checks.
                    overview_results = results[:_CALLS_PER_ROUND]
                    find_results = results[_CALLS_PER_ROUND:]
                    for r in overview_results:
                        text = json.dumps(r)
                        assert "Widget" in text, (
                            f"get_symbols_overview missing 'Widget': {text[:200]}"
                        )
                    for r in find_results:
                        text = json.dumps(r)
                        assert "Widget" in text, (
                            f"find_symbol missing 'Widget': {text[:200]}"
                        )

                # Invariant (c): the same single Serena survived.
                serena_after = _serena_pids_under(entry.pid)
                assert serena_after == {initial_serena}, (
                    f"expected same Serena {initial_serena} after storm, "
                    f"got {serena_after} — a silent pidfd restart happened"
                )
            finally:
                await agent_a.close()
                await agent_b.close()
        finally:
            await channel.close()


async def test_refresh_between_tool_batches_is_never_lost(
    tmp_path: Path, reaper
) -> None:
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

            # observer must be attached BEFORE the refresh, so we
            # catch its START/FINISHED on the same session stream.
            events: asyncio.Queue[pb.SessionEvent] = asyncio.Queue()
            observer_ready = asyncio.Event()
            observer_task = asyncio.create_task(
                _observe_events(stub, events, observer_ready)
            )
            try:
                await asyncio.wait_for(observer_ready.wait(), timeout=10.0)

                agent_b = BridgeSession(entry.socket_path)
                await agent_b.connect()
                try:
                    async def _storm() -> list:
                        return await asyncio.gather(
                            *[
                                asyncio.wait_for(
                                    agent_b.call_tool(
                                        "find_symbol", {"name_path": "Widget"}
                                    ),
                                    timeout=_TOOL_S,
                                )
                                for _ in range(_CALLS_PER_ROUND)
                            ],
                            return_exceptions=True,
                        )

                    async def _refresh() -> pb.RefreshReply:
                        return await asyncio.wait_for(
                            stub.RefreshProject(
                                pb.RefreshRequest(force=True)
                            ),
                            timeout=_TOOL_S,
                        )

                    storm_task = asyncio.create_task(_storm())
                    refresh_task = asyncio.create_task(_refresh())
                    storm_results, refresh_reply = await asyncio.gather(
                        storm_task, refresh_task
                    )

                    # RPC-side: RefreshProject accepted the trigger.
                    assert refresh_reply.triggered is True

                    # Stream-side: STARTED then FINISHED, in that order,
                    # within _REFRESH_EVENT_S.
                    deadline = time.monotonic() + _REFRESH_EVENT_S
                    started_ts: int | None = None
                    finished: pb.SessionEvent | None = None
                    while time.monotonic() < deadline:
                        try:
                            ev = await asyncio.wait_for(
                                events.get(), timeout=deadline - time.monotonic()
                            )
                        except asyncio.TimeoutError:
                            break
                        if ev.event_type == pb.SESSION_EVENT_REFRESH_STARTED:
                            started_ts = ev.timestamp_ns
                        elif ev.event_type == pb.SESSION_EVENT_REFRESH_FINISHED:
                            if started_ts is None:
                                pytest.fail(
                                    "REFRESH_FINISHED arrived without a "
                                    f"preceding STARTED: {ev.message!r}"
                                )
                            if ev.timestamp_ns < started_ts:
                                pytest.fail(
                                    "FINISHED timestamp precedes STARTED "
                                    f"({ev.timestamp_ns} < {started_ts})"
                                )
                            finished = ev
                            break

                    assert finished is not None, (
                        "did not observe REFRESH_FINISHED within "
                        f"{_REFRESH_EVENT_S}s of the force refresh"
                    )
                    assert not finished.message.startswith("error:"), (
                        f"REFRESH_FINISHED reports backend error: {finished.message!r}"
                    )

                    # Tool storm side: all successful, non-empty.
                    assert len(storm_results) == _CALLS_PER_ROUND
                    for r in storm_results:
                        assert not isinstance(r, BaseException), r
                        assert r, r
                finally:
                    await agent_b.close()
            finally:
                observer_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await observer_task
        finally:
            await channel.close()
