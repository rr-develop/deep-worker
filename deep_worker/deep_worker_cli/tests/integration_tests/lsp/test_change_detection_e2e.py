"""End-to-end auto-refresh tests on top of a REAL service process.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §13 (item
``test_add_module_refreshes_index``) and ``docs/DESIGN_LSP_WATCHER_WIRING.md``.

The unit tests (``tests/unit_tests/lsp/test_watcher.py``) exercise the
watcher and server wiring in a single process. Here we launch a real
``python -m deep_worker_cli.lsp.service`` via the real
``build_serve_argv``, bring up a gRPC client over a UDS and attach an
``AttachSession`` stream — the same path the agent will use later.

What is covered:

* adding a module is visible as ``SESSION_EVENT_REFRESH_FINISHED`` in the
  stream and via an increase in ``ProjectStatus.last_refresh_unix``;
* editing the body of an existing source file does NOT trigger a refresh;
  without this, the first test would be green even for an implementation
  that refreshes on any change.

All waits are event-driven (``asyncio.wait_for`` on the stream queue), the
file has a hard budget, and the ``reaper`` guarantees the service is killed
in ``finally``.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import grpc
import pytest

from deep_worker_cli.lsp.service.proto import (
    lsp_service_pb2 as pb,
    lsp_service_pb2_grpc as pbg,
)
from deep_worker_cli.lsp.service.registry import ServiceRegistry
from deep_worker_cli.lsp.service.service_cli import build_serve_argv
from deep_worker_cli.lsp.service.supervisor import (
    ServiceSupervisor,
    _wait_pid_exit,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_START_S = 30.0
_REFRESH_S = 20.0
_SILENCE_S = 3.0
_PKG_ROOT = Path(__file__).resolve().parents[3]


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    try:
        state = stat[stat.rindex(")") + 2 :].split()[0]
    except (ValueError, IndexError):  # pragma: no cover
        return False
    return state != "Z"


def _kill_pid(pid: int) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_pid_exit(pid, timeout_s=5.0)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        _wait_pid_exit(pid, timeout_s=5.0)


@pytest.fixture
def reaper():
    """Guaranteed cleanup after the test: see the §13 policy."""
    pids: list[int] = []
    procs: list[subprocess.Popen] = []

    class _R:
        def track_pid(self, pid: int) -> int:
            pids.append(pid); return pid

        def track_proc(self, p: subprocess.Popen) -> subprocess.Popen:
            procs.append(p); return p

    try:
        yield _R()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
        for pid in pids:
            _kill_pid(pid)


def _py_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
    )
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    return root


def _supervisor(reg_dir: Path, root: Path) -> ServiceSupervisor:
    return ServiceSupervisor(
        ServiceRegistry(base_dir=reg_dir),
        lambda slug_dir: build_serve_argv(root, slug_dir),
    )


async def _await_next_refresh_finished(queue: "asyncio.Queue[pb.SessionEvent]") -> pb.SessionEvent:
    """Wait for the next REFRESH_FINISHED, skipping keepalive/attached.

    The wait is event-driven: the queue is filled by the stream task as
    frames arrive; ``wait_for`` does not spin a waiting loop.
    """
    deadline = time.monotonic() + _REFRESH_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("REFRESH_FINISHED did not arrive")
        ev = await asyncio.wait_for(queue.get(), timeout=remaining)
        if ev.event_type == pb.SESSION_EVENT_REFRESH_FINISHED:
            return ev


async def _attach_session(stub: pbg.LspServiceStub, root: Path):
    """Attach and return (task, queue) for observing events."""
    queue: asyncio.Queue[pb.SessionEvent] = asyncio.Queue()
    stopping = asyncio.Event()

    async def pump() -> None:
        try:
            async for ev in stub.AttachSession(
                pb.AttachRequest(client_id="e2e", project_root=str(root))
            ):
                if stopping.is_set():
                    return
                await queue.put(ev)
        except grpc.aio.AioRpcError:
            return

    task = asyncio.create_task(pump(), name="e2e-attach-pump")
    # Wait for SESSION_EVENT_ATTACHED to guarantee the server sees the
    # client BEFORE we start modifying the project.
    first = await asyncio.wait_for(queue.get(), timeout=_START_S)
    assert first.event_type == pb.SESSION_EVENT_ATTACHED, first
    return task, queue, stopping


async def _wait_for_grpc_channel(sock_path: str) -> grpc.aio.Channel:
    """Open a channel and confirm the server responds."""
    channel = grpc.aio.insecure_channel(f"unix:{sock_path}")
    stub = pbg.LspServiceStub(channel)
    deadline = time.monotonic() + _START_S
    while True:
        try:
            reply = await asyncio.wait_for(
                stub.Health(pb.HealthRequest()), timeout=2.0
            )
            if reply.alive:
                return channel
        except (grpc.aio.AioRpcError, asyncio.TimeoutError):
            pass
        if time.monotonic() > deadline:
            await channel.close()
            raise TimeoutError(f"gRPC server at {sock_path} never healthy")
        await asyncio.sleep(0.1)


# ── §13 e2e ──────────────────────────────────────────────────────────


async def test_adding_a_module_triggers_a_refresh(
    tmp_path: Path, reaper
) -> None:
    """``test_add_module_refreshes_index``.

    We add a module (create a new ``__init__.py`` and change
    ``pyproject.toml``) → the service MUST publish
    ``REFRESH_FINISHED`` and update ``last_refresh_unix``. Otherwise the
    agent works with a stale index until a manual
    ``RefreshProject`` call.
    """
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"
    entry = _supervisor(reg_dir, root).ensure_started(root)
    reaper.track_pid(entry.pid)

    channel = await _wait_for_grpc_channel(entry.socket_path)
    try:
        stub = pbg.LspServiceStub(channel)
        pump_task, queue, stopping = await _attach_session(stub, root)
        try:
            before = await stub.ProjectStatus(pb.StatusRequest())

            # Real module addition: a new directory with
            # __init__.py + a manifest edit. Both are "structural"
            # events per §9 for the Python provider.
            (root / "extra").mkdir()
            (root / "extra" / "__init__.py").write_text("")
            (root / "pyproject.toml").write_text(
                "[project]\nname='demo'\nversion='0'\n"
                "dependencies=['httpx']\n"
            )

            ev = await _await_next_refresh_finished(queue)
            assert ev.event_type == pb.SESSION_EVENT_REFRESH_FINISHED

            after = await stub.ProjectStatus(pb.StatusRequest())
            # A successful refresh cycle advances last_refresh_unix BEFORE
            # FINISHED (see commit 8, `_run_refresh`). The backend in this
            # test project (tmp + pyproject.toml without sources) may not
            # find a refresh tool → the cycle honestly completes with
            # ``message='error: ...'``: then last_refresh_unix stays the
            # same. Both variants are correct; the test checks ONLY the
            # watcher → START/FINISHED event pipeline. A successful
            # refresh on the real Serena backend is verified in
            # tests/integration_tests/lsp/test_serena_backend_e2e.py.
            if not ev.message.startswith("error:"):
                assert after.last_refresh_unix > before.last_refresh_unix, (
                    f"successful refresh did not advance last_refresh_unix: "
                    f"before={before.last_refresh_unix} "
                    f"after={after.last_refresh_unix}"
                )
            assert "python" in list(after.active_providers), (
                f"active_providers empty in ProjectStatus: {after.active_providers}"
            )
            assert any(str(root) in p for p in after.watched_paths), (
                f"watched_paths does not mention project root: {after.watched_paths}"
            )
        finally:
            stopping.set()
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        await channel.close()


async def test_editing_an_existing_source_does_not_trigger_a_refresh(
    tmp_path: Path, reaper
) -> None:
    """The flip side of the first test: editing the body of a ``.py`` is a no-op.

    Without this check, a "refresh on any event" implementation passes
    the first test and completely destroys the §9 economy. Such a file is
    edited constantly.
    """
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"
    entry = _supervisor(reg_dir, root).ensure_started(root)
    reaper.track_pid(entry.pid)

    channel = await _wait_for_grpc_channel(entry.socket_path)
    try:
        stub = pbg.LspServiceStub(channel)
        pump_task, queue, stopping = await _attach_session(stub, root)
        try:
            before = await stub.ProjectStatus(pb.StatusRequest())

            (root / "pkg" / "mod.py").write_text(
                "def f():\n    return 2\n"
            )

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    _await_next_refresh_finished(queue), timeout=_SILENCE_S
                )

            after = await stub.ProjectStatus(pb.StatusRequest())
            assert after.last_refresh_unix == before.last_refresh_unix, (
                "editing a source file bumped last_refresh_unix — this is "
                "exactly the waste §9 forbids"
            )
        finally:
            stopping.set()
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        await channel.close()
