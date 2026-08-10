"""``deep-worker lsp service-status`` shows live metrics — commit 12.

See ``docs/DESIGN_LSP_OBSERVABILITY.md`` §5.

Why an integration test: the unit tests pin the servicer fields, but
the thing that actually broke in earlier commits was the *wiring* —
``service-status`` talking to a real UDS of a real service process and
rendering what it got back. That stitch is invisible to unit tests.

Amendment A6: ``cmd_service_status`` is invoked directly from Python
(not through a ``deep-worker`` shim on ``PATH``) so this exercises the
dev tree, not whatever master build happens to be installed.

Bounded per the integ-test policy (feedback 2026-07-28): one hard
budget for the whole test, event-driven readiness wait, and a reaper
fixture that guarantees SIGTERM→SIGKILL of the service process.
"""
from __future__ import annotations

import argparse
import asyncio
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
from deep_worker_cli.lsp.service.registry import RegistryEntry, ServiceRegistry
from deep_worker_cli.lsp.service.service_cli import cmd_service_status
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
_WHOLE_TEST_S = 150.0


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


def _status_args(root: Path, reg_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(root), registry_dir=str(reg_dir)
    )


async def test_service_status_reports_backend_and_metrics(
    tmp_path: Path, reaper, capsys: pytest.CaptureFixture[str]
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

            # Drive one real tool call so tool_calls_total is non-zero
            # — a metric block that only ever shows zeros would pass a
            # weaker assertion while being completely unwired.
            agent = BridgeSession(entry.socket_path)
            await agent.connect()
            try:
                await asyncio.wait_for(
                    agent.call_tool(
                        "get_symbols_overview",
                        {"relative_path": "pkg/core.py"},
                    ),
                    timeout=_TOOL_S,
                )
            finally:
                await agent.close()
        finally:
            await channel.close()

        # cmd_service_status is synchronous and opens its own loop, so
        # it must run off this one.
        rc = await asyncio.to_thread(
            cmd_service_status, _status_args(root, reg_dir)
        )
        out = capsys.readouterr().out

    assert rc == 0, f"service-status should succeed; output was:\n{out}"
    # Registry block (A6: always printed).
    assert f"pid      : {entry.pid}" in out
    assert "transport:" in out
    # Live block.
    assert "backend  : alive=True" in out, out
    assert "restarts=0" in out, out
    assert "serena=?" not in out, (
        "serena_version must be reported by a live backend, not '?'"
    )
    assert "tool_calls=1" in out, out
    assert "last_tool_call_unix=0" not in out, out


async def test_service_status_reports_unreachable_socket(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A6: a registered-but-dead service must fail loudly, not silently.

    We fabricate the exact pathological state the registry cannot tell
    apart on its own: a live pid (this test process) plus a socket file
    that nothing is listening on. ``service-status`` has to notice the
    difference by actually talking to it.
    """
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"
    registry = ServiceRegistry(base_dir=reg_dir)
    slug_dir = registry.slug_dir(root)
    slug_dir.mkdir(parents=True, exist_ok=True)
    sock = slug_dir / "grpc.sock"
    sock.write_bytes(b"")  # a regular file, not a listening socket
    registry.write(
        RegistryEntry(
            project_root=root.resolve(),
            pid=os.getpid(),  # alive, so the staleness check passes
            transport="uds",
            socket_path=str(sock),
            started_at_ns=time.time_ns(),
        )
    )

    async with asyncio.timeout(60.0):
        rc = await asyncio.to_thread(
            cmd_service_status, _status_args(root, reg_dir)
        )
    out = capsys.readouterr().out
    assert rc == 1, f"dead socket must exit non-zero; output:\n{out}"
    assert "unreachable:" in out, out
