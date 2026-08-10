"""Concurrent tool calls must not fail — tested against a LIVE Serena.

See ``docs/DESIGN_LSP_TRANSIENT_CANCEL_RETRY.md`` §6.

Why this test exists: the agent, on receiving a transient cancellation
``cancelled (-32800)`` from the language server, concluded that
"parallel calls cancel each other; serena must be called strictly
sequentially". The conclusion is wrong — calls are serialized by a
lock, not cancelled — but this can only be verified on the live stack,
without mocks: it is precisely the seam "bridge → gRPC → lock → MCP →
Serena" that unit tests do not cover.

Test invariant 1: N simultaneous calls from TWO independent sessions
complete WITHOUT a single exception. This is a direct check of the
user requirement "they must execute, not fail".

Honest caveat (accepted at review as non-blocking): the test does not
guarantee that ``-32800`` actually occurs during the run — the
reindexing window cannot be reproduced on demand. The evidential
burden for the retry is carried by unit tests and mutations; here we
catch concurrency regressions and verify that the metric reached the
CLI.

Integration-test policy: a strict overall budget, event-driven
readiness waiting, and a reaper with guaranteed SIGTERM→SIGKILL.
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
from deep_worker_cli.lsp.service.registry import ServiceRegistry
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
_WHOLE_TEST_S = 180.0

#: Simultaneous calls per round (3 from each session).
_CONCURRENT = 6


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


async def test_concurrent_tool_calls_all_succeed(
    tmp_path: Path, reaper
) -> None:
    """User requirement: simultaneous calls must execute.

    We catch exceptions EXPLICITLY: ``return_exceptions=True`` plus an
    explicit check, otherwise the first failed call would mask the rest.
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

        agent_a = BridgeSession(entry.socket_path)
        agent_b = BridgeSession(entry.socket_path)
        await agent_a.connect()
        await agent_b.connect()
        try:
            calls = []
            for _ in range(_CONCURRENT // 2):
                calls.append(
                    asyncio.wait_for(
                        agent_a.call_tool(
                            "get_symbols_overview",
                            {"relative_path": "pkg/core.py"},
                        ),
                        timeout=_TOOL_S,
                    )
                )
                calls.append(
                    asyncio.wait_for(
                        agent_b.call_tool(
                            "find_symbol", {"name_path": "Widget"}
                        ),
                        timeout=_TOOL_S,
                    )
                )
            results = await asyncio.gather(*calls, return_exceptions=True)
        finally:
            await agent_a.close()
            await agent_b.close()

    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, (
        f"{len(failures)}/{len(results)} concurrent tool calls failed; "
        f"concurrent calls must be QUEUED, never cancelled: {failures!r}"
    )
    assert all(r is not None for r in results), (
        "every concurrent call must return a payload"
    )


async def test_service_status_exposes_transient_retries(
    tmp_path: Path, reaper, capsys: pytest.CaptureFixture[str]
) -> None:
    """The metric reached the CLI (§4.5).

    ``N >= 0`` is deliberately a weak assertion: the reindexing window
    cannot be reproduced on demand. We verify the plumbing, not the
    value.
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

        rc = await asyncio.to_thread(
            cmd_service_status,
            argparse.Namespace(
                project_root=str(root), registry_dir=str(reg_dir)
            ),
        )
        out = capsys.readouterr().out

    assert rc == 0, out
    assert "transient_retries=" in out, out
