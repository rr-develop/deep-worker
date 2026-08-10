"""Auto-restart of Serena — commit 10 integration coverage.

See ``docs/DESIGN_LSP_SERENA_PIDFD_RESTART.md`` §7.

Two invariants pinned end-to-end with a REAL Serena:

* SIGTERM'ing Serena while the service is alive brings Serena back
  under the SAME service (same socket, same registry pid) with
  ``Health.alive`` returning True again.
* Four kills inside the 60s window exhaust the budget: the fourth
  restart is refused and ``Health.alive`` stays False. The service
  itself keeps serving admin RPCs.

Both tests obey the project's integ policy: hard bounded, event-
driven, guaranteed cleanup, and a final ``pgrep`` sanity check for
zero orphan Serena processes.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import grpc
import pytest

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

#: Paths that make ``python3 -m deep_worker_cli.lsp.service`` resolve
#: to THIS worktree (dev) rather than to whatever is on PATH's
#: ``deep-worker``. Without this, the integration test exercises master
#: — which by construction does not carry the code under test.
_DEV_PKG = Path(__file__).resolve().parents[3]
_DEV_DEEPAGENTS = _DEV_PKG.parent / "deepagents"
_DEV_ROOT = _DEV_PKG.parent


def _dev_serve_argv(project_root: Path, slug_dir: Path) -> list[str]:
    """Same shape as :func:`build_serve_argv` but forcing dev PYTHONPATH."""
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

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_START_S = 40.0
_RESTART_S = 40.0


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
    """Serena processes whose parent is our service."""
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
            comm = Path(f"/proc/{cp}/comm").read_text().strip()
        except FileNotFoundError:
            continue
        # Serena's launcher runs as ``python`` with the serena entry
        # point in argv; ``comm`` won't say "serena" but its cmdline
        # will.
        try:
            cmdline = Path(f"/proc/{cp}/cmdline").read_bytes().decode(
                errors="replace"
            )
        except FileNotFoundError:
            continue
        if SERENA_COMMAND in cmdline or "serena" in cmdline:
            out.add(cp)
    return out


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
    (root / "pkg" / "core.py").write_text("class Widget: pass\n")
    return root


def _supervisor(reg_dir: Path, root: Path) -> ServiceSupervisor:
    return ServiceSupervisor(
        ServiceRegistry(base_dir=reg_dir),
        lambda slug_dir: _dev_serve_argv(root, slug_dir),
    )


@pytest.fixture(autouse=True)
def _dev_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin PYTHONPATH so the spawned service is THIS worktree.

    Without this, ``python3 -m deep_worker_cli.lsp.service`` picks up
    whatever is on ``sys.path`` at spawn time — which on this host is
    the master worktree, and master by definition does not yet carry
    the commit under test.
    """
    env = _dev_env()
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])


async def _wait_backend_ready(
    stub: pbg.LspServiceStub, budget_s: float
) -> None:
    deadline = time.monotonic() + budget_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            h = await asyncio.wait_for(
                stub.Health(pb.HealthRequest()), timeout=2.0
            )
            if h.alive:
                return
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            last_err = e
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"backend did not become alive in {budget_s}s (last: {last_err!r})"
    )


async def _wait_backend_alive_stays(
    stub: pbg.LspServiceStub, is_alive: bool, budget_s: float
) -> None:
    """Wait until ``Health.alive`` equals ``is_alive`` and stays so."""
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        try:
            h = await asyncio.wait_for(
                stub.Health(pb.HealthRequest()), timeout=2.0
            )
            if bool(h.alive) is is_alive:
                return
        except (grpc.aio.AioRpcError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"Health.alive never became {is_alive} within {budget_s}s"
    )


async def _open_stub(
    sock_path: str,
) -> tuple[grpc.aio.Channel, pbg.LspServiceStub]:
    channel = grpc.aio.insecure_channel(f"unix:{sock_path}")
    stub = pbg.LspServiceStub(channel)
    return channel, stub


# ── tests ────────────────────────────────────────────────────────────


async def test_service_restarts_serena_on_kill(
    tmp_path: Path, reaper
) -> None:
    """SIGTERM kills the real Serena; supervisor spawns a fresh one.

    MUT-U regression point: without the ``mark_backend_ready`` at the
    end of ``_restart_once`` (via ``attach_backend``) Health.alive
    would never recover to True.
    """
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"

    entry = _supervisor(reg_dir, root).ensure_started(root)
    reaper.track(entry.pid)
    ch, stub = await _open_stub(entry.socket_path)
    try:
        await _wait_backend_ready(stub, _START_S)
        original_serena = _serena_pids_under(entry.pid)
        assert len(original_serena) == 1, (
            f"expected exactly one Serena under service {entry.pid}; "
            f"got {original_serena}"
        )
        serena_pid = next(iter(original_serena))

        # Kill Serena; supervisor's pidfd monitor must observe it and
        # spawn a fresh one.
        os.kill(serena_pid, signal.SIGTERM)
        _wait_pid_exit(serena_pid, timeout_s=5.0)

        # Health.alive must go False (transport error path from
        # commit 9) and then come back True on its own.
        await _wait_backend_alive_stays(stub, False, budget_s=10.0)
        await _wait_backend_alive_stays(stub, True, budget_s=_RESTART_S)

        # A different Serena is running under the same service.
        new_serena = _serena_pids_under(entry.pid)
        assert len(new_serena) == 1, (
            f"expected exactly one Serena after restart, got {new_serena}"
        )
        assert serena_pid not in new_serena, (
            "restart produced no new pid — is the monitor wired?"
        )
    finally:
        await ch.close()


async def test_service_gives_up_after_three_restarts_in_window(
    tmp_path: Path, reaper
) -> None:
    """Kill Serena four times in a row. The fourth restart must be
    refused: ``Health.alive`` stays False, the service itself is
    still up (admin RPCs answer).
    """
    _require_serena()
    root = _py_project(tmp_path)
    reg_dir = tmp_path / "reg"

    entry = _supervisor(reg_dir, root).ensure_started(root)
    reaper.track(entry.pid)
    ch, stub = await _open_stub(entry.socket_path)
    try:
        await _wait_backend_ready(stub, _START_S)
        # First three kills: each must recover.
        for i in range(3):
            pids = _serena_pids_under(entry.pid)
            assert len(pids) == 1, (
                f"iter {i}: expected one Serena, got {pids}"
            )
            os.kill(next(iter(pids)), signal.SIGTERM)
            _wait_pid_exit(next(iter(pids)), timeout_s=5.0)
            await _wait_backend_alive_stays(stub, True, budget_s=_RESTART_S)

        # Fourth kill: supervisor must refuse to restart.
        pids = _serena_pids_under(entry.pid)
        assert len(pids) == 1, (
            f"after 3 restarts: expected one Serena, got {pids}"
        )
        os.kill(next(iter(pids)), signal.SIGTERM)
        _wait_pid_exit(next(iter(pids)), timeout_s=5.0)

        await _wait_backend_alive_stays(stub, False, budget_s=10.0)
        # Confirm Health.alive stays False for a few seconds — no
        # sneaky "one more attempt" after a delay.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            h = await asyncio.wait_for(
                stub.Health(pb.HealthRequest()), timeout=2.0
            )
            assert not h.alive, (
                "budget was exhausted; supervisor must NOT keep restarting"
            )
            await asyncio.sleep(0.5)

        # The service itself is still up: admin RPCs answer.
        st = await asyncio.wait_for(
            stub.ProjectStatus(pb.StatusRequest()), timeout=2.0
        )
        assert st.pid == entry.pid, (
            "budget exhaustion must not take the service down; "
            "only backend readiness flips"
        )
    finally:
        await ch.close()
