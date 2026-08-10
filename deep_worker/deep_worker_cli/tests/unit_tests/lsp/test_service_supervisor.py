"""Supervisor contract for the per-project LSP service.

Covers the detach/lifecycle layer that lives BELOW the gRPC server.
gRPC + Serena wiring are covered by later commits — here we pin the
process-management invariants:

* ``spawn_detached()`` returns a pid that is NOT our child (setsid,
  double-fork), the pid is written to the registry, and killing our
  own process would not take the service with it.
* ``spawn_detached()`` is idempotent: a second call while the service
  is alive returns the existing pid, never spawns twice.
* Stale registry entries (pid dead) are reclaimed and a new service is
  started instead of returning the corpse.
* On ``stop()``, SIGTERM is sent to the pid, we wait bounded, then
  SIGKILL if still alive, and the registry is dropped.

No polling. Waits are event-driven (inotify on the socket path, or a
bounded ``os.waitpid``-style loop with a sentinel pipe).

These tests do NOT bring up Serena — they run the supervisor with a
sentinel child (a tiny sleep-loop) that writes its pid to state.json
and creates the ``grpc.sock`` file to signal readiness. That is
enough to exercise the entire supervision logic.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.registry import ServiceRegistry
from deep_worker_cli.lsp.service.supervisor import (
    ServiceAlreadyRunning,
    ServiceStartTimeout,
    ServiceSupervisor,
)


# Sentinel child: mimics a real service just enough for the supervisor
# tests. Reads the slug dir from argv, creates grpc.sock, then sleeps.
_SENTINEL = r"""
import os, sys, time, pathlib, signal
slug_dir = pathlib.Path(sys.argv[1])
sock = slug_dir / "grpc.sock"
# Simulate readiness: create the socket file the parent watches for.
sock.touch(mode=0o600)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
"""


def _sentinel_argv(slug_dir: Path) -> list[str]:
    return [sys.executable, "-c", _SENTINEL, str(slug_dir)]


@pytest.fixture
def sup(tmp_path: Path) -> ServiceSupervisor:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    # Fast start-timeout for tests (default 30s in prod).
    return ServiceSupervisor(
        registry=reg,
        spawn_argv=_sentinel_argv,
        start_timeout_s=5.0,
    )


def _wait_pid_gone(pid: int, timeout: float = 5.0) -> None:
    """Poll only for TEST teardown clarity — not the code under test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} still alive after {timeout}s")


# ── spawn & detach ───────────────────────────────────────────────────


def test_spawn_returns_pid_and_persists_registry(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    project = tmp_path / "proj_a"
    project.mkdir()
    entry = sup.ensure_started(project)
    try:
        assert entry.pid > 0
        assert os.kill(entry.pid, 0) is None  # alive

        loaded = sup.registry.read(project)
        assert loaded is not None and loaded.pid == entry.pid
    finally:
        sup.stop(project)
        _wait_pid_gone(entry.pid)


def test_spawn_detaches_from_current_process(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    """The service must NOT be a child of us — setsid + double-fork.

    Verified via /proc: the service's PPID must be 1 (init) or some
    non-us process. If the child's PPID were our own pid, killing us
    would take it down (that is exactly what this policy forbids).
    """
    project = tmp_path / "proj_b"
    project.mkdir()
    entry = sup.ensure_started(project)
    try:
        ppid = int(
            Path(f"/proc/{entry.pid}/status")
            .read_text()
            .split("PPid:")[1]
            .split()[0]
        )
        assert ppid != os.getpid(), (
            f"service pid {entry.pid} is our child (ppid={ppid}=us) — "
            f"setsid/double-fork did not detach it"
        )
    finally:
        sup.stop(project)
        _wait_pid_gone(entry.pid)


def test_spawn_never_forks_this_process(sup: ServiceSupervisor) -> None:
    """The supervisor must not call ``os.fork()`` in OUR address space.

    Callers are agents and bridges, which are always multi-threaded
    (asyncio plus provider stacks). ``fork()`` from a multi-threaded
    parent can deadlock the child — CPython warns about exactly this.
    Detaching is achieved with a throwaway launcher process instead,
    so this is an AST invariant, not a style preference.
    """
    import ast
    import inspect

    from deep_worker_cli.lsp.service import supervisor as sup_mod

    src = Path(inspect.getfile(sup_mod)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fork", "forkpty"}
        and getattr(node.func.value, "id", "") == "os"
    ]
    assert offenders == [], (
        f"os.fork() called at line(s) {offenders} — unsafe from a "
        f"multi-threaded parent; use the launcher-process pattern"
    )


def test_spawn_emits_no_fork_deprecation_warning(
    sup: ServiceSupervisor, tmp_path: Path, recwarn
) -> None:
    """Runtime counterpart to the AST guard above."""
    project = tmp_path / "proj_nofork"
    project.mkdir()
    entry = sup.ensure_started(project)
    try:
        forky = [
            w
            for w in recwarn
            if issubclass(w.category, DeprecationWarning)
            and "fork" in str(w.message).lower()
        ]
        assert forky == [], f"fork DeprecationWarning raised: {forky}"
    finally:
        sup.stop(project)
        _wait_pid_gone(entry.pid)


def test_ensure_started_is_idempotent(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    project = tmp_path / "proj_c"
    project.mkdir()
    first = sup.ensure_started(project)
    try:
        second = sup.ensure_started(project)
        assert second.pid == first.pid, (
            "ensure_started must return the existing service, not spawn twice"
        )
    finally:
        sup.stop(project)
        _wait_pid_gone(first.pid)


def test_stale_entry_is_reclaimed(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    """A registry pointing at a dead pid must be replaced, not honoured."""
    project = tmp_path / "proj_d"
    project.mkdir()

    from deep_worker_cli.lsp.service.registry import RegistryEntry

    # Manufacture a dead-pid entry.
    dead_pid = os.fork()
    if dead_pid == 0:
        os._exit(0)
    os.waitpid(dead_pid, 0)

    stale = RegistryEntry(
        project_root=project.resolve(),
        pid=dead_pid,
        transport="uds",
        socket_path=str(sup.registry.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )
    sup.registry.write(stale)

    fresh = sup.ensure_started(project)
    try:
        assert fresh.pid != dead_pid
        assert not sup.registry.is_stale(fresh)
    finally:
        sup.stop(project)
        _wait_pid_gone(fresh.pid)


# ── readiness wait is event-driven, not polling ──────────────────────


def test_ensure_started_times_out_if_socket_never_appears(
    tmp_path: Path,
) -> None:
    """If the child never creates the readiness marker, we must time
    out with a clear error — no infinite wait."""
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")

    def broken_argv(slug_dir: Path) -> list[str]:
        # Child sleeps forever, never touches grpc.sock.
        return [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ]

    sup = ServiceSupervisor(
        registry=reg, spawn_argv=broken_argv, start_timeout_s=0.5
    )
    project = tmp_path / "proj_e"
    project.mkdir()
    with pytest.raises(ServiceStartTimeout):
        sup.ensure_started(project)
    # The broken child must not linger.
    entries_left = list((tmp_path / "lsp").rglob("state.json"))
    assert entries_left == []


# ── stop ─────────────────────────────────────────────────────────────


def test_stop_terminates_service_and_drops_registry(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    project = tmp_path / "proj_f"
    project.mkdir()
    entry = sup.ensure_started(project)
    sup.stop(project)
    _wait_pid_gone(entry.pid)
    assert sup.registry.read(project) is None


def test_stop_is_noop_when_nothing_running(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    project = tmp_path / "proj_g"
    project.mkdir()
    sup.stop(project)  # must not raise
    assert sup.registry.read(project) is None


# ── ready() predicate ────────────────────────────────────────────────


def test_ready_reflects_socket_existence(
    sup: ServiceSupervisor, tmp_path: Path
) -> None:
    project = tmp_path / "proj_h"
    project.mkdir()
    assert sup.ready(project) is False
    entry = sup.ensure_started(project)
    try:
        assert sup.ready(project) is True
        # Removing the socket manually flips ready() back to False:
        Path(entry.socket_path).unlink()
        assert sup.ready(project) is False
    finally:
        sup.stop(project)
        _wait_pid_gone(entry.pid)


def test_start_race_second_caller_waits_and_reuses(
    tmp_path: Path,
) -> None:
    """Two concurrent ensure_started calls MUST NOT spawn two services.

    The supervisor takes an flock on <slug_dir>/.spawn.lock so a race
    between two agents (or two workflow steps) never doubles up.
    """
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_i"
    project.mkdir()

    import threading

    results: list = []

    def spawn(sup: ServiceSupervisor) -> None:
        try:
            results.append(sup.ensure_started(project))
        except Exception as e:  # noqa: BLE001
            results.append(e)

    sup_a = ServiceSupervisor(
        registry=reg, spawn_argv=_sentinel_argv, start_timeout_s=5.0
    )
    sup_b = ServiceSupervisor(
        registry=reg, spawn_argv=_sentinel_argv, start_timeout_s=5.0
    )
    t1 = threading.Thread(target=spawn, args=(sup_a,))
    t2 = threading.Thread(target=spawn, args=(sup_b,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert len(results) == 2 and all(hasattr(r, "pid") for r in results), results
    assert results[0].pid == results[1].pid, (
        f"race spawned two services: {results[0].pid} vs {results[1].pid}"
    )
    sup_a.stop(project)
    _wait_pid_gone(results[0].pid)
