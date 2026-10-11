"""Lifecycle of the shared service, against real processes (§14).

The unit suite pins these rules against a fake clock and a fake registry.
This file pins them against the thing that actually runs: a detached
process, a UDS on disk, and a registry record that has to survive the
death of whoever spawned it.

Scenarios (design §14):

* two ``ensure_started`` calls → exactly one service process;
* the service outlives the process that spawned it (detach);
* killing it between two requests → the next request still succeeds;
* ``stop`` leaves no record and no socket behind;
* an idle service shuts itself down;
* socket and directory are private to the user.
"""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from deep_worker_cli.semantic_memory.client import SemanticMemoryClient

from .conftest import (
    BUDGET_S,
    ServiceRig,
    assert_pid_exits,
    kill_pid,
    pid_alive,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(BUDGET_S)]


def test_second_ensure_started_reuses_the_same_process(
    service_rig: ServiceRig,
) -> None:
    """One service per registry — the second caller borrows, not spawns.

    This is the whole premise of the feature: if two agents each got their
    own process, each would load its own copy of the model.
    """
    first_socket = service_rig.ensure_started()
    first_pid = service_rig.current_pid()

    second_socket = service_rig.supervisor.ensure_started()
    second_pid = service_rig.current_pid()

    assert first_pid is not None
    assert second_socket == first_socket
    assert second_pid == first_pid
    assert pid_alive(first_pid)


def test_concurrent_ensure_started_spawns_once(service_rig: ServiceRig) -> None:
    """Racing starters must collapse to one process, not N.

    The lock is what makes this true, and a lock is exactly the kind of
    thing that looks correct until two callers actually collide.
    """

    async def race() -> list[Path]:
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(service_rig.supervisor.ensure_started)
                    for _ in range(4)
                )
            )
        )

    sockets = asyncio.run(race())

    assert len({str(s) for s in sockets}) == 1
    pid = service_rig.current_pid()
    assert pid is not None and pid_alive(pid)

    # The registry names one live process, and nothing else answers on
    # that socket: a second service would have had to replace the record.
    state = service_rig.registry.read()
    assert state is not None and state.pid == pid


def test_service_outlives_its_spawner(tmp_path: Path) -> None:
    """A service started by a short-lived process keeps running.

    Agents come and go; the service they share must not die with the
    first one that happened to need it. The spawner here is a separate
    interpreter that exits immediately after starting the service.
    """
    base_dir = tmp_path / "semantic_memory"
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from deep_worker_cli.semantic_memory.registry import ServiceRegistry
        from deep_worker_cli.semantic_memory.service.__main__ import build_serve_argv
        from deep_worker_cli.semantic_memory.service.supervisor import ServiceSupervisor

        registry = ServiceRegistry(base_dir=Path(sys.argv[1]))
        supervisor = ServiceSupervisor(
            registry=registry, spawn_argv=build_serve_argv, start_timeout_s=30.0
        )
        socket_path = supervisor.ensure_started()
        state = registry.read()
        print(json.dumps({"pid": state.pid, "socket": str(socket_path)}))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script, str(base_dir)],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert proc.returncode == 0, proc.stderr

    import json

    reported = json.loads(proc.stdout.strip().splitlines()[-1])
    service_pid = int(reported["pid"])

    try:
        # The spawner is gone (subprocess.run reaped it) and the service
        # is still there.
        assert pid_alive(service_pid)
        assert Path(reported["socket"]).exists()
    finally:
        kill_pid(service_pid)


def test_request_after_crash_restarts_the_service(
    service_rig: ServiceRig,
) -> None:
    """A killed service is an ordinary event, not a dead end.

    Idle-shutdown makes disappearance normal, so the client's retry has
    to cover a hard crash too: one UNAVAILABLE, one restart, one answer.
    """
    client = SemanticMemoryClient(supervisor=service_rig.supervisor)

    first = asyncio.run(client.health())
    assert first.ready
    victim = service_rig.current_pid()
    assert victim is not None

    kill_pid(victim)
    assert_pid_exits(victim, timeout_s=10.0, what="service to die")

    second = asyncio.run(client.health())
    assert second.ready

    survivor = service_rig.current_pid()
    assert survivor is not None
    assert survivor != victim, "a new process must have taken over"
    assert pid_alive(survivor)


def test_stop_removes_process_socket_and_record(
    service_rig: ServiceRig,
) -> None:
    """Stopping cleans up after itself, leaving nothing to reclaim."""
    socket_path = service_rig.ensure_started()
    pid = service_rig.current_pid()
    assert pid is not None

    service_rig.supervisor.stop()

    assert_pid_exits(pid, timeout_s=10.0, what="service to exit")
    assert not socket_path.exists()
    assert service_rig.registry.read() is None
    assert not service_rig.supervisor.ready()


def test_idle_service_shuts_itself_down(short_idle_rig: ServiceRig) -> None:
    """Nobody asking for a while → the model stops occupying RAM.

    The rig starts this one with a 3 s idle window; production uses 30
    minutes, which no test can wait out.
    """
    short_idle_rig.ensure_started()
    pid = short_idle_rig.current_pid()
    assert pid is not None

    assert_pid_exits(
        pid, timeout_s=30.0, what="idle service to shut down on its own"
    )


def test_socket_and_directory_are_private(service_rig: ServiceRig) -> None:
    """Only the owner may reach the socket.

    The environment is trusted, but a world-writable UDS would let any
    local user drive another user's index — a needless door to leave open.
    """
    socket_path = service_rig.ensure_started()
    base_dir = service_rig.registry.base_dir

    dir_mode = stat.S_IMODE(os.stat(base_dir).st_mode)
    assert dir_mode == 0o700, f"base dir is {dir_mode:o}, want 700"

    sock_mode = stat.S_IMODE(os.stat(socket_path).st_mode)
    assert not sock_mode & 0o077, f"socket is {sock_mode:o}, want no group/other"
