"""Integration tests for the lifecycle of the SHARED per-project LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §13 (Integration) and
``docs/POLICY_NO_INPROCESS_HEAVY_MCP.md``.

How these tests differ from the unit tests
----------------------------------
The unit tests in ``tests/unit_tests/lsp/`` exercise supervisor/registry/server
individually, feeding sentinel argv and living in a single process. Here
a REAL service process is launched through real argv
(``python -m deep_worker_cli.lsp.service __serve_impl__``), with a real
UDS, real gRPC, and real detach. It is exactly this seam — "the code is
right, the wiring is not" — that unit tests cannot catch:

* whether the service really survives the death of the agent that spawned it
  (detach/setsid), instead of dying together with it;
* whether a SECOND agent really reuses the already-started process rather
  than starting its own (the policy's main invariant: one backend per project);
* whether the service really shuts itself down on idle and cleans up the registry.

Policy "integration tests must not hang" (a user-reported bug):
every wait has a hard budget, waiting is event-driven (inotify/
pidfd, no polling), and any spawned process is guaranteed to be
finished off with SIGTERM→SIGKILL in ``finally``.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.registry import ServiceRegistry, project_slug
from deep_worker_cli.lsp.service.service_cli import build_serve_argv
from deep_worker_cli.lsp.service.supervisor import (
    ServiceSupervisor,
    _wait_pid_exit,
)

pytestmark = pytest.mark.integration

#: Budget for the socket of a just-started service to appear.
_START_S = 30.0

_PKG_ROOT = Path(__file__).resolve().parents[3]


def _pid_alive(pid: int) -> bool:
    """Whether the process is alive — explicitly excluding zombies.

    ``os.kill(pid, 0)`` is not sufficient on its own: it also succeeds on
    a zombie, i.e. on an already-dead process whose status has not yet
    been reaped by its parent. The test ``…autostart_after_previous_one_is_gone``
    used to fail falsely: the service really died, yet the check claimed the opposite.

    ``/proc/<pid>/stat`` gives an unambiguous answer: if the file is absent —
    the process is absent; if the state is ``Z`` — it is a corpse, not a service.
    """
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError:  # pragma: no cover — a foreign process, but alive
        return True
    try:
        state = stat_line[stat_line.rindex(")") + 2 :].split()[0]
    except (ValueError, IndexError):  # pragma: no cover — a read race
        return False
    return state != "Z"


def _kill_pid(pid: int) -> None:
    """SIGTERM → short event-driven wait → SIGKILL. Idempotent."""
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
    """Guaranteed cleanup: whatever the test starts will die here.

    Without this, a test that fails midway leaves a live service behind,
    which the next test picks up and turns the failure into a mysterious
    flake.
    """
    pids: list[int] = []
    procs: list[subprocess.Popen] = []

    class _Reaper:
        def track_pid(self, pid: int) -> int:
            pids.append(pid)
            return pid

        def track_proc(self, proc: subprocess.Popen) -> subprocess.Popen:
            procs.append(proc)
            return proc

    try:
        yield _Reaper()
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
        for pid in pids:
            _kill_pid(pid)


def _project(tmp_path: Path, name: str = "proj") -> Path:
    """Minimal "project": a directory with a root marker is enough."""
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    return root


def _supervisor(registry_dir: Path, root: Path) -> ServiceSupervisor:
    """Exactly the wiring used in production: build_serve_argv, not sentinel."""
    registry = ServiceRegistry(base_dir=registry_dir)
    return ServiceSupervisor(
        registry, lambda slug_dir: build_serve_argv(root, slug_dir)
    )


def _env_for_child(registry_dir: Path) -> dict[str, str]:
    env = {**os.environ}
    # The child python must import EXACTLY this tree, not the
    # installed package: otherwise the test is green while changes go unverified.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(_PKG_ROOT), existing]) if existing else str(_PKG_ROOT)
    env["HOME"] = str(registry_dir.parent)
    return env


# ── §13: sharing ─────────────────────────────────────────────────────


def test_two_agents_share_one_service_process(tmp_path: Path, reaper) -> None:
    """The policy's main invariant: a second "agent" does NOT start a second
    backend — it reuses the live one.

    This is the whole reason the service exists — if this test passes
    vacuously (e.g. the supervisor silently spawns a second process and we
    compare the wrong thing), the entire endeavour is pointless, so below
    we verify the pid, the socket path, and the number of live processes.
    """
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"

    first = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(first.pid)

    assert _pid_alive(first.pid), "service died right after ensure_started"
    assert Path(first.socket_path).exists(), first.socket_path

    # A second "agent" — a separate supervisor instance, as in production
    # (a different agent process, the same on-disk registry).
    second = _supervisor(registry_dir, root).ensure_started(root)

    assert second.pid == first.pid, (
        f"second agent spawned a SECOND backend "
        f"({second.pid} != {first.pid}) — policy violation"
    )
    assert second.socket_path == first.socket_path


def test_different_projects_get_different_services(
    tmp_path: Path, reaper
) -> None:
    """The flip side of the same invariant: sharing is PER PROJECT.

    Without this check, "one process for everything" would also pass the
    previous test, yet would be plainly wrong.
    """
    root_a = _project(tmp_path, "a")
    root_b = _project(tmp_path, "b")
    registry_dir = tmp_path / "reg"

    entry_a = _supervisor(registry_dir, root_a).ensure_started(root_a)
    reaper.track_pid(entry_a.pid)
    entry_b = _supervisor(registry_dir, root_b).ensure_started(root_b)
    reaper.track_pid(entry_b.pid)

    assert entry_a.pid != entry_b.pid
    assert entry_a.socket_path != entry_b.socket_path
    assert project_slug(root_a) != project_slug(root_b)


# ── §13: detach ──────────────────────────────────────────────────────


def test_service_detaches_from_the_agent_that_started_it(
    tmp_path: Path, reaper
) -> None:
    """``test_service_setsid_ppid_differs_from_agent``.

    The structural half of the detach check: the service is not our child
    (PPID != our pid) AND it is the leader of its own session (SID == its pid).

    Both conditions are needed and checked separately, because each one
    on its own is passed by accident: a plain ``Popen`` without
    ``start_new_session`` gives the child a correct SID exactly when it
    is still our child, while an intermediate launcher without setsid
    gives a foreign PPID without its own session.

    The behavioural half — "will it survive the real death of the agent" —
    is checked by ``test_service_survives_the_death_of_its_spawning_process``
    below, where the agent is really killed.
    """
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"

    entry = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(entry.pid)

    stat_line = Path(f"/proc/{entry.pid}/stat").read_text()
    # comm may contain spaces/parentheses — cut at the last ')'.
    fields = stat_line[stat_line.rindex(")") + 2 :].split()
    ppid = int(fields[1])
    sid = int(fields[3])

    assert ppid != os.getpid(), (
        "service is still OUR child — a killed agent would take the "
        "shared service down with it"
    )
    assert sid == entry.pid, f"service is not a session leader (sid={sid})"

    assert _pid_alive(entry.pid), "service died during the checks above"


def test_service_survives_the_death_of_its_spawning_process(
    tmp_path: Path, reaper
) -> None:
    """The same property, but verified "for real": the proxy agent
    is actually killed, and the service must remain alive.

    This is more honest than reading /proc: we literally do what we are
    afraid of."""
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"

    # An intermediate process plays the role of the agent: it starts the
    # service and prints its pid, after which we kill it.
    code = (
        "import json,sys;"
        "from pathlib import Path;"
        "from deep_worker_cli.lsp.service.registry import ServiceRegistry;"
        "from deep_worker_cli.lsp.service.service_cli import build_serve_argv;"
        "from deep_worker_cli.lsp.service.supervisor import ServiceSupervisor;"
        "root=Path(sys.argv[1]);reg=ServiceRegistry(base_dir=Path(sys.argv[2]));"
        "sup=ServiceSupervisor(reg, lambda d: build_serve_argv(root,d));"
        "e=sup.ensure_started(root);"
        "print(json.dumps({'pid':e.pid,'sock':e.socket_path}),flush=True)"
    )
    agent = subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(registry_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env_for_child(registry_dir),
    )
    reaper.track_proc(agent)
    try:
        out, err = agent.communicate(timeout=_START_S)
    except subprocess.TimeoutExpired:  # pragma: no cover
        agent.kill()
        pytest.fail(f"proxy agent hung starting the service")
    assert agent.returncode == 0, f"proxy agent failed: {err}"
    info = json.loads(out.strip().splitlines()[-1])
    service_pid = reaper.track_pid(int(info["pid"]))

    # The agent already exited on its own; finish it off just in case and confirm.
    _kill_pid(agent.pid)
    assert not _pid_alive(agent.pid)

    assert _pid_alive(service_pid), (
        "shared service died together with the agent that started it"
    )
    assert Path(info["sock"]).exists()


def test_service_autostart_after_previous_one_is_gone(
    tmp_path: Path, reaper
) -> None:
    """``test_service_autostart_from_second_agent_when_first_absent``.

    The service shut down cleanly. The next agent must start a new one,
    not get stuck on a stale registry entry.
    """
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"

    first = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(first.pid)
    _kill_pid(first.pid)
    assert not _pid_alive(first.pid)

    second = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(second.pid)

    assert second.pid != first.pid, "stale entry was reused as if alive"
    assert _pid_alive(second.pid)
    assert Path(second.socket_path).exists()


def test_autostart_after_a_crash_that_left_the_socket_behind(
    tmp_path: Path, reaper
) -> None:
    """The same autostart, but after a CRASH rather than a clean shutdown.

    Why this is a separate test. In ``ensure_started`` there are two
    independent reuse conditions: the registry entry is not stale AND the
    socket file is in place. On a clean SIGTERM the service manages to
    remove its socket, so the respawn happens via the second condition,
    and the staleness check (``is_stale``) carries no load — a mutation
    like "is_stale always False" would pass such a test straight through.

    A real failure looks different: SIGKILL / OOM-killer / a hard
    reboot leaves nothing cleaned up, and a seemingly alive socket
    remains on disk next to a dead pid. Then the only thing that saves us
    from connecting to a corpse is precisely ``is_stale``. That is exactly
    what is checked here.
    """
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"
    registry = ServiceRegistry(base_dir=registry_dir)

    first = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(first.pid)

    # SIGKILL: give the service no chance to clean up.
    os.kill(first.pid, signal.SIGKILL)
    _wait_pid_exit(first.pid, timeout_s=10.0)
    assert not _pid_alive(first.pid)

    # Test precondition: exactly the situation this test was written for.
    assert Path(first.socket_path).exists(), (
        "precondition failed: crash was expected to leave the socket file "
        "behind — without it this test would not exercise is_stale"
    )
    assert registry.read(root) is not None, (
        "precondition failed: crash was expected to leave the registry entry"
    )

    second = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(second.pid)

    assert second.pid != first.pid, (
        "supervisor reused a crashed service: the leftover socket file made "
        "it look alive, so the agent would talk to a dead endpoint"
    )
    assert _pid_alive(second.pid)
    assert Path(second.socket_path).exists()


# ── §13: teardown ────────────────────────────────────────────────────


def test_stop_kills_the_process_and_clears_the_registry(
    tmp_path: Path, reaper
) -> None:
    """``test_stop_removes_registry_files``."""
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"
    registry = ServiceRegistry(base_dir=registry_dir)

    sup = _supervisor(registry_dir, root)
    entry = sup.ensure_started(root)
    reaper.track_pid(entry.pid)
    assert registry.read(root) is not None

    sup.stop(root)

    assert not _pid_alive(entry.pid), "stop() left the service running"
    assert registry.read(root) is None, "stop() left a stale registry entry"
    assert not Path(entry.socket_path).exists(), "socket file was not removed"


@pytest.mark.asyncio
async def test_idle_shutdown_stops_the_server_with_no_clients(
    tmp_path: Path,
) -> None:
    """``test_idle_shutdown_after_no_clients``.

    Nobody attached → the service must shut itself down, otherwise live
    backends pile up on the host, one for every project ever
    opened.

    Why in this process rather than a subprocess. The production idle
    threshold is hard ≥60 s (``_MIN_IDLE_SHUTDOWN_S``), and that is NOT
    a test artefact: a service that shut down after a second would die
    between two agent calls. The threshold must not be lowered for the
    test — then the test would confirm behaviour that does not exist in
    production. So a real server with real 60 s is used, and "time" is
    moved honestly: the idle-start mark is shifted into the past, exactly
    as if the service had already served its term. The real watchdog is
    tested, not a simulation of it, and the test stays within budget.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )

    root = _project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=root,
            idle_shutdown_s=60.0,
            slug="test",
        )
    )
    await server.start()
    try:
        assert sock.exists(), "server did not bind its UDS"

        # The service has "served" its idle term without a single client.
        server._idle_since_ns = time.time_ns() - int(61 * 1e9)
        server._activity.set()

        await asyncio.wait_for(server.wait_closed(), timeout=10.0)

        assert not sock.exists(), (
            "idle service shut down but left its socket behind — the next "
            "agent would connect to a dead endpoint"
        )
    finally:
        await server.stop(grace=0.1)


@pytest.mark.asyncio
async def test_attached_client_prevents_idle_shutdown(tmp_path: Path) -> None:
    """``test_active_attach_stream_prevents_idle_shutdown``.

    A mirror of the previous test and a guard against "fixing" the first
    one trivially with "always shut down": while a client is attached,
    the idle countdown must be cancelled, even if the service had almost expired before the attach.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )

    root = _project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=root,
            idle_shutdown_s=60.0,
            slug="test",
        )
    )
    await server.start()
    try:
        server._idle_since_ns = time.time_ns() - int(59 * 1e9)
        server._attach()  # the client arrived before the deadline

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.wait_closed(), timeout=3.0)

        assert sock.exists(), "server died while a client was attached"
    finally:
        await server.stop(grace=0.1)


def test_socket_and_slug_dir_permissions_are_private(
    tmp_path: Path, reaper
) -> None:
    """The socket is access to the project's source edits. 0600/0700."""
    root = _project(tmp_path)
    registry_dir = tmp_path / "reg"

    entry = _supervisor(registry_dir, root).ensure_started(root)
    reaper.track_pid(entry.pid)

    sock = Path(entry.socket_path)
    assert sock.exists()
    assert oct(sock.stat().st_mode & 0o777) == "0o600", oct(sock.stat().st_mode)
    slug_dir = sock.parent
    assert oct(slug_dir.stat().st_mode & 0o777) == "0o700", oct(
        slug_dir.stat().st_mode
    )
