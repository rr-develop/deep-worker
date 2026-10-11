"""Auto-start supervisor for the semantic-memory service.

Design: docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md §7.

What this suite pins:

* ``flock`` serialises racing agents — N callers, ONE spawned process;
* a live service is reused, a stale record is reclaimed;
* the spawned process is NOT our child (detached) and outlives us;
* readiness is event-driven (inotify + pidfd), never a poll loop;
* a service that dies before readiness raises instead of hanging.

The spawn argv is injected, so none of this needs the real service or the
heavy stack. Every test that starts a real process reaps it in ``finally``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deep_worker_cli.semantic_memory.registry import (
    ServiceRegistry,
    ServiceState,
    read_start_ticks,
)
from deep_worker_cli.semantic_memory.service.supervisor import (
    ServiceStartError,
    ServiceSupervisor,
)

# Hard budget for anything that waits on a real process.
_BUDGET_S = 20.0


def _sock_toucher(sock: Path, hold_s: float = 30.0) -> list[str]:
    """argv for a process that creates the socket then sleeps.

    Creates a real AF_UNIX socket (not a plain file) so the supervisor's
    ``is_socket()`` check sees what it would see in production.
    """
    src = (
        "import socket,sys,time\n"
        "s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        f"s.bind({str(sock)!r})\n"
        "s.listen(1)\n"
        f"time.sleep({hold_s})\n"
    )
    return [sys.executable, "-c", src]


def _instant_exit(code: int = 1) -> list[str]:
    """argv for a process that dies without ever creating the socket."""
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


@pytest.fixture
def reaper():
    """Guarantee no test leaves a process behind, even on failure."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                break
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.05)
            else:
                continue
            break


def _supervisor(tmp_path: Path, argv_builder) -> ServiceSupervisor:
    registry = ServiceRegistry(base_dir=tmp_path / "semantic_memory")
    return ServiceSupervisor(
        registry=registry,
        spawn_argv=argv_builder,
        start_timeout_s=_BUDGET_S,
    )


class TestEnsureStarted:
    def test_spawns_and_records_state(self, tmp_path: Path, reaper) -> None:
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sock = sup.ensure_started()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)

        assert sock.is_socket()
        assert state.socket_path == str(sock)
        assert state.ready_at_ns > 0
        assert state.start_ticks == read_start_ticks(state.pid)

    def test_second_call_reuses_the_live_service(
        self, tmp_path: Path, reaper
    ) -> None:
        spawns: list[int] = []

        def argv(sock: Path) -> list[str]:
            spawns.append(1)
            return _sock_toucher(sock)

        sup = _supervisor(tmp_path, argv)
        sup.ensure_started()
        first = sup.registry.read()
        assert first is not None
        reaper.append(first.pid)

        sup.ensure_started()
        second = sup.registry.read()
        assert second is not None and second.pid == first.pid
        assert len(spawns) == 1, "a live service must not be re-spawned"

    def test_stale_record_is_reclaimed(self, tmp_path: Path, reaper) -> None:
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.registry.write(
            ServiceState(
                pid=4_194_304,  # above pid_max: cannot exist
                socket_path=str(sup.registry.socket_path),
                ready_at_ns=1,
                start_ticks=999,
            )
        )
        sup.ensure_started()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)
        assert state.pid != 4_194_304

    def test_recycled_pid_is_not_mistaken_for_the_service(
        self, tmp_path: Path, reaper
    ) -> None:
        # Our own pid is alive, but the recorded start_ticks belongs to a
        # different process. Without the start_ticks guard the supervisor
        # would hand back a socket nobody is listening on.
        #
        # A REAL bound socket is planted first, so the ``is_socket()`` check
        # cannot be what rejects this record — the start_ticks mismatch has
        # to be. Otherwise this test passes even with the guard removed.
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.registry.ensure_dir()
        import socket as _socket

        decoy = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        decoy.bind(str(sup.registry.socket_path))
        decoy.listen(1)
        try:
            assert sup.registry.socket_path.is_socket()
            sup.registry.write(
                ServiceState(
                    pid=os.getpid(),
                    socket_path=str(sup.registry.socket_path),
                    ready_at_ns=1,
                    start_ticks=read_start_ticks(os.getpid()) + 1,
                )
            )
            sup.ensure_started()
        finally:
            decoy.close()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)
        assert state.pid != os.getpid()

    def test_service_that_dies_before_readiness_raises(
        self, tmp_path: Path
    ) -> None:
        sup = _supervisor(tmp_path, lambda _sock: _instant_exit())
        started = time.monotonic()
        with pytest.raises(ServiceStartError, match="exited before readiness"):
            sup.ensure_started()
        elapsed = time.monotonic() - started
        # Death must be noticed at once, not by burning the start timeout.
        #
        # The tight bound is deliberate. A service that fails instantly is
        # the common case (missing extra, bad config), and it races: the
        # process is often already reaped by the time pidfd_open runs, so
        # pidfd_open raises ESRCH. Treating that as "pidfd unavailable"
        # rather than as proof of death made every failed start stall for
        # the full timeout — 20 s on each agent start. A loose bound here
        # would have let that through.
        assert elapsed < 5.0, (
            f"death detection took {elapsed:.1f}s — the ESRCH-means-dead "
            "path has regressed and startup failures now stall"
        )

    def test_stale_socket_without_state_is_removed(
        self, tmp_path: Path, reaper
    ) -> None:
        # A crash can leave the socket file behind. If the supervisor did
        # not unlink it, bind() in the new service would fail with EADDRINUSE.
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.registry.ensure_dir()
        sup.registry.socket_path.write_bytes(b"")
        sup.ensure_started()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)
        assert sup.registry.socket_path.is_socket()


class TestDetachment:
    def test_service_is_not_our_child(self, tmp_path: Path, reaper) -> None:
        """The service must survive the death of the agent that started it.

        If it stayed our child, killing one agent would take the shared
        service down for every other agent on the host.
        """
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.ensure_started()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)

        ppid = int(
            Path(f"/proc/{state.pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()[1]
        )
        assert ppid != os.getpid(), "service must be reparented, not our child"

    def test_service_is_in_its_own_session(
        self, tmp_path: Path, reaper
    ) -> None:
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.ensure_started()
        state = sup.registry.read()
        assert state is not None
        reaper.append(state.pid)
        assert os.getsid(state.pid) != os.getsid(os.getpid())


class TestConcurrency:
    def test_racing_agents_spawn_exactly_one_service(
        self, tmp_path: Path, reaper
    ) -> None:
        """N agents calling ensure_started at once → ONE service.

        This is the whole point of the flock. Run in real subprocesses so
        the file lock is contended across processes, not threads.
        """
        base = tmp_path / "semantic_memory"
        driver = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(Path(__file__).parents[3])!r})\n"
            "from pathlib import Path\n"
            "from deep_worker_cli.semantic_memory.registry import ServiceRegistry\n"
            "from deep_worker_cli.semantic_memory.service.supervisor import ServiceSupervisor\n"
            "import socket\n"
            "def argv(sock):\n"
            "    src = ('import socket,time\\n'\n"
            "           's=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\\n'\n"
            "           's.bind(%r)\\n' % str(sock) +\n"
            "           's.listen(1)\\n'\n"
            "           'time.sleep(30)\\n')\n"
            "    return [sys.executable, '-c', src]\n"
            f"reg = ServiceRegistry(base_dir=Path({str(base)!r}))\n"
            "sup = ServiceSupervisor(registry=reg, spawn_argv=argv, start_timeout_s=20.0)\n"
            "sup.ensure_started()\n"
            "st = reg.read()\n"
            "print(st.pid)\n"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", driver],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        try:
            outs = []
            for proc in procs:
                out, err = proc.communicate(timeout=_BUDGET_S * 2)
                assert proc.returncode == 0, f"driver failed: {err}"
                outs.append(int(out.strip()))
        finally:
            for proc in procs:
                if proc.poll() is None:  # pragma: no cover
                    proc.kill()
                    proc.wait()

        reaper.extend(set(outs))
        assert len(set(outs)) == 1, f"expected one service, got pids {set(outs)}"


class TestStop:
    def test_stop_kills_the_process_and_clears_the_record(
        self, tmp_path: Path
    ) -> None:
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.ensure_started()
        state = sup.registry.read()
        assert state is not None

        sup.stop()
        assert sup.registry.read() is None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(state.pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:  # pragma: no cover
            pytest.fail("service process survived stop()")

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        sup = _supervisor(tmp_path, lambda sock: _sock_toucher(sock))
        sup.stop()
        sup.stop()


class TestNoPolling:
    def test_module_contains_no_sleep_loop(self) -> None:
        """Policy: waiting is event-driven, never a poll.

        A ``sleep`` in the readiness path is the exact regression this
        forbids — it would burn CPU and add latency on every agent start.
        """
        from deep_worker_cli.semantic_memory.service import supervisor

        source = Path(supervisor.__file__).read_text(encoding="utf-8")
        # Strip comments/docstrings crudely: we only care about real calls.
        code_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        body = "\n".join(code_lines)
        assert "time.sleep(" not in body
        assert "asyncio.sleep(" not in body
