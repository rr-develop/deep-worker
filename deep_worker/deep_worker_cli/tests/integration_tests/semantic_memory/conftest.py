"""Shared rig for the semantic-memory integration suite.

Every test here drives the REAL service: a real subprocess, a real UDS, a
real gRPC channel, and — where the scenario needs it — the real heavy
stack behind it. No mocks, no stubs, no monkeypatched engine.

Two constraints shape this file.

*Isolation.* A test must never touch the operator's live service. Each
one gets a ``ServiceRegistry`` rooted in ``tmp_path``, so the socket,
lock, state file and log all live under the test's own directory and the
production registry at ``deep_worker_home()`` is never read or written.

*Bounded teardown.* Integration tests must not hang: the budget is 60 s
per test, and a service that outlives its test would leak a process
holding the heavy stack in RAM. ``service_rig`` therefore kills whatever
it spawned in ``finally`` — SIGTERM first, SIGKILL if the process has not
gone within the grace window — and it does so from the registry's own
record, so it also reaps a service that a *retry* spawned behind the
client's back.

*No polling.* Every wait in this rig is a kernel wait. The only thing
the suite ever waits for is a process exit, and Linux exposes that as a
readable fd via ``pidfd_open``, so ``wait_pid_exit`` blocks in ``select``
with a deadline instead of ticking a ``time.sleep`` loop. The suite's own
``test_rig_has_no_polling_loops`` enforces this — the ban is a test, not
a convention.

*Signals go through the descriptor, never the number.* ``kill_pid`` opens
a pidfd before it sends anything and drives the whole TERM → wait → KILL
→ wait escalation through that one fd, so a pid recycled mid-teardown can
never be signalled by mistake. ``test_rig_is_event_driven.py`` locks both
halves of that property.

*Teardown fails loudly or not at all.* "Guaranteed dead" is a guarantee
only if the helper checks: a signal that could not be delivered raises,
and a process still alive after SIGKILL raises. Reporting a cleanup that
did not happen would leave a service holding the heavy stack for the rest
of the session, with nothing in the test report to say so.
"""

from __future__ import annotations

import errno
import os
import select
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest

from deep_worker_cli.semantic_memory.registry import ServiceRegistry
from deep_worker_cli.semantic_memory.service.__main__ import build_serve_argv
from deep_worker_cli.semantic_memory.service.supervisor import ServiceSupervisor

#: Whole-test ceiling. The suite policy is 60 s; individual waits stay
#: well under it so a failure reports as an assertion, not a timeout.
BUDGET_S = 60.0

#: Starting the service imports grpc and builds the server. Model load is
#: lazy (first search), so readiness itself is fast.
START_TIMEOUT_S = 30.0


def pid_alive(pid: int) -> bool:
    """True while *pid* exists. Signal 0 checks liveness without sending."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _open_pidfd(pid: int) -> int | None:
    """A handle on *pid*, or None if it is already gone.

    ``ESRCH`` means the process no longer exists. Reading it as "pidfd
    unavailable" would be the bug documented in ``test_supervisor.py``:
    an instant death would then be discovered only by burning the whole
    deadline.
    """
    try:
        return os.pidfd_open(pid)  # type: ignore[attr-defined]
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return None
        raise


def _wait_fd_exit(pid_fd: int, *, timeout_s: float) -> bool:
    """Block on *pid_fd* until its process exits. False on timeout.

    A pidfd becomes readable exactly when the process it names dies, so
    ``select`` returns on the death itself rather than on the next tick of
    a sleep loop — the wait costs one syscall and ends without latency.
    """
    ready, _, _ = select.select([pid_fd], [], [], timeout_s)
    return bool(ready)


def wait_pid_exit(pid: int, *, timeout_s: float) -> bool:
    """Block until *pid* exits. True if it did, False on timeout."""
    pid_fd = _open_pidfd(pid)
    if pid_fd is None:
        return True
    try:
        return _wait_fd_exit(pid_fd, timeout_s=timeout_s)
    finally:
        os.close(pid_fd)


def assert_pid_exits(pid: int, *, timeout_s: float, what: str) -> None:
    """Wait for *pid* to die, failing with what was awaited.

    Failing here as an assertion — rather than letting the test hit a
    global timeout — keeps the report readable: it names the process that
    never went away.
    """
    if not wait_pid_exit(pid, timeout_s=timeout_s):
        raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def kill_pid(pid: int, *, grace_s: float = 5.0) -> None:
    """Stop *pid*, escalating to SIGKILL, without ever racing PID reuse.

    A pid number is not a stable identity. Between the moment a service
    exits and the moment teardown looks again, the kernel is free to hand
    that number to an unrelated process — and ``kill(pid, SIGKILL)`` would
    then land on a stranger. The registry guards its own records against
    exactly this with ``start_ticks``; a teardown helper that signalled by
    number would reintroduce the hole the registry closed.

    So identity is pinned once, up front: ``pidfd_open`` before any signal
    is sent, and every subsequent action — SIGTERM, the wait, SIGKILL, the
    confirming wait — goes through that same descriptor. A pidfd refers to
    the process it was opened on and to nothing else; once that process
    dies the fd reports the death and can never be re-pointed at whoever
    inherits the number. If the process is already gone, ``_open_pidfd``
    returns None and there is nothing to signal.

    Only one failure is benign here: the process having already exited.
    Anything else — EPERM, EINVAL, EBADF — means the signal did not land
    and the caller's belief that teardown happened is false, so it is
    raised rather than swallowed. Likewise the wait after SIGKILL is
    *checked*: a teardown that returns while the process is still running
    would be the "guaranteed dead" contract in name only, and the leaked
    service would hold the heavy stack in RAM for the rest of the session.

    Raises:
        OSError: the signal failed for any reason other than the target
            already being gone.
        AssertionError: the process was still alive after SIGKILL and the
            grace window.
    """
    pid_fd = _open_pidfd(pid)
    if pid_fd is None:
        return

    try:
        try:
            signal.pidfd_send_signal(pid_fd, signal.SIGTERM)
        except ProcessLookupError:
            # Died between the open and the signal — nothing left to do.
            return

        if _wait_fd_exit(pid_fd, timeout_s=grace_s):
            return

        try:
            signal.pidfd_send_signal(pid_fd, signal.SIGKILL)
        except ProcessLookupError:
            return

        # SIGKILL is not instantaneous — this wait is what makes teardown a
        # guarantee rather than a request, so its answer must be acted on.
        if not _wait_fd_exit(pid_fd, timeout_s=grace_s):
            raise AssertionError(
                f"pid {pid} survived SIGKILL for {grace_s}s — the service "
                "is still running and will hold the heavy stack in RAM"
            )
    finally:
        os.close(pid_fd)
    # The service is detached (double-forked), so it is not our child and
    # waitpid would raise; reaping is the init process's job.


class ServiceRig:
    """A supervisor bound to a throwaway registry, plus the PIDs it made.

    Tracking PIDs here rather than relying on the registry alone matters
    for the restart tests: once a service dies, its record is dropped, and
    a rig that only read the current record would forget the corpse it
    still has to reap.
    """

    def __init__(self, base_dir: Path, idle_shutdown_s: float | None = None) -> None:
        self.registry = ServiceRegistry(base_dir=base_dir)
        self._idle_shutdown_s = idle_shutdown_s
        self._seen_pids: set[int] = set()
        self.supervisor = ServiceSupervisor(
            registry=self.registry,
            spawn_argv=self._spawn_argv,
            start_timeout_s=START_TIMEOUT_S,
        )

    def _spawn_argv(self, socket_path: Path) -> list[str]:
        """Real serve argv, optionally with a shortened idle window.

        ``build_serve_argv`` is the production entry point; the idle test
        needs a service that gives up in seconds instead of half an hour,
        and ``serve`` takes that as a parameter rather than a flag, so the
        short-window variant runs the same module through ``-c``.
        """
        if self._idle_shutdown_s is None:
            return build_serve_argv(socket_path)
        import sys

        return [
            sys.executable,
            "-c",
            (
                "import asyncio, sys\n"
                "from pathlib import Path\n"
                "from deep_worker_cli.semantic_memory.service.grpc_server "
                "import serve\n"
                "asyncio.run(serve(Path(sys.argv[1]), "
                "idle_shutdown_s=float(sys.argv[2])))\n"
            ),
            str(socket_path),
            str(self._idle_shutdown_s),
        ]

    def current_pid(self) -> int | None:
        state = self.registry.read()
        if state is None:
            return None
        self._seen_pids.add(state.pid)
        return state.pid

    def ensure_started(self) -> Path:
        socket_path = self.supervisor.ensure_started()
        self.current_pid()
        return socket_path

    def reap(self) -> None:
        """Kill every service this rig ever recorded."""
        self.current_pid()
        for pid in self._seen_pids:
            kill_pid(pid)


@pytest.fixture
def service_rig(tmp_path: Path) -> Iterator[ServiceRig]:
    """A rig whose service is guaranteed dead when the test returns."""
    rig = ServiceRig(base_dir=tmp_path / "semantic_memory")
    try:
        yield rig
    finally:
        rig.reap()


@pytest.fixture
def short_idle_rig(tmp_path: Path) -> Iterator[ServiceRig]:
    """Same, but the service shuts itself down after a few idle seconds."""
    rig = ServiceRig(
        base_dir=tmp_path / "semantic_memory_idle", idle_shutdown_s=3.0
    )
    try:
        yield rig
    finally:
        rig.reap()
