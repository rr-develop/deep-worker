"""Auto-start supervisor for the shared semantic-memory service.

See ``docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md`` §7.

``ensure_started()`` is the whole contract:

1. create the state dir 0700 (and re-assert the mode — ``mkdir(mode=)``
   does nothing to an existing directory and is masked by the umask);
2. take an exclusive ``flock`` so N racing agents produce ONE service;
3. re-check liveness **under the lock** and reuse a healthy service;
4. arm the inotify watch **before** spawning, closing the race where the
   service becomes ready before the watch exists;
5. spawn detached, without ``fork()`` anywhere in our address space;
6. wait for readiness event-driven on inotify + pidfd — never a poll.

Two constraints shape the spawn, and they pull in opposite directions:

* **No ``fork()`` here.** The caller is an agent: asyncio plus provider
  stacks, always multi-threaded. ``fork()`` from a multi-threaded parent
  can deadlock the child, and CPython warns about it.
* **The service must not stay our child.** If it did, killing one agent
  would take the shared service down for every other agent on the host.

``Popen(start_new_session=True)`` satisfies the first but not the second.
So we ``Popen`` a tiny *launcher* which itself ``Popen``s the real service
and exits immediately; when it exits the service is reparented to init.
The launcher reports the service pid through a file rather than a pipe, so
we never read from a process we are about to reap.

This module lives inside ``service/`` on purpose (design §9.1): the
package is the allowlisted spawner under policy §8. Client lightness is
guaranteed by ``service/__init__.py`` being empty plus the transitive
import lock — not by where this file sits.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

from deep_worker_cli.semantic_memory.registry import (
    ServiceRegistry,
    ServiceState,
    read_start_ticks,
)

_DEFAULT_START_TIMEOUT_S: Final = 30.0
_STOP_TERM_WAIT_S: Final = 5.0
_FILE_MODE: Final = 0o600

#: Builds the argv for the real service given the socket path. Injected so
#: this module never imports the service (and therefore never the heavy
#: stack) just to know how to start it.
SpawnArgv = Callable[[Path], list[str]]

# inotify constants (linux/inotify.h). Used through libc directly so we do
# not take a dependency on a third-party inotify wrapper.
_IN_CREATE: Final = 0x00000100
_IN_MOVED_TO: Final = 0x00000080
_IN_NONBLOCK: Final = 0o4000

#: Source of the throwaway launcher. Kept as a string so there is no
#: import-time dependency and no module to keep in sync on disk.
_LAUNCHER_SRC: Final = """
import json, os, subprocess, sys, tempfile

argv = json.loads(sys.argv[1])
pid_path = sys.argv[2]
log_path = sys.argv[3]

# 0600 from creation: the log may carry paths and error detail, and the
# agent's ambient umask must not be allowed to widen it.
fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
try:
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=fd,
        stderr=fd,
        start_new_session=True,
        close_fds=True,
    )
finally:
    os.close(fd)

directory = os.path.dirname(pid_path)
tmp_fd, tmp_name = tempfile.mkstemp(prefix=".pid.", dir=directory)
with os.fdopen(tmp_fd, "w") as handle:
    handle.write(str(proc.pid))
os.replace(tmp_name, pid_path)
"""


class ServiceStartError(RuntimeError):
    """The service could not be started, or died before becoming ready."""


class _SlugLock:
    """Exclusive flock over the service's spawn lock file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_SlugLock":
        fd = os.open(
            str(self._path),
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
            _FILE_MODE,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


class ServiceSupervisor:
    """Starts the service on demand and reclaims stale records."""

    def __init__(
        self,
        registry: ServiceRegistry,
        spawn_argv: SpawnArgv,
        start_timeout_s: float = _DEFAULT_START_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._spawn_argv = spawn_argv
        self._start_timeout_s = start_timeout_s

    @property
    def registry(self) -> ServiceRegistry:
        return self._registry

    def ensure_started(self) -> Path:
        """Return the socket path of a live service, starting one if needed."""
        self._registry.ensure_dir()

        existing = self._live_socket()
        if existing is not None:
            return existing

        with _SlugLock(self._registry.lock_path):
            # Re-check under the lock: another agent may have started the
            # service while we waited for it.
            existing = self._live_socket()
            if existing is not None:
                return existing
            return self._spawn()

    def ready(self) -> bool:
        """True iff a live service is recorded and its socket exists."""
        return self._live_socket() is not None

    def stop(self) -> None:
        """Terminate the service and clear its record. Idempotent."""
        state = self._registry.read()
        if state is None:
            self._unlink_quiet(self._registry.socket_path)
            return
        if not ServiceRegistry.is_stale(state):
            self._kill_quiet(state.pid, signal.SIGTERM)
            if not self._wait_pid_exit(state.pid, _STOP_TERM_WAIT_S):
                self._kill_quiet(state.pid, signal.SIGKILL)
                self._wait_pid_exit(state.pid, _STOP_TERM_WAIT_S)
        self._registry.drop()
        self._unlink_quiet(self._registry.socket_path)

    # ── internals ────────────────────────────────────────────────────

    def _live_socket(self) -> Path | None:
        """Socket path iff the recorded service is alive and bound.

        Deliberately does NOT issue a Health RPC: that would make this
        module import grpc, and it runs on every agent start. The gRPC
        handshake happens in the client, which is where an unreachable
        service surfaces as a retryable error.
        """
        state = self._registry.read()
        if state is None:
            return None
        if ServiceRegistry.is_stale(state):
            self._registry.drop()
            self._unlink_quiet(Path(state.socket_path))
            return None
        socket_path = Path(state.socket_path)
        if not socket_path.is_socket():
            # Alive but not bound — a half-dead service. Clear it so the
            # caller spawns a healthy one.
            self._registry.drop()
            return None
        return socket_path

    def _spawn(self) -> Path:
        base = self._registry.base_dir
        socket_path = self._registry.socket_path
        pid_path = base / ".spawn.pid"

        # Pre-clean so "the socket exists" unambiguously means "the child
        # created it just now" and bind() cannot hit EADDRINUSE.
        self._unlink_quiet(socket_path)
        self._unlink_quiet(pid_path)

        # Arm the watch BEFORE spawning: otherwise a fast service can
        # become ready in the window before the watch exists, and we would
        # wait out the full timeout for an event that already happened.
        notify_fd = self._inotify_watch(base)
        try:
            argv = self._spawn_argv(socket_path)
            launcher = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _LAUNCHER_SRC,
                    json.dumps(argv),
                    str(pid_path),
                    str(self._registry.log_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            try:
                launcher.wait(timeout=self._start_timeout_s)
            except subprocess.TimeoutExpired as exc:
                launcher.kill()
                launcher.wait()
                raise ServiceStartError(
                    "launcher did not exit in time"
                ) from exc
            if launcher.returncode != 0:
                raise ServiceStartError(
                    f"launcher failed with exit code {launcher.returncode}"
                )

            try:
                service_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError) as exc:
                raise ServiceStartError(
                    f"launcher did not report a service pid in {pid_path}"
                ) from exc
            finally:
                self._unlink_quiet(pid_path)

            outcome = self._wait_socket_or_exit(
                notify_fd, socket_path, service_pid, self._start_timeout_s
            )
        finally:
            os.close(notify_fd)

        if outcome == "exited":
            raise ServiceStartError(
                f"service pid {service_pid} exited before readiness; "
                f"see {self._registry.log_path}"
            )
        if outcome == "timeout":
            self._kill_quiet(service_pid, signal.SIGKILL)
            raise ServiceStartError(
                f"service did not create {socket_path} within "
                f"{self._start_timeout_s}s; see {self._registry.log_path}"
            )

        self._registry.write(
            ServiceState(
                pid=service_pid,
                socket_path=str(socket_path),
                ready_at_ns=time.time_ns(),
                start_ticks=read_start_ticks(service_pid),
            )
        )
        return socket_path

    @staticmethod
    def _inotify_watch(directory: Path) -> int:
        """Watch ``directory`` for file creation; return the inotify fd."""
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(_IN_NONBLOCK)
        if fd < 0:  # pragma: no cover - inotify unavailable
            raise ServiceStartError("inotify_init1 failed")
        watch = libc.inotify_add_watch(
            fd, str(directory).encode(), _IN_CREATE | _IN_MOVED_TO
        )
        if watch < 0:  # pragma: no cover
            os.close(fd)
            raise ServiceStartError(f"inotify_add_watch failed on {directory}")
        return fd

    @staticmethod
    def _wait_socket_or_exit(
        notify_fd: int,
        socket_path: Path,
        pid: int,
        timeout_s: float,
    ) -> Literal["ready", "exited", "timeout"]:
        """Block until the socket appears, the service dies, or we time out.

        Three outcomes multiplexed in one kernel wait: inotify fires on
        socket creation, a pidfd fires if the service dies first, and
        ``select`` bounds the whole thing. No polling, no sleep.

        inotify wakeups are treated as advisory — the authoritative check
        is re-run on every wake, because the event may be for another file
        in the directory.
        """
        pid_fd: int | None
        try:
            pid_fd = os.pidfd_open(pid)  # type: ignore[attr-defined]
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                # The service is ALREADY gone — it died between the
                # launcher reporting its pid and us getting here, which is
                # exactly what a service that fails at startup does. ESRCH
                # is proof of death, not a missing capability: waiting for
                # the deadline here would turn an instant failure into a
                # full start-timeout stall on every agent start.
                #
                # It may still have created the socket before dying, so
                # readiness is checked first.
                return "ready" if socket_path.is_socket() else "exited"
            # Some other OS error (e.g. EPERM/EMFILE): degrade to
            # inotify + timeout. Still event-driven, just blind to death.
            pid_fd = None
        except AttributeError:
            # pidfd_open is unavailable on this build.
            pid_fd = None

        try:
            deadline = time.monotonic() + timeout_s
            while True:
                if socket_path.is_socket():
                    return "ready"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timeout"
                watch_fds = [notify_fd] + ([pid_fd] if pid_fd is not None else [])
                ready, _, _ = select.select(watch_fds, [], [], remaining)
                if not ready:
                    # select timed out; one last authoritative look so a
                    # ready-just-at-the-deadline service is not failed.
                    return "ready" if socket_path.is_socket() else "timeout"
                if pid_fd is not None and pid_fd in ready:
                    # The child may have created the socket and died in
                    # the same instant; readiness wins.
                    return "ready" if socket_path.is_socket() else "exited"
                if notify_fd in ready:
                    try:
                        os.read(notify_fd, 65536)
                    except BlockingIOError:  # pragma: no cover
                        pass
        finally:
            if pid_fd is not None:
                os.close(pid_fd)

    @staticmethod
    def _wait_pid_exit(pid: int, timeout_s: float) -> bool:
        """Wait for ``pid`` to exit. True if it did, False on timeout."""
        try:
            pid_fd = os.pidfd_open(pid)  # type: ignore[attr-defined]
        except (AttributeError, OSError) as exc:
            # ESRCH → already gone.
            return isinstance(exc, OSError) and exc.errno == errno.ESRCH
        try:
            ready, _, _ = select.select([pid_fd], [], [], timeout_s)
            return bool(ready)
        finally:
            os.close(pid_fd)

    @staticmethod
    def _kill_quiet(pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except OSError:
            pass

    @staticmethod
    def _unlink_quiet(path: Path) -> None:
        try:
            path.unlink()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
