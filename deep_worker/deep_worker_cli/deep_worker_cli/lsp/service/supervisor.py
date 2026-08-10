"""Per-project LSP service supervisor (spawn/reclaim/stop).

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §5 (lifecycle) and §14
(policy: only the service package may spawn Serena — this module is
in ``deep_worker_cli.lsp.service*`` which is the allowlisted path).

Responsibilities:

* Read the on-disk registry to decide whether a service is already
  live for a given project root.
* If not, spawn a new one detached (throwaway launcher + ``setsid``) so
  killing the invoking agent doesn't take the service with it.
* Wait event-driven for the child to become ready (a ``grpc.sock``
  file appearing under ``<slug_dir>/`` — the child creates it as its
  last init step).
* Serialize concurrent ``ensure_started`` calls per slug via an
  ``fcntl.flock`` on ``<slug_dir>/.spawn.lock`` — two agents starting
  at the same time never spawn twice.
* On ``stop()``: SIGTERM → bounded wait → SIGKILL → drop the registry
  entry. Idempotent (no-op if nothing runs).

Non-goals in this module (kept elsewhere for scope isolation):

* Building the gRPC server / servicer — that is the server process
  the supervisor merely LAUNCHES.
* Building the ``spawn_argv`` for real (production wires Serena);
  tests inject a sentinel argv. This is the ``spawn_argv`` injection
  point on the class constructor.
* Change detection (commit 5), CLI wiring (later in commit 4), and
  the bridge (commit 5).

Event-driven readiness (project policy forbids polling): every wait in
this module blocks in the kernel on a real event, never on a timer.

* ``_wait_socket_or_exit`` arms an inotify watch on the slug directory
  AND opens a ``pidfd`` for the service, then ``select``s on both — so
  it returns the instant the socket appears or the process dies,
  whichever happens first.
* ``_wait_pid_exit`` blocks on a ``pidfd`` alone.

Both are Linux-specific (inotify, ``pidfd_open``), which matches the
agent host. If inotify is unavailable the code degrades to a single
bounded ``select`` on the pidfd rather than falling back to a sleep
loop.

Spawning avoids ``os.fork()`` entirely: callers are agents and bridges,
which are always multi-threaded, and forking such a parent can deadlock
the child. See :meth:`ServiceSupervisor._spawn` for the launcher-process
pattern that detaches the service without forking us.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from deep_worker_cli.lsp.service.registry import (
    RegistryEntry,
    ServiceRegistry,
)

_DEFAULT_START_TIMEOUT_S: Final = 30.0
_STOP_TERM_WAIT_S: Final = 2.0
_LOCK_FILENAME: Final = ".spawn.lock"
_SOCK_FILENAME: Final = "grpc.sock"
_PIDFILE_NAME: Final = ".spawn.pid"

#: Source of the throwaway launcher process (see ``_spawn``).
#:
#: It starts the real service in its own session and exits immediately,
#: so the service is reparented to init instead of staying a child of
#: the agent. Kept deliberately tiny and dependency-free: it runs before
#: any of our package is importable in that interpreter.
#:
#: The pid is written atomically (tmp + ``os.replace``) so the parent
#: can never observe a half-written file.
_LAUNCHER_SRC: Final = """
import json, os, subprocess, sys

argv = json.loads(sys.argv[1])
pid_path = sys.argv[2]
proc = subprocess.Popen(
    argv,
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
tmp = pid_path + ".tmp"
with open(tmp, "w") as fh:
    fh.write(str(proc.pid))
os.replace(tmp, pid_path)
"""


class ServiceStartTimeout(RuntimeError):
    """Raised when the spawned service did not become ready in time."""


class ServiceAlreadyRunning(RuntimeError):
    """Raised if a caller wanted a fresh spawn but one exists.

    Not raised by ``ensure_started`` (which is idempotent by design);
    reserved for future explicit-spawn APIs.
    """


@dataclass(frozen=True)
class SpawnResult:
    """Outcome of a successful ``ensure_started`` — subset of the
    registry entry that callers usually care about."""

    pid: int
    socket_path: str


SpawnArgv = Callable[[Path], list[str]]
"""Given the per-slug directory, return the argv to exec.

The callback pattern lets tests inject a sentinel command instead of
Serena, and lets the CLI (commit 4 wiring) inject the real production
argv (`deep-worker lsp __serve_impl__ ...`) without importing Serena
here — which would leak the heavy dep into the client-side bridge."""


class ServiceSupervisor:
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

    # ── public API ───────────────────────────────────────────────────

    def ensure_started(self, project_root: Path) -> RegistryEntry:
        """Return a live entry for ``project_root``, spawning if needed.

        Idempotent, race-safe (flock), reclaims stale entries.
        """
        slug_dir = self._registry.slug_dir(project_root)
        lock_path = slug_dir / _LOCK_FILENAME
        # Hold an exclusive flock on the slug dir so two racing callers
        # do not both spawn. flock is released when the fd is closed.
        with _slug_lock(lock_path):
            existing = self._registry.read(project_root)
            if existing is not None and not self._registry.is_stale(existing):
                # Also require the readiness marker to be present —
                # otherwise the service is registered but not usable
                # (e.g. it crashed mid-init and left state.json). We
                # treat that as "stale" too.
                if Path(existing.socket_path).exists():
                    return existing
            if existing is not None:
                # Best-effort cleanup: drop the stale record before
                # spawning. If the socket file is stale too, remove it
                # so the readiness watcher below sees a fresh CREATE.
                self._registry.drop(project_root)
                try:
                    Path(existing.socket_path).unlink()
                except FileNotFoundError:
                    pass
            return self._spawn(project_root, slug_dir)

    def stop(self, project_root: Path) -> None:
        """Terminate the service if any; idempotent."""
        entry = self._registry.read(project_root)
        if entry is None:
            return
        pid = entry.pid
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            if e.errno == errno.ESRCH:
                # Already dead — just drop the registry and any socket.
                self._registry.drop(project_root)
                _unlink_quiet(Path(entry.socket_path))
                return
            raise
        # Bounded wait for graceful exit — event-driven via pidfd, so
        # we block on the kernel telling us the process died rather
        # than waking up to re-check (project policy forbids polling).
        if not _wait_pid_exit(pid, _STOP_TERM_WAIT_S):
            _kill_quiet(pid, signal.SIGKILL)
            # SIGKILL is not refusable; still wait on the pidfd rather
            # than assuming an interval is "long enough".
            _wait_pid_exit(pid, _STOP_TERM_WAIT_S)
        self._registry.drop(project_root)
        _unlink_quiet(Path(entry.socket_path))

    def ready(self, project_root: Path) -> bool:
        """True iff the registry points at a live pid AND the socket
        exists. Callers use this to decide whether to auto-start."""
        entry = self._registry.read(project_root)
        if entry is None:
            return False
        if self._registry.is_stale(entry):
            return False
        return Path(entry.socket_path).exists()

    # ── internals ────────────────────────────────────────────────────

    def _spawn(self, project_root: Path, slug_dir: Path) -> RegistryEntry:
        """Launch a detached service; wait event-driven for readiness.

        We must satisfy two constraints at once:

        1. **No ``fork()`` in this process.** The caller is an agent or
           a bridge, and those are always multi-threaded (asyncio plus
           provider stacks). ``fork()`` from a multi-threaded parent
           can deadlock the child, and CPython warns about it.
        2. **The service must NOT stay our child.** If it did, killing
           the agent would take the shared service down with it — and
           every other agent using that project loses its backend.

        A plain ``Popen(start_new_session=True)`` gives us (1) but not
        (2): the process is detached from our session yet still has us
        as its parent. So we ``Popen`` a tiny *launcher* which itself
        ``Popen``s the real service and exits immediately. When the
        launcher exits the service is reparented to init (or the
        nearest subreaper), giving us (2) with no ``fork()`` anywhere
        in our own address space.

        The launcher reports the service pid through a file rather
        than a pipe, so we never have to read from a process we are
        about to reap.
        """
        argv = self._spawn_argv(slug_dir)
        socket_path = slug_dir / _SOCK_FILENAME
        # Pre-clean the readiness marker so we can distinguish "child
        # created it just now" from "leftover from previous crash".
        _unlink_quiet(socket_path)

        pid_path = slug_dir / _PIDFILE_NAME
        _unlink_quiet(pid_path)

        launcher = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _LAUNCHER_SRC,
                json.dumps(argv),
                str(pid_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        # The launcher is short-lived by construction; waiting on it
        # both reaps it (no zombie) and guarantees the service has been
        # started and reparented by the time we look at the pid file.
        try:
            launcher.wait(timeout=self._start_timeout_s)
        except subprocess.TimeoutExpired as e:
            launcher.kill()
            launcher.wait()
            raise ServiceStartTimeout("launcher did not exit in time") from e
        if launcher.returncode != 0:
            raise ServiceStartTimeout(
                f"launcher failed with exit code {launcher.returncode}"
            )

        try:
            service_pid = int(pid_path.read_text().strip())
        except (OSError, ValueError) as e:
            raise ServiceStartTimeout(
                f"launcher did not report a service pid in {pid_path}"
            ) from e
        finally:
            _unlink_quiet(pid_path)

        # Wait for the readiness marker. Event-driven on both edges:
        # inotify fires when the socket is created, a pidfd fires if
        # the service dies first. See ``_wait_socket_or_exit``.
        outcome = _wait_socket_or_exit(
            socket_path, service_pid, self._start_timeout_s
        )
        if outcome == "exited":
            raise ServiceStartTimeout(
                f"service pid {service_pid} exited before readiness"
            )
        if outcome == "timeout":
            _kill_quiet(service_pid, signal.SIGKILL)
            raise ServiceStartTimeout(
                f"service did not create {socket_path} within "
                f"{self._start_timeout_s}s"
            )

        entry = RegistryEntry(
            project_root=project_root.resolve(),
            pid=service_pid,
            transport="uds",
            socket_path=str(socket_path),
            started_at_ns=time.time_ns(),
        )
        self._registry.write(entry)
        return entry


# ── helpers ──────────────────────────────────────────────────────────


class _slug_lock:
    """Context manager: exclusive flock on a per-slug lock file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_slug_lock":
        # 0600 lock file inside the 0700 slug dir.
        self._fd = os.open(
            str(self._path),
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
            0o600,
        )
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: object) -> None:
        fd = self._fd
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
                self._fd = None


def _wait_pid_exit(pid: int, timeout_s: float) -> bool:
    """Block until ``pid`` exits, or ``timeout_s`` elapses.

    Event-driven: ``pidfd_open`` gives a descriptor that becomes
    readable exactly when the process dies, so ``select`` blocks in the
    kernel with zero wakeups in between. No polling loop.

    Returns ``True`` if the process is gone, ``False`` on timeout.
    """
    try:
        pidfd = os.pidfd_open(pid, 0)
    except (OSError, AttributeError):
        # ESRCH → already dead. Anything else (very old kernel) means
        # we cannot observe it; report "gone" so the caller proceeds
        # to its SIGKILL path rather than hanging.
        return True
    try:
        readable, _, _ = select.select([pidfd], [], [], timeout_s)
        return bool(readable)
    finally:
        os.close(pidfd)


def _wait_socket_or_exit(
    socket_path: Path, pid: int, timeout_s: float
) -> str:
    """Wait for ``socket_path`` to appear, or ``pid`` to die first.

    Returns ``"ready"``, ``"exited"`` or ``"timeout"``.

    Event-driven on both edges:

    * a ``pidfd`` becomes readable the moment the child dies;
    * ``watchfiles`` (inotify) fires the moment the socket is created.

    ``select`` blocks on both descriptors at once, so this function
    consumes no CPU while waiting and reacts immediately to whichever
    happens first. No polling loop.
    """
    # The child may have won the race before we even got here.
    if socket_path.exists():
        return "ready"

    watch_dir = socket_path.parent
    watch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    try:
        pidfd = os.pidfd_open(pid, 0)
    except (OSError, AttributeError):
        return "ready" if socket_path.exists() else "exited"

    inotify_fd = -1
    try:
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            inotify_fd = libc.inotify_init1(os.O_NONBLOCK)
            if inotify_fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1 failed")
            # IN_CREATE | IN_MOVED_TO — the socket appears via bind().
            wd = libc.inotify_add_watch(
                inotify_fd, str(watch_dir).encode(), 0x00000100 | 0x00000080
            )
            if wd < 0:
                raise OSError(ctypes.get_errno(), "inotify_add_watch failed")
        except (OSError, AttributeError):
            # Without inotify we can still react to child death, and
            # fall back to a single bounded wait for the socket.
            readable, _, _ = select.select([pidfd], [], [], timeout_s)
            if socket_path.exists():
                return "ready"
            return "exited" if readable else "timeout"

        # Re-check after arming the watch: the file may have been
        # created between the first check and inotify_add_watch.
        if socket_path.exists():
            return "ready"

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "ready" if socket_path.exists() else "timeout"
            readable, _, _ = select.select(
                [pidfd, inotify_fd], [], [], remaining
            )
            if not readable:
                return "ready" if socket_path.exists() else "timeout"
            if inotify_fd in readable:
                try:
                    os.read(inotify_fd, 4096)  # drain; content unused
                except OSError:
                    pass
                if socket_path.exists():
                    return "ready"
                # A sibling file (e.g. the lock) was created — keep
                # blocking on the same descriptors, no timer involved.
                continue
            if pidfd in readable:
                # Child died. It may still have created the socket
                # microseconds earlier, so check before reporting.
                return "ready" if socket_path.exists() else "exited"
    finally:
        if inotify_fd >= 0:
            os.close(inotify_fd)
        os.close(pidfd)


def _kill_quiet(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
