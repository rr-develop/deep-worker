"""Auto-restart supervisor for the Serena backend.

See ``docs/DESIGN_LSP_SERENA_PIDFD_RESTART.md``.

Owns exactly one ``SerenaBackend`` and one ``os.pidfd_open`` fd. When
the child process dies (kernel marks the pidfd readable), a supervisor
task drops the old backend, creates a fresh one via ``backend_factory``,
re-runs the full ``server.attach_backend`` and arms a new pidfd — up to
3 attempts in any 60s window. Beyond that, the service stays with
``Health.alive=False`` and tool RPCs fail-loud (design §4).

Invariants realised here:

* **One backend per project, one monitor per backend.** The class
  holds a single ``_backend`` and a single ``_monitor`` task.
* **Event-driven, no polling.** Wait for process death via
  ``os.pidfd_open`` + ``loop.add_reader`` + ``asyncio.Event``. Zero
  ``sleep(interval)`` loops.
* **Attempts, not successes, count.** A failed restart still burns a
  budget slot — otherwise a broken environment would spin forever.
* **No restart on planned stop.** ``_stopped`` gate short-circuits
  ``_on_death`` so shutdown never races a fresh spawn.
* **Full attach on restart.** Every restart goes through
  ``server.attach_backend`` on a NEW ``SerenaBackend`` — that alone
  reinstalls tool handler + refresh handler + readiness flag. The dead
  backend object is never revived (see MUT-W).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Final

if TYPE_CHECKING:  # pragma: no cover
    from deep_worker_cli.lsp.service.grpc_server import LSPServiceServer
    from deep_worker_cli.lsp.service.serena_backend import SerenaBackend

_log = logging.getLogger(__name__)

#: Design §4: at most 3 restarts inside any 60s window. Both are
#: exported so unit tests can reference the same values.
_MAX_RESTARTS: Final = 3
_WINDOW_S: Final = 60.0


class SerenaSupervisor:
    def __init__(
        self,
        *,
        server: "LSPServiceServer",
        backend_factory: Callable[[], "SerenaBackend"],
        pidfd_open_fn: Callable[[int, int], int] | None = None,
        add_reader_fn: Callable[[int, Callable[[], None]], None] | None = None,
        remove_reader_fn: Callable[[int], None] | None = None,
        close_fd_fn: Callable[[int], None] | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._server = server
        self._factory = backend_factory
        # syscall / loop hooks — real by default, overrideable for tests.
        self._pidfd_open = pidfd_open_fn or self._default_pidfd_open
        self._close_fd = close_fd_fn or os.close
        self._add_reader = add_reader_fn
        self._remove_reader = remove_reader_fn
        self._now = now_fn or time.monotonic

        self._backend: "SerenaBackend | None" = None
        self._monitor_fd: int | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._death_event: asyncio.Event | None = None
        self._on_death_task: asyncio.Task[None] | None = None
        self._restart_lock = asyncio.Lock()
        self._restarts: list[float] = []
        self._stopped = asyncio.Event()

    # ── syscalls ─────────────────────────────────────────────────────

    @staticmethod
    def _default_pidfd_open(pid: int, flags: int = 0) -> int:
        # ``os.pidfd_open`` is Linux ≥5.3; the whole tree assumes Linux
        # 6.x so a hard AttributeError here is acceptable and loud.
        return os.pidfd_open(pid, flags)

    def _loop_add_reader(
        self, fd: int, cb: Callable[[], None]
    ) -> None:
        if self._add_reader is not None:
            self._add_reader(fd, cb)
            return
        asyncio.get_running_loop().add_reader(fd, cb)

    def _loop_remove_reader(self, fd: int) -> None:
        if self._remove_reader is not None:
            self._remove_reader(fd)
            return
        try:
            asyncio.get_running_loop().remove_reader(fd)
        except Exception:  # noqa: BLE001 — fd may already be dead
            pass

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> bool:
        """First attach. Returns whatever ``server.attach_backend`` did.

        On success, arms the pidfd monitor over the freshly attached
        backend's pid (if known). If ``backend.pid`` is ``None``, we
        still consider start successful — the backend just cannot be
        auto-restarted (tests exercise this branch via fake sessions;
        production always yields a real pid).
        """
        backend = self._factory()
        ok = await self._server.attach_backend(backend)
        if not ok:
            # attach_backend logs the exception; do not re-raise, do
            # not touch _restarts (this is the first attach, not a
            # restart attempt).
            return False
        self._backend = backend
        self._arm_monitor(backend)
        return True

    async def stop(self) -> None:
        """Idempotent shutdown."""
        # Set the gate FIRST so any in-flight _on_death exits before
        # even trying to acquire the restart lock.
        self._stopped.set()
        # Kill the monitor's fd/reader so no new death event fires.
        self._disarm_monitor()
        # If _on_death is already inside the lock (restart in progress),
        # wait for it to finish — otherwise it might race with
        # backend.stop() below.
        if self._on_death_task is not None:
            try:
                await self._on_death_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                _log.exception("supervisor: _on_death raised during stop")
            self._on_death_task = None
        # Cancel the monitor task (usually already returned).
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._monitor_task = None
        # Stop the current backend.
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                await backend.stop()
            except Exception:  # noqa: BLE001
                _log.exception("supervisor: backend stop failed")

    # ── monitor ──────────────────────────────────────────────────────

    def _arm_monitor(self, backend: "SerenaBackend") -> None:
        pid = getattr(backend, "pid", None)
        if not isinstance(pid, int):
            _log.warning(
                "supervisor: backend has no pid; pidfd monitor NOT armed "
                "(auto-restart disabled for this backend). This is expected "
                "only in test paths that fake the mcp session."
            )
            return
        try:
            fd = self._pidfd_open(pid, 0)
        except OSError as e:
            _log.error(
                "supervisor: pidfd_open(%d) failed: %s; auto-restart disabled",
                pid, e,
            )
            return
        self._monitor_fd = fd
        self._death_event = asyncio.Event()
        # Reader callback runs in the event-loop thread. Keep it
        # synchronous and idempotent: strip the reader, close the fd,
        # signal the event. The heavy lifting (restart) is scheduled
        # as its own task from _monitor_task below.
        fired = {"once": False}

        def _on_readable() -> None:
            if fired["once"]:
                return
            fired["once"] = True
            self._loop_remove_reader(fd)
            try:
                self._close_fd(fd)
            except OSError:
                pass
            if self._death_event is not None:
                self._death_event.set()

        self._loop_add_reader(fd, _on_readable)
        self._monitor_task = asyncio.create_task(self._monitor_wait())

    def _disarm_monitor(self) -> None:
        fd = self._monitor_fd
        self._monitor_fd = None
        if fd is not None:
            self._loop_remove_reader(fd)
            try:
                self._close_fd(fd)
            except OSError:
                pass
        # Wake any waiter so the monitor task exits promptly.
        if self._death_event is not None:
            self._death_event.set()

    async def _monitor_wait(self) -> None:
        event = self._death_event
        if event is None:
            return
        try:
            await event.wait()
        except asyncio.CancelledError:
            raise
        # If we were asked to stop, exit without triggering a restart.
        if self._stopped.is_set():
            return
        # Schedule _on_death as its own task; do not await inline so
        # cancelling the monitor task never leaves a half-run restart.
        self._on_death_task = asyncio.create_task(self._on_death())

    # ── restart decisions ────────────────────────────────────────────

    async def _on_death(self) -> None:
        if self._stopped.is_set():
            return
        async with self._restart_lock:
            if self._stopped.is_set():
                return
            now = self._now()
            if not self._restart_allowed(now):
                _log.error(
                    "serena: restart budget exhausted "
                    "(%d attempts within %.0fs); backend stays not-ready",
                    len(self._restarts), _WINDOW_S,
                )
                # Ensure the server explicitly reports not-ready even
                # though _mark_lost already fired on the dead backend
                # — belt-and-suspenders.
                self._notify_not_ready()
                return
            self._restarts.append(now)
            await self._restart_once()

    def _restart_allowed(self, now: float) -> bool:
        cutoff = now - _WINDOW_S
        # Slide the window forward — drop attempts older than 60s.
        self._restarts = [t for t in self._restarts if t > cutoff]
        return len(self._restarts) < _MAX_RESTARTS

    async def _restart_once(self) -> bool:
        # Best-effort tear-down of the dead instance. ``backend.stop``
        # is cross-task-safe (see SerenaBackend.stop docstring) —
        # anyio cancel-scope mismatches are downgraded to a warning
        # rather than crashing the service.
        old, self._backend = self._backend, None
        if old is not None:
            try:
                await old.stop()
            except Exception:  # noqa: BLE001
                _log.exception("supervisor: dead backend stop failed")
        # Fresh instance via factory — MUST run through attach_backend,
        # not just mark_backend_ready (MUT-W).
        new_backend = self._factory()
        try:
            ok = await self._server.attach_backend(new_backend)
        except Exception:  # noqa: BLE001
            _log.exception("supervisor: attach_backend on restart raised")
            ok = False
        if not ok:
            _log.error("supervisor: restart attach_backend returned False")
            self._notify_not_ready()
            return False
        self._backend = new_backend
        # A3/MUT-X: only a restart that actually produced a live
        # backend is counted. The first attach goes through
        # ``start()``, never here, so it can never be miscounted.
        self._note_restart(success=True)
        # New pidfd monitor over the new pid.
        self._arm_monitor(new_backend)
        _log.info(
            "serena: restarted successfully (%d/%d attempts in window)",
            len(self._restarts), _MAX_RESTARTS,
        )
        return True

    def _note_restart(self, *, success: bool) -> None:
        """Report a restart outcome to the server's metrics (A3).

        Tolerates a server without the counter (unit fakes) so the
        restart path can never fail because of observability.
        """
        fn = getattr(self._server, "note_restart", None)
        if callable(fn):
            try:
                fn(success=success)
            except Exception:  # noqa: BLE001 — metrics must not break restart
                _log.exception("supervisor: note_restart raised")

    def _notify_not_ready(self) -> None:
        # The dead backend already fired on_lost during _mark_lost;
        # still, poke the server directly in case someone re-attached
        # a new backend between events.
        fn = getattr(self._server, "mark_backend_not_ready", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                _log.exception("supervisor: mark_backend_not_ready raised")


__all__ = [
    "SerenaSupervisor",
    "_MAX_RESTARTS",
    "_WINDOW_S",
]
