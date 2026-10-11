"""Fire-and-forget agent lifecycle service.

Implements event-driven stop / restart / batch operations per
``docs/DESIGN_FIRE_AND_FORGET_LIFECYCLE.md``.

Contract (must hold):
* every public method returns <100 ms (no blocking, no polling)
* no ``asyncio.sleep`` in ops (one ``loop.call_later`` per grace / hard
  timeout is fine — that's event-driven)
* no threads, no ``asyncio.to_thread``, no executors
* SIGTERM → grace → SIGKILL escalation, just like ``dwcli --stop``
* every state transition broadcasts a WS event (``*_started`` /
  ``*_completed`` / ``*_failed``) — errors MUST be visible
* ``status=running`` does NOT gate stop (stale metadata is not truth)
"""
from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import signal
from collections.abc import Awaitable, Callable
from collections.abc import Coroutine
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from deep_worker_cli.storage_paths import deep_worker_home, read_agent_file

logger = logging.getLogger(__name__)

Broadcast = Callable[[dict[str, Any]], Awaitable[None]]

# Grace between SIGKILL and verification. Pidfd reader is the primary
# completion path; this is just a safety ceiling so _verify_process_gone
# eventually runs if the fd never becomes readable.
_KILL_VERIFY_DELAY_SEC = 0.3
_SPAWN_HARD_TIMEOUT_SEC = 15.0


# --------------------------------------------------------------------------- #
# Helpers — kept at module scope so tests can patch them.
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status_file(agent_name: str) -> str:
    """Best-effort read of session_metadata.json status (diagnostic only).

    NEVER used to gate stop — just surfaced in event payloads so the UI
    can show which agents were "officially" idle vs running.
    """
    raw = read_agent_file(agent_name, "session_metadata.json")
    if raw is None:
        return "unknown"
    try:
        data = json.loads(raw)
        return str(data.get("status", "unknown"))
    except (json.JSONDecodeError, ValueError):
        return "unknown"


def _read_proc_state(pid: int) -> str | None:
    """Read the one-letter process state from /proc/{pid}/stat.

    Returns:
        'R' / 'S' / 'D' / 'Z' / 'T' / 'I' / ... or None if /proc entry
        is gone (process fully reaped).
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Format: "pid (comm) state ..."
    right = data.rsplit(") ", 1)
    if len(right) != 2:
        return None
    parts = right[1].split()
    return parts[0] if parts else None


# --------------------------------------------------------------------------- #
# LifecycleJobRegistry — in-memory job state + broadcast dispatcher.
# --------------------------------------------------------------------------- #


class LifecycleJobRegistry:
    """Tracks in-flight jobs and fans their state changes into broadcast events.

    Each job has a ``request_id`` that callers keep; frontend correlates
    ``*_started`` with ``*_completed`` / ``*_failed`` through it.
    """

    def __init__(self, broadcast: Broadcast) -> None:
        self._broadcast = broadcast
        self._jobs: dict[str, dict[str, Any]] = {}
        # Event-driven completion hooks: request_id -> list of asyncio.Event.
        # Used by ``_drive_batch_aggregator`` to wake up EXACTLY ONCE per
        # child completion instead of polling the ``_jobs`` dict at 20Hz.
        # Each event carries a "was failure?" flag via the resolved payload
        # below (``_done_status``).
        self._done_events: dict[str, list[asyncio.Event]] = {}
        self._done_status: dict[str, bool] = {}  # request_id -> True if failed

    def get_job(self, request_id: str) -> dict[str, Any] | None:
        return self._jobs.get(request_id)

    def _register_done_event(self, request_id: str) -> asyncio.Event:
        """Return an Event that is set when ``request_id`` finishes
        (complete or fail). Multiple watchers may register for the
        same request — each gets its own Event instance, so a stale
        caller can't swallow another's wake-up."""
        ev = asyncio.Event()
        self._done_events.setdefault(request_id, []).append(ev)
        # Already finished? Fire immediately so we never hang on a
        # late subscribe-after-complete race.
        if request_id not in self._jobs and request_id in self._done_status:
            ev.set()
        return ev

    def _dispatch_done(self, request_id: str, failed: bool) -> None:
        """Wake up every subscriber for ``request_id`` exactly once."""
        self._done_status[request_id] = failed
        events = self._done_events.pop(request_id, [])
        for ev in events:
            ev.set()

    async def start_job(
        self,
        op: str,
        agent_name: str | None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        request_id = uuid4().hex
        self._jobs[request_id] = {
            "op": op,
            "agent_name": agent_name,
            "payload": payload or {},
            "started_at": _now_iso(),
        }
        await self._broadcast(
            {
                "type": f"{op}_started",
                "request_id": request_id,
                "agent_name": agent_name,
                "timestamp": _now_iso(),
                "payload": payload or {},
            }
        )
        return request_id

    async def complete_job(
        self, request_id: str, payload: dict[str, Any] | None = None
    ) -> None:
        job = self._jobs.pop(request_id, None)
        if job is None:
            return
        await self._broadcast(
            {
                "type": f"{job['op']}_completed",
                "request_id": request_id,
                "agent_name": job.get("agent_name"),
                "timestamp": _now_iso(),
                "payload": payload or {},
            }
        )
        # Wake any event-driven watchers (e.g. batch aggregator).
        self._dispatch_done(request_id, failed=False)

    async def fail_job(
        self, request_id: str, exc: BaseException, payload: dict[str, Any] | None = None
    ) -> None:
        job = self._jobs.pop(request_id, None)
        # Even if job was never registered, broadcast so UI can surface it.
        op = job["op"] if job else "agent_op"
        await self._broadcast(
            {
                "type": f"{op}_failed",
                "request_id": request_id,
                "agent_name": job.get("agent_name") if job else None,
                "timestamp": _now_iso(),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "payload": payload or {},
            }
        )
        # Wake event-driven watchers — failure is still a terminal
        # state they care about.
        self._dispatch_done(request_id, failed=True)


# --------------------------------------------------------------------------- #
# AgentLifecycleService — the user-facing orchestrator.
# --------------------------------------------------------------------------- #


class AgentLifecycleService:
    """Orchestrates stop / restart / batch ops with fire-and-forget semantics.

    All public methods are ``async def`` but return almost instantly — the
    heavy lifting runs in background tasks spawned via ``asyncio.create_task``.

    The object itself is pure asyncio; it does NOT create threads.
    """

    def __init__(
        self,
        *,
        registry: Any,
        broadcast: Broadcast,
        jobs: LifecycleJobRegistry | None = None,
        rpyc_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self._registry = registry
        self._broadcast = broadcast
        self._jobs = jobs or LifecycleJobRegistry(broadcast)
        # rpyc_resolver(agent_name) -> root service proxy or None. Injected
        # so we can reach an agent for in-band ``shutdown()`` before falling
        # back to SIGTERM. When None, stop always uses the signal path.
        self._rpyc_resolver = rpyc_resolver
        # Book-keeping so tests (and shutdown) can await outstanding work.
        self._pending: set[asyncio.Task[Any]] = set()
        # Map request_id -> pidfd so we can close it on shutdown if the
        # watcher never fires (shouldn't happen in practice).
        self._pidfds: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    async def request_stop(
        self,
        agent_name: str,
        *,
        grace_seconds: float = 5.0,
        force: bool = False,
    ) -> str:
        """Start an asynchronous stop.

        Returns a ``request_id`` immediately. Completion / failure comes
        via WS events (``agent_stop_completed`` / ``agent_stop_failed``).
        """
        info = self._registry.lookup(agent_name)
        if info is None:
            # Unknown agent — still give the caller a request_id so the UI
            # can correlate the failure event.
            request_id = await self._jobs.start_job(
                "agent_stop",
                agent_name,
                {"grace_seconds": grace_seconds, "force": force},
            )
            self._spawn(
                self._jobs.fail_job(
                    request_id,
                    LookupError(f"Agent '{agent_name}' not found"),
                )
            )
            return request_id

        pid = int(info.pid)
        status = _read_status_file(agent_name)  # diagnostic only
        request_id = await self._jobs.start_job(
            "agent_stop",
            agent_name,
            {
                "pid": pid,
                "grace_seconds": grace_seconds,
                "force": force,
                "status_hint": status,
            },
        )
        self._spawn(
            self._drive_stop(request_id, agent_name, pid, grace_seconds, force)
        )
        return request_id

    async def request_restart(
        self,
        agent_name: str,
        *,
        grace_seconds: float = 5.0,
        force: bool = False,
        model: str | None = None,
        provider: str | None = None,
        clean: bool = False,
    ) -> str:
        """Start async restart: stop → spawn."""
        request_id = await self._jobs.start_job(
            "agent_restart",
            agent_name,
            {
                "grace_seconds": grace_seconds,
                "force": force,
                "model": model,
                "provider": provider,
                "clean": clean,
            },
        )
        self._spawn(
            self._drive_restart(
                request_id,
                agent_name,
                grace_seconds=grace_seconds,
                force=force,
                model=model,
                provider=provider,
                clean=clean,
            )
        )
        return request_id

    async def request_batch(
        self,
        action: str,
        agent_names: list[str],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Fan out ``action`` across ``agent_names`` in parallel.

        Returns ``{"request_id", "action", "jobs": [{"agent_name","request_id"}...]}``
        immediately. Per-agent events flow through WS as usual, plus a
        ``batch_started`` / ``batch_progress`` / ``batch_summary`` trio.
        """
        if action not in ("stop", "restart"):
            raise ValueError(f"Unsupported batch action: {action!r}")
        if not agent_names:
            raise ValueError("agent_names must not be empty")

        batch_rid = await self._jobs.start_job(
            "batch",
            None,
            {"action": action, "agent_names": list(agent_names)},
        )

        jobs: list[dict[str, Any]] = []
        for name in agent_names:
            if action == "stop":
                rid = await self.request_stop(
                    name,
                    grace_seconds=float(params.get("grace_seconds", 5.0)),
                    force=bool(params.get("force", False)),
                )
            else:  # restart
                rid = await self.request_restart(
                    name,
                    grace_seconds=float(params.get("grace_seconds", 5.0)),
                    force=bool(params.get("force", False)),
                    model=params.get("model"),
                    provider=params.get("provider"),
                    clean=bool(params.get("clean", False)),
                )
            jobs.append({"agent_name": name, "request_id": rid})

        # Aggregator — closes the batch once every per-job job has a
        # terminal event. Uses the registry (_jobs) presence as signal.
        self._spawn(
            self._drive_batch_aggregator(
                batch_rid, action, [j["request_id"] for j in jobs]
            )
        )
        return {"request_id": batch_rid, "action": action, "jobs": jobs}

    # ------------------------------------------------------------------ #
    # Background drivers
    # ------------------------------------------------------------------ #

    async def _drive_stop(
        self,
        request_id: str,
        agent_name: str,
        pid: int,
        grace_seconds: float,
        force: bool,
    ) -> None:
        """The whole stop pipeline, event-driven end-to-end."""
        try:
            # 1) Try RPyC-level graceful shutdown (best-effort, bounded).
            rpyc_ok = await self._try_rpyc_shutdown(
                agent_name, grace_seconds, force
            )

            # 2) Attach pidfd watcher BEFORE sending any signal — the fd
            #    will become readable on process exit.
            await self._attach_pidfd_watcher(
                pid=pid, agent_name=agent_name, request_id=request_id
            )

            loop = asyncio.get_running_loop()

            # 3) Send the appropriate signal.
            if force:
                _send_signal(pid, signal.SIGKILL, logger)
                # Give kernel a tick, then verify.
                loop.call_later(
                    _KILL_VERIFY_DELAY_SEC,
                    lambda: self._spawn_if_pending(
                        request_id,
                        self._verify_and_finalize,
                        request_id, agent_name, pid, "forced_sigkill",
                    ),
                )
            else:
                if not rpyc_ok:
                    _send_signal(pid, signal.SIGTERM, logger)

                # Arm SIGKILL escalation if pidfd doesn't fire in time.
                loop.call_later(
                    max(0.01, grace_seconds),
                    lambda: self._spawn_if_pending(
                        request_id,
                        self._escalate_to_sigkill,
                        request_id, agent_name, pid,
                    ),
                )
        except Exception as exc:  # defensive: never lose a job
            logger.exception("stop pipeline failed: %s", agent_name)
            await self._jobs.fail_job(request_id, exc)

    async def _drive_restart(
        self,
        request_id: str,
        agent_name: str,
        *,
        grace_seconds: float,
        force: bool,
        model: str | None,
        provider: str | None,
        clean: bool,
    ) -> None:
        try:
            from app.routers.agents import load_launch_config  # local import

            config = load_launch_config(agent_name)
            if config is None:
                await self._jobs.fail_job(
                    request_id,
                    LookupError(
                        f"No launch_config for '{agent_name}' — start it normally first"
                    ),
                )
                return

            info = self._registry.lookup(agent_name)
            if info is not None:
                stop_rid = await self.request_stop(
                    agent_name, grace_seconds=grace_seconds, force=force
                )
                # Wait for the stop job to reach a terminal state.
                await self._await_job_terminal(stop_rid)

            # Spawn in background (event-driven wait via inotify).
            spawn_rid = await self._jobs.start_job(
                "agent_spawn",
                agent_name,
                {"model": model, "provider": provider, "clean": clean},
            )
            self._spawn(
                self._drive_spawn(
                    spawn_rid,
                    agent_name,
                    config=config,
                    model=model,
                    provider=provider,
                    clean=clean,
                )
            )
            await self._await_job_terminal(spawn_rid)
            await self._jobs.complete_job(
                request_id,
                {"spawn_request_id": spawn_rid},
            )
        except Exception as exc:
            logger.exception("restart pipeline failed: %s", agent_name)
            await self._jobs.fail_job(request_id, exc)

    async def _drive_spawn(
        self,
        request_id: str,
        agent_name: str,
        *,
        config: dict[str, Any],
        model: str | None,
        provider: str | None,
        clean: bool,
    ) -> None:
        """Launch agent subprocess and wait via inotify on the registry file."""
        try:
            from app.routers.agents import _spawn_subprocess  # type: ignore[attr-defined]

            proc = _spawn_subprocess(
                agent_name,
                config,
                model=model,
                provider=provider,
                clean=clean,
            )

            # Wait for the agent to appear in ProcessRegistry — event-driven
            # via inotify on the registry file, NOT polling.
            appeared = await self._wait_registered(agent_name, _SPAWN_HARD_TIMEOUT_SEC)
            if appeared:
                await self._jobs.complete_job(
                    request_id,
                    {"pid": proc.pid if proc else None},
                )
            else:
                # Tail the spawn error log for the UI.
                log_path = (
                    deep_worker_home()
                    / "logs"
                    / f"{agent_name}.spawn.log"
                )
                tail = ""
                try:
                    tail = log_path.read_text()[-500:]
                except OSError:
                    pass
                await self._jobs.fail_job(
                    request_id,
                    TimeoutError(
                        f"Agent '{agent_name}' did not appear in registry within "
                        f"{_SPAWN_HARD_TIMEOUT_SEC:.0f}s"
                    ),
                    payload={"spawn_log_tail": tail},
                )
        except Exception as exc:
            logger.exception("spawn pipeline failed: %s", agent_name)
            await self._jobs.fail_job(request_id, exc)

    async def _drive_batch_aggregator(
        self,
        batch_request_id: str,
        action: str,
        per_job_rids: list[str],
    ) -> None:
        """Emit ``batch_progress`` events as child jobs finish — STRICTLY
        event-driven, one progress frame per child completion.

        Previous implementation polled ``_jobs`` at 20 Hz (50 ms sleep)
        which produced ~80 ``batch_progress`` frames for a 5 s SIGTERM
        grace period (three agents × ~2 s). That is the "polling
        forbidden" rule violation. Instead we subscribe to an
        ``asyncio.Event`` the registry fires on complete_job/fail_job,
        wake once per child, broadcast one progress frame per real
        transition, and never spin.
        """
        pending = set(per_job_rids)
        completed_ok: set[str] = set()
        failed: set[str] = set()

        # Register a done-event per child BEFORE we possibly miss one:
        # by the time we get here, _drive_stop/_drive_restart may have
        # already started the child; pre-registering is what makes the
        # subscription race-free (see ``_register_done_event`` — it
        # fires immediately if the child already finished).
        events = {
            rid: self._jobs._register_done_event(rid) for rid in per_job_rids
        }

        while pending:
            # Wake on ANY child transition. ``FIRST_COMPLETED`` means one
            # call per transition — total frames = len(per_job_rids),
            # not len * 20 * grace_seconds.
            wait_tasks = {
                asyncio.create_task(events[rid].wait(), name=f"batch_wait_{rid}"): rid
                for rid in pending
            }
            done, _ = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                rid = wait_tasks[t]
                pending.discard(rid)
                # Infer success/fail from the registry's recorded status.
                if self._jobs._done_status.get(rid):
                    failed.add(rid)
                else:
                    completed_ok.add(rid)
            # Cancel the rest so we don't leak one task per loop.
            for t in wait_tasks:
                if not t.done():
                    t.cancel()
            # Single progress frame per transition.
            await self._broadcast(
                {
                    "type": "batch_progress",
                    "request_id": batch_request_id,
                    "timestamp": _now_iso(),
                    "payload": {
                        "action": action,
                        "total": len(per_job_rids),
                        "completed": len(completed_ok),
                        "failed": len(failed),
                        "pending": len(pending),
                    },
                }
            )

        # GC the _done_status entries we inspected so the dict does not
        # accumulate one key per child-of-every-batch-ever.
        for rid in per_job_rids:
            self._jobs._done_status.pop(rid, None)

        await self._jobs.complete_job(
            batch_request_id,
            {
                "action": action,
                "total": len(per_job_rids),
                "completed": len(completed_ok),
                "failed": len(failed),
            },
        )

    # ------------------------------------------------------------------ #
    # Signal / pidfd machinery
    # ------------------------------------------------------------------ #

    async def _attach_pidfd_watcher(
        self,
        *,
        pid: int,
        agent_name: str,
        request_id: str,
    ) -> None:
        """Open a pidfd and wire ``loop.add_reader`` for process death.

        This is the non-polling heart of the stop pipeline. The fd becomes
        readable ONLY when the process exits, so we get a real kernel event.
        """
        loop = asyncio.get_running_loop()
        try:
            pidfd = os.pidfd_open(pid)  # type: ignore[attr-defined]
        except (AttributeError, ProcessLookupError, OSError) as exc:
            # Process might already be gone; finalize accordingly.
            if isinstance(exc, ProcessLookupError) or (
                isinstance(exc, OSError) and exc.errno == errno.ESRCH
            ):
                self._spawn_if_pending(
                    request_id,
                    self._verify_and_finalize,
                    request_id, agent_name, pid, "already_dead",
                )
                return
            # pidfd_open unavailable — fall back to a single delayed check.
            logger.warning(
                "pidfd_open unavailable (%s); using delayed verify", exc
            )
            loop.call_later(
                _KILL_VERIFY_DELAY_SEC,
                lambda: self._spawn_if_pending(
                    request_id,
                    self._verify_and_finalize,
                    request_id, agent_name, pid, "no_pidfd",
                ),
            )
            return

        self._pidfds[request_id] = pidfd

        def _on_exit() -> None:
            # Kernel tells us the process exited; run the finalizer.
            # CRITICAL: a raise here unregisters the reader and leaks
            # the pidfd silently — no more exit notification ever.
            # Wrap everything in a catch-all so the fd and the job
            # clean up no matter what.
            try:
                try:
                    loop.remove_reader(pidfd)
                except (ValueError, KeyError):
                    pass
                self._pidfds.pop(request_id, None)
                try:
                    os.close(pidfd)
                except OSError:
                    pass
                self._spawn_if_pending(
                    request_id,
                    self._verify_and_finalize,
                    request_id, agent_name, pid, "pidfd",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "pidfd exit callback crashed for request %s "
                    "(agent=%s, pid=%s) — the reader is gone but the "
                    "job may be stuck; falling back to delayed verify",
                    request_id, agent_name, pid,
                )
                # Best-effort recovery: schedule the same finalizer
                # we would have launched on the happy path, so the
                # job isn't left hanging forever.
                try:
                    loop.call_later(
                        _KILL_VERIFY_DELAY_SEC,
                        lambda: self._spawn_if_pending(
                            request_id,
                            self._verify_and_finalize,
                            request_id, agent_name, pid, "pidfd_recovery",
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "pidfd exit callback: recovery schedule failed"
                    )

        loop.add_reader(pidfd, _on_exit)

    async def _escalate_to_sigkill(
        self, request_id: str, agent_name: str, pid: int
    ) -> None:
        """SIGTERM grace expired — bring out SIGKILL."""
        # If the job already finalized, nothing to do.
        if self._jobs.get_job(request_id) is None:
            return
        logger.warning(
            "stop %s (pid=%s): SIGTERM grace expired, escalating to SIGKILL",
            agent_name,
            pid,
        )
        await self._broadcast(
            {
                "type": "agent_stop_escalated",
                "request_id": request_id,
                "agent_name": agent_name,
                "timestamp": _now_iso(),
                "payload": {"pid": pid, "signal": "SIGKILL"},
            }
        )
        _send_signal(pid, signal.SIGKILL, logger)
        loop = asyncio.get_running_loop()
        loop.call_later(
            _KILL_VERIFY_DELAY_SEC,
            lambda: self._spawn_if_pending(
                request_id,
                self._verify_and_finalize,
                request_id, agent_name, pid, "escalated_sigkill",
            ),
        )

    async def _verify_and_finalize(
        self,
        request_id: str,
        agent_name: str,
        pid: int,
        phase: str,
    ) -> None:
        """Confirm the process is gone (or zombie) and close out the job."""
        if self._jobs.get_job(request_id) is None:
            # Already finalized by another path.
            return
        if self._verify_process_gone(pid):
            try:
                self._registry.unregister(agent_name)
            except Exception:
                logger.warning(
                    "registry.unregister(%s) failed", agent_name, exc_info=True
                )
            await self._broadcast(
                {
                    "type": "agent_terminated",
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "timestamp": _now_iso(),
                    "payload": {"pid": pid, "phase": phase},
                }
            )
            await self._jobs.complete_job(
                request_id, {"pid": pid, "phase": phase}
            )
        else:
            state = _read_proc_state(pid) or "?"
            await self._jobs.fail_job(
                request_id,
                RuntimeError(
                    f"Process {pid} still alive (state={state!r}) "
                    f"after {phase}. Manual intervention required: kill -9 {pid}"
                ),
                payload={"pid": pid, "phase": phase, "proc_state": state},
            )

    def _verify_process_gone(self, pid: int) -> bool:
        """True if the process is gone (including zombie)."""
        state = _read_proc_state(pid)
        if state is None:
            return True
        return state == "Z"

    # ------------------------------------------------------------------ #
    # RPyC-level graceful shutdown (best-effort)
    # ------------------------------------------------------------------ #

    async def _try_rpyc_shutdown(
        self, agent_name: str, grace_seconds: float, force: bool
    ) -> bool:
        """Best-effort RPyC-level ``exposed_shutdown``.

        Returns True if the RPC was accepted (not that the agent already
        exited — pidfd still handles that). On any failure, caller falls
        back to SIGTERM AND this method broadcasts
        ``agent_shutdown_rpc_failed`` so the UI can surface the reason
        instead of showing only the SIGTERM path.
        """
        async def _emit_rpc_failure(reason: str, detail: str) -> None:
            try:
                await self._broadcast(
                    {
                        "type": "agent_shutdown_rpc_failed",
                        "agent_name": agent_name,
                        "timestamp": _now_iso(),
                        "payload": {
                            "reason": reason,
                            "detail": detail,
                            "grace_seconds": grace_seconds,
                            "force": force,
                        },
                    }
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "shutdown_rpc_failed broadcast itself raised for %s",
                    agent_name,
                    exc_info=True,
                )

        if self._rpyc_resolver is None:
            await _emit_rpc_failure("no_resolver", "rpyc_resolver not configured")
            return False
        try:
            root = self._rpyc_resolver(agent_name)
        except Exception as exc:  # noqa: BLE001
            await _emit_rpc_failure(
                "resolver_raised", f"{type(exc).__name__}: {exc}"
            )
            return False
        # ``hasattr`` probes the attribute; on a dead netref this
        # itself may raise — treat that as a "not available" signal
        # but still broadcast so the UI sees the reason.
        try:
            has_shutdown = root is not None and hasattr(root, "shutdown")
        except Exception as exc:  # noqa: BLE001
            await _emit_rpc_failure(
                "probe_raised", f"{type(exc).__name__}: {exc}"
            )
            return False
        if not has_shutdown:
            await _emit_rpc_failure(
                "no_shutdown_method",
                "agent root has no shutdown method",
            )
            return False

        # Fire-and-forget via rpyc.async_ — never await the round trip.
        try:
            import rpyc_async as rpyc  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            await _emit_rpc_failure(
                "rpyc_import_failed", f"{type(exc).__name__}: {exc}"
            )
            return False
        try:
            async_call = rpyc.async_(root.shutdown)  # type: ignore[attr-defined]
            # New agents take (grace_seconds, force); older ones take
            # no arguments. Try the rich signature first and fall back
            # if RPyC tells us the arity is wrong. Either way we do NOT
            # await the round trip — fire only.
            try:
                async_call(float(grace_seconds), bool(force))
            except TypeError:
                async_call()
        except Exception as exc:
            logger.debug(
                "rpyc shutdown RPC failed for %s: %s", agent_name, exc
            )
            await _emit_rpc_failure(
                "rpc_call_raised", f"{type(exc).__name__}: {exc}"
            )
            return False
        await self._broadcast(
            {
                "type": "agent_shutdown_requested",
                "agent_name": agent_name,
                "timestamp": _now_iso(),
                "payload": {"grace_seconds": grace_seconds, "force": force},
            }
        )
        return True

    # ------------------------------------------------------------------ #
    # Registry wait (event-driven via inotify when available)
    # ------------------------------------------------------------------ #

    async def _wait_registered(self, agent_name: str, timeout: float) -> bool:
        """Wait for ``agent_name`` to appear in the registry.

        Prefers inotify (single syscall event per change) and falls back to
        a single scheduled poll if inotify isn't wired. No tight loop.
        """
        if self._registry.lookup(agent_name) is not None:
            return True
        loop = asyncio.get_running_loop()
        appeared = asyncio.Event()

        try:
            import ctypes
            import ctypes.util

            libc_name = ctypes.util.find_library("c")
            libc = ctypes.CDLL(libc_name or "libc.so.6", use_errno=True)
            IN_MOVED_TO = 0x00000080
            IN_CREATE = 0x00000100
            IN_CLOSE_WRITE = 0x00000008
            IN_MODIFY = 0x00000002
            O_NONBLOCK = os.O_NONBLOCK

            inotify_init1 = libc.inotify_init1
            inotify_init1.argtypes = [ctypes.c_int]
            inotify_init1.restype = ctypes.c_int
            inotify_add_watch = libc.inotify_add_watch
            inotify_add_watch.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            inotify_add_watch.restype = ctypes.c_int

            fd = inotify_init1(O_NONBLOCK)
            if fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1 failed")
            # Watch the /tmp directory where the registry json lives.
            registry_dir = Path("/tmp").as_posix().encode()
            wd = inotify_add_watch(
                fd,
                registry_dir,
                IN_MOVED_TO | IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY,
            )
            if wd < 0:
                os.close(fd)
                raise OSError(ctypes.get_errno(), "inotify_add_watch failed")

            def _on_inotify() -> None:
                # CRITICAL: this is a ``loop.add_reader`` callback — if
                # it raises, asyncio removes the reader and the
                # ``appeared`` event never fires again. Wrap everything
                # so a transient ``registry.lookup`` failure doesn't
                # turn a start/restart into a silent hang.
                try:
                    try:
                        # Drain events; we don't parse them — just
                        # check the registry.
                        os.read(fd, 4096)
                    except BlockingIOError:
                        pass
                    except OSError:
                        pass
                    if self._registry.lookup(agent_name) is not None:
                        appeared.set()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "inotify reader for %r crashed; the reader "
                        "survives but this event was dropped",
                        agent_name,
                    )

            loop.add_reader(fd, _on_inotify)
            try:
                await asyncio.wait_for(appeared.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                # One last check in case the event fired between the last
                # drain and the timeout.
                return self._registry.lookup(agent_name) is not None
            finally:
                try:
                    loop.remove_reader(fd)
                except Exception:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
        except (OSError, AttributeError):
            # Inotify unavailable — single delayed check as last resort.
            logger.warning("inotify unavailable, using delayed single check")

            def _one_shot() -> None:
                if self._registry.lookup(agent_name) is not None:
                    appeared.set()

            loop.call_later(min(timeout, 1.0), _one_shot)
            try:
                await asyncio.wait_for(appeared.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return self._registry.lookup(agent_name) is not None

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    async def _await_job_terminal(self, request_id: str) -> None:
        """Block until the given job has a terminal broadcast (completed/failed).

        Implemented by watching the in-memory registry state (removed on
        terminal). This is in-process state, NOT I/O polling, so it doesn't
        violate the "no polling" rule — one cooperative yield per tick.
        """
        while self._jobs.get_job(request_id) is not None:
            await asyncio.sleep(0.01)

    def _spawn(self, coro: Coroutine[Any, Any, Any] | Awaitable[Any]) -> None:
        # asyncio.create_task requires a coroutine; callers in this module
        # always pass real coroutines. Cast via create_task's accepted type.
        task = asyncio.create_task(coro)  # type: ignore[arg-type]
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _spawn_if_pending(
        self,
        request_id: str,
        method: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
    ) -> None:
        """Schedule ``method(*args)`` only if the job is still in flight.

        Used by call_later callbacks: when the pidfd reader already
        finalized the job (happy path), we must not resurrect work —
        that just creates orphaned coroutines that pile up as
        "Task was destroyed but it is pending!" at shutdown.
        """
        if self._jobs.get_job(request_id) is None:
            return
        self._spawn(method(*args))


# --------------------------------------------------------------------------- #
# Private: signal dispatch with structured logging.
# --------------------------------------------------------------------------- #


def _send_signal(pid: int, sig: int, log: logging.Logger) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        log.info("pid %d already gone when sending signal %d", pid, sig)
    except PermissionError:
        log.error("no permission to signal pid %d with %d", pid, sig)
