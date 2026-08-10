"""gRPC server for the shared per-project LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §4 (contract) and §5
(lifecycle). Commit 4 wires the administrative surface — Health,
AttachSession, ProjectStatus, RefreshProject — and stubs the 11
Serena tool methods with ``UNIMPLEMENTED`` so a client immediately
sees the shape. Full tool proxying lands in commit 5 (bridge/proxy).

Runtime shape:

* Binds a Unix Domain Socket at ``<slug_dir>/grpc.sock`` (600 file
  inside a 700 directory). Loopback TCP + PSK is the fallback path
  documented in the design; not needed on Linux where UDS is native.
* Idle-shutdown timer per §16.7: after ``idle_shutdown_s`` seconds
  with zero attached clients, the server calls ``stop()`` on itself.
  Configurable, ≥60 s enforced; 0 forbidden.
* Keepalive: AttachSession pushes ``SESSION_EVENT_KEEPALIVE`` frames
  on the same stream at ``keepalive_interval_s`` (default 25 s to
  stay under gRPC's default 30 s http/2 idle timeout). This is
  server-driven — the client never polls.
* Backend readiness: ``mark_backend_ready()`` flips Health from
  ``alive=False`` to ``alive=True``. In commit 5 the bridge/Serena
  supervisor calls this; here it's a public method so tests can
  exercise both states.

This module is INSIDE the allowlisted package
``deep_worker_cli.lsp.service*`` (policy §8) — only this package is
permitted to spawn Serena in later commits.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Final

import grpc
from grpc import aio as grpc_aio

from deep_worker_cli.lsp.service.change_detector import ChangeDetector
from deep_worker_cli.lsp.service.proto import (
    lsp_service_pb2 as pb,
    lsp_service_pb2_grpc as pbg,
)
from deep_worker_cli.lsp.service.serena_backend import (
    _DISPATCH_TIMEOUT_MSG,
    _LS_TERMINATED_MSG,
    _LS_TERMINATED_MUTATING_MSG,
    _RPC_TO_TOOL,
    _is_backend_terminated,
)
from deep_worker_cli.lsp.service.watcher import ProjectWatcher

_SERVICE_VERSION: Final = "0.1.0-commit4"
_DEFAULT_KEEPALIVE_S: Final = 25.0
_MIN_IDLE_SHUTDOWN_S: Final = 60.0

#: How long ``AttachSession`` holds the ATTACHED event back waiting for
#: the backend to come online. Covers Serena's ~4–8s startup with a wide
#: margin. On expiry we emit ATTACHED anyway, tagged ``backend-not-ready``
#: so the bridge unblocks and tool calls fail loudly with the real
#: reason instead of hanging at connect().
_DEFAULT_BACKEND_READY_WAIT_S: Final = 60.0

#: Tool RPCs that MUTATE the workspace. Replaying one of these after a
#: language-server death is unsafe: Serena writes the edit to disk
#: BEFORE it asks the language server for diagnostics
#: (``serena/tools/symbol_tools.py`` + ``tools_base.DiagnosticsContext``),
#: and the transport can also die after the edit was applied. Either way
#: the ToolResult is lost while the change may already be on disk — an
#: ambiguous commit. Replaying ``insert_*`` would duplicate the body.
#:
#: Spelled out by hand rather than derived, so adding a new mutating
#: tool forces a deliberate decision here. See
#: ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md`` §4.2.0.
_MUTATING_RPCS: Final[frozenset[str]] = frozenset(
    {
        "RenameSymbol",
        "ReplaceSymbolBody",
        "InsertBeforeSymbol",
        "InsertAfterSymbol",
        "SafeDeleteSymbol",
    }
)

#: Tools that only READ the workspace, and are therefore safe to replay
#: after a language-server death.
#:
#: Spelled out explicitly rather than derived as ``all - mutating``.
#: A derived set fails OPEN: adding a new mutating tool to
#: ``_RPC_TO_TOOL`` would silently land it here and enable replay — the
#: exact accident the design forbids, and one no test could catch
#: because the union assertion would still hold by construction.
#: With two independent whitelists an unclassified RPC belongs to
#: neither, so ``_dispatch_tool`` refuses to replay it (fail closed)
#: and ``test_rpc_classification_is_total_and_disjoint`` fails loudly.
_READ_ONLY_RPCS: Final[frozenset[str]] = frozenset(
    {
        "FindSymbol",
        "FindReferencingSymbols",
        "FindDeclaration",
        "FindImplementations",
        "GetSymbolsOverview",
        "GetDiagnosticsForFile",
    }
)

#: Ceiling on the WHOLE tool RPC, retry included. Must stay below the
#: bridge's client deadline (``bridge_loop._CALL_TIMEOUT_S`` = 120s),
#: otherwise the retry would never be delivered: the bridge would
#: already have failed the call with DEADLINE_EXCEEDED and the agent
#: would see a raw transport error — precisely what this feature
#: exists to remove.
#:
#: The 10s reserve costs us nothing measurable: across 3003 recorded
#: Serena task durations in ``/home/<user>/.serena/logs/`` the slowest
#: call was 20.7s, i.e. the 110-120s band is empty in practice.
_DISPATCH_BUDGET_S: Final = 110.0

#: Ceiling on waiting for the supervisor to bring a NEW backend up
#: after a transport death. Cold Kotlin LS start measures ~4s in the
#: same logs; 30s is a wide margin that still leaves budget for the
#: replay itself.
_LS_TERMINATED_READY_WAIT_S: Final = 30.0

#: Below this many seconds left there is no point starting another
#: attempt — mirrors ``serena_backend._MIN_ATTEMPT_S``.
_MIN_ATTEMPT_S: Final = 0.05

_log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Input to :class:`LSPServiceServer`. Immutable per instance."""

    socket_path: Path
    project_root: Path
    idle_shutdown_s: float = _MIN_IDLE_SHUTDOWN_S
    # Where ``grpc_server`` is instantiated in production, both the
    # slug and the socket path are derived from ``project_root`` by
    # the CLI. We accept them separately so tests can bind to a
    # tmp_path without going through the registry.
    slug: str = ""
    #: Off-switch for the filesystem watcher. Production leaves it on;
    #: unit tests that only exercise gRPC surface flip it off to skip
    #: inotify setup.
    watch_enabled: bool = True


class LSPServiceServer:
    """gRPC server + attached-client bookkeeping + idle-shutdown timer.

    Owns exactly one ``grpc.aio.server`` instance. Not reentrant.
    """

    def __init__(self, config: ServerConfig) -> None:
        if config.idle_shutdown_s < _MIN_IDLE_SHUTDOWN_S:
            # Policy: minimum 60s. Zero forbidden.
            # Tests may still pass a small value if they use it purely
            # to speed a specific check — but the constructor is where
            # production callers get pinned.
            raise ValueError(
                f"idle_shutdown_s must be >= {_MIN_IDLE_SHUTDOWN_S}s, "
                f"got {config.idle_shutdown_s}"
            )
        self._config = config
        self._server: grpc_aio.Server | None = None
        self._servicer = _Servicer(self)
        self._started_ns = time.time_ns()
        self._backend_ready = False
        self._attached_clients = 0
        self._keepalive_interval_s = _DEFAULT_KEEPALIVE_S
        self._idle_since_ns: int | None = self._started_ns
        # Set whenever the attached-client count changes, so the idle
        # watchdog can block on it instead of ticking (no polling).
        self._activity = asyncio.Event()
        #: One outbound queue per attached AttachSession stream.
        self._sessions: list[asyncio.Queue[pb.SessionEvent]] = []
        #: Installed by the backend supervisor (commit 5). None => the
        #: tool RPCs abort UNIMPLEMENTED instead of faking a result.
        self._tool_handler: Callable[[str, str], Awaitable[pb.ToolResult]] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_refresh_unix = 0
        self._last_refresh_reason = ""
        #: Wired in :meth:`start`; None means "watcher disabled" (only
        #: used from tests that pin ``watch_enabled=False``).
        self._watcher: "ProjectWatcher | None" = None
        self._detector: "ChangeDetector | None" = None
        #: Optional backend refresh hook — the Serena binding installs
        #: it in stage 2. Runs BETWEEN REFRESH_STARTED and
        #: REFRESH_FINISHED so a subscriber observing FINISHED can
        #: trust that the backend already re-scanned.
        self._backend_refresh: Callable[[str], Awaitable[None]] | None = None
        #: Single-flight lock for refresh cycles. Bursts of triggers
        #: (auto + manual, or overlapping notifies from the detector)
        #: must not run the backend twice in parallel — that would
        #: race a language server's project reload against itself and
        #: interleave STARTED/FINISHED pairs on the session stream.
        self._refresh_lock = asyncio.Lock()
        #: Attached backend, if any. Owned by the server so ``stop()``
        #: can tear it down between the watcher and the gRPC surface
        #: (A6 shutdown order).
        self._backend: Any | None = None
        #: Optional backend owner (SerenaSupervisor) — when set, the
        #: server's shutdown delegates backend teardown to it instead
        #: of calling ``backend.stop()`` directly, so the pidfd monitor
        #: and rate-limit state are gone before we release the socket
        #: (design commit 10 §6).
        self._backend_owner: Any | None = None
        #: Event-driven readiness gate for AttachSession. Set by
        #: ``mark_backend_ready``, cleared by ``mark_backend_not_ready``.
        #: Lets the bridge block on the AttachSession stream instead of
        #: polling Health (project policy forbids poll loops).
        self._backend_ready_event = asyncio.Event()
        #: Ceiling for the readiness gate above. A knob (not a constant
        #: read inline) so tests can pin it without paying the full
        #: production wait.
        self._backend_ready_wait_s = _DEFAULT_BACKEND_READY_WAIT_S
        # Commit 12 (observability) — see docs/DESIGN_LSP_OBSERVABILITY.md.
        #: Nanoseconds when the CURRENT live backend was successfully
        #: attached. None while not ready; Health returns 0 in that case.
        self._backend_started_ns: int | None = None
        #: Successful auto-restarts performed by SerenaSupervisor in
        #: THIS service process. First attach is NOT a restart.
        self._restarts_total: int = 0
        #: POSIX seconds of the last successful (is_error=False) tool call.
        self._last_tool_call_unix: int = 0
        #: Total successful tool calls this process has answered.
        self._tool_calls_total: int = 0
        #: Cached ``serverInfo.version`` from the backend's initialize
        #: response. Populated in attach_backend, cleared on
        #: mark_backend_not_ready. Kept as an empty string when unknown.
        self._serena_version: str = ""
        #: Transient LSP cancellations retried by the backend instead
        #: of being surfaced to the agent. See
        #: docs/DESIGN_LSP_TRANSIENT_CANCEL_RETRY.md §4.5.
        self._transient_retries_total: int = 0
        #: Language-server deaths ridden out by replaying a read-only
        #: call. See docs/DESIGN_LSP_LS_TERMINATED_RETRY.md §4.6.
        self._ls_terminated_retries_total: int = 0
        #: Monotonic "a backend became ready" counter. A retry waits for
        #: this to CHANGE, not merely for the event to be set: the event
        #: object itself is replaced on every not-ready transition, so
        #: identity alone cannot tell "still the old backend" from "a
        #: new one came up" (ABA).
        self._backend_ready_generation: int = 0
        #: Knobs (not inline constants) so tests can pin the budgets
        #: without paying the production wait.
        self._dispatch_budget_s: float = _DISPATCH_BUDGET_S
        self._ls_terminated_ready_wait_s: float = _LS_TERMINATED_READY_WAIT_S

    # ── public knobs ─────────────────────────────────────────────────

    @property
    def config(self) -> ServerConfig:
        return self._config

    def mark_backend_ready(self) -> None:
        self._backend_ready = True
        # Bump BEFORE setting the event: a waiter woken by ``set()``
        # reads the generation to decide whether a new backend really
        # arrived, so it must never observe the old value.
        self._backend_ready_generation += 1
        # Wake any pending AttachSession streams that were parked
        # waiting on backend readiness (event-driven; no poll loop
        # in the bridge).
        self._backend_ready_event.set()

    def mark_backend_not_ready(self) -> None:
        """Flip Health.alive back off after the backend was lost.

        Introduced with the Serena attach (A3): when the backend dies
        and restart attempts are exhausted, Health.alive must return
        False so clients see the real state instead of a stale True.
        """
        self._backend_ready = False
        # Wake whoever is parked on the CURRENT event before swapping it
        # out. Without this, a waiter holding a reference to the old
        # object would sleep forever: the next ``mark_backend_ready``
        # sets the NEW event, which the old waiter is not watching
        # (lost wakeup, reproduced in
        # test_wait_survives_event_replacement). Woken waiters re-read
        # the attribute and re-park on the fresh event, still bounded by
        # their own deadline.
        self._backend_ready_event.set()
        # Fresh event so future waiters block again until the next
        # ``mark_backend_ready``. Existing waiters that were already
        # unblocked are unaffected; only new AttachSession streams
        # after this call will see the not-ready state.
        self._backend_ready_event = asyncio.Event()
        # A2: uptime describes the CURRENT live backend. Once it is
        # gone there is no live backend, so the counter must read 0
        # rather than keep ageing a corpse. Same for the version — a
        # stale version string would misreport what is running.
        self._backend_started_ns = None
        self._serena_version = ""

    async def attach_backend(self, backend: Any) -> bool:
        """Bring ``backend`` up and wire it into this server.

        Returns True on success (handlers installed, alive=True) and
        False on failure (handlers left absent, alive stays False,
        server keeps serving administrative RPCs) — A2. The exception
        from ``backend.start()`` is logged, never re-raised: the
        service must not die on backend startup failure because the
        registry entry would then point at a dead pid.

        Installs a **not-ready hook** on the backend (A3): when the
        backend notices its subprocess is gone (transport error on
        a tool call, or explicit ``stop``), it invokes the hook and
        we flip ``Health.alive`` back to False. Without this, the
        service would keep reporting a stale alive=True against a
        dead Serena — the exact regression A1/A3 were meant to kill.
        """
        # Install the not-ready hook BEFORE start() so a very early
        # crash still flips readiness back off (start() takes the
        # backend from lost=False to lost=False; the first tool call
        # or stop() is what surfaces death).
        try:
            backend.set_on_transient_retry(self.note_transient_retry)
        except AttributeError:
            # Older backend without the observability hook — the tool
            # path still works, only the counter stays at zero.
            pass
        try:
            backend.set_on_lost(self.mark_backend_not_ready)
        except AttributeError:
            # Older backend without the hook — should never happen in
            # production, but a graceful no-op is nicer than raising
            # from the wiring path.
            pass
        try:
            await backend.start()
        except Exception:  # noqa: BLE001 — fail-loud in log, not by dying
            _log.exception(
                "serena backend start failed for %s; service continues "
                "with alive=False",
                self._config.project_root,
            )
            return False
        self.set_tool_handler(backend.call_tool)
        self.set_backend_refresh_handler(backend.refresh)
        self._backend = backend
        # A2: the uptime clock starts at the successful attach, not at
        # process start — a restarted backend must report ITS age.
        self._backend_started_ns = time.time_ns()
        # A1: the version was captured inside the backend's single
        # ``initialize()`` handshake; we only copy the cached value.
        # An older backend without the property is not an error.
        self._serena_version = getattr(backend, "serena_version", "") or ""
        # Readiness LAST: ``mark_backend_ready`` wakes every parked
        # AttachSession stream, so anything those clients may read must
        # already be in place. Flipping it first leaves a window where
        # Health answers alive=True with an empty version / zero uptime.
        self.mark_backend_ready()
        return True

    def note_restart(self, success: bool) -> None:
        """Record an auto-restart attempt made by the supervisor.

        A3: only **successful** restarts are counted. A failed attempt
        already shows up as ``alive=False``; counting it here too
        would make ``restarts_total`` read as "how often did we try",
        which is a different (and less actionable) signal.
        """
        if success:
            self._restarts_total += 1

    def note_ls_terminated_retry(self) -> None:
        """Count one language-server death we replayed past (§4.6).

        Incremented once per retry that actually happens — never for
        mutating RPCs (they are not replayed) and never for a dispatch
        budget timeout (no termination was observed). A rising number
        means the LS is crashing while agents saw nothing.
        """
        self._ls_terminated_retries_total += 1

    def note_transient_retry(self) -> None:
        """Count one retried transient LSP cancellation (§4.5).

        Deliberately unconditional: unlike restarts, every retry is
        worth counting — the number is how we tell "the language
        server reindexes constantly" from "it never does", which is
        currently only a hypothesis.
        """
        self._transient_retries_total += 1

    def set_keepalive_interval_s(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("keepalive interval must be > 0")
        self._keepalive_interval_s = seconds

    def set_backend_ready_wait_s(self, seconds: float) -> None:
        """Pin how long AttachSession waits for backend readiness."""
        if seconds <= 0:
            raise ValueError("backend ready wait must be > 0")
        self._backend_ready_wait_s = seconds

    def set_backend_owner(self, owner: Any | None) -> None:
        """Delegate backend teardown to a supervisor.

        When set, :meth:`stop` calls ``owner.stop()`` instead of
        ``backend.stop()``. The supervisor owns the pidfd monitor,
        rate-limit state and the actual backend object; letting the
        server tear the backend down directly would strand the
        monitor (design commit 10 §6).
        """
        self._backend_owner = owner

    def note_refresh(self, reason: str) -> None:
        self._last_refresh_unix = int(time.time())
        self._last_refresh_reason = reason

    @property
    def watcher(self) -> "ProjectWatcher | None":
        return self._watcher

    def set_backend_refresh_handler(
        self, handler: "Callable[[str], Awaitable[None]] | None"
    ) -> None:
        """Install a coroutine the server should run BETWEEN
        REFRESH_STARTED and REFRESH_FINISHED. Stage 2 uses this to
        drive Serena's ``refresh_project``.
        """
        self._backend_refresh = handler

    def _trigger_refresh(self, reason: str) -> None:
        """Fire a refresh cycle observable to session subscribers.

        Called from the detector's ``on_refresh`` hook. Runs as an
        asyncio task so the detector can go straight back to
        debouncing the next burst instead of waiting for the backend.
        """
        asyncio.create_task(
            self._run_refresh(reason), name=f"lsp-refresh:{reason}"
        )

    async def _run_refresh(self, reason: str) -> None:
        # Serialize on ``_refresh_lock`` — see field docstring. STARTED
        # is emitted only after we hold the lock so subscribers never
        # see interleaved START/FINISH pairs.
        async with self._refresh_lock:
            self.publish_session_event(
                pb.SessionEvent(
                    event_type=pb.SESSION_EVENT_REFRESH_STARTED,
                    message=reason,
                    timestamp_ns=time.time_ns(),
                )
            )
            try:
                if self._backend_refresh is not None:
                    await self._backend_refresh(reason)
            except Exception as e:  # noqa: BLE001 — must not swallow silently
                _log.exception("backend refresh failed: %s", e)
                self.publish_session_event(
                    pb.SessionEvent(
                        event_type=pb.SESSION_EVENT_REFRESH_FINISHED,
                        message=f"error: {e}",
                        timestamp_ns=time.time_ns(),
                    )
                )
                return
            # note_refresh BEFORE FINISHED so a client that reacts to
            # FINISHED by hitting ProjectStatus reads the fresh value.
            self.note_refresh(reason)
            self.publish_session_event(
                pb.SessionEvent(
                    event_type=pb.SESSION_EVENT_REFRESH_FINISHED,
                    message=reason,
                    timestamp_ns=time.time_ns(),
                )
            )

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = grpc_aio.server()
        pbg.add_LspServiceServicer_to_server(self._servicer, self._server)
        # Bind UDS; grpc.aio accepts ``unix:<path>``.
        _prepare_socket_path(self._config.socket_path)
        addr = f"unix:{self._config.socket_path}"
        self._server.add_insecure_port(addr)
        await self._server.start()
        # Force 0600 on the socket file (kernel default depends on
        # umask; we already umask-fix in daemonize but the CLI-invoked
        # path may not, so be explicit).
        try:
            os.chmod(self._config.socket_path, 0o600)
        except OSError:  # pragma: no cover
            pass
        self._idle_task = asyncio.create_task(
            self._idle_watchdog(), name="lsp-idle-watchdog"
        )
        if self._config.watch_enabled:
            self._detector = ChangeDetector(
                self._config.project_root,
                on_refresh=self._trigger_refresh,
            )
            self._watcher = ProjectWatcher(
                self._config.project_root, self._detector
            )
            await self._watcher.start()

    async def stop(self, grace: float = 1.0) -> None:
        if self._server is None:
            return
        try:
            # A6 shutdown order: watcher → backend → gRPC. Watcher
            # first so no more auto-refreshes queue up; backend next
            # so its child process (Serena + LSP) is gone before we
            # release the socket; gRPC last so in-flight admin calls
            # can drain.
            if self._watcher is not None:
                try:
                    await self._watcher.stop()
                except Exception:  # noqa: BLE001 — never block shutdown
                    _log.exception("watcher stop failed")
                self._watcher = None
                self._detector = None
            # A6-updated (commit 10 §6): if a supervisor owns the
            # backend, delegate to it so its pidfd monitor and rate-
            # limit state are torn down BEFORE we call backend.stop().
            # Otherwise the monitor might briefly try to restart a
            # backend the server is already shutting down.
            owner = self._backend_owner
            if owner is not None:
                try:
                    await owner.stop()
                except Exception:  # noqa: BLE001 — never block shutdown
                    _log.exception("backend supervisor stop failed")
                self._backend_owner = None
                self._backend = None
                self._tool_handler = None
                self._backend_refresh = None
                self._backend_ready = False
            elif self._backend is not None:
                try:
                    await self._backend.stop()
                except Exception:  # noqa: BLE001 — same reason
                    _log.exception("backend stop failed")
                self._backend = None
                self._tool_handler = None
                self._backend_refresh = None
                self._backend_ready = False
            if self._idle_task is not None:
                self._idle_task.cancel()
                try:
                    await self._idle_task
                except (asyncio.CancelledError, BaseException):
                    pass
                self._idle_task = None
            await self._server.stop(grace=grace)
        finally:
            self._server = None
            _unlink_quiet(self._config.socket_path)
            self._stop_event.set()

    async def wait_closed(self) -> None:
        """Block until :meth:`stop` completes. Used by the CLI serve
        command as the top-level ``await`` point."""
        await self._stop_event.wait()

    # ── bookkeeping used by the servicer ─────────────────────────────

    def _attach(self) -> None:
        self._attached_clients += 1
        self._idle_since_ns = None
        self._activity.set()

    def _detach(self) -> None:
        self._attached_clients = max(0, self._attached_clients - 1)
        if self._attached_clients == 0:
            self._idle_since_ns = time.time_ns()
        self._activity.set()

    # ── session fan-out (event-driven; no polling) ───────────────────

    def _register_session(self) -> "asyncio.Queue[pb.SessionEvent]":
        """Create an outbound queue for one attached client."""
        queue: asyncio.Queue[pb.SessionEvent] = asyncio.Queue()
        self._sessions.append(queue)
        return queue

    def _unregister_session(self, queue: "asyncio.Queue[pb.SessionEvent]") -> None:
        try:
            self._sessions.remove(queue)
        except ValueError:
            pass

    async def _next_session_event(
        self, queue: "asyncio.Queue[pb.SessionEvent]", keepalive_s: float
    ) -> pb.SessionEvent:
        """Return the next real event, or a keepalive if none arrives.

        Blocks on the queue — a real event is delivered the instant it
        is published, and the timeout only exists to satisfy the gRPC
        http/2 idle window.
        """
        try:
            return await asyncio.wait_for(queue.get(), timeout=keepalive_s)
        except TimeoutError:
            return pb.SessionEvent(
                event_type=pb.SESSION_EVENT_KEEPALIVE,
                timestamp_ns=time.time_ns(),
            )

    def publish_session_event(self, event: pb.SessionEvent) -> None:
        """Fan an event out to every attached client immediately."""
        for queue in list(self._sessions):
            queue.put_nowait(event)

    # ── tool dispatch ────────────────────────────────────────────────

    def set_tool_handler(
        self, handler: "Callable[[str, str], Awaitable[pb.ToolResult]]"
    ) -> None:
        """Install the backend that actually executes tool calls.

        Until a handler is installed every tool RPC aborts with
        ``UNIMPLEMENTED`` rather than returning an empty result — a
        client must never mistake "no backend" for "no matches".
        """
        self._tool_handler = handler

    async def _run_attempt(
        self, rpc_name: str, args_json: str, remaining: float
    ) -> "tuple[pb.ToolResult | None, str]":
        """One attempt at the backend, capped by ``remaining``.

        Returns ``(result, failure)`` where exactly one is meaningful:
        ``failure`` is ``""`` on success, ``"timeout"`` when the budget
        cancelled the attempt, or the exception text when the handler
        raised.

        ``_tool_handler`` is re-read by the caller on every attempt (see
        ``_dispatch_tool``): after a transport death the supervisor
        installs a NEW backend's ``call_tool``, and a cached reference
        would send the replay straight back into the corpse.
        """
        handler = self._tool_handler
        if handler is None:  # pragma: no cover — guarded by caller
            return None, "serena backend not running"
        try:
            result = await asyncio.wait_for(
                handler(rpc_name, args_json), timeout=remaining
            )
        except asyncio.TimeoutError:
            return None, "timeout"
        except Exception as e:  # noqa: BLE001 — classified by the caller
            return None, f"{e}"
        return result, ""

    async def _await_new_backend(self, deadline: float, gen0: int) -> bool:
        """Block until a NEW backend is ready, or the deadline passes.

        ``gen0`` is the readiness generation the CALLER observed at the
        moment it saw ``_backend_ready is False``. It must be captured
        there, not here: between the caller's check and this method's
        first statement the supervisor can complete the restart and call
        ``mark_backend_ready``. Snapshotting the generation locally would
        then capture the already-bumped value and make us wait for a
        SECOND recovery that never comes — a spurious timeout on a
        backend that is, in fact, healthy.

        Event-driven by construction: every iteration parks on an
        ``asyncio.Event``. This is NOT a poll loop — there is no timer
        tick and no periodic re-check; we go round again only because
        ``mark_backend_not_ready`` deliberately woke us to hand over a
        replacement event (see its comment), and the whole series is
        bounded by one shared ``deadline``.
        """
        loop = asyncio.get_running_loop()
        while True:
            if self._backend_ready and self._backend_ready_generation != gen0:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                # The generation already moved past ``gen0``, so a
                # replacement backend WAS built during this call; it is
                # merely not ready again at this instant (a second
                # flap). Retrying is still the right move: the handler
                # now points at the new backend, and one more attempt
                # is cheaper than telling the agent the server died
                # when it did not.
                return self._backend_ready_generation != gen0
            event = self._backend_ready_event  # re-read every iteration
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return self._backend_ready_generation != gen0

    async def _dispatch_tool(
        self, rpc_name: str, args_json: str, context: grpc_aio.ServicerContext
    ) -> pb.ToolResult:
        """Execute one tool RPC, riding out a language-server death.

        See ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md``. The shape is:
        one attempt, and — for READ-ONLY tools only — at most one replay
        once the backend is back. Both attempts run under a single
        ``_dispatch_budget_s`` deadline so the whole RPC still fits
        inside the bridge's client deadline; otherwise the explanation
        below would never reach the agent.
        """
        if self._tool_handler is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{rpc_name}: no tool backend attached to this service",
            )
        tool = _RPC_TO_TOOL.get(rpc_name, rpc_name)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._dispatch_budget_s

        # Readiness generation as it was BEFORE the attempt. If the
        # backend dies and is rebuilt while we are inside the call, the
        # generation moves past this value, which is exactly the signal
        # "a new backend exists" — even when the whole death-and-restart
        # cycle completes before we get to look at readiness again.
        gen_before_call = self._backend_ready_generation

        result, failure = await self._run_attempt(
            rpc_name, args_json, max(0.0, deadline - loop.time())
        )

        if failure == "timeout":
            # NOT a termination: nothing said the server died, so
            # calling it one would be a false diagnostic.
            _log.warning(
                "%s: cancelled after the %.0fs dispatch budget",
                rpc_name, self._dispatch_budget_s,
            )
            return pb.ToolResult(
                is_error=True,
                error_message=_DISPATCH_TIMEOUT_MSG.format(
                    tool=tool, budget=self._dispatch_budget_s
                ),
            )

        # Both shapes of death are classified the same way: an
        # is_error ToolResult carrying the text, or a raised exception.
        raw = failure if result is None else (
            result.error_message if result.is_error else ""
        )
        if not _is_backend_terminated(raw):
            if result is not None:
                self._note_successful_tool_call(result)
                return result
            # A genuine, non-terminated failure of the handler must not
            # look like an empty result to the model.
            await context.abort(
                grpc.StatusCode.INTERNAL, f"{rpc_name} failed: {failure}"
            )
            raise  # unreachable; abort raises

        _log.warning("%s: language server terminated (%s)", rpc_name, raw)

        # Ambiguous commit: the edit may already be on disk, so a
        # replay could apply it twice. Tell the truth instead.
        #
        # FAIL CLOSED: only an explicitly whitelisted read-only RPC is
        # replayed. An RPC that is in neither whitelist (a tool added to
        # _RPC_TO_TOOL without classifying it) is treated like a
        # mutating one — never replayed — because assuming "probably
        # safe" is how a duplicate edit gets shipped.
        if rpc_name not in _READ_ONLY_RPCS:
            if rpc_name not in _MUTATING_RPCS:
                _log.error(
                    "%s is not classified as read-only or mutating; "
                    "refusing to replay it. Add it to one of the two "
                    "whitelists in grpc_server.py.",
                    rpc_name,
                )
            return pb.ToolResult(
                is_error=True,
                error_message=_LS_TERMINATED_MUTATING_MSG.format(tool=tool),
            )

        # Path (a): the transport died, readiness was flipped off and
        # the supervisor is building a replacement — wait for it. Path
        # (b): Serena is alive and only its language server died, so
        # readiness never moved and there is nothing to wait for.
        #
        # The generation is the one captured BEFORE the attempt, so a
        # restart that finished while we were inside the call (or
        # between this check and the helper's first statement) already
        # counts as recovery. Snapshotting it inside the helper would
        # race the supervisor and time out on a healthy backend.
        if not self._backend_ready:
            wait_until = min(
                deadline,
                loop.time() + self._ls_terminated_ready_wait_s,
            )
            if not await self._await_new_backend(wait_until, gen_before_call):
                return pb.ToolResult(
                    is_error=True,
                    error_message=_LS_TERMINATED_MSG.format(tool=tool),
                )

        if deadline - loop.time() < _MIN_ATTEMPT_S:
            # Delivering the explanation beats starting a replay we
            # cannot finish inside the client's deadline.
            return pb.ToolResult(
                is_error=True,
                error_message=_LS_TERMINATED_MSG.format(tool=tool),
            )

        self.note_ls_terminated_retry()
        retry_result, retry_failure = await self._run_attempt(
            rpc_name, args_json, max(0.0, deadline - loop.time())
        )
        retry_raw = retry_failure if retry_result is None else (
            retry_result.error_message if retry_result.is_error else ""
        )
        if retry_result is not None and not retry_result.is_error:
            self._note_successful_tool_call(retry_result)
            return retry_result
        if retry_result is not None and not _is_backend_terminated(retry_raw):
            # The replay reached a live server and got a real answer
            # (e.g. "symbol not found") — that belongs to the agent.
            return retry_result
        if (
            retry_result is None
            and retry_failure != "timeout"
            and not _is_backend_terminated(retry_failure)
        ):
            # The replay raised a genuine, unrelated exception. Dressing
            # that up as "the language server died" would be a false
            # diagnostic and would hide a real bug; the first attempt
            # aborts INTERNAL in this case, and the replay must behave
            # identically.
            await context.abort(
                grpc.StatusCode.INTERNAL, f"{rpc_name} failed: {retry_failure}"
            )
            raise  # unreachable; abort raises
        return pb.ToolResult(
            is_error=True,
            error_message=_LS_TERMINATED_MSG.format(tool=tool),
        )

    def _note_successful_tool_call(self, result: pb.ToolResult) -> None:
        """Advance the success counters (MUT-Y: errors must not count).

        A stream of backend errors must not look like healthy traffic,
        which is exactly what ``last_tool_call_unix`` is read for.
        """
        if not result.is_error:
            self._tool_calls_total += 1
            self._last_tool_call_unix = int(time.time())

    def _uptime_seconds(self) -> int:
        return max(0, (time.time_ns() - self._started_ns) // 1_000_000_000)

    def _backend_uptime_seconds(self) -> int:
        """Seconds since the current backend attached; 0 when absent."""
        if self._backend_started_ns is None:
            return 0
        return max(
            0, (time.time_ns() - self._backend_started_ns) // 1_000_000_000
        )

    async def _idle_watchdog(self) -> None:
        """Fire ``stop`` once idle for ``idle_shutdown_s``.

        Fully event-driven (project policy forbids polling): the task
        blocks on ``self._activity`` — an ``asyncio.Event`` set by
        :meth:`_attach` / :meth:`_detach` — and only ever wakes on a
        real state change or on the single deadline that actually
        matters. There is no periodic tick.
        """
        try:
            while True:
                if self._idle_since_ns is None:
                    # Someone is attached: block until a detach sets
                    # the event. No timeout → no wakeups meanwhile.
                    self._activity.clear()
                    await self._activity.wait()
                    continue
                idle_s = (time.time_ns() - self._idle_since_ns) / 1e9
                remaining = self._config.idle_shutdown_s - idle_s
                if remaining <= 0:
                    _log.info(
                        "LSP service %s idle for %.1fs → self-shutdown",
                        self._config.slug or self._config.project_root,
                        idle_s,
                    )
                    asyncio.create_task(self.stop(grace=1.0))
                    return
                # Wait exactly until the deadline, but wake early if a
                # client attaches (which cancels the countdown).
                self._activity.clear()
                try:
                    await asyncio.wait_for(self._activity.wait(), timeout=remaining)
                except TimeoutError:
                    pass  # deadline reached; re-evaluate above
        except asyncio.CancelledError:
            return


class _Servicer(pbg.LspServiceServicer):
    """Wire-side surface. Delegates state to :class:`LSPServiceServer`."""

    def __init__(self, owner: LSPServiceServer) -> None:
        self._owner = owner

    # ── administrative ──────────────────────────────────────────────

    async def Health(  # type: ignore[override]
        self, request: pb.HealthRequest, context: grpc_aio.ServicerContext
    ) -> pb.HealthReply:
        return pb.HealthReply(
            alive=self._owner._backend_ready,
            service_version=_SERVICE_VERSION,
            serena_version=self._owner._serena_version,
            project_root=str(self._owner.config.project_root),
            uptime_seconds=self._owner._uptime_seconds(),
            attached_clients=self._owner._attached_clients,
            backend_uptime_seconds=self._owner._backend_uptime_seconds(),
            restarts_total=self._owner._restarts_total,
        )

    async def ProjectStatus(  # type: ignore[override]
        self, request: pb.StatusRequest, context: grpc_aio.ServicerContext
    ) -> pb.StatusReply:
        return pb.StatusReply(
            project_root=str(self._owner.config.project_root),
            slug=self._owner.config.slug,
            transport="uds",
            transport_address=str(self._owner.config.socket_path),
            pid=os.getpid(),
            uptime_seconds=self._owner._uptime_seconds(),
            attached_clients=self._owner._attached_clients,
            last_refresh_unix=self._owner._last_refresh_unix,
            last_refresh_reason=self._owner._last_refresh_reason,
            active_providers=(
                self._owner._watcher.active_providers
                if self._owner._watcher is not None
                else []
            ),
            watched_paths=(
                [str(p) for p in self._owner._watcher.watched_paths]
                if self._owner._watcher is not None
                else []
            ),
            backend_uptime_seconds=self._owner._backend_uptime_seconds(),
            restarts_total=self._owner._restarts_total,
            last_tool_call_unix=self._owner._last_tool_call_unix,
            tool_calls_total=self._owner._tool_calls_total,
            serena_version=self._owner._serena_version,
            transient_retries_total=self._owner._transient_retries_total,
            ls_terminated_retries_total=(
                self._owner._ls_terminated_retries_total
            ),
        )

    async def RefreshProject(  # type: ignore[override]
        self, request: pb.RefreshRequest, context: grpc_aio.ServicerContext
    ) -> pb.RefreshReply:
        # Fires the same START→backend→FINISHED cycle as the watcher,
        # so subscribers see manual and automatic refreshes uniformly.
        reason = "forced" if request.force else "manual"
        self._owner._trigger_refresh(reason)
        return pb.RefreshReply(triggered=True, reason=reason)

    async def AttachSession(  # type: ignore[override]
        self,
        request: pb.AttachRequest,
        context: grpc_aio.ServicerContext,
    ) -> AsyncIterator[pb.SessionEvent]:
        self._owner._attach()
        try:
            # Delay ATTACHED until the backend is actually ready — the
            # bridge treats the first stream event as "safe to call
            # tools now", which is what lets it stay event-driven and
            # never poll Health. If the backend never comes up we still
            # emit ATTACHED (tagged ``backend-not-ready``) so the
            # stream is not blocked forever and tool calls fail loudly.
            #
            # A wake is NOT proof of readiness. ``mark_backend_not_ready``
            # deliberately sets the current event before replacing it (so
            # a parked retry can pick up the new one), which means this
            # waiter can be released by a NOT-ready transition. Reporting
            # that as "safe to call tools now" would be a lie the bridge
            # acts on. So: re-check actual readiness after every wake and
            # go back to waiting on the CURRENT event, all under the one
            # bounded deadline that was already there.
            ready_note = " backend-not-ready"
            attach_deadline = (
                asyncio.get_running_loop().time()
                + self._owner._backend_ready_wait_s
            )
            while True:
                if self._owner._backend_ready:
                    ready_note = ""
                    break
                remaining = attach_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                event = self._owner._backend_ready_event
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            yield pb.SessionEvent(
                event_type=pb.SESSION_EVENT_ATTACHED,
                message=(
                    f"attached client_id={request.client_id!r}{ready_note}"
                ),
                timestamp_ns=time.time_ns(),
            )
            # Event-driven: block on the session's outbound queue and
            # only emit a keepalive when the wait times out with no
            # real event to send. Real events (refresh, diagnostics)
            # wake this immediately instead of waiting out the tick.
            queue = self._owner._register_session()
            try:
                while True:
                    event = await self._owner._next_session_event(
                        queue, self._owner._keepalive_interval_s
                    )
                    yield event
            finally:
                self._owner._unregister_session(queue)
        except asyncio.CancelledError:
            raise
        finally:
            self._owner._detach()

    # ── Serena tool proxies (stubbed until commit 5) ────────────────
    #
    # Each returns UNIMPLEMENTED so a misconfigured client sees the
    # exact reason immediately. Commit 5 replaces each body with a
    # call into the internal Serena SSE session.

    async def FindSymbol(  # type: ignore[override]
        self, request: pb.FindSymbolRequest, context: grpc_aio.ServicerContext
    ) -> pb.FindSymbolReply:
        return pb.FindSymbolReply(result=await self._owner._dispatch_tool("FindSymbol", request.args_json, context))

    async def FindReferencingSymbols(  # type: ignore[override]
        self, request: pb.RefRequest, context: grpc_aio.ServicerContext
    ) -> pb.RefReply:
        return pb.RefReply(result=await self._owner._dispatch_tool("FindReferencingSymbols", request.args_json, context))

    async def FindDeclaration(  # type: ignore[override]
        self, request: pb.DeclRequest, context: grpc_aio.ServicerContext
    ) -> pb.DeclReply:
        return pb.DeclReply(result=await self._owner._dispatch_tool("FindDeclaration", request.args_json, context))

    async def FindImplementations(  # type: ignore[override]
        self, request: pb.ImplRequest, context: grpc_aio.ServicerContext
    ) -> pb.ImplReply:
        return pb.ImplReply(result=await self._owner._dispatch_tool("FindImplementations", request.args_json, context))

    async def GetSymbolsOverview(  # type: ignore[override]
        self, request: pb.OverviewRequest, context: grpc_aio.ServicerContext
    ) -> pb.OverviewReply:
        return pb.OverviewReply(result=await self._owner._dispatch_tool("GetSymbolsOverview", request.args_json, context))

    async def GetDiagnosticsForFile(  # type: ignore[override]
        self, request: pb.DiagRequest, context: grpc_aio.ServicerContext
    ) -> pb.DiagReply:
        return pb.DiagReply(result=await self._owner._dispatch_tool("GetDiagnosticsForFile", request.args_json, context))

    async def RenameSymbol(  # type: ignore[override]
        self, request: pb.RenameRequest, context: grpc_aio.ServicerContext
    ) -> pb.RenameReply:
        return pb.RenameReply(result=await self._owner._dispatch_tool("RenameSymbol", request.args_json, context))

    async def ReplaceSymbolBody(  # type: ignore[override]
        self, request: pb.ReplaceRequest, context: grpc_aio.ServicerContext
    ) -> pb.ReplaceReply:
        return pb.ReplaceReply(result=await self._owner._dispatch_tool("ReplaceSymbolBody", request.args_json, context))

    async def InsertBeforeSymbol(  # type: ignore[override]
        self, request: pb.InsertRequest, context: grpc_aio.ServicerContext
    ) -> pb.InsertReply:
        return pb.InsertReply(result=await self._owner._dispatch_tool("InsertBeforeSymbol", request.args_json, context))

    async def InsertAfterSymbol(  # type: ignore[override]
        self, request: pb.InsertRequest, context: grpc_aio.ServicerContext
    ) -> pb.InsertReply:
        return pb.InsertReply(result=await self._owner._dispatch_tool("InsertAfterSymbol", request.args_json, context))

    async def SafeDeleteSymbol(  # type: ignore[override]
        self, request: pb.DeleteRequest, context: grpc_aio.ServicerContext
    ) -> pb.DeleteReply:
        return pb.DeleteReply(result=await self._owner._dispatch_tool("SafeDeleteSymbol", request.args_json, context))


# ── helpers ──────────────────────────────────────────────────────────


def _prepare_socket_path(path: Path) -> None:
    """Ensure the parent dir exists (0700) and the socket file is
    absent so ``bind()`` doesn't hit EADDRINUSE from a stale sock."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:  # pragma: no cover
        pass
    _unlink_quiet(path)


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
