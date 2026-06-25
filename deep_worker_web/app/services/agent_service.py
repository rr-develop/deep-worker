"""AgentService for managing deep-worker agent connections.

This is the core service that:
1. Discovers agents from ProcessRegistry
2. Manages RPyC connections to agents
3. Subscribes to agent message callbacks
4. Broadcasts agent events to WebSocket clients

CRITICAL ARCHITECTURAL PRINCIPLES:
==================================
1. SINGLE EVENT LOOP: All async operations run in ONE event loop (no threads)
2. DIRECT RPyC: Connect directly to agents (no DwSession abstraction)
3. AGENT IS SOURCE OF TRUTH: All messages come from agent via callbacks
4. LOCALHOST ONLY: Only connect to 127.0.0.1 (security requirement)
5. NO SYNCHRONOUS BLOCKING: ALL RPyC calls MUST use fire_and_forget_async
   with callbacks. NEVER call rpyc methods synchronously — it blocks the
   event loop and freezes the entire web UI. Connections are opened via
   ``await async_connect(host, port, ...)`` (natively non-blocking) and
   closed via ``await conn.aclose()``. Status queries, message sends, and
   all other RPyC operations MUST go through fire_and_forget_async.
6. ONE CONNECTION PER (CLIENT, AGENT) PAIR — NO DUPLICATES.
   The rule is about *duplicate* connections, NOT about how many
   distinct clients may talk to one agent.

   * Two or more independent clients (separate web servers, separate
     ``dwcli`` invocations, a web server AND a TaskManager, etc.)
     MAY connect to the same agent concurrently. The agent's
     ``AsyncioServer`` is explicitly designed to accept multiple
     concurrent clients — that is a supported configuration, not a
     forbidden one. This module imposes NO single-client cap on the
     agent side.

   * What IS forbidden is a single client opening more than one
     ``rpyc.Connection`` to the *same* agent. Within one
     ``AgentService`` instance, ``self._connections[agent_name]``
     MUST hold at most one connection at any time. All concurrent
     API requests routed through this service for that agent
     (``/agents/<n>``, ``/status``, ``/history``, ``/message``,
     ``/interrupt``, ...) MUST multiplex over that single
     connection. RPyC supports many in-flight async calls per
     connection, so duplicating connections from the same client
     buys nothing and only wastes a port plus doubles handshake
     load on the agent.

   * Operationally this means connection establishment in this
     service is effectively single-flight per agent name: concurrent
     ``connect_to_agent(<name>)`` callers from the SAME
     ``AgentService`` instance must end up sharing one in-progress
     connect, not racing independent ``async_connect`` calls. A
     different ``AgentService`` (or any other RPyC client process)
     opening its own connection in parallel is fine — that is a
     different (client, agent) pair.
"""

import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.schemas.agent_schemas import AgentInfo, AgentStatus, AgentStatusInfo
from app.utils.supervisor import supervise

# ============================================================================
# CRITICAL: RPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP!
# ============================================================================
# Import rpyc from the /path/to/rpyc_async source directory to ensure AsyncioServer support
# ============================================================================
_rpyc_async_path = Path("/path/to/rpyc_async")
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc  # type: ignore[import-not-found]
from rpyc.core.async_connect import async_connect  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _decode_subscriber_payload(msg: Any) -> Any:
    """Turn a subscriber callback payload into a plain local value.

    The agent is expected to ``json.dumps`` broadcast messages on its
    side, so the fast path is just ``json.loads``. We accept several
    shapes for backwards compatibility:

    * ``str``: parsed as JSON; if that fails, returned as-is.
    * ``dict`` (real local dict): returned unchanged.
    * ``bytes`` / ``bytearray``: decoded as UTF-8 and parsed as JSON.
    * anything else (including RPyC netrefs from older agents):
      stringified. We intentionally do NOT iterate netref items —
      that would trigger ``sync_request()`` from the async loop that
      serves the connection and (a) leak remote objects, (b) spam
      ``RuntimeError`` in the log, (c) drop the broadcast entirely.
    * ``None``: returned as ``None`` (RPyC can send a null callback).

    Returns a plain Python value ready to be handed to
    ``_broadcast_message`` (which will still ``json.dumps(default=str)``
    at the WS edge as today).
    """
    import json

    if msg is None:
        return None
    if isinstance(msg, dict):
        # Already local — the happy new-agent path actually returns a
        # str, so this branch protects ``None``-surrogates from older
        # code paths that build a dict locally.
        return msg
    if isinstance(msg, (bytes, bytearray)):
        try:
            msg = msg.decode("utf-8")
        except UnicodeDecodeError:
            return str(msg)
    if isinstance(msg, str):
        try:
            return json.loads(msg)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — hand the raw text down as a minimal envelope
            # so downstream code doesn't need to special-case strings.
            return {"type": "raw", "content": msg}
    # Foreign object (RPyC netref or anything else) — DO NOT touch
    # ``.items()`` / ``.keys()`` / subscripting. Just stringify.
    return {"type": "raw", "content": str(msg)}


class AgentService:
    """Service for managing connections to deep-worker agents.
    
    This service uses ProcessRegistry to discover agents and RPyC AsyncioServer
    to communicate with them directly. All operations are async and run in the
    main event loop.
    
    Architecture:
    - Discovers agents from /tmp/deep_worker_registry.json
    - Establishes RPyC connections to agent processes
    - Subscribes to agent message callbacks
    - Broadcasts events to registered callbacks (WebSocket connections)
    """

    # Upper bound on a SINGLE background connect attempt (TCP + the eager
    # RPyC ``HANDLE_GETROOT`` handshake). ``async_connect``'s own timeout
    # only covers ``sock_connect``; the handshake is otherwise unbounded
    # and parks forever against a deadlocked peer. This bound lives in the
    # BACKGROUND connect task only — never in an HTTP handler — so it does
    # not block the UI. See the 2026-06-15 incident.
    _CONNECT_TIMEOUT: float = 5.0
    # Reconnect backoff: first failure waits this long before another
    # attempt is allowed, doubling each consecutive failure up to the cap.
    _RECONNECT_BACKOFF_START: float = 1.0
    _RECONNECT_BACKOFF_MAX: float = 30.0

    def __init__(
        self,
        registry: Any | None = None,
    ) -> None:
        """Initialize AgentService.
        
        Args:
            registry: ProcessRegistry instance (or mock for testing).
                     If None, will create real ProcessRegistry from deep_worker_cli.
        """
        # Import ProcessRegistry only if not provided (allows testing with mocks)
        if registry is None:
            import sys
            from pathlib import Path

            # Tree-local resolution: import the deep_worker_cli that lives
            # in the SAME tree as this web server. Hard-coding the prod
            # path used to make the dev web server load prod-tree
            # ProcessRegistry, defeating --dev. See
            # docs/DESIGN_DEV_FLAG.md.
            #
            # __file__ here =
            #   <TREE>/<web-app>/app/services/agent_service.py
            # parent x3 = <TREE>.
            _tree_root = Path(__file__).resolve().parent.parent.parent.parent
            deep_worker_path = (
                _tree_root / "deep_worker" / "deep_worker_cli"
            )
            if str(deep_worker_path) not in sys.path:
                sys.path.insert(0, str(deep_worker_path))

            from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]

            self._registry: Any = ProcessRegistry()
        else:
            self._registry = registry

        # Agent tracking
        # IMPORTANT: We store basic agent info from registry, not full connections yet
        # Full RPyC connections will be established on-demand or during monitoring
        self._agents: dict[str, AgentInfo] = {}

        # RPyC connections (agent_name -> RPyC connection object)
        # These are established when we connect to agents
        self._connections: dict[str, Any] = {}
        
        # RPyC root objects (agent_name -> root service proxy)
        # Used to call exposed methods on the agent
        self._roots: dict[str, Any] = {}
        
        # Message callback services for RPyC subscriptions
        # (agent_name -> callback service instance)
        self._callback_services: dict[str, Any] = {}

        # Process identity the live connection/subscription was established
        # against (agent_name -> (pid, rpyc_port)). Used to detect an
        # ungraceful restart: rpyc's ``_closed`` flag is NOT flipped when the
        # peer dies abruptly and respawns on a new port, so we compare the
        # registry's current identity against this to know the held connection
        # is stale. See DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 1).
        self._conn_identity: dict[str, tuple[int, int]] = {}

        # Message callbacks (for WebSocket broadcasting)
        # Each callback is an async function that receives message dict
        self._message_callbacks: list[Callable[[dict[str, Any]], Awaitable[None]]] = []

        # Auto-approve agents (agents where user enabled auto-approve)
        self._auto_approve_agents: set[str] = set()

        # Background monitoring task
        self._monitor_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        # ── Single-flight connect machinery ──────────────────────────
        # The 2026-06-15 incident: a stuck peer used to spawn a fresh connect
        # per HTTP request and per registry event, each parking forever on
        # the unbounded eager RPyC handshake. That stormed connections,
        # leaked half-open sockets, and (because handlers awaited the
        # connect) hung the per-agent UI.
        #
        # Invariants enforced here, ALL on the existing event loop — no
        # threads, no polling, no sleeps:
        #   * at most ONE in-flight connect task per agent;
        #   * HTTP handlers never await connect establishment
        #     (``ensure_connecting`` is non-blocking);
        #   * a dead/stuck peer is retried with capped exponential backoff,
        #     never in an infinite per-event loop.
        #
        # ``_connecting`` holds the live connect task per agent (a STRONG
        # ref, dropped in the task's done-callback — same anti-GC
        # discipline as fire_and_forget_async._INFLIGHT).
        self._connecting: dict[str, asyncio.Task[bool]] = {}
        # ``_reconnect_not_before`` is a per-agent loop-time deadline; no
        # new connect is launched before it. Set on failure, cleared on
        # success.
        self._reconnect_not_before: dict[str, float] = {}
        # Per-agent current backoff (seconds), grows on each consecutive
        # failure up to ``_RECONNECT_BACKOFF_MAX``.
        self._reconnect_backoff: dict[str, float] = {}

        # Lazily initialised lifecycle orchestrator (requires a running
        # event loop to spawn tasks — created on first access via
        # ``get_lifecycle()``). Routers depend on this to schedule
        # fire-and-forget stop / restart / batch ops.
        self._lifecycle: Any = None

        logger.info("AgentService initialized")

    def get_lifecycle(self) -> Any:
        """Return (and lazily create) the AgentLifecycleService.

        Must be called inside an async context (there must be a running
        asyncio loop). The lifecycle service schedules fire-and-forget
        tasks that rely on the same loop we run under.
        """
        if self._lifecycle is None:
            # Local import avoids any circulars at module load.
            from app.services.lifecycle import AgentLifecycleService

            self._lifecycle = AgentLifecycleService(
                registry=self._registry,
                broadcast=self._broadcast_message,
                rpyc_resolver=self.get_root,
            )
        return self._lifecycle

    async def initialize(self) -> None:
        """Initialize service and discover agents.
        
        This discovers all agents currently registered in ProcessRegistry
        and prepares the service for operation.
        """
        logger.info("Initializing AgentService - discovering agents")
        await self._discover_agents()
        logger.info(f"AgentService initialized with {len(self._agents)} agents")

    async def _discover_agents(self) -> None:
        """Discover agents from ProcessRegistry.
        
        This reads the ProcessRegistry and updates our internal agent tracking.
        Runs in async context but ProcessRegistry operations are sync (file-based).
        """
        # ProcessRegistry is sync (file-based), so we run it in executor
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        agent_process_infos = await loop.run_in_executor(
            None, self._registry.list_all
        )

        logger.info(f"Discovered {len(agent_process_infos)} agents from registry")

        # Convert to AgentInfo (our internal format)
        # NOTE: We don't know provider/model yet - will query via RPyC when needed
        for proc_info in agent_process_infos:
            agent_info = AgentInfo(
                name=proc_info.name,
                pid=proc_info.pid,
                status=AgentStatus.IDLE,  # Default, will update from agent
                provider="unknown",  # Will query from agent
                model="unknown",  # Will query from agent
                rpyc_port=proc_info.rpyc_port,
                working_dir=proc_info.working_dir,
                started_at=proc_info.started_at,
            )
            # Seed status + error_message from the agent's persisted
            # session_metadata.json. This is the DURABLE source: it
            # survives even when the agent's live in-memory status has
            # diverged (e.g. error persisted to disk but exposed_get_status
            # reverted to idle — the case observed live on 2026-05-20), so the web
            # UI shows the error detail on PAGE LOAD instead of an
            # error-with-no-explanation. The live RPyC poll later refreshes
            # this; the disk read just guarantees the snapshot is never
            # blank on the error cause.
            self._seed_status_from_metadata(agent_info)
            self._agents[proc_info.name] = agent_info
            logger.debug(f"Registered agent: {proc_info.name} (PID {proc_info.pid}, port {proc_info.rpyc_port})")

    def _seed_status_from_metadata(self, info: AgentInfo) -> None:
        """Best-effort: fill ``status`` + ``error_message`` from the
        agent's ``session_metadata.json`` under its data directory in the
        user's home.

        Never raises — a missing/corrupt file just leaves the defaults.
        """
        import json  # local import — matches this module's convention
        try:
            meta_file = (
                Path.home() / Path("~/.deep_worker").name / info.name / "session_metadata.json"
            )
            if not meta_file.exists():
                return
            data = json.loads(meta_file.read_text())
            status_str = data.get("status")
            if status_str:
                try:
                    info.status = self._convert_status(status_str)
                except Exception:  # noqa: BLE001 — unknown value → keep default
                    pass
            err = data.get("error_message")
            # Only attach the error text when the persisted state is an
            # error; otherwise leave it None so a stale message can't
            # linger on a recovered agent.
            if info.status == AgentStatus.ERROR and err:
                info.error_message = err
            else:
                info.error_message = None
            raw_status_metadata = data.get("status_metadata")
            if info.status == AgentStatus.IDLE and isinstance(raw_status_metadata, dict):
                info.status_metadata = raw_status_metadata
            else:
                info.status_metadata = None
        except Exception:  # noqa: BLE001 — disk read must never break discover
            logger.debug("could not seed status from metadata for %s", info.name, exc_info=True)

    async def list_agents(self) -> list[AgentInfo]:
        """Get list of all known agents.
        
        Returns:
            List of AgentInfo for all discovered agents.
        """
        return list(self._agents.values())

    async def get_agent(self, agent_name: str) -> AgentInfo | None:
        """Get info for a specific agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            AgentInfo if found, None otherwise.
        """
        return self._agents.get(agent_name)

    async def is_connected(self, agent_name: str) -> bool:
        """Check if we have an active RPyC connection to an agent.

        Defence-in-depth against the AsyncResult leak documented in
        a related internal incident analysis (not included here): a Connection
        whose channel is torn down already has ``conn._closed = True``
        (rpyc maintains this flag itself — no probe / round-trip /
        polling needed). Treat such a Connection as not-connected and
        evict it from ``_connections`` so the next caller triggers a
        clean reconnect instead of pumping more
        ``fire_and_forget_async(root.method())`` calls onto a dead
        channel.

        Eviction is limited to the local dict; we never call
        ``aclose`` from a probe path because rpyc's own teardown is
        what set ``_closed`` in the first place.

        Args:
            agent_name: Name of the agent.

        Returns:
            True if connected, False otherwise.
        """
        conn = self._connections.get(agent_name)
        if conn is None:
            return False

        # Identity check FIRST: an ungraceful restart respawns the agent on a
        # new (pid, rpyc_port) WITHOUT flipping the old connection's
        # ``_closed`` flag. If the registry's current identity differs from the
        # one this connection was opened against, the held connection points at
        # a dead process — evict everything so the next connect_to_agent dials
        # the new port and resubscribes. See
        # DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 1).
        recorded = self._conn_identity.get(agent_name)
        if recorded is not None:
            current = self._registry_identity(agent_name)
            if current is not None and current != recorded:
                self._evict_connection(agent_name)
                return False

        # rpyc maintains ``_closed`` as a strict ``bool`` and only
        # flips it to ``True`` on real teardown (see
        # ``Connection.close`` / ``aclose`` in rpyc_async). Use an
        # identity-check against ``True`` so that test doubles
        # (``MagicMock`` etc.) which auto-generate attribute access
        # do not get evicted on every probe — only a genuine
        # ``conn._closed = True`` triggers eviction.
        if getattr(conn, "_closed", False) is True:
            self._evict_connection(agent_name)
            return False
        return True

    def _record_conn_identity(
        self, agent_name: str, *, pid: int, rpyc_port: int
    ) -> None:
        """Remember the (pid, rpyc_port) the live connection was opened
        against, so a later restart (new identity) is detected as stale."""
        self._conn_identity[agent_name] = (pid, rpyc_port)

    def _registry_identity(self, agent_name: str) -> tuple[int, int] | None:
        """Current (pid, rpyc_port) for ``agent_name`` from in-memory
        discovery state, or None if unknown."""
        info = self._agents.get(agent_name)
        if info is None:
            return None
        try:
            return (int(info.pid), int(info.rpyc_port))
        except (TypeError, ValueError, AttributeError):
            return None

    def _evict_connection(self, agent_name: str) -> None:
        """Drop all local state for a stale/dead connection without any RPC
        (the peer is gone; touching the netref would just hang/raise). The
        next connect_to_agent makes a clean connection + subscription.

        CRITICAL — take the connection's asyncio reader OFF the loop before
        dropping our last reference. The rpyc Connection registers an
        ``add_reader`` callback on the running uvloop; that callback closes
        over the Connection, so merely ``pop``-ing it from these dicts does
        NOT free it — the reader stays armed forever. Once the agent's fd is
        recycled by uvicorn's own transport, the orphaned reader fires on a
        foreign fd, ``remove_reader`` then raises 'fd is used by transport',
        and the loop spins at ~100% CPU serving zero recv (the 7887 spin,
        2026-05-27). ``disable_asyncio_serving()`` is synchronous, does no
        RPC/IO, and unregisters the reader — exactly what the "peer is gone"
        path needs."""
        conn = self._connections.pop(agent_name, None)
        if conn is not None:
            try:
                conn.disable_asyncio_serving()
            except Exception:
                logger.debug(
                    "disable_asyncio_serving during evict of %s failed",
                    agent_name, exc_info=True,
                )
        self._roots.pop(agent_name, None)
        self._callback_services.pop(agent_name, None)
        self._conn_identity.pop(agent_name, None)

    def get_root(self, agent_name: str) -> Any | None:
        """Get RPyC root proxy for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            RPyC root proxy if connected, None otherwise.
        """
        return self._roots.get(agent_name)

    def ensure_connecting(self, agent_name: str) -> "asyncio.Future[bool]":
        """NON-BLOCKING connectivity trigger for HTTP handlers.

        Returns a *resolved* awaitable carrying the CURRENT connectivity:

        * ``True``  — a live connection already exists; the caller may
          use ``get_root`` immediately.
        * ``False`` — not connected (yet). A single-flight BACKGROUND
          connect has been (re)triggered (subject to backoff). The
          caller MUST NOT wait for it — return ``503 connecting`` (or a
          cached last-known value) so the UI never blocks on a slow or
          deadlocked peer.

        This is the method HTTP handlers call. It awaits ``is_connected``
        (a flag read — no RPC, no round-trip) and otherwise returns
        immediately, honoring the mandate that NO handler may block on
        connect establishment — not even for the duration of a timeout.

        Implemented as ``async`` so call sites keep the ``await`` shape,
        but it contains no blocking await on the connect itself.
        """
        return asyncio.ensure_future(self._ensure_connecting(agent_name))

    async def _ensure_connecting(self, agent_name: str) -> bool:
        if await self.is_connected(agent_name):
            return True
        # Not connected → kick a single-flight background connect (no await).
        self._start_background_connect(agent_name)
        return False

    def _start_background_connect(self, agent_name: str) -> None:
        """Ensure (at most) one in-flight connect task exists for an
        agent, respecting the reconnect backoff. Pure scheduling — never
        awaits the connect. Runs entirely on the current event loop."""
        # Already connecting? single-flight — do nothing.
        existing = self._connecting.get(agent_name)
        if existing is not None and not existing.done():
            return
        # Backoff gate: don't relaunch before the deadline.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (shouldn't happen in the web server) — skip.
            return
        not_before = self._reconnect_not_before.get(agent_name)
        if not_before is not None and loop.time() < not_before:
            return
        if agent_name not in self._agents:
            logger.error(f"Agent {agent_name} not found in registry")
            return

        task: asyncio.Task[bool] = loop.create_task(
            self._do_connect(agent_name),
            name=f"connect_to_agent:{agent_name}",
        )
        # STRONG ref while in flight; dropped + outcome handled in the
        # done-callback (the event-driven "connect finished" hook). Same
        # anti-GC discipline as fire_and_forget_async._INFLIGHT.
        self._connecting[agent_name] = task
        task.add_done_callback(
            lambda t, n=agent_name: self._on_connect_done(n, t)
        )

    def _on_connect_done(
        self, agent_name: str, task: "asyncio.Task[bool]"
    ) -> None:
        """Done-callback for a background connect. Clears the in-flight
        slot and arms/clears reconnect backoff based on the outcome.
        Event-driven — invoked by the loop when the task completes."""
        # Only clear the slot if it is still THIS task (a newer one may
        # have replaced it after a backoff window).
        if self._connecting.get(agent_name) is task:
            self._connecting.pop(agent_name, None)

        ok = False
        if task.cancelled():
            ok = False
        else:
            exc = task.exception()
            if exc is None:
                ok = bool(task.result())
            else:
                logger.warning(
                    "Background connect to %s failed: %s", agent_name, exc
                )
        if ok:
            self._clear_reconnect_backoff(agent_name)
        else:
            self._arm_reconnect_backoff(agent_name)

    def _arm_reconnect_backoff(self, agent_name: str) -> None:
        """Grow the per-agent backoff and set the next-attempt deadline.
        Caps at ``_RECONNECT_BACKOFF_MAX`` so a permanently dead agent is
        retried at a bounded rate, never in an infinite tight loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        cur = self._reconnect_backoff.get(agent_name)
        nxt = (
            self._RECONNECT_BACKOFF_START
            if cur is None
            else min(cur * 2.0, self._RECONNECT_BACKOFF_MAX)
        )
        self._reconnect_backoff[agent_name] = nxt
        self._reconnect_not_before[agent_name] = loop.time() + nxt

    def _clear_reconnect_backoff(self, agent_name: str) -> None:
        self._reconnect_backoff.pop(agent_name, None)
        self._reconnect_not_before.pop(agent_name, None)

    async def connect_to_agent(self, agent_name: str) -> bool:
        """Establish (or reuse) an RPyC connection to an agent.

        Single-flight gate. This is now a thin coordinator over the
        background ``_do_connect`` worker:

        * If already connected → return ``True`` (fast path, flag read).
        * Otherwise (re)trigger the one background connect task and
          AWAIT IT. Callers that genuinely need the connection outcome
          (lifecycle/monitor paths) use this. HTTP handlers must instead
          use ``ensure_connecting`` (non-blocking).

        Concurrent callers for the SAME agent share the ONE in-flight
        task — this is the single-flight gate the old docstring said a
        future change "MUST add". It eliminates the connect storm and
        half-open socket pile-up against a stuck peer
        (the 2026-06-15 incident).

        This rule does NOT forbid *other* clients (a different
        ``AgentService`` instance, another web server, a ``dwcli``
        process, etc.) from connecting to the same agent — those are
        separate (client, agent) pairs and the agent's ``AsyncioServer``
        accepts them.

        Args:
            agent_name: Name of the agent to connect to.

        Returns:
            True if connection successful, False otherwise.
        """
        if await self.is_connected(agent_name):
            logger.debug(f"Already connected to {agent_name}")
            return True

        if agent_name not in self._agents:
            logger.error(f"Agent {agent_name} not found in registry")
            return False

        # Reuse the in-flight connect if one exists; otherwise start it.
        self._start_background_connect(agent_name)
        task = self._connecting.get(agent_name)
        if task is None:
            # Suppressed by backoff (recent failure). Report current state
            # without blocking on a fresh attempt.
            return await self.is_connected(agent_name)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    async def _do_connect(self, agent_name: str) -> bool:
        """Background connect worker (one per agent via the single-flight
        gate). Bounds the WHOLE connect — TCP + eager RPyC handshake — in
        ``asyncio.wait_for`` so a deadlocked peer cannot park it forever.
        This timeout lives here, in the background task, never in an HTTP
        handler."""
        agent = self._agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found in registry")
            return False

        try:
            logger.info(f"Connecting to agent {agent_name} on port {agent.rpyc_port}")

            # ⚠️ MANDATORY: ALL RPyC operations MUST be non-blocking.
            # async_connect() is natively non-blocking (uses
            # loop.sock_connect under the hood), auto-enables asyncio
            # serving, and eager-fetches conn.root — so we don't need
            # run_in_executor, a separate enable_asyncio_serving call,
            # or a root access afterwards.
            #
            # async_connect's own ``timeout`` covers only sock_connect;
            # the eager HANDLE_GETROOT handshake is unbounded and would
            # park forever on a deadlocked peer (the 2026-06-15 incident).
            # The outer wait_for bounds the ENTIRE connect.
            conn = await asyncio.wait_for(
                async_connect(
                    "127.0.0.1",
                    agent.rpyc_port,
                    config={"allow_public_attrs": True},
                    timeout=self._CONNECT_TIMEOUT,
                ),
                timeout=self._CONNECT_TIMEOUT,
            )
            root = conn.root

            # Store connection and root
            self._connections[agent_name] = conn
            self._roots[agent_name] = root
            # Pin the process identity this connection is bound to, so a later
            # ungraceful restart (new pid/port) is detected as stale by
            # is_connected. See DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 1).
            try:
                self._record_conn_identity(
                    agent_name, pid=int(agent.pid), rpyc_port=int(agent.rpyc_port)
                )
            except (TypeError, ValueError):
                pass

            # Subscribe to agent messages
            await self._subscribe_to_agent(agent_name, root)

            # Fetch metadata (provider, model) — once at connect time
            await self._fetch_agent_metadata(agent_name, root)

            # Query and update agent status
            await self._update_agent_status(agent_name, root)

            logger.info(f"Successfully connected to agent {agent_name}")
            return True

        except asyncio.TimeoutError:
            logger.warning(
                "Connect to %s timed out after %.1fs (peer unresponsive); "
                "will retry with backoff",
                agent_name,
                self._CONNECT_TIMEOUT,
            )
            self._evict_connection(agent_name)
            return False
        except Exception as e:
            logger.error(f"Failed to connect to agent {agent_name}: {e}")
            # Cleanup partial connection. Use _evict_connection so any
            # armed asyncio reader is unregistered before we drop the
            # last ref (the 2026-05-27 incident).
            self._evict_connection(agent_name)
            return False

    async def disconnect_from_agent(self, agent_name: str) -> None:
        """Disconnect from an agent using fire-and-forget for cleanup.

        Args:
            agent_name: Name of the agent to disconnect from.
        """
        if not await self.is_connected(agent_name):
            return

        from rpyc.utils.helpers import fire_and_forget_async

        try:
            # Unsubscribe via fire-and-forget (don't block on cleanup)
            if agent_name in self._callback_services and agent_name in self._roots:
                root = self._roots[agent_name]
                callback = self._callback_services[agent_name]

                async def on_success(result: Any) -> None:
                    logger.info(f"Unsubscribed from {agent_name}")

                async def on_error(exc: BaseException) -> None:
                    # "Stream has been closed" / EOFError is the
                    # expected outcome after SIGKILL — the agent is
                    # gone, unsubscribe can't be delivered. Log at
                    # DEBUG so those cleanup paths don't look like
                    # real errors in the ops log.
                    msg = str(exc)
                    if isinstance(exc, (EOFError, BrokenPipeError)) or (
                        "stream has been closed" in msg.lower()
                        or "connection" in msg.lower()
                        and "closed" in msg.lower()
                    ):
                        logger.debug(
                            "unsubscribe from %s: peer already gone (%s)",
                            agent_name, exc,
                        )
                    else:
                        logger.warning(
                            "Error unsubscribing from %s: %s", agent_name, exc
                        )
                    # Don't fail disconnect on unsubscribe error

                # Fire and forget with short timeout (cleanup)
                fire_and_forget_async(
                    root.unsubscribe_messages(callback),
                    timeout=2.0,
                    success_callback=on_success,
                    error_callback=on_error,
                    name=f"unsubscribe_{agent_name}",
                )

                # Don't wait - continue with disconnect immediately

            # async close — matches async_connect's non-blocking
            # contract. sync close() here would hit rpyc_async's
            # sync_request guard from the running FastAPI loop.
            conn = self._connections[agent_name]
            await conn.aclose()

            logger.info(f"Disconnected from agent {agent_name}")

        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            # Post-SIGKILL aclose: the socket is already torn down,
            # there is nothing to handshake. Benign.
            logger.debug(
                "disconnect_from_agent(%s): peer already gone (%s)",
                agent_name, e,
            )
        except Exception as e:
            # Peer-gone paths also surface as a generic Exception with
            # "stream has been closed" message from rpyc_async. Treat
            # that as benign — the agent is dead, that is the whole
            # point of the disconnect.
            msg = str(e).lower()
            if "stream has been closed" in msg or "connection closed" in msg:
                logger.debug(
                    "disconnect_from_agent(%s): peer already gone (%s)",
                    agent_name, e,
                )
            else:
                logger.error(f"Error disconnecting from {agent_name}: {e}")
        finally:
            # Cleanup
            if agent_name in self._connections:
                del self._connections[agent_name]
            if agent_name in self._roots:
                del self._roots[agent_name]
            if agent_name in self._callback_services:
                del self._callback_services[agent_name]
            self._conn_identity.pop(agent_name, None)

    async def _subscribe_to_agent(self, agent_name: str, root: Any) -> None:
        """Subscribe to message callbacks from an agent using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        # Create callback service for this agent
        agent_service = self

        class MessageCallbackService(rpyc.Service):  # type: ignore[misc]
            """RPyC service for receiving messages from agent."""

            async def exposed_on_message(self, msg: Any) -> None:
                """Receive message from agent (async exposed for RPyC).

                Delegates to ``AgentService._dispatch_subscriber_payload``
                so the exact same path is covered by unit tests without
                having to reach into RPyC internals.
                """
                await agent_service._dispatch_subscriber_payload(
                    agent_name, msg
                )

        # Create and store callback instance
        callback = MessageCallbackService()
        self._callback_services[agent_name] = callback

        async def on_success(success: bool) -> None:
            """Handle successful subscription."""
            if success:
                logger.info(f"Subscribed to messages from {agent_name}")
                await self._broadcast_message({
                    "type": "subscription_success",
                    "agent_name": agent_name,
                })
            else:
                logger.error(f"Failed to subscribe to messages from {agent_name}")
                await self._broadcast_message({
                    "type": "subscription_failed",
                    "agent_name": agent_name,
                    "error": "Agent returned False",
                })

        async def on_error(exc: BaseException) -> None:
            """Handle subscription errors."""
            logger.error(f"Error subscribing to {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "subscription_error",
                "agent_name": agent_name,
                "error": str(exc),
            })
            # Clean up callback service on error
            if agent_name in self._callback_services:
                del self._callback_services[agent_name]

        # Fire and forget with 10s timeout
        fire_and_forget_async(
            root.subscribe_messages(callback),
            timeout=10.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"subscribe_{agent_name}",
        )

    async def _fetch_agent_metadata(self, agent_name: str, root: Any) -> None:
        """Fetch agent metadata (provider, model) once at connect time.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        async def on_success(response_str: str) -> None:
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse metadata response: {response_str}")
                return

            if response.get("success"):
                metadata = response.get("metadata", {})
                provider = metadata.get("provider")
                model = metadata.get("model")
                raw_status_metadata = metadata.get("status_metadata")

                if agent_name in self._agents:
                    info = self._agents[agent_name]
                    if provider:
                        info.provider = provider
                    if model:
                        info.model = model
                    info.status_metadata = (
                        raw_status_metadata
                        if isinstance(raw_status_metadata, dict)
                        else None
                    )
                    logger.info(f"Fetched metadata for {agent_name}: {provider}/{model}")

                    # Broadcast metadata update so frontend picks it up
                    await self._broadcast_message({
                        "type": "agent_metadata",
                        "agent_name": agent_name,
                        "provider": info.provider,
                        "model": info.model,
                        "status_metadata": info.status_metadata,
                    })

        async def on_error(exc: BaseException) -> None:
            logger.error(f"Error fetching metadata for {agent_name}: {exc}")

        try:
            fire_and_forget_async(
                root.get_metadata(),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"get_metadata_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to call get_metadata on {agent_name}: {e}")

    async def _update_agent_status(self, agent_name: str, root: Any) -> None:
        """Query agent status and update AgentInfo using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        async def on_success(response_str: str) -> None:
            """Handle successful status query.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse status response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            if response.get("success"):
                # Agent's exposed_get_status() returns only status:
                #   {"success": true, "status": "working", "error": null}
                # Provider/model are fetched once via get_metadata() at connect time.
                status_str = response.get("status", "idle")

                if agent_name in self._agents:
                    info = self._agents[agent_name]
                    info.status = self._convert_status(status_str)

                    logger.info(f"Updated status for {agent_name}: {status_str}")

                    # Broadcast status update
                    broadcast_msg: dict[str, Any] = {
                        "type": "agent_status",
                        "agent_name": agent_name,
                        "status": status_str,
                    }
                    # Forward the agent's error message so the UI's
                    # persistent error panel can show it on the polling
                    # path too (not only the subscribe-push path). The
                    # agent's exposed_get_status returns the human-
                    # readable cause under the "error" key when
                    # status == "error" (observed live, 2026-05-20).
                    err = response.get("error")
                    if status_str == "error" and err:
                        broadcast_msg["error_message"] = err
                        # Persist into the AgentInfo mirror so
                        # GET /api/agents (page load) and a UI re-select
                        # carry the error detail — not only the live WS
                        # push. observed live, 2026-05-20.
                        info.error_message = err
                    else:
                        # Recovered (or no error) → clear so a stale
                        # error panel doesn't linger after recovery.
                        info.error_message = None
                    # Include task_manager fields only if present in response.
                    # Also mirror into AgentInfo so ``GET /api/agents`` is
                    # synced and frontend reconnects see the TM row.
                    if "task_manager_active" in response:
                        tm_active = bool(response["task_manager_active"])
                        broadcast_msg["task_manager_active"] = tm_active
                        info.task_manager_active = tm_active
                        info.has_task_manager = True
                    if "task_manager_disabled" in response:
                        tm_disabled = bool(response["task_manager_disabled"])
                        broadcast_msg["task_manager_disabled"] = tm_disabled
                        info.task_manager_disabled = tm_disabled
                        info.has_task_manager = True
                    await self._broadcast_message(broadcast_msg)

        async def on_error(exc: BaseException) -> None:
            """Handle status query errors."""
            logger.error(f"Error updating status for {agent_name}: {exc}")
            # Set defaults on error
            if agent_name in self._agents:
                self._agents[agent_name].provider = "error"
                self._agents[agent_name].model = "unavailable"

        # Fire and forget with 5s timeout (status queries should be fast)
        fire_and_forget_async(
            root.get_status(),
            timeout=5.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"get_status_{agent_name}",
        )

    async def send_message(
        self,
        agent_name: str,
        message: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to an agent using fire-and-forget pattern.

        Returns immediately. Results delivered via WebSocket callbacks.

        Args:
            agent_name: Name of the agent.
            message: Message to send.
            request_id: Optional opaque correlation id returned in callbacks.

        Returns:
            Immediate response indicating message was sent. Actual agent response
            will be delivered via WebSocket with type="agent_response" or "agent_error".
        """
        # Connect if not connected
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {
                "success": False,
                "error": "Agent not connected (connecting…)",
                "request_id": request_id,
            }

        from rpyc.utils.helpers import fire_and_forget_async

        root = self._roots[agent_name]

        # Define async callbacks
        async def on_success(response_str: str) -> None:
            """Handle successful agent response.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse agent response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            # A user-initiated interrupt comes back as a clean
            # ``{"success": False, "interrupted": True}`` result (NOT a
            # remote exception — see exposed_invoke_agent). It is not an
            # error and not a real response: the agent already broadcast
            # ``agent_interrupted``. Do NOT show a "processed
            # successfully" toast and do NOT broadcast an agent_response
            # for it (2026-05-20 interrupt-as-error fix).
            if isinstance(response, dict) and response.get("interrupted"):
                logger.info(f"Agent {agent_name} invocation interrupted by user")
                return

            logger.info(f"Agent {agent_name} responded successfully")
            logger.debug(f"Response type: {type(response)}, value: {response}")

            # Extract text from response (handle different formats)
            response_text = response
            if isinstance(response, dict):
                # Try common keys for response text
                response_text = (
                    response.get("text") or
                    response.get("content") or
                    response.get("response") or
                    str(response)
                )

            # Ensure response_text is JSON serializable
            if not isinstance(response_text, (str, int, float, bool, type(None))):
                response_text = str(response_text)

            logger.debug(f"Extracted response_text: {str(response_text)[:200]}")

            # Broadcast success to WebSocket clients
            await self._broadcast_message({
                "type": "agent_response",
                "agent_name": agent_name,
                "request_id": request_id,
                "success": True,
                "response": response_text,  # Send only serializable text
            })

        async def on_error(exc: BaseException) -> None:
            """Handle RPC errors (timeout, connection, remote exception)."""
            logger.error(f"Agent {agent_name} error: {exc}")
            # Broadcast error to WebSocket clients
            await self._broadcast_message({
                "type": "agent_error",
                "agent_name": agent_name,
                "request_id": request_id,
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })

        # ================================================================
        # DO NOT ADD A TIMEOUT HERE!
        # ================================================================
        # Agent invocation can take minutes or even tens of minutes:
        # - Complex tasks with multiple tool calls
        # - TaskManager evaluation loops (multiple LLM iterations)
        # - Interrupt/approval waiting (blocks until user decides)
        # - Web search and other slow tools
        #
        # A timeout here does NOT cancel the agent — it only makes the
        # web UI show a false error while the agent keeps working.
        # Connection errors (agent crash, network) are already handled
        # by RPyC raising EOFError/ConnectionError → on_error callback.
        # ================================================================
        fire_and_forget_async(
            root.invoke_agent(message, stream=False),
            timeout=None,
            success_callback=on_success,
            error_callback=on_error,
            name=f"invoke_agent_{agent_name}",
        )

        # Return immediately - actual response comes via WebSocket
        return {
            "success": True,
            "message": "Message sent to agent (response will arrive via WebSocket)",
            "agent_name": agent_name,
            "request_id": request_id,
        }

    async def send_approval(
        self,
        agent_name: str,
        approval_id: str,
        decision: str,
    ) -> dict[str, Any]:
        """Send approval/rejection decision for a tool interrupt.

        Args:
            agent_name: Agent name
            approval_id: Unique ID of the approval request
            decision: "approve", "reject", or "auto_approve_all"

        Returns:
            Response dict with success status.
        """
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        root = self._roots[agent_name]

        decision_dict = {"type": decision}
        if decision == "reject":
            decision_dict["message"] = "Rejected by user via web UI"

        # Handle auto_approve_all: approve this one and set flag
        if decision == "auto_approve_all":
            decision_dict = {"type": "auto_approve_all"}
            self._auto_approve_agents.add(agent_name)

        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response: Any) -> None:
                logger.info(f"Approval sent for {agent_name}: {decision}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error sending approval for {agent_name}: {exc}")

            fire_and_forget_async(
                root.respond_to_approval(approval_id, decision_dict),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"approve_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to send approval to {agent_name}: {e}")
            return {"success": False, "error": str(e)}

        return {"success": True, "message": f"Decision '{decision}' sent"}

    async def set_task_manager_enabled(
        self, agent_name: str, enabled: bool
    ) -> dict[str, Any]:
        """Enable or disable TaskManager for an agent."""
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        root = self._roots[agent_name]
        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response_str: str) -> None:
                import json
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
                logger.info(f"TaskManager {'enabled' if enabled else 'disabled'} for {agent_name}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error toggling TaskManager for {agent_name}: {exc}")

            fire_and_forget_async(
                root.set_task_manager_enabled(enabled),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"tm_toggle_{agent_name}",
            )

            # CRITICAL: propagate the new disabled state to ALL caches and
            # clients NOW — do not wait for the next status poll.
            #
            # The agent process flips ``task_manager.disabled`` itself, but
            # it does NOT push a status_change for it. If we only fire the
            # RPyC call and return, the web AgentInfo mirror and every
            # connected browser keep the stale value. Symptom (user report
            # 2026-05-20): uncheck TM, switch agents, switch back → the box
            # is checked again, because the JS cache was never told.
            #
            # ``disabled = not enabled``. We update the in-memory mirror so
            # GET /api/agents and a UI re-select read the truth, and
            # broadcast ``agent_status`` so every client updates live. The
            # agent remains the source of truth — the next poll merely
            # confirms what we already published.
            disabled = not enabled
            info = self._agents.get(agent_name)
            if info is not None:
                info.task_manager_disabled = disabled
                info.has_task_manager = True
            await self._broadcast_message({
                "type": "agent_status",
                "agent_name": agent_name,
                "status": info.status.value if info is not None else "idle",
                "task_manager_disabled": disabled,
            })
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": True, "enabled": enabled}

    async def reset_task_manager(self, agent_name: str) -> dict[str, Any]:
        """Reset TaskManager for an agent."""
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        root = self._roots[agent_name]
        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response_str: str) -> None:
                logger.info(f"TaskManager reset for {agent_name}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error resetting TaskManager for {agent_name}: {exc}")

            fire_and_forget_async(
                root.reset_task_manager(),
                timeout=10.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"tm_reset_{agent_name}",
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": True, "message": "TaskManager reset requested"}

    async def interrupt_agent(self, agent_name: str) -> dict[str, Any]:
        """Interrupt currently running agent execution.

        Cancels the agent's current worker task via RPyC. The agent then
        broadcasts ``agent_interrupted`` to all subscribers, which clients
        receive via WebSocket and use to update UI state.

        Returns immediately (fire-and-forget). The actual interrupt result
        arrives via WebSocket as ``agent_interrupt_result`` (success or error).

        Args:
            agent_name: Name of the agent to interrupt.

        Returns:
            Dict with ``success`` and ``message`` (or ``error``) keys.
        """
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        root = self._roots[agent_name]
        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response_str: str) -> None:
                """Handle successful interrupt RPC.

                Note: the agent itself broadcasts ``agent_interrupted`` to all
                subscribers. We additionally broadcast a service-level
                ``agent_interrupt_result`` so the originating client receives
                an explicit acknowledgment regardless of subscription state.
                """
                import json as _json
                try:
                    parsed = _json.loads(response_str) if isinstance(response_str, str) else {}
                except (ValueError, TypeError):
                    parsed = {"success": True, "message": str(response_str)}
                logger.info(
                    f"Interrupt requested for {agent_name}: {parsed}"
                )
                await self._broadcast_message({
                    "type": "agent_interrupt_result",
                    "agent_name": agent_name,
                    "success": bool(parsed.get("success", True)),
                    "message": parsed.get("message", ""),
                })

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error interrupting agent {agent_name}: {exc}")
                await self._broadcast_message({
                    "type": "agent_interrupt_result",
                    "agent_name": agent_name,
                    "success": False,
                    "error": str(exc),
                })

            fire_and_forget_async(
                root.interrupt_agent(),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"interrupt_agent_{agent_name}",
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "message": "Interrupt requested (result will arrive via WebSocket)",
        }

    async def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        offset: int = 0,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get agent history using fire-and-forget pattern.

        Args:
            agent_name: Agent name
            limit: Max messages
            offset: Skip recent messages
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual history delivered via WebSocket with
            type="history_response" or "history_error".
        """
        # Connect if not connected
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        # Generate request ID if not provided
        req_id = request_id or str(uuid.uuid4())

        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful history retrieval.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse history response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            logger.info(f"Got history for {agent_name}: {len(response.get('history', []))} messages")
            await self._broadcast_message({
                "type": "history_response",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "history": response.get("history", []),
            })

        async def on_error(exc: BaseException) -> None:
            """Handle history retrieval errors."""
            logger.error(f"Error getting history from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "history_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        # Fire and forget with 15s timeout (history can be large)
        fire_and_forget_async(
            root.get_history(limit, offset),
            timeout=15.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"get_history_{agent_name}",
        )

        return {
            "success": True,
            "message": "History request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    async def get_agent_last_response(
        self,
        agent_name: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get last assistant response using fire-and-forget pattern.

        Args:
            agent_name: Agent name
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual data delivered via WebSocket with
            type="last_response_result" or "last_response_error".
        """
        # Connect if not connected
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        req_id = request_id or str(uuid.uuid4())
        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful last response retrieval."""
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse last response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            logger.info(f"Got last response for {agent_name}")
            await self._broadcast_message({
                "type": "last_response_result",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "last_response": response.get("last_response"),
            })

        async def on_error(exc: BaseException) -> None:
            """Handle last response retrieval errors."""
            logger.error(f"Error getting last response from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "last_response_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        try:
            fire_and_forget_async(
                root.get_last_response(),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"get_last_response_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to call get_last_response on {agent_name}: {e}")
            return {"success": False, "error": f"Agent does not support get_last_response: {e}"}

        return {
            "success": True,
            "message": "Last response request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    async def get_agent_detailed_status(
        self,
        agent_name: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed agent status using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual status delivered via WebSocket with
            type="detailed_status_response" or "detailed_status_error".
        """
        # Connect if not connected
        # NON-BLOCKING connect gate: never wait on a slow/stuck peer.
        # ensure_connecting returns the current connectivity and kicks a
        # single-flight background connect if needed. A stuck agent yields
        # an immediate "connecting" result instead of hanging the handler.
        if not await self.ensure_connecting(agent_name):
            return {"success": False, "error": "Agent not connected (connecting…)"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        req_id = request_id or str(uuid.uuid4())
        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful status retrieval.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse detailed status response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            if not response.get("success"):
                await on_error(Exception("Agent returned success=False"))
                return

            session_info = response.get("session_info", {})
            agent = self._agents.get(agent_name)

            if not agent:
                await on_error(Exception("Agent not found in tracking"))
                return

            # Build status info dict
            status_info = {
                "name": agent.name,
                "pid": agent.pid,
                "status": self._convert_status(session_info.get("status", "idle")).value,
                "provider": session_info.get("provider", agent.provider),
                "model": session_info.get("model", agent.model),
                "rpyc_port": agent.rpyc_port,
                "working_dir": agent.working_dir,
                "started_at": agent.started_at.isoformat(),
                "current_thread_id": session_info.get("current_thread_id"),
                "message_count": session_info.get("message_count", 0),
                "last_activity": agent.started_at.isoformat(),
                "task_manager_active": session_info.get("task_manager_active", False),
            }

            logger.info(f"Got detailed status for {agent_name}")
            await self._broadcast_message({
                "type": "detailed_status_response",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "status": status_info,
            })

        async def on_error(exc: BaseException) -> None:
            """Handle detailed-status retrieval errors."""
            logger.error(f"Error getting detailed status from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "detailed_status_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        # Fire-and-forget: actual status arrives via WebSocket in on_success.
        fire_and_forget_async(
            root.get_status(),
            timeout=10.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"detailed_status_{agent_name}",
        )

        return {
            "success": True,
            "message": "Status request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    async def get_agent_dialog_formatted(
        self,
        agent_name: str,
        verbosity: str = "final",
    ) -> str:
        """Get formatted dialog text for an agent at the requested verbosity.

        Returns the entire conversation (no artificial limit). Three sources of
        user-side messages are all rendered as first-class turns
        (User / TaskManager / agent:<name>). See
        ``app.services.dialog_formatter`` for the rendering rules and
        ``docs/DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md`` for the design.

        Args:
            agent_name: Name of the agent.
            verbosity: ``"final"`` (default) | ``"text"`` | ``"full"``.

        Returns:
            Formatted dialog text as plain string in format:
            --- User ---
            <user message>

            --- Assistant ---
            <assistant response>

        Raises:
            ConnectionError: If agent is not connected.
            ValueError: If history retrieval fails.
            asyncio.TimeoutError: If agent doesn't respond within timeout.
        """
        if agent_name not in self._roots:
            raise ConnectionError(f"Agent '{agent_name}' is not connected")

        root = self._roots[agent_name]

        import asyncio
        import json

        from app.services.dialog_formatter import (
            render_dialog,
            render_system_prompt,
            render_tools,
            split_into_turns,
        )

        # Introspection modes (§9 of
        # docs/DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md): the system prompt
        # and tool list do NOT live in the checkpoint and are NOT conversation
        # turns. Bypass the history fetch entirely and call the dedicated RPyC
        # methods, then render via the pure formatters.
        # Marker substring of the AttributeError raised when the running agent
        # process PREDATES this feature (its RPyC service has no
        # ``get_system_prompt`` / ``get_tools``). RPyC surfaces it as an
        # AttributeError on the netref attribute access. We turn it into a
        # clear, plain-text instruction instead of a raw 500 + remote
        # traceback. (Same spirit as the LEGACY_NONE_MARKER path below.)
        def _is_missing_method_error(exc: Exception) -> bool:
            msg = str(exc)
            return (
                "has no attribute 'get_system_prompt'" in msg
                or "has no attribute 'get_tools'" in msg
                or "has no attribute 'exposed_get_system_prompt'" in msg
                or "has no attribute 'exposed_get_tools'" in msg
            )

        _STALE_AGENT_HINT = (
            "This view requires a newer agent runtime than the one currently "
            "running for '{name}'. Restart the agent (🔄 Restart, or "
            "`deep-worker --restart {name}`) to enable it."
        )

        if verbosity == "system_prompt":
            try:
                # ⚠️ NON-BLOCKING RPyC — DO NOT REGRESS. ⚠️
                # MUST be ``await async_(root.M)()``, NEVER ``await root.M()``.
                # ``root`` is an RPyC netref; ``await root.get_system_prompt()``
                # evaluates the call EAGERLY and INLINE on the event loop,
                # dropping into ``AsyncResult.value -> wait() -> while …:
                # conn.serve()`` which READS THE SOCKET on the loop thread and
                # busy-spins at 100% CPU. Against a slow/stuck peer it NEVER
                # returns → the whole web server deadlocks (regression
                # 2026-06-23: a frozen web server whose restart then hung).
                # ``async_(proxy)()`` issues ``asyncreq`` (no inline serve) and
                # returns an AsyncResult whose ``__await__`` yields to the loop
                # — cancel-aware, no arbitrary ``timeout=`` band-aid needed.
                # See module header rule #5, POLICIES.md §1/§3/§7, and
                # tests/integration/test_no_blocking_rpyc_in_handlers.py.
                raw = await rpyc.async_(root.get_system_prompt)()
            except Exception as e:
                if _is_missing_method_error(e):
                    # Stale agent (no RPyC method). The invariant system prompt
                    # is a PURE function of agent config, so compute it
                    # web-side from session_metadata.json — no restart needed.
                    local = self._compute_invariant_system_prompt_locally(agent_name)
                    if local is not None:
                        return local
                    return (
                        "--- System Prompt (unavailable) ---\n"
                        + _STALE_AGENT_HINT.format(name=agent_name)
                    )
                raise
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if not payload.get("success"):
                error_msg = str(payload.get("error", "Unknown error"))
                raise ValueError(f"Agent returned error: {error_msg}")
            return render_system_prompt(
                payload.get("system_prompt", ""),
                provider=payload.get("provider"),
                model=payload.get("model"),
            )

        if verbosity == "tools":
            try:
                # ⚠️ NON-BLOCKING RPyC — DO NOT REGRESS. ⚠️ See the identical
                # banner on the ``get_system_prompt`` call above. NEVER write
                # ``await root.get_tools()`` — it blocks the event loop in
                # AsyncResult.value/serve (100% CPU deadlock vs a slow peer,
                # regression 2026-06-23). The non-blocking await path is
                # ``await async_(root.get_tools)()``. No ``timeout=`` band-aid:
                # the async path yields to the loop and is cancel-aware. See
                # test_no_blocking_rpyc_in_handlers.py.
                raw = await rpyc.async_(root.get_tools)()
            except Exception as e:
                if _is_missing_method_error(e):
                    return (
                        "--- Available Tools (unavailable) ---\n"
                        + _STALE_AGENT_HINT.format(name=agent_name)
                    )
                raise
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if not payload.get("success"):
                error_msg = str(payload.get("error", "Unknown error"))
                raise ValueError(f"Agent returned error: {error_msg}")
            return render_tools(
                payload.get("tools", []),
                provider=payload.get("provider"),
            )

        # Marker for legacy agents that crash on ``limit=None``. Pre-fix
        # ``_handle_get_history`` does ``total - limit - offset``, which
        # raises ``unsupported operand type(s) for -: 'int' and 'NoneType'``.
        # We retry with a huge positive int so Full Dialog stays reliable
        # even if the running agent process predates the no-limit fix.
        LEGACY_NONE_MARKER = "unsupported operand type(s) for -:"
        # 1 billion is "everything" in practice — far above any context
        # window. Still a positive int that legacy agents accept.
        LEGACY_HUGE_LIMIT = 1_000_000_000

        async def _fetch_history_payload(limit_arg: Any) -> dict[str, Any]:
            """Single RPyC round-trip with the given ``limit``.

            Returns the parsed payload dict (success/history/...).
            """
            history_call = root.get_history(limit=limit_arg, offset=0)
            raw = await asyncio.wait_for(history_call, timeout=30.0)
            if isinstance(raw, str):
                return json.loads(raw)
            return raw

        try:
            # Preferred path: ask for the ENTIRE history. No artificial
            # limit — per design, the only ceiling is the agent's own
            # context window. (DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §6)
            payload = await _fetch_history_payload(None)

            if not payload.get('success'):
                error_msg = str(payload.get('error', 'Unknown error'))

                # Legacy-agent compatibility: if the running agent process
                # predates the no-limit fix, it returns the arithmetic
                # error above. Retry once with a huge positive int so the
                # endpoint stays usable WITHOUT requiring a restart of
                # every long-running agent.
                if LEGACY_NONE_MARKER in error_msg:
                    logger.warning(
                        f"Agent '{agent_name}' uses legacy get_history "
                        f"contract; retrying with limit={LEGACY_HUGE_LIMIT}"
                    )
                    payload = await _fetch_history_payload(LEGACY_HUGE_LIMIT)
                    if not payload.get('success'):
                        error_msg2 = str(payload.get('error', 'Unknown error'))
                        logger.error(
                            f"Legacy fallback for '{agent_name}' also failed: {error_msg2}"
                        )
                        raise ValueError(f"Agent returned error: {error_msg2}")
                else:
                    logger.error(f"Failed to get history for '{agent_name}': {error_msg}")
                    raise ValueError(f"Agent returned error: {error_msg}")

            history = payload.get('history', [])
            turns = split_into_turns(history)
            return render_dialog(turns, verbosity)

        except asyncio.TimeoutError:
            logger.error(f"Timeout getting history for '{agent_name}' after 30s")
            raise asyncio.TimeoutError(f"Agent '{agent_name}' did not respond within 30s")

        except ValueError:
            # Already wrapped above — preserve the message as-is.
            raise

        except Exception as e:
            logger.error(f"Error getting formatted history for '{agent_name}': {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve dialog: {e}")

    def _compute_invariant_system_prompt_locally(self, agent_name: str) -> str | None:
        """Compute the invariant system prompt WEB-SIDE for a stale agent.

        The invariant (startup) system prompt is a pure function of agent
        configuration (assistant_id + working_dir), so when the running agent
        process predates the RPyC ``get_system_prompt`` method we can still
        produce the exact same text by calling ``get_system_prompt`` here,
        seeding working_dir / provider / model from ``session_metadata.json``.
        Returns None if config can't be read (then the caller shows the hint).
        """
        import json
        from pathlib import Path

        from app.services.dialog_formatter import render_system_prompt

        try:
            data_dir = Path.home() / Path("~/.deep_worker").name / agent_name
            meta_file = data_dir / "session_metadata.json"
            if not meta_file.exists():
                return None
            data = json.loads(meta_file.read_text())
            cfg = data.get("launch_config")
            if not isinstance(cfg, dict):
                return None
            working_dir = cfg.get("working_dir")
            provider = cfg.get("provider")
            model = cfg.get("model")
            from deep_worker_cli.agent import get_system_prompt
            project_memory_dir = Path(working_dir) if working_dir else None
            system_prompt = get_system_prompt(
                assistant_id=agent_name,
                sandbox_type=None,
                project_memory_dir=project_memory_dir,
            )
            return render_system_prompt(
                system_prompt, provider=provider, model=model
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"web-side system prompt computation failed for "
                f"'{agent_name}': {e}"
            )
            return None

    def register_message_callback(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a callback for agent messages.
        
        Callbacks are called when messages are received from agents.
        Typically used to broadcast to WebSocket clients.
        
        Args:
            callback: Async function that receives message dict.
        """
        self._message_callbacks.append(callback)
        logger.debug(f"Registered message callback: {callback}")

    def unregister_message_callback(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Unregister a message callback.
        
        Args:
            callback: The callback to remove.
        """
        if callback in self._message_callbacks:
            self._message_callbacks.remove(callback)
            logger.debug(f"Unregistered message callback: {callback}")

    async def _dispatch_subscriber_payload(
        self, agent_name: str, raw: Any
    ) -> None:
        """Route a payload received on the subscribe-messages channel.

        This is the single seam that decides what to do with every
        broadcast coming from an agent. Carved out of the inline
        ``MessageCallbackService.exposed_on_message`` closure so unit
        tests can exercise the full routing without touching RPyC
        machinery.

        Recognised payload ``type`` values:

        * ``status_change`` — translate to a ``agent_status``
          broadcast and update ``self._agents[agent_name]`` in
          place. Only fields present in the payload are copied;
          absent TM fields leave the in-memory mirror at its
          existing value (honest-unknown preserved).
        * ``approval_request`` / ``approval_result`` — forwarded as
          dedicated top-level WS events.
        * anything else — wrapped in ``agent_message`` for the
          frontend router.
        """
        plain_msg = _decode_subscriber_payload(raw)
        if not isinstance(plain_msg, dict):
            return

        msg_type = plain_msg.get("type")

        if msg_type == "status_change":
            status_str = plain_msg.get("status", "idle")
            broadcast: dict[str, Any] = {
                "type": "agent_status",
                "agent_name": agent_name,
                "status": status_str,
            }
            raw_status_metadata = plain_msg.get("status_metadata")
            status_metadata = (
                raw_status_metadata if isinstance(raw_status_metadata, dict) else None
            )
            if status_metadata is not None:
                broadcast["status_metadata"] = status_metadata
            # Mirror into AgentInfo cache so REST callers see the
            # new state immediately. ``_convert_status`` maps the
            # agent's enum to our web enum; unknown values fall
            # through to IDLE.
            info = self._agents.get(agent_name)
            if info is not None:
                try:
                    info.status = self._convert_status(status_str)
                except Exception:
                    logger.debug(
                        "could not convert status '%s' for %s",
                        status_str,
                        agent_name,
                        exc_info=True,
                    )
                info.status_metadata = status_metadata

            metadata_broadcast: dict[str, Any] | None = None
            provider = plain_msg.get("provider")
            model = plain_msg.get("model")
            if info is not None and (provider or model):
                if provider:
                    info.provider = str(provider)
                if model:
                    info.model = str(model)
                metadata_broadcast = {
                    "type": "agent_metadata",
                    "agent_name": agent_name,
                    "provider": info.provider,
                    "model": info.model,
                }

            # TaskManager fields: only propagate when the agent
            # actually sent them.
            if "task_manager_active" in plain_msg:
                val = bool(plain_msg["task_manager_active"])
                broadcast["task_manager_active"] = val
                if info is not None:
                    info.task_manager_active = val
                    info.has_task_manager = True
            if "task_manager_disabled" in plain_msg:
                val = bool(plain_msg["task_manager_disabled"])
                broadcast["task_manager_disabled"] = val
                if info is not None:
                    info.task_manager_disabled = val
                    info.has_task_manager = True

            # If the agent told us it has no TM at all (both flags
            # absent) we deliberately leave ``has_task_manager`` as
            # ``None`` / existing — we don't invent a False.

            error_message = plain_msg.get("error_message")
            if error_message is not None:
                broadcast["error_message"] = error_message

            await self._broadcast_message(broadcast)
            if metadata_broadcast is not None:
                await self._broadcast_message(metadata_broadcast)
            return

        if msg_type == "approval_request":
            if agent_name in self._auto_approve_agents:
                approval_id = plain_msg.get("approval_id")
                if approval_id:
                    await self.send_approval(agent_name, approval_id, "approve")
                return
            await self._broadcast_message({
                "type": "approval_request",
                "agent_name": agent_name,
                "approval_id": plain_msg.get("approval_id"),
                "actions": plain_msg.get("actions", []),
            })
            return

        if msg_type == "approval_result":
            await self._broadcast_message({
                "type": "approval_result",
                "agent_name": agent_name,
                "approval_id": plain_msg.get("approval_id"),
                "decision": plain_msg.get("decision"),
            })
            return

        if msg_type == "channel_quarantined":
            # A client RPyC channel to the agent was flooded and put into
            # quarantine. This is a non-blocking WARNING: the agent keeps
            # working and its status is unchanged. Forward as a dedicated
            # top-level event so the frontend can show a warning banner
            # WITHOUT touching the agent's displayed status / error panel.
            # observed live, 2026-05-22.
            await self._broadcast_message({
                "type": "channel_quarantined",
                "agent_name": agent_name,
                "level": "warning",
                "connid": plain_msg.get("connid"),
                "peer": plain_msg.get("peer"),
                "content": plain_msg.get("content"),
            })
            return

        # Default: wrap for the frontend router.
        await self._broadcast_message({
            "type": "agent_message",
            "agent_name": agent_name,
            "message": plain_msg,
        })

    async def _broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all registered callbacks.

        This is called when we receive messages from agents via RPyC callbacks.

        Args:
            message: Message dict to broadcast.
        """
        import json

        # Add uptime data to every message
        from app.utils.uptime import get_uptime_data
        message["uptime"] = get_uptime_data()

        # CRITICAL: Ensure message is JSON serializable before broadcasting
        # This handles RPyC proxies, datetime objects, and other non-serializable types
        try:
            # Try to serialize and deserialize to catch any issues early
            json_str = json.dumps(message, default=str)
            serializable_message = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to serialize message for broadcast: {e}")
            logger.error(f"Message keys: {list(message.keys())}")
            # Fallback: convert all values to serializable types
            serializable_message = {}
            for key, value in message.items():
                try:
                    json.dumps({key: value})
                    serializable_message[key] = value
                except (TypeError, ValueError):
                    serializable_message[key] = str(value)
                    logger.warning(f"Converted non-serializable field '{key}' to string")

        logger.debug(f"Broadcasting message to {len(self._message_callbacks)} callbacks")

        # Call all callbacks concurrently with serializable message.
        # ``return_exceptions=True`` prevents a single broken
        # subscriber from killing the others — but historically the
        # results were discarded, which meant a broken subscriber
        # stayed invisible until a user noticed missing events.
        # Log every exception so silent drops leave a trail.
        if self._message_callbacks:
            results = await asyncio.gather(
                *[callback(serializable_message) for callback in self._message_callbacks],
                return_exceptions=True,
            )
            for callback, result in zip(self._message_callbacks, results):
                if isinstance(result, BaseException):
                    cb_name = getattr(callback, "__qualname__", repr(callback))
                    logger.error(
                        "message callback %s raised during broadcast "
                        "(type=%s): %s",
                        cb_name,
                        type(result).__name__,
                        result,
                        exc_info=(type(result), result, result.__traceback__),
                    )

    def _convert_status(self, status_str: str) -> AgentStatus:
        """Convert string status to AgentStatus enum.

        Args:
            status_str: Status string from agent.

        Returns:
            AgentStatus enum value.
        """
        # Map agent SessionStatus values to web AgentStatus
        status_lower = status_str.lower()
        if status_lower == "ready":
            return AgentStatus.IDLE
        if status_lower == "working":
            return AgentStatus.RUNNING

        # Try direct mapping
        try:
            return AgentStatus(status_lower)
        except ValueError:
            # Unknown status - default to IDLE
            logger.warning(f"Unknown agent status: {status_str}, defaulting to IDLE")
            return AgentStatus.IDLE

    async def _reconcile_from_registry(self) -> None:
        """Diff in-memory agent map against the live ProcessRegistry.

        Emits ``agent_discovered`` / ``agent_terminated`` broadcasts for
        anything that changed. Does NOT poll ``get_status`` — that would
        bring back the old 5-second polling loop. Status changes come
        from the subscribe-messages channel, which is already wired by
        ``_subscribe_to_agent``.
        """
        loop = asyncio.get_event_loop()
        current_agents = await loop.run_in_executor(
            None, self._registry.list_all
        )
        current_names = {agent.name for agent in current_agents}
        known_names = set(self._agents.keys())

        # New agents
        for name in current_names - known_names:
            logger.info(f"Discovered new agent: {name}")
            proc_info = next(a for a in current_agents if a.name == name)
            agent_info = AgentInfo(
                name=proc_info.name,
                pid=proc_info.pid,
                status=AgentStatus.IDLE,
                provider="unknown",
                model="unknown",
                rpyc_port=proc_info.rpyc_port,
                working_dir=proc_info.working_dir,
                started_at=proc_info.started_at,
            )
            self._agents[name] = agent_info

            await self._broadcast_message({
                "type": "agent_discovered",
                "agent": agent_info.model_dump(mode="json"),
            })

            # Single-flight background connect: pure scheduling, returns
            # instantly so the registry reader keeps the inotify fd
            # drained. At most one in-flight connect per agent; failures
            # arm a capped backoff (no per-event connect storm against a
            # stuck peer — the 2026-06-15 incident). The connect task's own
            # done-callback handles outcome/logging.
            self._start_background_connect(name)

        # Terminated agents
        for name in known_names - current_names:
            logger.info(f"Agent terminated: {name}")
            if await self.is_connected(name):
                await self.disconnect_from_agent(name)
            self._agents.pop(name, None)
            await self._broadcast_message({
                "type": "agent_terminated",
                "agent_name": name,
            })

        # Already-known agents: refresh the in-memory pid/rpyc_port from the
        # registry FIRST, then reconnect only when we've lost the connection.
        # The refresh is what lets identity-aware ``is_connected`` notice an
        # ungraceful restart (same name, new pid/port): without it the stale
        # ``_agents[name]`` still matches the recorded connection identity and
        # the dead channel would never be evicted. NO periodic get_status poll
        # — status updates flow through the subscribe channel.
        # See DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 1).
        proc_by_name = {a.name: a for a in current_agents}
        for name in known_names & current_names:
            proc_info = proc_by_name.get(name)
            info = self._agents.get(name)
            if proc_info is not None and info is not None:
                info.pid = proc_info.pid
                info.rpyc_port = proc_info.rpyc_port
            if not await self.is_connected(name):
                logger.debug(f"Attempting to reconnect to {name}")
                # Single-flight + backoff-aware (see above). A stuck agent
                # is NOT reconnected on every inotify event — the backoff
                # deadline gates relaunches (the 2026-06-15 incident).
                self._start_background_connect(name)

    async def monitor_agents(self) -> None:
        """Event-driven agent-registry monitor.

        Watches the registry file directly (``IN_CLOSE_WRITE | IN_DELETE_SELF``)
        and its parent directory for atomic renames (``IN_MOVED_TO`` only).
        On every relevant event we re-read the file and reconcile in
        ``_reconcile_from_registry``.

        No ``asyncio.sleep``-polling, no periodic status RPC.
        No wakeup on unrelated filesystem activity — the directory watch
        is restricted to ``IN_MOVED_TO`` (rename) only.

        If inotify is unavailable (non-Linux), we fall back to a single
        long-period safety tick (``loop.call_later(60s)``) that just
        triggers a reconcile — still not a tight poll.

        Runs until ``self._shutdown_event`` is set.
        """
        logger.info("Starting agent monitoring task (inotify-driven)")

        # Initial reconcile so agents discovered before the watcher was
        # armed show up immediately.
        try:
            await self._reconcile_from_registry()
        except Exception as e:
            logger.error(f"Initial registry reconcile failed: {e}", exc_info=True)

        loop = asyncio.get_running_loop()
        inotify_fd: int | None = None
        watch_dir: Path | None = None
        next_safety_handle: asyncio.TimerHandle | None = None

        try:
            try:
                import ctypes
                import ctypes.util

                libc_name = ctypes.util.find_library("c")
                libc = ctypes.CDLL(libc_name or "libc.so.6", use_errno=True)
                IN_CLOSE_WRITE = 0x00000008
                IN_MOVED_TO = 0x00000080
                IN_DELETE_SELF = 0x00000400
                O_NONBLOCK = os.O_NONBLOCK

                inotify_init1 = libc.inotify_init1
                inotify_init1.argtypes = [ctypes.c_int]
                inotify_init1.restype = ctypes.c_int
                inotify_add_watch = libc.inotify_add_watch
                inotify_add_watch.argtypes = [
                    ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32
                ]
                inotify_add_watch.restype = ctypes.c_int

                registry_file = getattr(
                    self._registry, "registry_file", Path("/tmp/deep_worker_registry.json")
                )
                watch_dir = Path(registry_file).parent
                watch_dir.mkdir(parents=True, exist_ok=True)
                registry_basename = Path(registry_file).name.encode()
                registry_tmp_basename = registry_basename + b".tmp"

                fd = inotify_init1(O_NONBLOCK)
                if fd < 0:
                    raise OSError(ctypes.get_errno(), "inotify_init1 failed")

                # Watch the directory for atomic rename (write .tmp → rename .json).
                # Only IN_MOVED_TO — no IN_CREATE / IN_CLOSE_WRITE / IN_MODIFY / IN_DELETE,
                # so we do NOT wake on every temp-file write in /tmp.
                dir_wd = inotify_add_watch(fd, str(watch_dir).encode(), IN_MOVED_TO)
                if dir_wd < 0:
                    os.close(fd)
                    raise OSError(ctypes.get_errno(), "inotify_add_watch(/tmp) failed")

                # Watch the registry file directly for close-write (non-atomic write)
                # and delete-self (inode replaced by atomic rename, triggers re-arm).
                file_wd = inotify_add_watch(
                    fd,
                    str(registry_file).encode(),
                    IN_CLOSE_WRITE | IN_DELETE_SELF,
                )
                # ENOENT means the file doesn't exist yet — that's OK,
                # the dir watch will catch the first atomic rename.
                _ENOENT = 2  # os.errno.ENOENT
                if file_wd < 0 and ctypes.get_errno() != _ENOENT:
                    os.close(fd)
                    raise OSError(ctypes.get_errno(), "inotify_add_watch(registry.json) failed")

                inotify_fd = fd

                # Debounced reconcile: one pending reconcile at a time.
                reconcile_pending = False
                reconcile_task: asyncio.Task[None] | None = None

                # inotify_event: struct { int wd; uint32 mask; uint32 cookie;
                #                         uint32 len; char name[]; }
                import struct
                _header = struct.Struct("iIII")  # 16 bytes

                async def _debounced_reconcile() -> None:
                    nonlocal reconcile_pending, reconcile_task
                    await asyncio.sleep(0)
                    reconcile_pending = False
                    reconcile_task = None
                    try:
                        await self._reconcile_from_registry()
                    except Exception as exc:
                        logger.error(
                            "Registry reconcile failed: %s", exc, exc_info=True
                        )

                def _rearm_file_watch() -> None:
                    """Re-arm the inotify watch on the registry file.

                    After an atomic rename (write .tmp → rename .json) the old
                    inode is gone and its watch fires IN_DELETE_SELF +
                    IN_IGNORED.  We need a fresh watch on whatever inode the
                    path points to now.
                    """
                    nonlocal file_wd
                    try:
                        new_wd = inotify_add_watch(
                            inotify_fd,
                            str(registry_file).encode(),
                            IN_CLOSE_WRITE | IN_DELETE_SELF,
                        )
                        if new_wd >= 0:
                            file_wd = new_wd
                    except Exception:
                        pass  # file may not exist yet; the dir watch covers it.

                def _on_inotify_event() -> None:
                    """Kernel says something changed in watch_dir OR on the
                    registry file itself.

                    Two independent watches:
                    * ``dir_wd`` — directory ``/tmp`` with ``IN_MOVED_TO``
                      (atomic rename).
                    * ``file_wd`` — the registry file itself with
                      ``IN_CLOSE_WRITE | IN_DELETE_SELF`` (direct write or
                      inode replaced).

                    When ``file_wd`` fires ``IN_DELETE_SELF`` (inode unlinked
                    by rename) we re-arm on the new inode.

                    CRITICAL: this is a ``loop.add_reader`` callback — if
                    it raises, asyncio removes the reader and the whole
                    monitoring pipeline silently dies. We wrap EVERY
                    failure mode in a catch-all so a short read, a
                    ``struct.error`` on a truncated event, or any other
                    unexpected exception logs-and-survives instead of
                    killing agent discovery."""
                    nonlocal reconcile_pending, reconcile_task
                    try:
                        matched = False
                        rearm = False
                        try:
                            while True:
                                try:
                                    buf = os.read(inotify_fd, 4096)  # type: ignore[arg-type]
                                except BlockingIOError:
                                    break
                                # Parse every event in the buffer.
                                offset = 0
                                while offset + _header.size <= len(buf):
                                    _wd, _mask, _cookie, name_len = _header.unpack_from(
                                        buf, offset
                                    )
                                    name = buf[
                                        offset + _header.size : offset + _header.size + name_len
                                    ].rstrip(b"\x00")
                                    offset += _header.size + name_len

                                    # ── File watch (close-write / delete-self) ──
                                    if _wd == file_wd:
                                        if _mask & IN_CLOSE_WRITE:
                                            matched = True
                                        if _mask & IN_DELETE_SELF:
                                            # Inode replaced by atomic rename —
                                            # re-arm on the new inode.
                                            rearm = True
                                            matched = True
                                        # File watches have no name field (name is empty).
                                        continue

                                    # ── Directory watch (IN_MOVED_TO) ──
                                    if name == registry_basename or name == registry_tmp_basename:
                                        matched = True
                        except OSError:
                            # Expected on a torn-down fd; skip this tick.
                            return
                        if not matched:
                            return
                        # Re-arm file watch if the old inode was replaced by rename.
                        if rearm:
                            _rearm_file_watch()
                        # Debounce — one task per burst.
                        if reconcile_pending:
                            return
                        reconcile_pending = True
                        reconcile_task = supervise(
                            _debounced_reconcile(),
                            name="debounced_reconcile",
                        )
                    except Exception:  # noqa: BLE001
                        # Never let a reader callback propagate — that
                        # unregisters the fd and freezes monitoring.
                        logger.exception(
                            "inotify reader callback crashed — the "
                            "reader survives, but the triggering event "
                            "(if any) was dropped"
                        )

                # fd is guaranteed non-negative here (we raised on < 0 above).
                assert inotify_fd is not None
                loop.add_reader(inotify_fd, _on_inotify_event)
                logger.info(
                    f"Inotify watcher installed: dir={watch_dir} (IN_MOVED_TO), "
                    f"file={registry_basename.decode()} (IN_CLOSE_WRITE|IN_DELETE_SELF), "
                    f"fd={inotify_fd}"
                )
            except (OSError, AttributeError) as exc:
                logger.warning(
                    f"Inotify unavailable ({exc}); falling back to a slow 60s safety tick"
                )

                def _safety_tick() -> None:
                    nonlocal next_safety_handle
                    if self._shutdown_event.is_set():
                        return
                    supervise(
                        self._reconcile_from_registry(),
                        name="safety_tick_reconcile",
                    )
                    next_safety_handle = loop.call_later(60.0, _safety_tick)

                next_safety_handle = loop.call_later(60.0, _safety_tick)

            # Park the task until shutdown is requested. No polling, no work.
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("Agent monitoring task cancelled")
        finally:
            if next_safety_handle is not None:
                next_safety_handle.cancel()
            if inotify_fd is not None:
                try:
                    loop.remove_reader(inotify_fd)
                except Exception:
                    pass
                try:
                    os.close(inotify_fd)
                except OSError:
                    pass
            logger.info("Agent monitoring task stopped")

    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources.
        
        This:
        - Stops monitoring task
        - Closes all RPyC connections
        - Clears callbacks and agent tracking
        """
        logger.info("Shutting down AgentService")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Disconnect from all agents
        for agent_name in list(self._connections.keys()):
            await self.disconnect_from_agent(agent_name)

        # Clear state
        self._agents.clear()
        self._message_callbacks.clear()

        logger.info("AgentService shutdown complete")
