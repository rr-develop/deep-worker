"""Agent process implementation for deep_worker."""

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

# ============================================================================
# CRITICAL: RPYC_ASYNC MUST BE IMPORTED FROM SOURCE, NOT FROM PIP!
# ============================================================================
# Add rpyc_async to sys.path to use AsyncioServer from the rpyc_async source directory
# This enables true bidirectional async communication (156x faster!)
# DO NOT use 'pip install rpyc' - it will use classic RPyC 5.0 without AsyncioServer
# ============================================================================
_rpyc_async_path = Path("/path/to/rpyc_async")
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

from rpyc.core.protocol import DEFAULT_CONFIG
from rpyc.utils.async_server import AsyncioServer  # AsyncioServer from rpyc_async!

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus


def _resolve_persisted_codex_display_model(agent_name: str) -> str | None:
    try:
        state_path = Path.home() / Path("~/.deep_worker").name / agent_name / "codex_session_state.json"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(state_data, dict):
        return None
    session_state = state_data.get(agent_name)
    if not isinstance(session_state, dict):
        return None
    session_uuid = session_state.get("session_uuid")
    if not isinstance(session_uuid, str) or not session_uuid.strip():
        return None

    raw_codex_home = os.environ.get("CODEX_HOME")
    codex_home = Path(raw_codex_home).expanduser() if raw_codex_home else Path.home() / ".codex"
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return None
    try:
        candidates = sorted(
            sessions_dir.rglob(f"*{session_uuid}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict) or event.get("type") != "turn_context":
                        continue
                    payload = event.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    model = payload.get("model")
                    if not isinstance(model, str) or not model.strip():
                        continue
                    model_name = model.strip()
                    if model_name.lower() == "unknown":
                        return None
                    return model_name
        except OSError:
            continue
    return None


def _find_free_port(start_port: int = 0, max_attempts: int = 100) -> int:
    """Find a free port dynamically.

    ⚠️ CRITICAL: NEVER USE FIXED PORTS IN TESTS! ⚠️
    ============================================
    Using fixed ports (like 18861) causes race conditions and test failures
    when multiple agents/tests run concurrently. The port may be in use by:
    - Another test running in parallel
    - A previous test that didn't clean up
    - System services

    This function uses port 0 by default, which tells the OS to automatically
    select a free ephemeral port. This is the ONLY CORRECT approach for testing.

    DO NOT change start_port to a fixed value unless you have a very good reason
    and understand the consequences!

    Args:
        start_port: Starting port number (default: 0 = OS chooses automatically)
                   ⚠️ MUST be 0 for tests to avoid port conflicts!
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        Free port number assigned by the OS

    Raises:
        RuntimeError: If no free port found (extremely rare with port=0)
    """
    # ⚠️ CRITICAL: Use port 0 for automatic OS port selection
    # This is the ONLY way to avoid port conflicts in tests!
    if start_port == 0:
        # Let the OS choose a free ephemeral port (RECOMMENDED for tests)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            # Get the port that OS assigned
            port = s.getsockname()[1]
            return port

    # Fallback: sequential search (NOT RECOMMENDED for tests)
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port to check if it's free
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                # If binding succeeds, port is free
                return port
        except OSError:
            # Port is in use, try next one
            continue

    raise RuntimeError(
        f"No free ports available in range {start_port}-{start_port + max_attempts - 1}"
    )


def _utc_log_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _message_attr(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _log_error_context(
    *,
    error: str,
    input_message_id: str | None,
    messages: list[Any] | None = None,
) -> None:
    fields = [f"error={error}"]
    if input_message_id:
        fields.append(f"input_message_id={input_message_id}")
    if messages:
        last_message = messages[-1]
        last_message_id = _message_attr(last_message, "id")
        last_message_type = _message_attr(last_message, "type") or _message_attr(last_message, "role")
        if last_message_id:
            fields.append(f"last_message_id={last_message_id}")
        if last_message_type:
            fields.append(f"last_message_type={last_message_type}")
    print(f"[{_utc_log_timestamp()}] [INVOKE_ERROR] {' '.join(fields)}", file=sys.stderr)


class AgentProcess:
    """
    Main class for a deep-worker agent process.

    Responsibilities:
    - Register in ProcessRegistry
    - Create PID file
    - Initialize agent (LangGraph)
    - Setup RPyC server
    - Handle signals
    - Manage lifecycle
    """

    @staticmethod
    def _is_outgoing_inter_agent_echo(agent_name: str, response: Any) -> bool:
        """Return True iff ``response`` is this agent's own outgoing
        inter-agent echo (produced by ``SendMessageCompletionMiddleware``).

        An outgoing echo has the exact shape
        ``"<self.name> -> <receiver> :> ..."`` at the very start of the
        string — it's not a reply to the user but a message addressed
        to a peer agent. TaskManager must skip evaluation of such turns
        because they aren't task responses.

        The check only anchors on the sender side (``"<self.name> -> "``)
        because any receiver name is valid for an outgoing echo — what
        matters for the TaskManager-skip semantics is that THIS agent
        was the sender.

        Args:
            agent_name: This agent's own name (typically ``self.name``).
            response: The ``full_response`` captured from the agent's
                stream / ainvoke output. Non-string inputs (``None``,
                tool-block lists, etc.) return False safely.

        Returns:
            True iff the string starts with ``"<agent_name> -> "`` AND
            contains the ``" :> "`` separator (so we don't accidentally
            match free-form text that happens to begin with the name).
        """
        if not isinstance(response, str):
            return False
        sender_prefix = f"{agent_name} -> "
        if not response.startswith(sender_prefix):
            return False
        # After "<sender> -> " there must be a receiver name and the
        # " :> " separator before the message body.
        tail = response[len(sender_prefix):]
        return " :> " in tail

    def _idle_metadata_for_response(self, response: Any) -> dict[str, str] | None:
        if not self._is_outgoing_inter_agent_echo(self.name, response):
            return None
        return {
            "substatus": "waiting_agent",
            "source_tool": "send_message_to_agent",
            "reason": "tool_call",
        }

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        registry: ProcessRegistry | None = None,
        pid_file: Path | None = None,
        data_dir: Path | None = None,
        skip_permissions: bool = False,
        enable_task_manager: bool = False,
        rpyc_port: int | bool | None = True,
        memory_only: bool = False,
        use_sqlite: bool = False,
        launch_config: dict | None = None,
        auto_approve_send_message: bool = False,
    ):
        """Initialize the agent process.

        Args:
            name: Unique name for this agent
            config: Agent configuration
            registry: Optional ProcessRegistry instance
            pid_file: Optional custom PID file path
            data_dir: Optional custom data directory
            skip_permissions: Skip permission checks (auto-approve)
            enable_task_manager: Enable TaskManager for task evaluation
            rpyc_port: RPyC server port configuration:
                - True (default): Auto-assign free port starting from 18861
                - int: Use specific port number
                - False/None: Disable RPyC server (NOT RECOMMENDED)
            memory_only: Use in-memory storage only (no persistence to database)
            use_sqlite: Force SQLite persistence (even in test environment)
            launch_config: Optional launch config dict persisted to
                session_metadata.json for --restart.
            auto_approve_send_message: If True, auto-approve
                `send_message_to_agent` tool calls without prompting,
                while keeping HITL gates for all other side-effect tools.
                Has no effect when ``skip_permissions`` is True
                (everything is already auto-approved).

        Raises:
            ValueError: If both memory_only and use_sqlite are True
        """
        # Validation: both flags cannot be True simultaneously
        if memory_only and use_sqlite:
            raise ValueError("Cannot specify both --memory-only and --use-sqlite")

        self.name = name
        self.config = config
        self.skip_permissions = skip_permissions
        self.auto_approve_send_message = auto_approve_send_message
        self.enable_task_manager = enable_task_manager
        self.memory_only = memory_only
        self.use_sqlite = use_sqlite
        self.launch_config = launch_config
        self._checkpointer_ctx = None  # For lifecycle management

        # Auto-assign RPyC port if enabled
        if rpyc_port is True:
            self.rpyc_port = _find_free_port()
        elif isinstance(rpyc_port, int):
            self.rpyc_port = rpyc_port
        else:
            self.rpyc_port = None

        # Paths
        self.pid_file = pid_file or Path(f"/tmp/deep_worker_{name}.pid")
        self.data_dir = data_dir or (Path("~/.deep_worker").expanduser() / name)
        from deep_worker_cli.storage_paths import (
            detect_storage_version,
            ensure_private_dir,
        )
        self.storage_version = detect_storage_version(name)

        ensure_private_dir(self.data_dir)

        # Components
        self.registry = registry or ProcessRegistry()
        self.agent = None  # Will be initialized in start()
        self.backend = None  # Will be initialized in start()
        self.session_manager: SessionManager | None = None
        self.task_manager = None  # Will be initialized in start() if enabled

        # RPyC components
        self.rpyc_server: AsyncioServer | None = None
        self.rpyc_service_instance: Any = None
        self._agent_event_loop: Any = None  # Event loop for RPyC callbacks

        # State
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.started_at: datetime | None = None  # Set when agent starts

        # Track how many messages we've broadcasted per thread
        # This prevents re-broadcasting old messages from LangGraph checkpointer
        # Similar to sent_count pattern in ClaudeCodeChatModel
        # Key: thread_id, Value: number of messages already broadcasted
        self._last_broadcasted_count: dict[str, int] = {}

        # ═══════════════════════════════════════════════════════════════
        # Message-dict cache — see DESIGN_CONTEXT_DELTA_CACHE.md §4.7.
        # ``_handle_get_history`` serializes each LangChain message into
        # a JSON-ready dict on every call. Messages are immutable once
        # their id is assigned, so we memoize by id and only build dicts
        # for messages we have not seen before. Invalidated by
        # summarization via ``MessageDictCache.prune_to()``.
        # ═══════════════════════════════════════════════════════════════
        from deep_worker_cli.process.message_dict_cache import MessageDictCache
        self._message_dict_cache = MessageDictCache()

        # Agent-side command state
        # These are used by slash commands (/clear, /tokens, etc.)
        self.current_thread_id: str = "default"  # Current conversation thread
        self.last_token_count: int = 0  # Token count from last agent response

        # Agent interruption state
        # These are used by exposed_interrupt_agent to cancel running tasks
        self.is_running: bool = False  # True when agent is executing
        self.current_worker_task: Any = None  # Current asyncio task for cancellation

        # Invoke tasks whose RPyC dispatch task was cancelled (channel drain /
        # inbound-quarantine) while the invoke kept running. Strong refs here
        # keep them alive until they finish; they self-remove via a done
        # callback. See AgentService.exposed_invoke_agent (decoupling).
        self._detached_invoke_tasks: set[Any] = set()

        # Approval decisions cache (for backward compatibility)
        # Used by AgentService to store approval decisions when not using HITL framework
        # Maps approval_id -> decision dict (with "type" field: "approve", "reject", etc.)
        self._approval_decisions: dict[str, dict[str, Any]] = {}

        # Message subscribers (shared across ALL RPyC connections)
        # Each connection (TUI, headless, web UI) gets its own AgentService instance,
        # but they all share this set of subscribers to ensure ALL clients receive
        # messages regardless of which connection triggered the agent invocation.
        # This upholds the "agent is single source of truth" architecture principle.
        self._message_subscribers: set[Any] = set()

    def _current_status_snapshot(self) -> dict[str, Any]:
        """Return the ``status_change`` envelope for the agent's current state.

        Subscribers use it:

        * ``session_manager`` calls it (via the notifier wiring below)
          on every transition, so status changes propagate to UI
          clients immediately without polling;
        * ``exposed_subscribe_messages`` delivers it as the first
          message to every new subscriber, so a UI reloaded at any
          point gets accurate state without waiting for the next
          transition.

        TaskManager fields are included ONLY when TaskManager is
        configured on this process — we never fabricate them.
        """
        status_value = (
            self.session_manager.status.value  # type: ignore[union-attr]
            if self.session_manager is not None
            else "idle"
        )
        error_message = (
            self.session_manager.error_message  # type: ignore[union-attr]
            if self.session_manager is not None
            else None
        )
        payload: dict[str, Any] = {
            "type": "status_change",
            "status": status_value,
            "error_message": error_message,
        }
        status_metadata = (
            getattr(self.session_manager, "status_metadata", None)
            if self.session_manager is not None
            else None
        )
        if status_metadata is not None:
            payload["status_metadata"] = status_metadata
        if self.task_manager is not None:
            payload["task_manager_active"] = (
                self.task_manager.state != "inactive"
            )
            payload["task_manager_disabled"] = bool(self.task_manager.disabled)
        return payload

    async def _dispatch_status_event(self, _payload: dict[str, Any]) -> None:
        """Notifier wired into ``SessionManager``.

        ``session_manager`` hands us its own minimal payload; we ignore
        it and rebuild the full envelope via
        ``_current_status_snapshot()`` so TaskManager fields — which
        live outside SessionManager — are always fresh.

        Each subscriber is pushed to via ``fire_and_forget_async`` with
        a bounded timeout. Three reasons this matters, all from
        production incidents:

        1. *Caller never blocks.* ``SessionManager.set_status`` is on
           the hot path; we must not let one dead WS peer freeze every
           future status change.
        2. *Pinned in ``_INFLIGHT``.* ``fire_and_forget_async`` adds
           the dispatched Task to ``rpyc.utils.helpers._INFLIGHT`` with
           an auto-discard ``add_done_callback``, so a GC pass cannot
           silently drop a half-finished outbound RPC and pin its
           ``AsyncResult`` chain in ``Connection._request_callbacks``
           (incidents 2026-05-08, 2026-05-11).
        3. *Timeout breaks the hang.* A subscriber whose ``on_message``
           never resolves used to accumulate one parked Task per
           ``set_status`` for the rest of the process's life
           (incident observed 2026-05-19). With the timeout, each dispatched
           Task settles within ``DWCLI_STATUS_PUSH_TIMEOUT_SEC``
           (default 10 s) and releases its slot.

        Do NOT wrap with ``rpyc.async_`` — that forces ``HANDLE_CALL``
        on the peer, which calls ``async def exposed_on_message``
        without awaiting, drops the payload and emits the telltale
        "Boxing coroutine object … Did you forget to await?" warning.
        The async-flagged netref already picks ``HANDLE_ASYNC_CALL``.
        """
        import json as _json
        import os as _os

        if not self._message_subscribers:
            return
        payload_str = _json.dumps(self._current_status_snapshot(), default=str)

        try:
            push_timeout = float(_os.environ.get("DWCLI_STATUS_PUSH_TIMEOUT_SEC", "10"))
        except ValueError:
            push_timeout = 10.0

        from rpyc.utils.helpers import fire_and_forget_async  # noqa: PLC0415

        for subscriber in list(self._message_subscribers):
            subscriber_id = id(subscriber)

            async def _on_error(
                exc: BaseException,
                _sid: int = subscriber_id,
                _sub: Any = subscriber,
            ) -> None:
                # Use the captured id() instead of touching the netref;
                # attribute access on a dying peer would re-enter the
                # same broken connection.
                print(
                    f"[AGENT] status push to subscriber id={_sid} failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                # EVICT the dead subscriber. on_disconnect deliberately
                # does NOT clean up — it relies on a graceful
                # exposed_unsubscribe_messages that never arrives when the
                # web client dies/restarts ungracefully. Without eviction
                # the dead MessageCallbackService netref lingers forever
                # and EVERY status push goes only to the corpse (10 s
                # timeout each), so the live UI never updates — the
                # "stuck at idle while actually working" bug
                # (incident observed 2026-05-20). A live web re-subscribes on its next
                # connect_to_agent, so eviction is safe.
                #
                # Evict by IDENTITY: ``set.discard(netref)`` would call
                # ``netref.__hash__`` → a sync RPyC round-trip over the
                # already-dead channel → EOFError, which on the invoke path
                # escaped and forced the agent to ERROR (incident observed 2026-05-22).
                # ``evict_subscriber_by_identity`` never crosses the wire.
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, _sub)

            try:
                awaitable = subscriber.on_message(payload_str)
            except Exception as exc:  # noqa: BLE001
                # Construction-time failure (rare): the netref is so
                # dead it can't even produce an AsyncResult. Log and
                # move on; do not let one broken subscriber abort the
                # whole snapshot fan-out.
                print(
                    f"[AGENT] status push to subscriber id={subscriber_id} "
                    f"could not be dispatched: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                # Same identity-based eviction as the _on_error path (a
                # ``discard`` here would re-raise EOFError via __hash__).
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, subscriber)
                continue

            fire_and_forget_async(
                awaitable,
                timeout=push_timeout,
                error_callback=_on_error,
                name=f"status-push-{subscriber_id}",
            )

    def _on_inbound_quarantine(self, info: dict[str, Any]) -> None:
        """rpyc_async ``on_inbound_quarantine`` hook (sync, runs on the loop).

        A client's RPyC channel was put into inbound quarantine because it
        flooded the agent past ``max_inbound_inflight``. This is purely a
        WARNING: the agent keeps working and its status MUST NOT change. We
        only log it and broadcast a non-status warning event to the live
        subscribers (the quarantined channel itself won't receive it — it's
        the spammer — but other live channels, e.g. the web client, will).

        Quarantine (vs. closing the channel) is intentional: it stops the
        spammer from immediately opening a fresh connection. See
        rpyc_async/docs/DESIGN_INBOUND_BACKPRESSURE.md.
        """
        connid = info.get("connid")
        peer = info.get("peer")
        print(
            f"[RPyC][WARNING] inbound channel quarantined: connid={connid} "
            f"peer={peer} inbound_inflight={info.get('inbound_inflight')} "
            f"threshold={info.get('threshold')}. A client flooded this "
            f"channel; it is now ignored. The agent keeps working — status "
            f"unchanged.",
            file=sys.stderr,
            flush=True,
        )
        # Schedule the async fan-out without blocking the dispatch path.
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._broadcast_channel_quarantine_warning(info))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[RPyC][WARNING] could not schedule quarantine broadcast: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    async def _broadcast_channel_quarantine_warning(self, info: dict[str, Any]) -> None:
        """Fan a ``channel_quarantined`` warning out to all live subscribers.

        CRITICAL: this does NOT touch ``session_manager`` / agent status. It
        is a warning only; the agent continues its task. Reuses the same
        fire-and-forget + dead-subscriber-eviction pattern as
        ``_dispatch_status_event`` so one dead peer can't block the rest.
        """
        import json as _json
        import os as _os

        if not self._message_subscribers:
            return

        payload_str = _json.dumps({
            "type": "channel_quarantined",
            "role": "system",
            "level": "warning",
            "connid": info.get("connid"),
            "peer": str(info.get("peer")),
            "inbound_inflight": info.get("inbound_inflight"),
            "threshold": info.get("threshold"),
            "content": (
                "A client RPyC channel was flooded and put into quarantine. "
                "The agent keeps working; this is a warning, not an error."
            ),
        }, default=str)

        try:
            push_timeout = float(_os.environ.get("DWCLI_STATUS_PUSH_TIMEOUT_SEC", "10"))
        except ValueError:
            push_timeout = 10.0

        from rpyc.utils.helpers import fire_and_forget_async  # noqa: PLC0415

        for subscriber in list(self._message_subscribers):
            subscriber_id = id(subscriber)

            async def _on_error(
                exc: BaseException,
                _sid: int = subscriber_id,
                _sub: Any = subscriber,
            ) -> None:
                print(
                    f"[AGENT] quarantine-warning push to subscriber id={_sid} "
                    f"failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                # Evict by identity — never ``discard`` a netref (its
                # __hash__ crosses the dead wire → EOFError, incident
                # observed 2026-05-22).
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, _sub)

            try:
                awaitable = subscriber.on_message(payload_str)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AGENT] quarantine-warning push to subscriber "
                    f"id={subscriber_id} could not be dispatched: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, subscriber)
                continue

            fire_and_forget_async(
                awaitable,
                timeout=push_timeout,
                error_callback=_on_error,
                name=f"quarantine-warning-{subscriber_id}",
            )

    async def start(self) -> None:
        """Start the agent process.

        This is the main entry point for the agent.
        """
        try:
            # 0. Reset shutdown event and set started_at timestamp
            self.shutdown_event.clear()
            self.started_at = datetime.now()
            # Capture the running loop up front so signal handling (and the
            # fallback path) can safely schedule shutdown onto it. Also set
            # later in the RPyC start path for callbacks; setting it here is
            # harmless and earlier.
            self._agent_event_loop = asyncio.get_running_loop()

            # 1. Create PID file
            await self._create_pid_file()

            # 2. Setup signal handlers
            self._setup_signal_handlers()

            # 2.5 Auto-migrate storage to latest version if needed
            # Must happen before session manager init so metadata reflects v4.
            from deep_worker_cli.storage_paths import CURRENT_STORAGE_VERSION, migrate_agent_storage_to_v4

            if self.storage_version < CURRENT_STORAGE_VERSION:
                migrate_result = migrate_agent_storage_to_v4(self.name)
                self.storage_version = CURRENT_STORAGE_VERSION
                msg = (
                    f"[STORAGE] Auto-migrated agent '{self.name}' "
                    f"from v{migrate_result['migrated_from']} to v{CURRENT_STORAGE_VERSION}\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

            # 3. Initialize session manager (with launch_config for post-mortem inspection)
            # ``notifier=self._dispatch_status_event`` wires every
            # ``set_status`` / ``set_error`` transition into the
            # subscribe-messages broadcast, so UI clients learn of
            # state changes without any polling.
            if isinstance(self.launch_config, dict):
                self.launch_config.setdefault("storage_version", self.storage_version)
            self.session_manager = SessionManager(
                data_dir=self.data_dir,
                launch_config=self.launch_config,
                notifier=self._dispatch_status_event,
            )

            # 4. Initialize agent with create_cli_agent
            try:
                import dotenv
                from pathlib import Path as PathLib

                msg = f"[AGENT_INIT] Initializing agent '{self.name}'...\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)

                # Set up logging to stderr so log_token_usage() output is visible
                _log = logging.getLogger("deep_worker_cli")
                if not _log.handlers:
                    _handler = logging.StreamHandler(sys.stderr)
                    _handler.setFormatter(logging.Formatter("%(message)s"))
                    _log.addHandler(_handler)
                    _log.setLevel(logging.INFO)

                # Load environment variables from the agent's data directory (.env) BEFORE importing config
                # This is critical because config.py initializes Settings at module level
                # and needs OPENAI_MODEL, OPENAI_BASE_URL, etc. to be set before import
                deepagents_env = PathLib.home() / ".deepagents" / ".env"
                if deepagents_env.exists():
                    dotenv.load_dotenv(deepagents_env)
                    msg = f"[AGENT_INIT] Loaded environment from {deepagents_env}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

                # ================================================================
                # CRITICAL: Add deepagents to sys.path BEFORE imports
                # ================================================================
                # deepagents must be imported from source, not from pip/venv —
                # this ensures correct version and avoids pip-vs-local
                # divergence.
                #
                # The path is derived from __file__ so the SAME line works
                # for BOTH the production tree and the developer tree.
                # Hard-coding
                # the prod path here used to defeat the --dev launcher
                # dispatch: even after the bash launcher correctly re-execs
                # into the dev tree, this insertion would prepend the prod
                # ``deepagents`` and shadow the dev one. See
                # docs/DESIGN_DEV_FLAG.md.
                #
                # __file__ here =
                #   deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py
                # parent x4 =
                #   deep_worker
                # ================================================================
                deepagents_path = (
                    PathLib(__file__).resolve().parent.parent.parent.parent
                    / "deepagents"
                )
                if deepagents_path.exists() and str(deepagents_path) not in sys.path:
                    sys.path.insert(0, str(deepagents_path))

                # Now import AFTER environment is loaded and deepagents is in path
                from deep_worker_cli.agent import create_cli_agent
                from deep_worker_cli.config import create_model
                from deep_worker_cli import tools as cli_tools
                from deep_worker_cli.tools import (
                    http_request,
                    fetch_url,
                    make_send_message_to_agent_tool,
                    web_search_delegation_tool,
                    read_image,
                )

                # Create model using configuration from AgentConfig
                # Pass provider and model explicitly from config.
                # Pass user=self.name for KV-cache isolation so the main
                # agent and its TaskManager (:tm) use separate user_ids
                # and do not evict each other's DeepSeek prefix-cache.
                model = create_model(
                    provider=self.config.provider,
                    model_name=self.config.model,
                    user=self.name,
                )

                # Set working_dir on model for MCP direct tool execution
                # (e.g. project_memory needs to know the project root)
                # Uses setattr() because create_model() returns BaseChatModel
                # but only ClaudeCodeChatModel has _working_dir attribute.
                if hasattr(model, '_working_dir'):
                    setattr(model, '_working_dir', str(self.config.working_dir))

                set_resolved_model_callback = getattr(
                    model, "set_resolved_model_callback", None
                )
                if callable(set_resolved_model_callback):
                    async def _on_resolved_model(model_name: str) -> None:
                        set_display_model = getattr(
                            self.config, "set_dynamic_display_model", None
                        )
                        if callable(set_display_model):
                            set_display_model(model_name)
                        if self.session_manager is not None:
                            update_model = getattr(
                                self.session_manager,
                                "update_launch_config_model",
                                None,
                            )
                            if callable(update_model):
                                await update_model(model_name)

                    set_resolved_model_callback(_on_resolved_model)
                    if self.config.provider == "codex" and self.config.model is None:
                        persisted_model = _resolve_persisted_codex_display_model(self.name)
                        if persisted_model:
                            await _on_resolved_model(persisted_model)

                restart_tool_factory = getattr(cli_tools, "make_restart_agent_tool")

                # Prepare web tools
                # - http_request and fetch_url work without API keys
                # - web_search_delegation_tool uses Claude Code SDK for universal web search
                # - send_message_to_agent: inter-agent messaging; sender name
                #   is baked into the factory from self.name so the LLM
                #   cannot forge it. See docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md
                web_tools: list[Any] = [
                    http_request,
                    fetch_url,
                    web_search_delegation_tool,
                    read_image,
                    restart_tool_factory(self.name),
                    make_send_message_to_agent_tool(self.name),
                ]

                # Codex-specific: register the native-tool passthroughs.
                # Codex CLI hard-codes a few native tools (apply_patch,
                # update_plan, view_image, request_user_input, mcp
                # resource introspection) that no --disable flag can
                # remove from its API tools list. When the model calls
                # one of them, ``codex_provider.event_pump`` records the
                # call as an honest LangGraph tool_call referencing the
                # matching BaseTool from ``codex_provider.native_tools``.
                # The body parks on an asyncio.Future and the pump
                # resolves it at item.completed — event-driven, no
                # polling. See docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md.
                if self.config.provider == "codex":
                    from codex_provider.native_tools import NATIVE_TOOLS
                    web_tools.extend(NATIVE_TOOLS)

                # Determine checkpointer based on flags and environment
                # Priority order:
                # 1. --use-sqlite flag: Use SQLite (overrides test detection)
                # 2. --memory-only flag: Use memory (overrides test detection)
                # 3. Test environment: Use memory (default for tests)
                # 4. Normal run: Use SQLite (default for production)
                from deep_worker_cli.utils import is_running_tests
                from deep_worker_cli.sessions import get_checkpointer

                checkpointer = None
                if self.use_sqlite:
                    # Explicit --use-sqlite flag: force SQLite even in tests
                    # Per-agent DB to avoid "database is locked" with concurrent agents
                    self._checkpointer_ctx = get_checkpointer(agent_name=self.name)
                    checkpointer = await self._checkpointer_ctx.__aenter__()
                elif self.memory_only:
                    # Explicit --memory-only flag: force memory even in production
                    checkpointer = None  # Will fall back to InMemorySaver
                elif is_running_tests():
                    # Test environment: default to memory (unless --use-sqlite specified)
                    checkpointer = None  # Will fall back to InMemorySaver
                else:
                    # Normal production run: use per-agent SQLite DB
                    # Each agent gets its own file (sessions_{name}.db) to avoid
                    # "database is locked" errors from concurrent writes.
                    self._checkpointer_ctx = get_checkpointer(agent_name=self.name)
                    checkpointer = await self._checkpointer_ctx.__aenter__()

                # Create the agent with auto-approve for testing

                self.agent, self.backend = create_cli_agent(
                    model=model,
                    assistant_id=self.name,
                    tools=web_tools,
                    auto_approve=self.skip_permissions,
                    auto_approve_send_message=self.auto_approve_send_message,
                    enable_memory=True,
                    enable_skills=True,
                    enable_shell=True,
                    project_memory_dir=Path(self.config.working_dir),
                    checkpointer=checkpointer,
                )

            except Exception as e:
                msg = f"[AGENT_INIT] Error initializing agent: {e}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                # Agent remains None, fallback to echo mode
                self.agent = None

            # Initialize TaskManager if enabled and agent is available
            # This is done OUTSIDE the agent creation try/except so that
            # TaskManager failures don't prevent the main agent from working
            if self.enable_task_manager and self.agent:
                try:
                    from deep_worker_cli.task_manager import TaskManager

                    # Pass the actual checkpointer instance (not context manager)
                    # TaskManager MUST use same checkpointer type as main agent:
                    # - InMemorySaver if main agent uses InMemorySaver (test mode)
                    # - AsyncSqliteSaver if main agent uses AsyncSqliteSaver (production)
                    self.task_manager = TaskManager(
                        agent_name=self.name,
                        config=self.config,
                        session_manager=self.session_manager,
                        checkpointer=checkpointer,  # Same as main agent
                    )
                    await self.task_manager.start()

                    msg = f"[AGENT_INIT] TaskManager started (manager_id: {self.name}:tm)\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

                except Exception as e:
                    msg = f"[AGENT_INIT] Error initializing TaskManager: {e}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    # TaskManager remains None, agent continues without TaskManager
                    self.task_manager = None

            # DEBUG: Log before RPyC server start
            msg = f"[AGENT_INIT] Reached line before RPyC server start...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 5. Start RPyC server if port is configured
            msg = f"[AGENT_INIT] About to start RPyC server (port={self.rpyc_port})...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            if self.rpyc_port is not None:
                await self._start_rpyc_server()  # Now async!
            else:
                raise RuntimeError("RPyC port must be configured - IPC server has been removed")

            msg = f"[AGENT_INIT] RPyC server started successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 6. Register in registry (ONLY after RPyC server is ready!)
            # This ensures clients can connect immediately after seeing the agent in registry
            msg = f"[AGENT_INIT] About to register in registry...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            await self._register()

            msg = f"[AGENT_INIT] Registered successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 7. Enter event loop
            self.running = True
            print(f"Agent '{self.name}' started (PID {os.getpid()})")
            print(f"RPyC port: {self.rpyc_port}")
            print(f"Working directory: {self.config.working_dir}")

            # Main event loop: wait for shutdown signal - EVENT DRIVEN, NO POLLING
            # Wait for shutdown event instead of polling self.running
            await self.shutdown_event.wait()
            
            print(f"Agent '{self.name}' event loop exiting...")

            # Graceful shutdown after event loop exits
            await self.shutdown()

        except Exception as e:
            print(f"Failed to start agent: {e}")
            await self.shutdown()
            raise

    async def cleanup(self) -> None:
        """Clean up resources, including checkpointer.

        This method should be called on agent shutdown to properly close
        database connections and release resources.
        """
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error closing checkpointer: {e}", file=sys.stderr)
            finally:
                self._checkpointer_ctx = None

    async def shutdown(self) -> None:
        """Graceful shutdown of the agent."""
        if not self.running:
            return

        print(f"Shutting down agent '{self.name}'...")
        self.running = False
        self.shutdown_event.set()

        # 1. Stop TaskManager if enabled
        if self.task_manager:
            await self.task_manager.stop()

        # 2. Stop AsyncioServer
        if self.rpyc_server:
            print(f"[RPyC] Stopping AsyncioServer...")
            try:
                await self.rpyc_server.close()  # AsyncioServer.close() is async
            except Exception as e:
                print(f"[RPyC] Error stopping AsyncioServer: {e}")

        # 3. Persist state
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.SHUTDOWN)

        # 4. Cleanup checkpointer
        await self.cleanup()

        # 5. Unregister
        await self._unregister()

        # 6. Cleanup files
        await self._remove_pid_file()

        print(f"Agent '{self.name}' stopped")

    async def _register(self) -> None:
        """Register agent in ProcessRegistry."""
        if self.started_at is None:
            self.started_at = datetime.now()

        # RPyC port is required for communication
        if self.rpyc_port is None:
            raise RuntimeError("RPyC port must be configured - RPyC is required for agent communication")

        process_info = AgentProcessInfo(
            name=self.name,
            pid=os.getpid(),
            pid_file=self.pid_file,
            working_dir=str(self.config.working_dir),
            started_at=self.started_at,
            rpyc_port=self.rpyc_port,  # RPyC port (REQUIRED for communication)
        )
        self.registry.register(process_info)

    async def _unregister(self) -> None:
        """Unregister agent from ProcessRegistry."""
        self.registry.unregister(self.name)

    async def _create_pid_file(self) -> None:
        """Create PID file."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    async def _remove_pid_file(self) -> None:
        """Remove PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown.

        Uses ``loop.add_signal_handler`` so the callback runs INSIDE the event
        loop and can safely set ``shutdown_event``. A plain ``signal.signal``
        handler runs between bytecodes on the main thread and must not touch
        asyncio primitives — and historically it only flipped ``self.running``,
        which the main loop (blocked on ``await self.shutdown_event.wait()``)
        never observed, so SIGTERM printed "shutting down" but the process hung
        forever still serving RPyC (stop_agent then timed out / kill -9 needed;
        2026-05-25). Setting the event is what actually unblocks shutdown.
        """
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._handle_shutdown_signal, sig)
        except (NotImplementedError, RuntimeError):
            # No running loop or platform without add_signal_handler: fall back
            # to signal.signal. _signal_handler then schedules the event set
            # onto the stored loop (it cannot set it directly from here).
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

    def _handle_shutdown_signal(self, signum: int) -> None:
        """Loop-native shutdown callback: unblock the main loop's wait().

        Only sets the event — it does NOT flip ``self.running``. ``shutdown()``
        guards on ``if not self.running: return`` for idempotency, so clearing
        ``running`` here would make the awaited ``shutdown()`` skip all cleanup
        (RPyC close, unregister) and the process would still not exit. Let
        ``shutdown()`` own the ``running`` transition.
        """
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown_event.set()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Fallback signal handler (no running loop at setup time).

        Runs on the main thread between bytecodes, so it must NOT call
        ``shutdown_event.set()`` directly; instead it hops onto the agent's
        event loop, which sets the event in a loop-safe way. As with
        ``_handle_shutdown_signal``, it does not flip ``self.running`` — see
        that method's note.
        """
        print(f"\nReceived signal {signum}, shutting down...")
        loop = self._agent_event_loop
        if loop is not None:
            loop.call_soon_threadsafe(self.shutdown_event.set)

    async def _start_rpyc_server(self) -> None:
        """Start RPyC AsyncioServer (native asyncio, no threads!).

        MIGRATED FROM ThreadedServer TO AsyncioServer for:
        - True bidirectional async support (no deadlocks!)
        - 156x better performance (event-driven vs polling)
        - Single event loop (no thread synchronization needed)
        - Native async/await integration

        SECURITY REQUIREMENTS:
        ====================
        1. Server MUST listen ONLY on localhost (127.0.0.1), NEVER on 0.0.0.0
        2. No external network access is allowed - local trusted environment only
        3. Authentication is NOT implemented - relies on localhost isolation
        4. allow_pickle is enabled - safe ONLY because of localhost-only binding
        5. This is NOT suitable for untrusted networks or remote access

        The security model assumes:
        - All connections come from the local machine
        - The local environment is trusted
        - No malicious processes can connect to localhost
        """
        if self.rpyc_port is None:
            # ``rpyc_port`` is explicitly opted-out (None). ``0`` is a
            # valid value meaning "let the OS pick a free port" — do
            # NOT treat it as falsy here. The actual assigned port is
            # read back below from ``self.rpyc_server.listener``.
            return

        from deep_worker_cli.rpyc_service.agent_service import AgentService

        # SECURITY VALIDATION: Ensure we're binding to localhost only
        SAFE_HOSTNAME = "127.0.0.1"

        print(f"[RPyC] Starting AsyncioServer on {SAFE_HOSTNAME}:{self.rpyc_port}...")

        # Create service class factory that captures agent_process reference
        agent_process = self
        # Store the event loop for RPyC callbacks (we're already in the loop!)
        self._agent_event_loop = asyncio.get_running_loop()

        class BoundAgentService(AgentService):
            """AgentService bound to specific AgentProcess instance."""

            def on_connect(self, conn: Any) -> None:
                """Called when client connects."""
                super().on_connect(conn)
                # Set agent_process and event_loop references for this connection
                self.set_agent_process(agent_process)
                self.set_event_loop(agent_process._agent_event_loop)

        # Extend safe_attrs with standard Python magic methods for transparent proxies
        # These methods are used by built-in Python operations (len(), reversed(), bytes(), etc.)
        # and must work transparently through RPyC NetRef proxies for dwcli interactive mode
        extended_safe_attrs = set(DEFAULT_CONFIG["safe_attrs"])
        extended_safe_attrs.update([
            # Iteration & sequence operations
            "__reversed__",      # reversed(obj) - used in interactive.py:_load_more_history()
            "__missing__",       # dict[key] when key not found (defaultdict)

            # Type conversion & representation
            "__bytes__",         # bytes(obj)
            "__fspath__",        # os.fspath(obj) for path-like objects
            "__complex__",       # complex(obj)

            # Introspection (safe, read-only)
            "__dir__",           # dir(obj)
            "__sizeof__",        # sys.getsizeof(obj)

            # Context managers (already have __enter__/__exit__)
            "__aenter__",        # async with obj
            "__aexit__",         # async with obj

            # Async iteration
            "__aiter__",         # async for
            "__anext__",         # async for

            # Descriptors (safe for data access)
            "__get__",           # descriptor protocol
            "__set__",           # descriptor protocol
            "__delete__",        # descriptor protocol

            # Copy operations
            "__copy__",          # copy.copy(obj)
            "__deepcopy__",      # copy.deepcopy(obj)

            # Pickling (already allow_pickle=True, so expose these)
            "__reduce__",        # pickle support
            "__reduce_ex__",     # pickle support
            "__getnewargs__",    # pickle support
            "__getnewargs_ex__", # pickle support
            "__getstate__",      # pickle support
            "__setstate__",      # pickle support
        ])

        # Create AsyncioServer (native asyncio - no threads!)
        # SECURITY: Listen ONLY on localhost (127.0.0.1), NEVER on external interfaces!
        # This prevents remote access and ensures the server is only accessible locally.
        # ============================================================================
        # CRITICAL: NEVER ADD TIMEOUTS TO RPyC SERVER!
        # ============================================================================
        # deep-worker agents work with dwcli for DAYS or WEEKS continuously.
        # RPyC connection lifetime timeouts are ABSOLUTELY FORBIDDEN.
        #
        # FORBIDDEN:
        # - "sync_request_timeout": <any value>  # NEVER ADD THIS
        # - Any other timeout on connection lifetime
        #
        # Agent may be busy for extended periods with legitimate work:
        # - TaskManager evaluation loops (minutes to hours)
        # - Complex multi-step operations
        # - Large file processing
        #
        # If you add a timeout, the connection WILL drop during normal operation.
        # This is a BUG, not a feature!
        # ============================================================================
        self.rpyc_server = AsyncioServer(
            BoundAgentService,
            hostname=SAFE_HOSTNAME,  # CRITICAL: localhost only, no external access
            port=self.rpyc_port,
            protocol_config={
                # SECURITY WARNING: These settings are ONLY safe because we bind to localhost!
                # allow_public_attrs: Allows access to public attributes without authentication
                # allow_pickle: Enables pickle serialization (unsafe if exposed externally)
                # safe_attrs: Extended list of safe magic methods for transparent proxy operations
                # DO NOT use these settings if the server is accessible from untrusted networks!
                "allow_public_attrs": True,
                "allow_pickle": True,
                "safe_attrs": extended_safe_attrs,
                # NO TIMEOUT - connection can live for days/weeks
                # Surface inbound-quarantine as a non-blocking WARNING (log +
                # broadcast to live subscribers). Does NOT change agent status
                # — the agent keeps working. Observed 2026-05-22.
                "on_inbound_quarantine": self._on_inbound_quarantine,
            }
        )

        # Start AsyncioServer (non-blocking, returns immediately)
        await self.rpyc_server.start()

        # Update port if it was 0 (auto-assigned)
        if self.rpyc_port == 0:
            self.rpyc_port = self.rpyc_server.listener.getsockname()[1]
            print(f"[RPyC] Auto-assigned port: {self.rpyc_port}")

        print(f"[RPyC] AsyncioServer started on {SAFE_HOSTNAME}:{self.rpyc_port} (localhost only)")
        print(f"[RPyC] ✅ Native asyncio - no threads, true bidirectional async, 156x faster!")

    def _handle_status(self, command: dict[str, Any]) -> dict[str, Any]:
        """Handle status query command.

        Returns full session information compatible with BaseSession protocol.
        """
        if not self.session_manager:
            return {"success": False, "error": "Session manager not initialized"}

        # Import here to avoid circular dependency
        from deep_worker_cli.basesession.session_adapter import get_session_info

        try:
            session_info = get_session_info(self)
            return {
                "success": True,
                "session_info": session_info.model_dump(mode="json"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_history(
        self, command: dict[str, Any], connection: Any = None
    ) -> dict[str, Any]:
        """Handle get_history command to retrieve conversation history.

        Args:
            command: Command dictionary with optional 'limit' and 'offset' keys
            connection: Legacy parameter (unused, kept for compatibility)

        Returns:
            Response dictionary with history data:
            {
                "success": True,
                "history": [list of message dicts],
                "total": total message count,
                "has_more": bool indicating if more messages exist
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        # ⚠️ NO ARTIFICIAL LIMITS POLICY (per design doc
        # DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §6).
        # The default is "return everything" — the only ceiling is the
        # agent's own context window. ``limit`` may be:
        #   - absent / None / 0 / negative → return all messages
        #   - positive int N → return last N messages (legacy contract)
        raw_limit = command.get("limit", None)
        try:
            int_limit = int(raw_limit) if raw_limit is not None else 0
        except (TypeError, ValueError):
            int_limit = 0
        # 0 or negative ⇒ no cap
        limit: int | None = int_limit if int_limit > 0 else None
        offset = command.get("offset", 0)

        # ⚠️ LOG-STORM GUARD — DO NOT repr the state / agent / config here.
        # get_history is POLLED at high frequency by web/CLI clients. The old
        # code logged the FULL StateSnapshot (every message body, tool output,
        # file dump) and the Pregel ``agent`` repr at INFO on EVERY call:
        #   logger.info(f"... Successfully got state: {state}")
        #   logger.info(f"... Agent is initialized: {self.agent}")
        # That grew the agent log to 5.2 GB at ~46 KB/s and spiked RssAnon per
        # call (observed 2026-05-23; first seen 2026-05-08, see
        # a related internal incident analysis (not included here)). Keep ONLY
        # bounded breadcrumbs (sizes/counts, never content), and at DEBUG so a
        # polled hot path is silent by default. NEVER interpolate ``state``,
        # ``self.agent``, ``config``, or any message into a log record.
        logger.debug("[GET_HISTORY] start (limit=%s, offset=%s)", limit, offset)

        if not self.agent:
            logger.error("[GET_HISTORY] Agent not initialized")
            return {
                "success": False,
                "error": "Agent not initialized",
            }

        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
            from langchain_core.runnables import RunnableConfig

            # Use agent name as thread_id (same as invoke)
            config: RunnableConfig = {
                "configurable": {"thread_id": self.name},
                "metadata": {
                    "assistant_id": self.name,
                    "agent_name": self.name,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            }

            # Get state from agent checkpoint. Do NOT log ``state`` — see the
            # log-storm guard above; only a bounded message COUNT is logged
            # after we extract it.
            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return {
                    "success": True,
                    "history": [],
                    "total": 0,
                    "has_more": False,
                }

            messages = state.values.get("messages", [])
            total_count = len(messages)
            # Bounded breadcrumb: COUNT only, never content. (DEBUG so the
            # polled hot path is silent by default.) See log-storm guard above.
            logger.debug("[GET_HISTORY] got %d message(s)", total_count)

            # Apply offset and limit for pagination.
            # ``limit is None`` ⇒ return everything (no cap).
            end_idx = total_count - offset if offset > 0 else total_count
            if limit is None:
                start_idx = 0
            else:
                start_idx = max(0, end_idx - limit)
            selected_messages = messages[start_idx:end_idx]

            # Convert LangChain messages to JSON-serializable format.
            # Serialization is memoized per-message-id via MessageDictCache —
            # messages are immutable once their id is assigned, so the same
            # call on the same history returns the same dicts without
            # allocation. See DESIGN_CONTEXT_DELTA_CACHE.md §4.7.
            history = [
                self._message_dict_cache.serialize(msg)
                for msg in selected_messages
            ]
            # Prune stale entries left over from summarization: keep only
            # ids that are in the full (not sliced) history. Cheap — just
            # a set-diff + dict pops.
            live_ids: set[str] = set()
            for msg in messages:
                msg_id = getattr(msg, "id", None)
                if isinstance(msg_id, str) and msg_id:
                    live_ids.add(msg_id)
            self._message_dict_cache.prune_to(live_ids)

            has_more = start_idx > 0

            return {
                "success": True,
                "history": history,
                "total": total_count,
                "has_more": has_more,
            }

        except Exception as e:
            import traceback
            print(f"[GET_HISTORY] Exception: {e}", file=sys.stderr)
            print(f"[GET_HISTORY] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            return {
                "success": False,
                "error": f"Failed to retrieve history: {str(e)}",
            }

    async def _handle_get_last_response(self) -> dict[str, Any]:
        """Get the last assistant (AIMessage) response from conversation history.

        Returns only the last AIMessage with non-empty text content,
        skipping tool-only messages. This is much more efficient than
        loading full history when only the latest response is needed.

        Returns:
            Response dictionary:
            {
                "success": True,
                "last_response": {
                    "type": "AIMessage",
                    "id": "...",
                    "content": "...",
                    "tool_calls": [...]  # optional
                }
            }
            Or {"success": True, "last_response": None} if no AIMessage found.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.agent:
            return {"success": False, "error": "Agent not initialized"}

        try:
            from langchain_core.messages import AIMessage
            from langchain_core.runnables import RunnableConfig

            config: RunnableConfig = {
                "configurable": {"thread_id": self.name},
                "metadata": {
                    "assistant_id": self.name,
                    "agent_name": self.name,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            }

            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return {"success": True, "last_response": None}

            messages = state.values.get("messages", [])

            # Walk backwards to find last AIMessage with text content
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue

                # Extract text content
                content = msg.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                else:
                    text = str(content)

                # Skip AIMessages with empty text (tool-call-only messages)
                if not text.strip():
                    continue

                msg_dict: dict[str, Any] = {
                    "type": "AIMessage",
                    "id": getattr(msg, "id", None),
                    "content": text,
                }

                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args", {}),
                        }
                        for tc in tool_calls
                    ]

                return {"success": True, "last_response": msg_dict}

            return {"success": True, "last_response": None}

        except Exception as e:
            logger.error(f"Error getting last response: {e}")
            return {"success": False, "error": f"Failed to get last response: {str(e)}"}

    async def _handle_invoke(
        self,
        command: dict[str, Any],
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
    ) -> dict[str, Any]:
        """Handle agent invocation command.

        Args:
            command: Command dictionary with 'message' and optional 'stream' keys
            connection: IPC connection for streaming responses
            notify_callback: Optional callable(msg: dict) that broadcasts a message
                to all subscribers (e.g., AgentService._notify_subscribers).
                When provided, the TaskManager evaluation loop calls this for each
                intermediate assistant response and each corrective prompt so that
                ALL connected dwcli instances see the full conversation in real time.
                This upholds the "agent is single source of truth" architecture
                principle (see ARCHITECTURE.md).

        Returns:
            Response dictionary with assistant message
        """
        message = command.get("message")
        stream = command.get("stream", True)  # Default to streaming to match deepagents
        input_message_id: str | None = None
        last_error_messages: list[Any] = []

        if not message:
            return {
                "success": False,
                "error": "No message provided",
            }

        # Set status to RUNNING before execution
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.RUNNING)

        # Track execution state for interruption support
        self.is_running = True
        self.current_worker_task = asyncio.current_task()

        # ============================================================
        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
        # ============================================================
        # Notify user message at the start (unless in streaming mode,
        # which handles this separately to ensure proper ordering).
        # This ensures ALL messages flow through the agent and are
        # broadcast to ALL connected clients.
        # ============================================================
        if not stream and notify_callback is not None:
            await notify_callback({
                "type": "user",
                "role": "user",
                "content": message,
            })

        try:
            if self.agent:
                from langchain_core.messages import HumanMessage
                from langchain_core.runnables import RunnableConfig
                from deep_worker_cli.process.status_callback import StatusTrackingCallback

                def _next_user_message(content: str) -> HumanMessage:
                    nonlocal input_message_id
                    input_message_id = f"user-{uuid.uuid4().hex}"
                    return HumanMessage(content=content, id=input_message_id)

                # Create status tracking callback
                status_callback = StatusTrackingCallback(session_manager=self.session_manager)

                # Configure thread_id for checkpointer (use agent name as thread_id)
                config: RunnableConfig = {
                    "configurable": {"thread_id": self.name},
                    "callbacks": [status_callback],
                    "metadata": {
                        "assistant_id": self.name,
                        "agent_name": self.name,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                }

                # Use actual LangGraph agent if available
                if stream:
                    # ============================================================
                    # STREAMING MODE: Full message streaming with all types
                    # ============================================================
                    # Uses LangGraph's stream_mode=["messages", "updates"] and
                    # subgraphs=True to capture ALL message types in real-time:
                    # - Text chunks from assistant
                    # - Tool calls
                    # - Tool results
                    # All messages are broadcast via notify_callback to uphold
                    # the "agent as single source of truth" architecture.
                    #
                    # If TaskManager is enabled, wrap streaming in evaluation loop
                    # to allow corrective prompts.
                    # ============================================================
                    from deep_worker_cli.process.stream_processor import process_message_chunk

                    # Initialize TaskManager if enabled
                    if self.task_manager:
                        await self.task_manager.init_task(
                            initial_request=message,
                            project_path=str(self.config.working_dir),
                            session_file_path=None
                        )

                    # TaskManager evaluation loop or single iteration
                    # TaskManager has NO iteration limit — it runs until done/unable.
                    iteration = 0
                    current_message = message
                    full_response = ""

                    while True:
                        iteration += 1
                        if self.task_manager:
                            print(f"[STREAM] TaskManager iteration {iteration}", file=sys.stderr)

                        # Notify user message at start (agent as source of truth)
                        # For TaskManager: only send on first iteration, subsequent corrective
                        # prompts are already sent at the end of previous iteration (lines 1040-1044)
                        if notify_callback is not None and iteration == 1:
                            await notify_callback({
                                "type": "user",
                                "role": "user",
                                "content": current_message,
                            })

                        # Tool call accumulation buffer for streaming chunks
                        tool_call_buffer: dict[str, Any] = {}
                        full_response = ""

                        # DEBUG: Log to track chunks arriving from LangGraph
                        # ALWAYS enabled for debugging (no env var check)
                        import os
                        import time as time_mod
                        chunk_count = 0
                        debug_enabled = True  # ALWAYS log for debugging

                        if debug_enabled:
                            with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                f.write(f"\n{'='*80}\n")
                                f.write(f"[{time_mod.time():.3f}] STARTING astream loop\n")
                                if self.task_manager:
                                    f.write(f"  TaskManager iteration: {iteration}\n")
                                f.write(f"  stream_mode=['messages', 'updates']\n")
                                f.write(f"  durability=None (default — checkpoints on every node)\n")
                                f.write(f"  Env DWCLI_DEBUG_ASTREAM={os.environ.get('DWCLI_DEBUG_ASTREAM')}\n")
                                f.write(f"{'='*80}\n")

                        # Stream with full message visibility
                        # ⚠️ DO NOT add durability="exit" here!
                        # With durability="exit", checkpoints are written ONLY when
                        # astream completes. If agent is killed/restarted mid-stream,
                        # all data is lost (stuck in writes table, unrecoverable).
                        # Without it, checkpoints are written after each node —
                        # streaming still works (chunks appear immediately), and
                        # data is safely persisted. With DeltaSqliteCheckpointer,
                        # each checkpoint is a small delta, not a full state copy.
                        async for chunk in self.agent.astream(
                            {"messages": [_next_user_message(current_message)]},
                            stream_mode=["messages", "updates"],
                            subgraphs=True,
                            config=config,
                        ):
                            chunk_count += 1

                            # DEBUG: Log every chunk received
                            if debug_enabled:
                                with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                    f.write(f"\n[{time_mod.time():.3f}] CHUNK #{chunk_count} received:\n")
                                    f.write(f"  Type: {type(chunk)}\n")
                                    f.write(f"  Is tuple: {isinstance(chunk, tuple)}\n")
                                    if isinstance(chunk, tuple):
                                        f.write(f"  Length: {len(chunk)}\n")
                                        if len(chunk) >= 3:
                                            f.write(f"  namespace: {chunk[0]}\n")
                                            f.write(f"  stream_mode: {chunk[1]}\n")
                                            f.write(f"  data type: {type(chunk[2])}\n")

                            # LangGraph stream format: (namespace, stream_mode, data)
                            if not isinstance(chunk, tuple) or len(chunk) != 3:
                                if debug_enabled:
                                    with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                        f.write(f"  ⚠ SKIPPING: Invalid format\n")
                                continue

                            namespace, current_stream_mode, data = chunk

                            # DEBUG: Log stream mode
                            if debug_enabled:
                                with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                    f.write(f"  Stream mode: {current_stream_mode}\n")

                            # Handle MESSAGES stream - for actual message content
                            if current_stream_mode == "messages":
                                message_obj, metadata = data

                                # DEBUG: Log message details
                                if debug_enabled:
                                    with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                        f.write(f"  MESSAGE chunk:\n")
                                        f.write(f"    Message type: {type(message_obj)}\n")
                                        f.write(f"    Has content: {hasattr(message_obj, 'content')}\n")
                                        if hasattr(message_obj, 'content'):
                                            content = message_obj.content
                                            f.write(f"    Content type: {type(content)}\n")
                                            if isinstance(content, list):
                                                f.write(f"    Content blocks: {len(content)}\n")
                                                for i, block in enumerate(content):
                                                    if isinstance(block, dict):
                                                        # `content` here is AIMessage.content — LangChain
                                                        # types it as `list[str | dict[Any, Any]]`. The
                                                        # unparameterized dict confuses ty's overload
                                                        # resolution for `.get()`; the cast pins it.
                                                        block_typed = cast("dict[str, Any]", block)
                                                        block_type = block_typed.get('type', 'unknown')
                                                        f.write(f"      Block {i}: type={block_type}\n")
                                                        if block_type in {'tool_call_chunk', 'tool_call'}:
                                                            f.write(f"        ✅ TOOL_CALL_CHUNK/CALL DETECTED!\n")
                                                            f.write(f"        id={block.get('id')}\n")
                                                            f.write(f"        name={block.get('name')}\n")
                                        f.write(f"    Calling process_message_chunk...\n")

                                # Process message with stream_processor
                                if notify_callback is not None:
                                    await process_message_chunk(
                                        message=message_obj,
                                        metadata=metadata,
                                        notify_callback=notify_callback,
                                        tool_call_buffer=tool_call_buffer,
                                    )

                                    if debug_enabled:
                                        with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                            f.write(f"    ✓ process_message_chunk completed\n")

                                # Track full response for return value (accumulate text chunks)
                                if hasattr(message_obj, "content"):
                                    content = message_obj.content
                                    if isinstance(content, str):
                                        full_response += content

                            # Handle UPDATES stream - for interrupts and state changes
                            # (Currently we don't process updates, but this is where they would go)

                        # CRITICAL: Flush any remaining buffered text chunks
                        # This ensures all batched text is sent before returning
                        from deep_worker_cli.process.stream_processor import flush_all_text_buffers
                        await flush_all_text_buffers()

                        # If TaskManager enabled, evaluate response and check for corrective prompt
                        if self.task_manager:
                            # NOTE: In streaming mode, assistant response is already broadcast
                            # via stream_processor chunks. No need to send it again here.

                            # Skip TaskManager for outgoing inter-agent echoes:
                            # the turn's "final response" is a message addressed to another
                            # agent (produced by SendMessageCompletionMiddleware), not a
                            # reply to the user's task. Evaluating it would make the manager
                            # LLM complain that the subordinate didn't do the work.
                            # See docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md §5.4.
                            if self._is_outgoing_inter_agent_echo(self.name, full_response):
                                print(
                                    f"[STREAM] Skipping TaskManager: outgoing inter-agent message",
                                    file=sys.stderr,
                                )
                                break

                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=full_response,
                                send_to_subordinate_callback=None
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                print(f"[STREAM] TaskManager evaluation complete (done/unable)", file=sys.stderr)
                                break

                            # Broadcast TaskManager corrective prompt to ALL subscribers
                            # BEFORE sending it back to the agent. This ensures dwcli TUI
                            # shows the "TaskManager:> ..." message in real time.
                            if notify_callback is not None:
                                try:
                                    await notify_callback({
                                        "type": "user",
                                        "content": corrective_prompt,
                                        "role": "user",
                                    })
                                except Exception as e:
                                    # Don't let notification failures stop TaskManager loop
                                    print(f"[STREAM] Warning: Failed to notify subscribers of TaskManager prompt: {e}", file=sys.stderr)

                            # Continue with corrective prompt
                            print(f"[STREAM] TaskManager requesting corrections: {corrective_prompt[:100]}...", file=sys.stderr)
                            current_message = corrective_prompt
                        else:
                            # No TaskManager - exit after first iteration
                            break

                    # Set status back to IDLE after successful completion
                    if self.session_manager:
                        await self.session_manager.set_status(
                            AgentStatus.IDLE,
                            metadata=self._idle_metadata_for_response(full_response),
                        )

                    # Log token usage with cache info from agent state
                    try:
                        from deep_worker_cli.utils import log_token_usage
                        state = await self.agent.aget_state(config)
                        if state and state.values:
                            msgs = state.values.get("messages", [])
                            if msgs:
                                last = msgs[-1]
                                rm = getattr(last, 'response_metadata', None) or {}
                                log_token_usage(logging.getLogger(__name__), self.name, rm, "stream")
                    except Exception:
                        pass

                    # Send final response
                    return {
                        "success": True,
                        "response": {
                            "role": "assistant",
                            "content": full_response,
                        },
                    }
                else:
                    # Non-streaming mode: wait for complete response
                    # If TaskManager enabled, wrap in evaluation loop
                    if self.task_manager:
                        # Initialize task with user's message
                        await self.task_manager.init_task(
                            initial_request=message,
                            project_path=str(self.config.working_dir),
                            session_file_path=None  # Could add session file tracking
                        )

                        # TaskManager evaluation loop
                        # NO iteration limit — TaskManager runs until done/unable.
                        iteration = 0
                        current_message = message

                        while True:
                            iteration += 1
                            print(f"[INVOKE] TaskManager iteration {iteration}", file=sys.stderr)

                            # Invoke agent
                            result = await self.agent.ainvoke(
                                {"messages": [_next_user_message(current_message)]},
                                config
                            )

                            # Extract response
                            messages = result.get("messages", [])
                            last_error_messages = list(messages) if isinstance(messages, list) else []
                            if not messages:
                                break

                            last_msg = messages[-1]
                            content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")

                            # ============================================================
                            # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
                            # ============================================================
                            # Broadcast this intermediate assistant response to ALL
                            # subscribers (including dwcli TUI instances) so the full
                            # conversation is visible in real time.
                            # This is the fix for the TaskManager loop bypassing
                            # _notify_subscribers() for intermediate messages.
                            # ============================================================
                            if notify_callback is not None:
                                await notify_callback(
                                    {"type": "assistant", "content": content, "role": "assistant"}
                                )

                            # Skip TaskManager for outgoing inter-agent echoes:
                            # the turn's "final response" is a message addressed to another
                            # agent (produced by SendMessageCompletionMiddleware), not a
                            # reply to the user's task.
                            # See docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md §5.4.
                            if self._is_outgoing_inter_agent_echo(self.name, content):
                                print(
                                    f"[INVOKE] Skipping TaskManager: outgoing inter-agent message",
                                    file=sys.stderr,
                                )
                                break

                            # Send to TaskManager for evaluation
                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=content,
                                send_to_subordinate_callback=None  # Not used in loop approach
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                print(f"[INVOKE] TaskManager evaluation complete (done/unable)", file=sys.stderr)
                                break

                            # Broadcast TaskManager corrective prompt to ALL subscribers
                            # before sending it back to the agent. This ensures dwcli TUI
                            # shows the "TaskManager:> ..." message in real time.
                            if notify_callback is not None:
                                await notify_callback(
                                    {
                                        "type": "user",
                                        "content": corrective_prompt,
                                        "role": "user",
                                    }
                                )

                            # Continue with corrective prompt
                            print(f"[INVOKE] TaskManager requesting corrections: {corrective_prompt[:100]}...", file=sys.stderr)
                            current_message = corrective_prompt

                        # Return final response
                        messages = result.get("messages", [])
                        last_error_messages = list(messages) if isinstance(messages, list) else []
                        if messages:
                            last_msg = messages[-1]
                            content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
                            role = getattr(last_msg, "type", "assistant")

                            # Notify final assistant response (TaskManager mode)
                            if notify_callback is not None:
                                await notify_callback({
                                    "type": "assistant",
                                    "role": role,
                                    "content": content,
                                })

                            # Log token usage with cache info
                            try:
                                from deep_worker_cli.utils import log_token_usage
                                if messages:
                                    last = messages[-1]
                                    rm = getattr(last, 'response_metadata', None) or {}
                                    print(f"[INVOKE_DEBUG] log_token_usage: rm={rm}", file=sys.stderr)
                                    log_token_usage(logging.getLogger(__name__), self.name, rm, "invoke")
                            except Exception:
                                pass

                            if self.session_manager:
                                await self.session_manager.set_status(
                                    AgentStatus.IDLE,
                                    metadata=self._idle_metadata_for_response(content),
                                )

                            return {
                                "success": True,
                                "response": {
                                    "role": role,
                                    "content": content,
                                },
                            }
                        else:
                            # Terminal failure: settle status so the agent
                            # is not left dangling at RUNNING forever (the
                            # incident observed 2026-05-20, "forever working" bug).
                            if self.session_manager:
                                _log_error_context(
                                    error="No messages returned from agent",
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                )
                                await self.session_manager.set_error(
                                    "No messages returned from agent"
                                )
                            return {
                                "success": False,
                                "error": "No messages returned from agent",
                            }

                    else:
                        # Standard invocation without TaskManager
                        # CRITICAL FIX: In non-streaming mode, we must wait for FINAL response
                        # If agent has interrupts (tool approvals), ainvoke() may return early
                        # with empty content. We need to keep invoking until we get final answer.

                        max_iterations = 20  # Prevent infinite loops
                        iteration = 0
                        final_content = ""

                        # Initial ainvoke with user message
                        result = await self.agent.ainvoke(
                            {"messages": [_next_user_message(message)]},
                            config
                        )
                        print(f"[INVOKE] Initial ainvoke: Got result with {len(result.get('messages', []))} messages", file=sys.stderr)

                        while iteration < max_iterations:
                            iteration += 1

                            messages = result.get("messages", [])
                            last_error_messages = list(messages) if isinstance(messages, list) else []
                            if not messages:
                                print(f"[INVOKE] No messages in result", file=sys.stderr)
                                break

                            # CRITICAL FIX: Notify ONLY NEW messages (not entire history)
                            # This ensures tool_call and tool_result messages are visible to users
                            # WITHOUT re-broadcasting old messages from previous queries
                            # Similar to streaming mode behavior in stream_processor.py
                            # Pattern: analogous to sent_count in ClaudeCodeChatModel
                            if notify_callback is not None:
                                # Extract thread_id from config
                                thread_id = config.get("configurable", {}).get("thread_id", "default")

                                # Determine which messages are NEW (not yet broadcasted)
                                last_broadcasted = self._last_broadcasted_count.get(thread_id, 0)
                                new_messages = messages[last_broadcasted:]

                                print(f"[INVOKE] Iteration {iteration}: Broadcasting {len(new_messages)} new messages (total={len(messages)}, last_broadcasted={last_broadcasted})", file=sys.stderr)

                                # Broadcast ONLY new messages
                                for msg in new_messages:
                                    # Extract message data
                                    if isinstance(msg, dict):
                                        msg_role = msg.get("role", "")
                                        msg_content = msg.get("content", "")
                                        msg_tool_calls = msg.get("tool_calls")
                                    elif hasattr(msg, "type"):
                                        msg_role = msg.type
                                        msg_content = getattr(msg, "content", "")
                                        msg_tool_calls = getattr(msg, "tool_calls", None)
                                    else:
                                        continue

                                    # Send tool_call notifications
                                    if msg_tool_calls:
                                        for tool_call in msg_tool_calls:
                                            if isinstance(tool_call, dict):
                                                await notify_callback({
                                                    "type": "tool_call",
                                                    "name": tool_call.get("name", ""),
                                                    "args": tool_call.get("args", {}),
                                                    "id": tool_call.get("id", ""),
                                                })
                                            else:
                                                # LangChain ToolCall object
                                                await notify_callback({
                                                    "type": "tool_call",
                                                    "name": getattr(tool_call, "name", ""),
                                                    "args": getattr(tool_call, "args", {}),
                                                    "id": getattr(tool_call, "id", ""),
                                                })

                                    # Send tool_result notifications
                                    if msg_role == "tool":
                                        # Extract status from ToolMessage (default: "success")
                                        # LangChain ToolMessage has status attribute ("success" or "error")
                                        if isinstance(msg, dict):
                                            msg_status = msg.get("status", "success")
                                        else:
                                            msg_status = getattr(msg, "status", "success")

                                        await notify_callback({
                                            "type": "tool_result",
                                            "content": msg_content,
                                            "status": msg_status,
                                            "tool_call_id": (
                                                msg.get("tool_call_id", "")
                                                if isinstance(msg, dict)
                                                else getattr(msg, "tool_call_id", "")
                                            ),
                                        })

                                # Update last_broadcasted_count AFTER broadcasting
                                # This ensures we don't re-broadcast these messages in future iterations
                                self._last_broadcasted_count[thread_id] = len(messages)

                            last_msg = messages[-1]

                            # Check if there are interrupts pending
                            has_interrupt = "__interrupt__" in result and result["__interrupt__"]

                            # Extract content from last message
                            if isinstance(last_msg, dict):
                                content = last_msg.get("content", "")
                                role = last_msg.get("role", "")
                            elif hasattr(last_msg, "content"):
                                content = last_msg.content
                                role = getattr(last_msg, "type", "")
                            else:
                                content = ""
                                role = ""

                            # Check if this message has tool calls
                            has_tool_calls = False
                            if isinstance(last_msg, dict):
                                has_tool_calls = bool(last_msg.get("tool_calls"))
                            elif hasattr(last_msg, "tool_calls"):
                                has_tool_calls = bool(last_msg.tool_calls)

                            print(f"[INVOKE] Iteration {iteration}: content='{content[:100] if content else '(empty)'}', role={role}, has_tool_calls={has_tool_calls}, has_interrupt={has_interrupt}", file=sys.stderr)

                            # If we have real content and no pending tool calls AND it's from assistant, this is final response
                            # CRITICAL: Must check role in ('assistant', 'ai') to avoid treating tool results as final response!
                            # LangGraph uses 'ai' for AIMessage.type, while dicts may use 'assistant'
                            if content and not has_tool_calls and role in ('assistant', 'ai'):
                                final_content = content
                                print(f"[INVOKE] Got final response: {content[:100]}...", file=sys.stderr)
                                break

                            # If there's an interrupt, handle it properly:
                            # 1. Get approval decision (via callback or auto-approve)
                            # 2. Resume with Command(resume=...) — do NOT re-send HumanMessage
                            if has_interrupt:
                                interrupts = result["__interrupt__"]
                                print(f"[INVOKE] Interrupt detected: {len(interrupts)} pending approvals", file=sys.stderr)

                                # NOTE: Do NOT broadcast here — exposed_request_approval()
                                # handles broadcasting with its own approval_id.

                                # Get decisions: either from callback or auto-approve
                                if approval_callback is not None:
                                    hitl_response = await approval_callback(interrupts)
                                else:
                                    # Auto-approve all (no callback = headless/web without approval UI)
                                    hitl_response = {}
                                    for intr in interrupts:
                                        actions = intr.value.get("action_requests", [])
                                        hitl_response[intr.id] = {
                                            "decisions": [{"type": "approve"} for _ in actions]
                                        }
                                    print(f"[INVOKE] Auto-approving {len(interrupts)} interrupts (no approval callback)", file=sys.stderr)

                                # Resume graph with decisions (NOT with HumanMessage!)
                                from langgraph.types import Command
                                resume_input = Command(resume=hitl_response)
                                result = await self.agent.ainvoke(resume_input, config)
                                print(f"[INVOKE] Resumed after interrupt, got {len(result.get('messages', []))} messages", file=sys.stderr)

                                # Re-evaluate the new result (loop continues from evaluation)
                                continue

                            # If we have tool calls but no interrupt, they should have been
                            # executed by ainvoke already. Break to avoid re-sending.
                            if has_tool_calls:
                                print(f"[INVOKE] Tool calls detected without interrupt — ainvoke should have handled them", file=sys.stderr)
                                break

                            # If we got here with empty content and no tool calls,
                            # the agent finished but has nothing to say (shouldn't happen normally)
                            if not content:
                                print(f"[INVOKE] Empty response with no tool calls at iteration {iteration}", file=sys.stderr)
                                break

                        if iteration >= max_iterations:
                            print(f"[INVOKE] Max iterations reached without final response", file=sys.stderr)
                            # Settle status — do not leave the agent
                            # dangling at RUNNING (incident observed 2026-05-20).
                            if self.session_manager:
                                _log_error_context(
                                    error="Agent exceeded maximum iterations without producing final response",
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                )
                                await self.session_manager.set_error(
                                    "Agent exceeded maximum iterations "
                                    "without producing final response"
                                )
                            return {
                                "success": False,
                                "error": "Agent exceeded maximum iterations without producing final response"
                            }

                        if not final_content:
                            print(f"[INVOKE] No final content found after {iteration} iterations", file=sys.stderr)
                            # Settle status — do not leave the agent
                            # dangling at RUNNING (incident observed 2026-05-20).
                            # This is the exact path the codex API-400
                            # failure took: ainvoke returned an empty
                            # assistant message and we fell through here.
                            if self.session_manager:
                                _log_error_context(
                                    error="Agent did not produce a response",
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                )
                                await self.session_manager.set_error(
                                    "Agent did not produce a response"
                                )
                            return {
                                "success": False,
                                "error": "Agent did not produce a response"
                            }

                        # ============================================================
                        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
                        # ============================================================
                        # In non-streaming mode WITHOUT TaskManager, we must explicitly
                        # broadcast the final assistant response to ensure ALL connected
                        # clients (TUI, web UI, etc.) see it via real-time callbacks.
                        # This is critical for maintaining consistency across all subscribers.
                        # ============================================================
                        if notify_callback is not None:
                            await notify_callback({
                                "type": "assistant",
                                "role": "assistant",
                                "content": final_content,
                            })

                        # Set status back to IDLE after successful completion
                        if self.session_manager:
                            await self.session_manager.set_status(
                                AgentStatus.IDLE,
                                metadata=self._idle_metadata_for_response(final_content),
                            )

                        # Log token usage with cache info from last invocation
                        try:
                            from deep_worker_cli.utils import log_token_usage
                            msgs = result.get("messages", [])
                            if msgs:
                                last = msgs[-1]
                                rm = getattr(last, 'response_metadata', None) or {}
                                log_token_usage(logging.getLogger(__name__), self.name, rm, "invoke")
                        except Exception:
                            pass

                        return {
                            "success": True,
                            "response": {
                                "role": "assistant",
                                "content": final_content,
                            },
                        }
            else:
                # STRICT FAILURE: Agent MUST be initialized, NO SILENT FALLBACKS
                error_msg = (
                    "CRITICAL: Agent not initialized! Cannot process requests.\n"
                    "This indicates a fatal initialization error that must be fixed.\n"
                    f"Agent state: {self.agent}\n"
                    "Check agent process logs for initialization errors."
                )
                print(f"[INVOKE] {error_msg}", file=sys.stderr)

                if self.session_manager:
                    _log_error_context(
                        error="Agent not initialized",
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                    )
                    await self.session_manager.set_error("Agent not initialized")

                raise RuntimeError(error_msg)

        except asyncio.CancelledError:
            # Task was cancelled (interrupted) — this is a NORMAL state
            # transition initiated by the user, not an error. The agent
            # process is fully alive and ready to accept the next user
            # message, so we land in INTERRUPTED (not ERROR, not
            # WAITING_INPUT).
            #
            # History of this branch:
            # - Originally called set_error("Agent interrupted"); the UI
            #   then refused to talk to the agent until manual restart.
            # - Then switched to WAITING_INPUT — but that name is
            #   overloaded: ``StatusTrackingCallback.on_tool_start`` also
            #   uses WAITING_INPUT for the in-turn tool-running state.
            #   Same status, two meanings — confusing in the UI.
            # - Now INTERRUPTED is a first-class state. UI renders it
            #   in amber (same colour as the Interrupt button) so the
            #   user can see at a glance that the agent is paused
            #   mid-turn waiting for their next message — typically a
            #   correction.
            print("[INVOKE] Agent task cancelled (interrupted)", file=sys.stderr)

            # Notify cancellation via callback (if provided)
            if notify_callback is not None:
                await notify_callback({
                    "type": "agent_interrupted",
                    "role": "system",
                    "content": "Interrupted",
                })

            # Transition to INTERRUPTED — agent is alive, awaiting the
            # user's next message (typically a correction). NOT an error.
            if self.session_manager:
                await self.session_manager.set_status(AgentStatus.INTERRUPTED)

            # Re-raise to properly handle cancellation
            raise

        except Exception as e:
            import traceback
            print(f"[INVOKE] Exception occurred: {e}", file=sys.stderr)
            print(f"[INVOKE] Traceback:\n{traceback.format_exc()}", file=sys.stderr)

            # Usage/session-limit gets a clean, user-facing message instead
            # of the raw "Agent invocation failed: ..." wrapper. The provider
            # raises ClaudeUsageLimitError whose str() is the CLI marker (incl.
            # the reset time). Detected by class name to keep this module
            # provider-agnostic (codex agents never import the claude module).
            # Observed 2026-05-21: the limit used to fail silently to idle.
            if type(e).__name__ == "ClaudeUsageLimitError":
                limit_msg = f"Claude usage limit reached — {str(e)}"
                print(f"[INVOKE] {limit_msg}", file=sys.stderr)
                if notify_callback is not None:
                    try:
                        await notify_callback({
                            "type": "error",
                            "role": "system",
                            "content": limit_msg,
                        })
                    except Exception as notify_err:
                        print(f"[INVOKE] Warning: Failed to notify limit to TUI: {notify_err}", file=sys.stderr)
                if self.session_manager:
                    _log_error_context(
                        error=limit_msg,
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                    )
                    await self.session_manager.set_error(limit_msg)
                return {
                    "success": False,
                    "error": limit_msg,
                }

            # CRITICAL: Notify error to TUI so users see what went wrong
            # This matches deepagents architecture where errors are displayed via ErrorMessage
            error_msg = f"Agent error: {str(e)}"
            if notify_callback is not None:
                try:
                    await notify_callback({
                        "type": "error",
                        "role": "system",
                        "content": error_msg,
                    })
                except Exception as notify_err:
                    print(f"[INVOKE] Warning: Failed to notify error to TUI: {notify_err}", file=sys.stderr)

            # Set error status on exception
            if self.session_manager:
                invocation_error = f"Agent invocation failed: {str(e)}"
                _log_error_context(
                    error=invocation_error,
                    input_message_id=input_message_id,
                    messages=last_error_messages,
                )
                await self.session_manager.set_error(invocation_error)

            return {
                "success": False,
                "error": f"Agent invocation failed: {str(e)}",
            }

        finally:
            # Always reset execution state
            self.is_running = False
            self.current_worker_task = None
