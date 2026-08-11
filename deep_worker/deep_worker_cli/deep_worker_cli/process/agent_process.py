"""Agent process implementation for deep_worker."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, cast

# Timestamp helper for log messages (shared format with the web app)
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

def _ts() -> str:
    return datetime.now(UTC).strftime(_TS_FORMAT)


def _get_git_commit_info(tree_root: str | None) -> tuple[str, str]:
    """Return (commit_sha, describe) for the git tree at *tree_root*.

    Both values default to ``"unknown"`` on any error (no .git, git not
    in PATH, permission denied, etc.) so the caller never crashes.
    """
    if not tree_root:
        return "unknown", "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", tree_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not sha:
            sha = "unknown"
    except Exception:
        sha = "unknown"
    try:
        describe = subprocess.run(
            ["git", "-C", tree_root, "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not describe:
            describe = sha
    except Exception:
        describe = sha
    return sha, describe

# ============================================================================
# RPYC_ASYNC — IMPORTED FROM THE INSTALLED PACKAGE (PyPI)
# ============================================================================
# rpyc-async is a regular PyPI dependency (>=1.0.0, owner decision
# 2026-08-11) providing the AsyncioServer-based protocol:
#   - rpyc.core.async_connect.async_connect  (async client)
#   - Connection.aclose() / wait_closed()    (async close)
# The previous source-checkout sys.path injection was removed: on a
# developer machine it shadowed the PyPI package with sources, which made
# "clean env" wheel installs test the wrong code.
# ============================================================================

from rpyc_async.core.protocol import DEFAULT_CONFIG
from rpyc_async.utils.async_server import AsyncioServer  # AsyncioServer from rpyc_async!

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus

clear_history_index = None
index_all_history_files = None


def _history_indexer_funcs() -> tuple[Any, Any]:
    global clear_history_index, index_all_history_files
    # Do not rely on middleware import order having injected the semantic_search
    # wrapper dir onto sys.path; ensure it explicitly. See
    # deep_worker_cli/semantic_search_path.py.
    from deep_worker_cli.semantic_search_path import (
        ensure_semantic_search_importable,
        semantic_search_wrapper_dir,
    )

    if not ensure_semantic_search_importable():
        raise ModuleNotFoundError(
            "semantic_search package not found: expected wrapper dir at "
            f"{semantic_search_wrapper_dir()}"
        )
    if clear_history_index is None:
        from semantic_search.history_indexer import clear_history_index as _clear

        clear_history_index = _clear
    if index_all_history_files is None:
        from semantic_search.history_indexer import index_all_history_files as _index

        index_all_history_files = _index
    return clear_history_index, index_all_history_files


def _clear_then_index_history(summaries_dir: Path) -> Any:
    clear_func, index_func = _history_indexer_funcs()
    clear_func(summaries_dir)
    return index_func(summaries_dir)


def _resolve_persisted_grok_display_model(agent_name: str) -> str | None:
    """Return the modelId grok used in the most recent persisted session for
    ``agent_name``, or ``None`` if it cannot be determined.

    Grok stores per-session data under ``~/.grok/sessions/<urlencoded-cwd>/
    <session_id>/`` (events.jsonl, updates.jsonl, chat_history.jsonl). Every
    stream entry tags ``model_id`` / ``modelId``; we read the last one we can
    find. Used at startup so the TUI displays the real model immediately
    instead of "unknown" until the first session/prompt response lands.

    Best-effort: never raises. Mirrors
    :func:`_resolve_persisted_codex_display_model`.
    """
    try:
        state_path = Path.home() / Path("~/.deep_worker").name / agent_name / "grok_session_state.json"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(state_data, dict):
        return None
    session_state = state_data.get(agent_name)
    if not isinstance(session_state, dict):
        return None
    session_id = session_state.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None

    sessions_dir = Path.home() / ".grok" / "sessions"
    if not sessions_dir.exists():
        return None
    try:
        candidates = sorted(
            sessions_dir.rglob(session_id),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        # updates.jsonl is the richest stream; events.jsonl and
        # chat_history.jsonl are good fallbacks. Latest-wins per file.
        for filename in ("updates.jsonl", "events.jsonl", "chat_history.jsonl"):
            path = candidate / filename
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            model_name: str | None = None
            for line in text.splitlines():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = _extract_grok_model_id(obj)
                if isinstance(m, str) and m.strip() and m.strip().lower() != "unknown":
                    model_name = m.strip()
            if model_name is not None:
                return model_name
    return None


def _extract_grok_model_id(obj: Any) -> str | None:
    """Pull a ``modelId`` / ``model_id`` value out of any nesting in ``obj``.

    Grok places the model name at slightly different keys depending on the
    stream — top-level for chat_history events, under ``_meta`` for ACP
    session/update wrappers, etc. Recurses through dicts/lists to find one.
    """
    if isinstance(obj, dict):
        for key in ("modelId", "model_id"):
            v = obj.get(key)
            if isinstance(v, str):
                return v
        for v in obj.values():
            found = _extract_grok_model_id(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract_grok_model_id(item)
            if found:
                return found
    return None


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


def _make_resolved_model_handler(
    *,
    config: Any,
    session_manager: Any,
) -> Callable[..., Awaitable[None]]:
    """Build the ``_on_resolved_model`` async callback used by codex/grok
    providers and the startup display seed.

    ``persist`` controls whether the resolved model is written back to
    ``launch_config["model"]`` (durable execution config). For an agent launched
    with ``config.model is None`` (unpinned), the resolved value is DISPLAY-ONLY
    and must NOT be persisted — persisting it re-pins execution and defeats the
    operator rule "no explicit --model ⇒ provider default". See
    ``DESIGN_FIX_CROSS_PROVIDER_MODEL_POISON.md`` and a related internal
    incident analysis (not included here):
    codex's session jsonl had recorded a foreign grok model name (because the
    original broken ``-pp`` bug had fed codex that string), and the previous
    handler persisted that back into ``launch_config["model"]`` on every restart.

    - **Startup session-log seed**: ``persist=False`` (display only).
    - **Live provider callback**: ``persist=True``, BUT only persists when the
      agent was launched with an explicit model (``config.model is not None``).
      Unpinned agents never silently pin from a session log.

    Extracted from ``AgentProcess.start`` to be unit-testable
    (``test_resolved_model_handler.py``).
    """

    async def _on_resolved_model(
        model_name: str, *, persist: bool = True
    ) -> None:
        set_display_model = getattr(config, "set_dynamic_display_model", None)
        if callable(set_display_model):
            set_display_model(model_name)
        if not persist:
            return
        if getattr(config, "model", None) is None:
            # Agent launched unpinned — never re-pin execution config from a
            # resolved value. Display already updated above.
            return
        if session_manager is None:
            return
        update_model = getattr(
            session_manager, "update_launch_config_model", None
        )
        if callable(update_model):
            await update_model(model_name)

    return _on_resolved_model


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


_logger = logging.getLogger(__name__)


# NEVER use print() in this module's hot paths (model invoke / catch-all /
# notify) — see docs/POLICIES.md §1 and docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md.
# ``stderr`` can block under load and ``BlockingIOError`` propagates up to the
# invoke catch-all, which used to flip the whole agent into AgentStatus.ERROR.
# Always use ``_logger.*`` (non-blocking handler).
def _log_error_context(
    *,
    error: str,
    input_message_id: str | None,
    messages: list[Any] | None = None,
    recoverable: bool | None = None,
) -> None:
    fields = [f"error={error}"]
    if recoverable is not None:
        fields.append(f"recoverable={'true' if recoverable else 'false'}")
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
    _logger.error("[INVOKE_ERROR] %s", " ".join(fields))


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
        mcp_config_paths: list[Path] | None = None,
        mcp_server_inline: list[str] | None = None,
        startup_query: str | None = None,
        startup_sum_query: str | None = None,
        startup_summarize: bool = False,
        startup_summary_prompt: str | None = None,
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
            mcp_config_paths: Additional layered MCP configuration files.
            mcp_server_inline: Highest-priority inline MCP server specs.
            startup_query: Optional plain starting query to execute right
                after startup (one-shot, never persisted). See
                ``DESIGN_STARTUP_QUERY_OPTIONS.md``.
            startup_sum_query: Optional pseudo-summary starting query — the
                current thread history is offloaded to ``<agent>.jsonl`` and
                the active context wiped before this becomes a summary-shape
                starting message (one-shot, never persisted).

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
        self.history_memory = bool(
            (launch_config or {}).get("history_memory", False)
        )
        self.use_sqlite = use_sqlite
        self.launch_config = launch_config
        self.mcp_config_paths = list(mcp_config_paths or [])
        self.mcp_server_inline = list(mcp_server_inline or [])
        self._checkpointer_ctx = None  # For lifecycle management
        # Restart notice consumed on start() — delivered to the model
        # as a SystemMessage on the next invoke. None when there is no
        # pending notice (the common case).
        self._pending_restart_notice: dict[str, Any] | None = None

        # Startup query fields (one-shot per invocation, never persisted to
        # disk / launch_config). Consumed by ``_auto_run_startup_query`` after
        # start(). See ``DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.2.
        self._pending_startup_query: str | None = startup_query
        self._pending_startup_sum_query: str | None = startup_sum_query
        self._pending_summarize = startup_summarize
        self._pending_summary_prompt = startup_summary_prompt

        # The underlying chat model. Assigned in ``start()`` after
        # ``create_model`` returns. Used by ``_auto_continue_after_restart``
        # to wait for MCP tool registration before triggering the first
        # post-restart LLM turn — otherwise the model runs blind without
        # tools and the agent can only respond in plain text.
        self._model: Any | None = None

        # Auto-assign RPyC port if enabled
        if rpyc_port is True:
            self.rpyc_port = _find_free_port()
        elif isinstance(rpyc_port, int):
            self.rpyc_port = rpyc_port
        else:
            self.rpyc_port = None

        # Paths
        self.pid_file = pid_file or Path(f"/tmp/deep_worker_{name}.pid")
        self.data_dir = data_dir or (Path.home() / Path("~/.deep_worker").name / name)
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
        # Token-stats recorder (token-stats design §5.1). Lazily
        # created on first LLM invocation; writes to
        # ``<data_dir>/token_stats.jsonl``.
        self._token_stats_recorder_obj: Any | None = None
        # Invariant (startup) system prompt, captured at agent-build time in
        # start(). Surfaced by the Full Dialog "system_prompt" introspection
        # mode (§9 of DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md).
        self._invariant_system_prompt: str | None = None

        # External MCP client pool — brought up in start(), shut down in
        # stop(). See docs/DESIGN_EXTERNAL_MCP_TOOLS.md.
        self._external_mcp_pool: Any = None  # ExternalMCPClientPool | None

        # RPyC components
        self.rpyc_server: AsyncioServer | None = None
        self.rpyc_service_instance: Any = None
        self._agent_event_loop: Any = None  # Event loop for RPyC callbacks

        # State
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.started_at: datetime | None = None  # Set when agent starts

        # A2A protocol server handle (None when disabled / unavailable). Set in
        # start() just before _register() so a2a_port/a2a_url can be written
        # atomically with the rest of the registration (RC-6).
        self._a2a_handle: Any = None

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
        # One conversation thread has one mutable LangGraph/provider state.
        # Serialize complete top-level invocations, including every model/tool
        # iteration inside them; a provider-level ainvoke lock is too narrow.
        self._invoke_lock = asyncio.Lock()

        # Invoke tasks whose RPyC dispatch task was cancelled (channel drain /
        # inbound-quarantine) while the invoke kept running. Strong refs here
        # keep them alive until they finish; they self-remove via a done
        # callback. See AgentService.exposed_invoke_agent (decoupling).
        self._detached_invoke_tasks: set[Any] = set()

        # ── Durable inter-agent inbox (DESIGN_DURABLE_INTER_AGENT_DELIVERY.md)
        # The drain task is held in its own attribute and is NOT managed like
        # _detached_invoke_tasks: that set only holds references, while this
        # consumer must be cancelled and awaited before the checkpointer closes
        # (§3.3.4). Callback injection is a SEPARATE method from task creation
        # so the idempotence guard can never discard a late callback (§3.3.3).
        self._inbox_drain_task: Any = None
        self._inbox_event: asyncio.Event = asyncio.Event()
        self._inbox_seq_lock = asyncio.Lock()
        self._inbox_next_seq: int = 0
        # False until a FULL spool scan has primed _inbox_next_seq. Publishing
        # against an unprimed counter would allocate seq 0 while records with
        # higher seqs survive on disk, breaking the
        # next_seq = max(pending ∪ quarantined) + 1 contract and the attempt
        # order that depends on it. Enqueue therefore primes on demand and
        # fails loudly if it cannot (design §3.1.1).
        self._inbox_seq_primed: bool = False
        self._inbox_notify: Any = None
        self._shutting_down: bool = False

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
           (a related internal incident analysis (not included here)).
        3. *Timeout breaks the hang.* A subscriber whose ``on_message``
           never resolves used to accumulate one parked Task per
           ``set_status`` for the rest of the process's life
           (a related internal incident analysis, not included here). With
           the timeout, each dispatched
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

        from rpyc_async.utils.helpers import fire_and_forget_async  # noqa: PLC0415

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
                _logger.info(f"[AGENT] status push to subscriber id={_sid} failed: {exc!r}")
                # EVICT the dead subscriber. on_disconnect deliberately
                # does NOT clean up — it relies on a graceful
                # exposed_unsubscribe_messages that never arrives when the
                # web client dies/restarts ungracefully. Without eviction
                # the dead MessageCallbackService netref lingers forever
                # and EVERY status push goes only to the corpse (10 s
                # timeout each), so the live UI never updates — the
                # the "stuck at idle while actually working" bug.
                # A live web re-subscribes on its next
                # connect_to_agent, so eviction is safe.
                #
                # Evict by IDENTITY: ``set.discard(netref)`` would call
                # ``netref.__hash__`` → a sync RPyC round-trip over the
                # already-dead channel → EOFError, which on the invoke path
                # escaped and forced the agent to ERROR.
                # ``evict_subscriber_by_identity`` never crosses the wire.
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, _sub)

            try:
                # Use async-flagged callattr directly instead of
                # ``subscriber.on_message(payload_str)``, which goes
                # through netref.__getattribute__ → syncreq(HANDLE_GETATTR)
                # then method_netref.__call__ → syncreq(HANDLE_CALL) /
                # asyncreq(HANDLE_ASYNC_CALL) depending on the peer's
                # ____is_async__ flag. For a peer whose on_message is NOT
                # a coroutine function, the method netref gets
                # ____is_async__=False → syncreq(HANDLE_CALL) →
                # sync_request(HANDLE_CALL) → GUARD FIRES → RuntimeError
                # ("sync_request() was called from the asyncio loop").
                #
                # ``asyncreq(proxy, HANDLE_CALLATTR, ...)`` calls
                # ``conn.async_request(HANDLE_CALLATTR, ...)`` — a single
                # async message that never enters sync_request at all. The
                # peer's _dispatch_request_async handles both sync and
                # async on_message correctly (see §6.1 of
                # DESIGN_ASYNC_DISPATCH). Fixes issue #48.
                from rpyc_async.core.netref import asyncreq as _asyncreq  # noqa: PLC0415
                awaitable = _asyncreq(
                    subscriber,
                    8,  # HANDLE_CALLATTR — get attr + call
                    "on_message",
                    (payload_str,),
                    (),  # kwargs
                )
            except Exception as exc:  # noqa: BLE001
                # Construction-time failure (rare): the netref is so
                # dead it can't even produce an AsyncResult. Log and
                # move on; do not let one broken subscriber abort the
                # whole snapshot fan-out.
                _logger.info(f"[AGENT] status push to subscriber id={subscriber_id} "
                    f"could not be dispatched: {exc!r}")
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

        # Restore-on-idle (§7.2 of DESIGN_PROVIDER_FAILOVER): after
        # pushing status to all subscribers, check whether we should
        # restore the preferred provider now that the agent is idle.
        await self._maybe_restore_preferred_provider(_payload)

    # ------------------------------------------------------------------
    # Restore-on-idle (§7.2)
    # ------------------------------------------------------------------
    async def _maybe_restore_preferred_provider(
        self, payload: dict[str, Any]
    ) -> None:
        """Check if the agent should be restored to its preferred provider.

        Fire-and-forget: this runs on every status change but only acts
        when all conditions are met — never blocks the caller.

        Fires only on ``status → idle`` / ``→ waiting_input`` transitions
        and only when:
        - ``current_provider != preferred_provider``
        - ``preferred_provider`` is not ``"deepseek"``
        - ``availability_tracker.is_restore_eligible(preferred_provider)``
          is True (stricter than ``is_available`` — active limit blocks restore)
        - No dedicated ``_restore_scheduled`` flag is already set (one per reopen).

        When all conditions are met, schedules ``deep-worker --restart
        <name> --provider <preferred> [--model <preferred_model>]
        --no-update-preferred`` via ``asyncio.create_subprocess_exec``.
        """
        status = payload.get("status")
        if status not in ("idle", "waiting_input"):
            return

        if not isinstance(self.launch_config, dict):
            return
        current_provider = self.launch_config.get("provider", "")
        preferred_provider = self.launch_config.get("preferred_provider", "")
        if not preferred_provider or preferred_provider == current_provider:
            return
        if preferred_provider == "deepseek":
            return

        from deep_worker_cli.runtime.availability_tracker import (  # noqa: PLC0415
            AvailabilityTracker,
        )
        tracker = AvailabilityTracker()
        if not tracker.is_restore_eligible(preferred_provider):
            return

        # Guard: only fire once per reopen. After a failover,
        # ``record_failure`` stores ``available_after_utc``. When the
        # reset window passes, ``is_restore_eligible`` flips back to True.
        # Clear the entry so we don't re-fire on every idle transition
        # until the NEXT failover.
        tracker.clear(preferred_provider)

        preferred_model = self.launch_config.get("preferred_model")
        restore_cmd: list[str] = [
            "deep-worker",
            "--restart",
            self.name,
            "--provider",
            preferred_provider,
            "--no-update-preferred",
        ]
        if preferred_model:
            restore_cmd.extend(["--model", str(preferred_model)])

        _logger.info(f"[RESTORE-ON-IDLE] Agent '{self.name}' on {current_provider} is "
            f"{status}; restoring preferred provider {preferred_provider} …")
        try:
            proc = await asyncio.create_subprocess_exec(
                *restore_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            # Detach — do NOT await. The restart will kill this process.
            asyncio.ensure_future(proc.wait())
        except Exception as exc:
            _logger.info(f"[RESTORE-ON-IDLE] Failed to schedule restore for "
                f"'{self.name}': {exc!r}")

    def _on_inbound_quarantine(self, info: dict[str, Any]) -> None:
        """rpyc_async ``on_inbound_quarantine`` hook (sync, runs on the loop).

        A client's RPyC channel was put into inbound quarantine because it
        flooded the agent past ``max_inbound_inflight``. This is purely a
        WARNING: the agent keeps working and its status MUST NOT change. We
        only log it and broadcast a non-status warning event to the live
        subscribers (the quarantined channel itself won't receive it — it's
        the spammer — but other live channels, e.g. the web app, will).

        Quarantine (vs. closing the channel) is intentional: it stops the
        spammer from immediately opening a fresh connection. See
        rpyc_async's design doc for inbound backpressure.
        """
        connid = info.get("connid")
        peer = info.get("peer")
        _logger.info(f"[RPyC][WARNING] inbound channel quarantined: connid={connid} "
            f"peer={peer} inbound_inflight={info.get('inbound_inflight')} "
            f"threshold={info.get('threshold')}. A client flooded this "
            f"channel; it is now ignored. The agent keeps working — status "
            f"unchanged.")
        # Schedule the async fan-out without blocking the dispatch path.
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._broadcast_channel_quarantine_warning(info))
        except Exception as exc:  # noqa: BLE001
            _logger.info(f"[RPyC][WARNING] could not schedule quarantine broadcast: {exc!r}")

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

        from rpyc_async.utils.helpers import fire_and_forget_async  # noqa: PLC0415

        for subscriber in list(self._message_subscribers):
            subscriber_id = id(subscriber)

            async def _on_error(
                exc: BaseException,
                _sid: int = subscriber_id,
                _sub: Any = subscriber,
            ) -> None:
                _logger.info(f"[AGENT] quarantine-warning push to subscriber id={_sid} "
                    f"failed: {exc!r}")
                # Evict by identity — never ``discard`` a netref (its
                # __hash__ crosses the dead wire → EOFError).
                from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
                    evict_subscriber_by_identity,
                )
                evict_subscriber_by_identity(self._message_subscribers, _sub)

            try:
                # Same fix as _dispatch_status_event: use asyncreq +
                # HANDLE_CALLATTR to avoid sync_request guard (issue #48).
                from rpyc_async.core.netref import asyncreq as _asyncreq  # noqa: PLC0415
                awaitable = _asyncreq(
                    subscriber,
                    8,  # HANDLE_CALLATTR
                    "on_message",
                    (payload_str,),
                    (),  # kwargs
                )
            except Exception as exc:  # noqa: BLE001
                _logger.info(f"[AGENT] quarantine-warning push to subscriber "
                    f"id={subscriber_id} could not be dispatched: {exc!r}")
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

    @property
    def _token_stats_recorder(self) -> Any:
        """Lazy token-stats recorder for this agent.

        Created on first access (i.e. first instrumented LLM invocation);
        appends to ``<data_dir>/token_stats.jsonl`` (token-stats design
        §4.2). The same recorder instance is shared with the TaskManager so
        both write to one file, tagged ``kind="agent"`` / ``kind="tm"``.
        """
        if self._token_stats_recorder_obj is None:
            from deep_worker_cli.stats.token_stats import TokenUsageRecorder

            self._token_stats_recorder_obj = TokenUsageRecorder(self.data_dir)
        return self._token_stats_recorder_obj

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

            # Emit a one-line startup diagnostic to the agent's stdout log
            # (daemonized → <name>.out.log in the agent's log directory).
            # Includes git commit SHA and the full launch configuration so
            # every agent start produces a grep-able breadcrumb.
            lc = self.launch_config or {}
            tree = str(lc.get("tree_root", ""))
            commit, describe = _get_git_commit_info(tree)
            pairs = [
                ("agent", self.name),
                ("provider", str(lc.get("provider", "unknown"))),
                ("model", str(lc.get("model", "unknown"))),
                ("tree", tree),
                ("commit", commit),
                ("describe", describe),
                ("pid", str(os.getpid())),
                ("tree_root", tree),
                ("working_dir", str(lc.get("working_dir", ""))),
                ("skip_permissions", str(lc.get("skip_permissions", False))),
                ("task_manager", str(lc.get("task_manager", False))),
                ("memory_only", str(lc.get("memory_only", False))),
                ("history_memory", str(lc.get("history_memory", False))),
                ("use_sqlite", str(lc.get("use_sqlite", False))),
                ("dev_tree", str(lc.get("dev_tree", False))),
                ("clean", str(lc.get("clean", False))),
                ("auto_approve_send_message", str(lc.get("auto_approve_send_message", False))),
                ("preferred_provider", str(lc.get("preferred_provider", ""))),
                ("started_at", str(lc.get("started_at", ""))),
            ]
            line = " ".join(f"{k}={v}" for k, v in pairs)
            print(f"[STARTUP] {line}", flush=True)

            # 1. Create PID file
            await self._create_pid_file()

            # 2. Setup signal handlers
            self._setup_signal_handlers()

            # 2.1 Clear any stale legacy .restart_self flag from old code.
            # The previous (deferred) restart_self design wrote a flag file
            # in data_dir; it has been removed in favour of the synchronous
            # restart-notice mechanism. A stale flag from a pre-upgrade
            # process is harmless but should be cleaned up.
            from deep_worker_cli.cli.agent_command import _get_agent_data_dir
            stale_flag = _get_agent_data_dir(self.name) / ".restart_self"
            if stale_flag.exists():
                stale_flag.unlink()
                msg = (
                    f"{_ts()} [STARTUP] Cleared stale legacy .restart_self "
                    f"flag from {stale_flag}\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

            # 2.2 Consume restart-notice if present. The new restart_self
            # tool writes this file synchronously before spawning the
            # restart subprocess; the new process picks it up here and
            # will inject it as a SystemMessage on the next invoke.
            self._pending_restart_notice = self._load_and_consume_restart_notice()
            if self._pending_restart_notice is not None:
                msg = (
                    f"{_ts()} [STARTUP] Loaded restart-notice "
                    f"(cause={self._pending_restart_notice.get('cause')!r}, "
                    f"ts={self._pending_restart_notice.get('ts')!r}). "
                    f"It will be delivered to the next invoke as a "
                    f"SystemMessage.\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

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

                msg = f"{_ts()} [AGENT_INIT] Initializing agent '{self.name}'...\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                _logger.info(msg)

                # Set up logging to stderr so log_token_usage() output is visible
                _log = logging.getLogger("deep_worker_cli")
                if not _log.handlers:
                    _handler = logging.StreamHandler(sys.stderr)
                    _handler.setFormatter(
                        logging.Formatter(
                            "%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                        )
                    )
                    _log.addHandler(_handler)
                    _log.setLevel(logging.INFO)

                # Load environment variables from ~/.deepagents/.env BEFORE importing config
                # This is critical because config.py initializes Settings at module level
                # and needs OPENAI_MODEL, OPENAI_BASE_URL, etc. to be set before import
                deepagents_env = PathLib.home() / ".deepagents" / ".env"
                if deepagents_env.exists():
                    dotenv.load_dotenv(deepagents_env)
                    msg = f"{_ts()} [AGENT_INIT] Loaded environment from {deepagents_env}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    _logger.info(msg)

                # Drop-in env files in ~/.deepagents/env.d/*.env (each 0o600).
                # Loaded AFTER the main .env so it stays authoritative — drop-ins
                # only fill blanks. See DESIGN_EXTERNAL_MCP_TOOLS.md §7 and
                # tests/unit_tests/process/test_env_d_loader.py.
                from deep_worker_cli.process.env_loader import load_env_d

                env_d_dir = PathLib.home() / ".deepagents" / "env.d"
                applied = load_env_d(env_d_dir)
                if applied:
                    msg = (
                        f"{_ts()} [AGENT_INIT] Loaded {len(applied)} variable(s) "
                        f"from {env_d_dir}: {sorted(applied.keys())}\n"
                    )
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    _logger.info(msg)

                # ================================================================
                # CRITICAL: Add deepagents to sys.path BEFORE imports
                # ================================================================
                # deepagents must be imported from source, not from pip/venv —
                # this ensures correct version and avoids pip-vs-local
                # divergence.
                #
                # The path is derived from __file__ so the SAME line works
                # for BOTH the production tree and the developer tree.
                # Hard-coding the prod path here used to defeat the --dev
                # launcher dispatch: even after the bash launcher correctly
                # re-execs into the dev tree, this insertion would prepend
                # the prod ``deepagents`` and shadow the dev one. See
                # docs/DESIGN_DEV_FLAG.md.
                #
                # __file__ here =
                #   <TREE>/deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py
                # parent x4 =
                #   <TREE>/<package-root>
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
                # kind="agent" + factories wire the token-stats recorder
                # (token-stats design §5.1).
                from deep_worker_cli.stats.callback import (
                    make_stats_callback_factory,
                    make_usage_event_callback_factory,
                )

                model = create_model(
                    provider=self.config.provider,
                    model_name=self.config.model,
                    user=self.name,
                    kind="agent",
                    stats_callback_factory=make_stats_callback_factory(self._token_stats_recorder),
                    usage_event_callback_factory=make_usage_event_callback_factory(
                        self._token_stats_recorder
                    ),
                )

                # Set working_dir on model for MCP direct tool execution
                # (e.g. project_memory needs to know the project root)
                # Uses setattr() because create_model() returns BaseChatModel
                # but only ClaudeCodeChatModel has _working_dir attribute.
                if hasattr(model, '_working_dir'):
                    setattr(model, '_working_dir', str(self.config.working_dir))

                # Expose the model on ``self`` so any post-init helper
                # that needs it (e.g. provider failover, codex display-
                # model resolution) can reach it. After a related internal
                # incident (analysis not included here), the MCP-tools-ready
                # barrier that used to live here was replaced by a
                # deterministic
                # prewarm of ClientManager — see
                # ``_extract_final_tool_metadata`` and
                # ``_prewarm_client_manager`` below, and the call site
                # immediately after ``create_cli_agent`` returns.
                self._model = model

                set_resolved_model_callback = getattr(
                    model, "set_resolved_model_callback", None
                )
                if callable(set_resolved_model_callback):
                    _on_resolved_model = _make_resolved_model_handler(
                        config=self.config,
                        session_manager=self.session_manager,
                    )
                    set_resolved_model_callback(_on_resolved_model)
                    if self.config.provider == "codex" and self.config.model is None:
                        persisted_model = _resolve_persisted_codex_display_model(self.name)
                        if persisted_model:
                            await _on_resolved_model(persisted_model, persist=False)
                    elif self.config.provider == "grok" and self.config.model is None:
                        # Seed the TUI display with the modelId from grok's
                        # own per-session log (~/.grok/sessions/...). Without
                        # this seed the user sees "unknown" until the first
                        # session/prompt response arrives, because the hot
                        # path only updates the display from the response
                        # _meta.modelId. See a related internal incident
                        # analysis (not included here).
                        persisted_model = _resolve_persisted_grok_display_model(self.name)
                        if persisted_model:
                            await _on_resolved_model(persisted_model, persist=False)

                restart_tool_factory = getattr(cli_tools, "make_restart_agent_tool")
                restart_self_tool_factory = getattr(cli_tools, "make_restart_self_tool")

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
                    restart_self_tool_factory(self.name),
                    make_send_message_to_agent_tool(self.name),
                ]

                # A2A outbound tools (a2a_*): only registered when A2A is
                # enabled for this agent. Lets the LLM initiate A2A protocol
                # calls to other A2A-compliant agents. Import/build failures
                # are non-fatal (A2A is optional) — the agent keeps its other
                # tools. See the A2A protocol design documentation.
                try:
                    from deepworker_a2a.config import A2AConfig as _A2AConfig

                    if _A2AConfig.from_env(self.name).enabled:
                        from deepworker_a2a.tools import make_a2a_tools

                        web_tools.extend(make_a2a_tools())
                        _logger.info("Registered outbound a2a_* tools")
                except Exception:  # noqa: BLE001 — A2A tools are optional
                    _logger.info(
                        "deepworker_a2a tools unavailable; skipping a2a_* tools"
                    )

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

                # Grok-specific: register the built-in tool passthroughs.
                # Grok CLI in ``agent stdio`` always runs its built-ins
                # (Shell/Read/Edit/Glob/Grep/WebSearch/Task/Plan/...)
                # locally and cannot be told not to (Q1 live-check). When
                # the model calls one, ``grok_provider.chat_model`` records
                # it as an honest LangGraph tool_call referencing the
                # matching BaseTool from
                # ``grok_provider.builtin_tool_passthrough``. The body parks
                # on an asyncio.Future resolved by the ACP reader at
                # ``tool_call_update status=completed`` — event-driven, no
                # polling. See DESIGN_GROK_PROVIDER.md §11A.4.
                if self.config.provider == "grok":
                    from grok_provider.builtin_tool_passthrough import (
                        build_passthrough_tools,
                    )
                    web_tools.extend(build_passthrough_tools())

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

                # Start the external MCP servers (see
                # docs/DESIGN_EXTERNAL_MCP_TOOLS.md) BEFORE create_cli_agent,
                # so their tools enter the graph from the very beginning and
                # land in _extract_final_tool_metadata (then Full Dialog
                # verbosity=tools will show them, and the provider prewarm
                # will see the correct final set).
                self._external_mcp_pool = await self._start_external_mcp_pool()

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
                    enable_history_memory=self.history_memory,
                    external_mcp_tools=(
                        self._external_mcp_pool.tools
                        if self._external_mcp_pool is not None
                        else None
                    ),
                    external_mcp_auto_approved_names=(
                        self._external_mcp_pool.auto_approved_tool_names
                        if self._external_mcp_pool is not None
                        else None
                    ),
                )

                # Capture the INVARIANT (startup) system prompt so the
                # Full Dialog "system_prompt" introspection mode (§9 of the
                # Full Dialog design doc) reports EXACTLY what the agent was
                # built with. We pass no
                # ``system_prompt`` override above, so ``create_cli_agent``
                # resolves it via ``get_system_prompt`` with these same args —
                # recomputing here yields the identical provider-agnostic text.
                try:
                    from deep_worker_cli.agent import get_system_prompt
                    self._invariant_system_prompt = get_system_prompt(
                        assistant_id=self.name,
                        sandbox_type=None,
                        project_memory_dir=Path(self.config.working_dir),
                        provider=getattr(self.config, "provider", None),
                    )
                except Exception:  # noqa: BLE001 — never block start()
                    self._invariant_system_prompt = None

                # ════════════════════════════════════════════════════════
                # PREWARM ClientManager with the FINAL tool inventory.
                # ════════════════════════════════════════════════════════
                # Root-cause fix for a related internal incident (analysis
                # not included here). The compiled
                # agent already knows its final tool set (LangGraph's
                # ToolNode.tools_by_name) — read it synchronously and
                # seed ClientManager.clients[self.name] with a
                # ProxyMCPServer built from that exact set. This makes
                # the first ainvoke (including the one fired by
                # _auto_continue_after_restart) hit a cache slot with
                # the CORRECT proxy, instead of racing whichever
                # bind_tools call happens to land first.
                #
                # ⚠ DO NOT replace this with a polling barrier on
                # model._tools_metadata. Polling for "any non-empty
                # set" is what caused the regression in the first
                # place. See:
                #   - a related internal incident analysis (not included here)
                #   - claude_code_provider/client_manager.py
                #     (rotate-on-mismatch — defence in depth).
                # Both halves of the fix MUST coexist.
                _final_metadata = self._extract_final_tool_metadata(self.agent)
                _client_manager = getattr(model, "_client_manager", None)
                if _client_manager is not None and _final_metadata:
                    await self._prewarm_client_manager(
                        client_manager=_client_manager,
                        thread_id=self.name,
                        tools_metadata=_final_metadata,
                        working_dir=self.config.working_dir,
                    )

            except Exception as e:
                msg = f"{_ts()} [AGENT_INIT] Error initializing agent: {e}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                _logger.info(msg)
                _logger.info(traceback.format_exc())
                # Agent remains None, fallback to echo mode
                self.agent = None
                # Per No-Agent-Error-State policy + user requirement: init failures
                # (e.g. missing MCP env vars, bad config) are FATAL ("agent not initialised").
                # Make the error VISIBLE and PERSISTENT in Web UI as ERROR status + error_message.
                # It must not be hidden in logs only. The error banner will show prominently
                # for the selected agent.
                if self.session_manager:
                    try:
                        await self.session_manager.set_error(
                            f"Agent initialization failed: {e}"
                        )
                    except Exception:
                        pass  # best effort; don't block start continuation

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
                        token_stats_recorder=self._token_stats_recorder,
                    )
                    await self.task_manager.start()

                    msg = f"{_ts()} [AGENT_INIT] TaskManager started (manager_id: {self.name}:tm)\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    _logger.info(msg)

                except Exception as e:
                    msg = f"{_ts()} [AGENT_INIT] Error initializing TaskManager: {e}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    _logger.info(msg)
                    _logger.info(traceback.format_exc())
                    # TaskManager remains None, agent continues without TaskManager
                    self.task_manager = None

            # DEBUG: Log before RPyC server start
            msg = f"{_ts()} [AGENT_INIT] Reached line before RPyC server start...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            _logger.info(msg)

            # 4.9 Prime the durable inbox BEFORE the RPyC server can accept an
            # enqueue. publish() would create the spool lazily, but _next_seq
            # must be primed from disk first or a surviving record could have
            # its filename reused. The drain itself is started later
            # (bring_up_inbox), once the graph and checkpointer are ready.
            # DESIGN_DURABLE_INTER_AGENT_DELIVERY.md §3.2.1.
            try:
                self.init_inbox_spool()
            except OSError as exc:
                _logger.exception("[INBOX] could not initialise spool: %s", exc)

            # 5. Start RPyC server if port is configured
            msg = f"{_ts()} [AGENT_INIT] About to start RPyC server (port={self.rpyc_port})...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            _logger.info(msg)

            if self.rpyc_port is not None:
                await self._start_rpyc_server()  # Now async!
            else:
                raise RuntimeError("RPyC port must be configured - IPC server has been removed")

            msg = f"{_ts()} [AGENT_INIT] RPyC server started successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            _logger.info(msg)

            # 6. Register in registry (ONLY after RPyC server is ready!)
            # This ensures clients can connect immediately after seeing the agent in registry
            # 5.1 Start optional A2A protocol server BEFORE registering, so
            # a2a_port/a2a_url are written atomically with registration (RC-6).
            # Disabled by default; failure here is logged, never fatal (RC-7).
            await self._maybe_start_a2a_server()

            msg = f"{_ts()} [AGENT_INIT] About to register in registry...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            _logger.info(msg)

            await self._register()

            msg = f"{_ts()} [AGENT_INIT] Registered successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            _logger.info(msg)

            # 7. Enter event loop
            self.running = True
            print(f"{_ts()} Agent '{self.name}' started (PID {os.getpid()})")
            print(f"{_ts()} RPyC port: {self.rpyc_port}")
            print(f"{_ts()} Working directory: {self.config.working_dir}")

            # 7.1 Auto-continue after restart. If a restart-notice was
            # consumed at startup, the agent must NOT sit idle waiting
            # for a new user message — it must immediately continue
            # working on the pending task on the freshly-loaded code.
            # Schedule a self-initiated invoke that delivers the
            # SystemMessage; the model takes it from there.
            if self._pending_restart_notice is not None:
                asyncio.create_task(self._auto_continue_after_restart())

            # 7.2 Auto-run a startup query. Symmetric to the restart-notice
            # hook above: if --sum-query / -q was supplied on this launch, the
            # agent immediately begins executing it instead of sitting idle.
            # One-shot; the pending fields are cleared inside the coroutine.
            # See DESIGN_STARTUP_QUERY_OPTIONS.md §4.3.
            if (
                self._pending_startup_query is not None
                or self._pending_startup_sum_query is not None
                or self._pending_summarize
            ):
                startup_task = asyncio.create_task(
                    self._auto_run_startup_query()
                )
                self._detached_invoke_tasks.add(startup_task)
                startup_task.add_done_callback(
                    self._detached_invoke_tasks.discard
                )
                startup_task.add_done_callback(
                    self._log_startup_task_result
                )

            # Bring up the durable inter-agent inbox: create/recover the spool
            # and start the drain. Owned by AgentProcess (NOT AgentService) so
            # it runs whenever the process is alive, including the startup edge
            # that recovers records left by a restart — which happens before
            # any subscriber attaches. The notify callback is injected
            # separately via set_inbox_notify.
            # See DESIGN_DURABLE_INTER_AGENT_DELIVERY.md §3.3.3/§3.3.4.
            recovered = self.bring_up_inbox()
            if recovered:
                msg = (
                    f"{_ts()} [STARTUP] Recovered {recovered} pending "
                    f"inter-agent message(s) from the durable inbox; they "
                    f"will be processed by the drain.\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

            # Main event loop: wait for shutdown signal - EVENT DRIVEN, NO POLLING
            # Wait for shutdown event instead of polling self.running
            await self.shutdown_event.wait()
            
            print(f"{_ts()} Agent '{self.name}' event loop exiting...")

            # Graceful shutdown after event loop exits
            await self.shutdown()

        except Exception as e:
            print(f"{_ts()} Failed to start agent: {e}")
            await self.shutdown()
            raise

    def _load_and_consume_restart_notice(self) -> dict[str, Any] | None:
        """Load and remove the ``.restart_notice.json`` file on startup.

        The ``restart_self`` tool writes this file right before spawning
        ``deep-worker --restart``. The newly-started process picks it up
        here, removes it from disk, and stores the dict so it can be
        injected as a ``SystemMessage`` on the next ``ainvoke``.

        Returns:
            The notice dict (``{ts, cause, message}``) on success, or
            ``None`` if no notice exists or the file is corrupt. In
            both error paths the file is unlinked so it cannot
            re-fire on the next start.
        """
        notice_path = self.data_dir / ".restart_notice.json"
        if not notice_path.exists():
            return None

        try:
            import json as _json
            notice = _json.loads(notice_path.read_text())
        except (OSError, ValueError):
            # json.JSONDecodeError is a subclass of ValueError so it is
            # caught here. Corrupt notice = no notice.
            notice = None

        # Always remove the file (success or corrupt) — single-shot.
        try:
            notice_path.unlink()
        except OSError:
            pass

        if not isinstance(notice, dict):
            return None
        return notice

    def _format_restart_notice_message(self, notice: dict[str, Any]):
        """Format a ``SystemMessage`` describing the restart event.

        The message is purely informational — it does NOT ask the agent
        to acknowledge or confirm. The agent is expected to continue
        whatever work was in progress.
        """
        from langchain_core.messages import SystemMessage

        ts = notice.get("ts", "<unknown>")
        cause = notice.get("cause", "unknown")
        extra = notice.get("message", "")

        if cause == "self":
            human_cause = "by its own restart_self tool call (self-initiated)"
        else:
            human_cause = f"by cause={cause}"

        body = (
            f"NOTICE: This agent process was restarted at {ts} "
            f"{human_cause}. The new process is running with the latest "
            f"on-disk code. Conversation history is preserved via the "
            f"checkpointer. Continue with the work that was in progress."
        )
        if extra:
            body += f" Details: {extra}"

        return SystemMessage(content=body)

    @staticmethod
    def _extract_final_tool_metadata(agent: Any) -> list[Any]:
        """Extract the FINAL tool inventory from a compiled LangGraph agent.

        ════════════════════════════════════════════════════════════════
        ⚠ WHY THIS EXISTS — DO NOT REPLACE WITH POLLING.
        ════════════════════════════════════════════════════════════════
        After ``create_cli_agent`` returns, the LangGraph factory has
        already computed the FINAL tool set the model_node will use on
        every turn. See ``langchain/agents/factory.py:880-905``:

            available_tools = middleware_tools + regular_tools
            tool_node = ToolNode(tools=available_tools, ...)
            default_tools = list(tool_node.tools_by_name.values()) + ...

        That ``tool_node`` is attached to the compiled graph at
        ``graph.nodes["tools"]`` (LangGraph wraps it in a PregelNode but
        the underlying object remains reachable). We read it
        SYNCHRONOUSLY here — no polling, no waiting, no race with
        ``bind_tools`` side effects.

        Earlier code (now deleted) used ``_wait_for_mcp_tools_ready``
        — a 30s polling barrier on ``model.tools_metadata`` that
        returned True on the FIRST non-empty set. That was the EXACT
        failure mode of a related internal incident (analysis not included
        here): the first non-empty
        ``tools_metadata`` came from ``_bind_tools_for_middleware``
        with the original caller-supplied 7 ``web_tools``, NOT the
        final 23 tools. The barrier released early and the cached
        ``ProxyMCPServer`` permanently held a stale, 4-of-7 inventory.

        Reading the compiled graph is the deterministic alternative:
        the value is known at agent-build time and never changes.

        Args:
            agent: A compiled LangGraph agent returned from
                ``create_cli_agent`` (which itself wraps
                ``create_agent`` from langchain).

        Returns:
            ``list[ToolMetadata]`` — one entry per tool found in the
            agent's ``tools`` node, ``[]`` when the agent has no
            ``tools`` node (legitimately tool-less agents). Each entry
            has the same shape as ``ClaudeCodeChatModel._tools_metadata``
            so it can be handed straight to
            ``ClientManager.get_or_create_client``.
        """
        # Best-effort: the agent shape can drift between LangGraph
        # versions. We tolerate missing pieces by returning [] (which
        # means "no prewarm" — the first turn will go through the
        # normal path) rather than raising and crashing start().
        try:
            nodes = getattr(agent, "nodes", None)
            if not nodes:
                return []
            tools_node = nodes.get("tools") if hasattr(nodes, "get") else None
            if tools_node is None:
                return []
            tools_by_name = getattr(tools_node, "tools_by_name", None)
            if tools_by_name is None:
                # LangGraph may wrap the ToolNode in a PregelNode that
                # exposes the underlying object on ``.bound`` (or other
                # attributes — verified empirically on this version).
                # Probe a few common shapes before giving up.
                for attr in ("bound", "node", "runnable"):
                    inner = getattr(tools_node, attr, None)
                    if inner is not None and hasattr(inner, "tools_by_name"):
                        tools_by_name = inner.tools_by_name
                        break
            if tools_by_name is None:
                return []
        except Exception:  # noqa: BLE001 — never block start() over extraction
            _logger.warning(
                "[PREWARM] tool_node extraction failed; skipping prewarm "
                "(first real turn will create the CLI client normally)",
                exc_info=True,
            )
            return []

        # Build ToolMetadata triples (name, description, input_schema).
        # We reuse ``ClaudeCodeChatModel`` to derive the schema so the
        # prewarm-created ProxyMCPServer advertises EXACTLY the same
        # surface a regular turn's ``bind_tools`` would produce. No
        # divergence between prewarm and turn paths.
        from claude_code_provider.chat_model import ClaudeCodeChatModel  # noqa: PLC0415

        tools_list = list(tools_by_name.values())
        if not tools_list:
            return []
        try:
            probe = ClaudeCodeChatModel(base_options=None).bind_tools(tools_list)
            return list(probe.tools_metadata)
        except Exception:  # noqa: BLE001
            _logger.warning(
                "[PREWARM] bind_tools probe failed; skipping prewarm",
                exc_info=True,
            )
            return []

    @staticmethod
    async def _prewarm_client_manager(
        client_manager: Any,
        thread_id: str,
        tools_metadata: list[Any],
        working_dir: str | None,
    ) -> None:
        """Deterministically prime the ClientManager cache with the
        FINAL tool inventory before any ``ainvoke`` can run.

        ════════════════════════════════════════════════════════════════
        ⚠ WHY THIS EXISTS — ROOT-CAUSE FIX FOR A RELATED INTERNAL INCIDENT (ANALYSIS NOT INCLUDED HERE).
        ════════════════════════════════════════════════════════════════
        ``ClientManager`` caches one ``(ClaudeSDKClient, ProxyMCPServer)``
        pair per ``thread_id``. The cached ``ProxyMCPServer`` is built
        from whatever ``tools_metadata`` was passed on the FIRST
        ``get_or_create_client`` call. After a related internal incident
        (analysis not included here) we also rotate-on-mismatch (see
        ``ClientManager``), but the
        ROOT prevention is to make sure the first call lands with the
        right inventory.

        ``_extract_final_tool_metadata(self.agent)`` gives us that
        inventory synchronously at start-time. Calling this prewarm
        here, BEFORE ``_auto_continue_after_restart`` (or any other
        code path that can trigger ``model.ainvoke``), guarantees the
        cache slot is correct from the start.

        Best-effort: any failure is logged and swallowed. The agent
        still starts; the first real turn will go through the normal
        create path and the rotate-on-mismatch guard will correct any
        residual mistake.

        ⚠ DO NOT replace this with a polling barrier. The whole point
        is determinism. See the related internal incident analysis (not
        included here).
        """
        if not tools_metadata:
            # Nothing to prewarm — likely a tool-less agent or extraction
            # gave up. Leave the cache slot empty; first real turn fills
            # it normally.
            return
        # ════════════════════════════════════════════════════════════════
        # ⚠ PREWARM MUST NOT OPEN THE CLAUDE CLI SESSION — A RELATED
        # INTERNAL INCIDENT (ANALYSIS NOT INCLUDED HERE).
        # ════════════════════════════════════════════════════════════════
        # Previously this called ``get_or_create_client(...)``, which
        # actually spawned the Claude CLI subprocess with
        # ``resume=None``. Because that set ``has_client(thread_id) =
        # True`` BEFORE the first ``ainvoke``, the history-restore block
        # in ``ClaudeCodeChatModel.ainvoke`` (gated on
        # ``not has_client(thread_id)``) NEVER ran on restart, and the
        # agent forgot its prior conversation.
        #
        # ``prepare_proxy_only`` does the prewarm half that we actually
        # need (deterministic ``ProxyMCPServer`` with the FINAL tool
        # inventory — a related internal incident (analysis not included
        # here)) WITHOUT opening the CLI
        # session. The first real ``ainvoke`` computes
        # ``resume_session_id`` and the subsequent
        # ``get_or_create_client(...)`` adopts our reserved proxy and
        # finally connects WITH that resume.
        #
        # ⚠ DO NOT switch this back to ``get_or_create_client``. See
        # ``test_client_manager_prewarm_no_connect.py``.
        try:
            await client_manager.prepare_proxy_only(
                thread_id=thread_id,
                tools_metadata=tools_metadata,
                working_dir=working_dir,
            )
            _logger.info(
                f"[PREWARM] ClientManager proxy reserved for "
                f"thread_id={thread_id} with {len(tools_metadata)} tools "
                f"(no CLI connect — first ainvoke will resume)"
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                f"[PREWARM] prepare_proxy_only failed for "
                f"thread_id={thread_id} ({exc!r}); first real turn will "
                f"build the proxy from scratch. Agent start proceeds.",
                exc_info=True,
            )

    def _find_summarization_middleware(self) -> Any:
        """Locate the ``SummarizationMiddleware`` instance in the compiled graph.

        The startup pseudo-summary path (``--sum-query``) needs the live
        middleware instance to call ``apply_pseudo_summary``. The compiled
        LangGraph agent binds each middleware's ``before_model`` /
        ``abefore_model`` as a graph node whose bound callable's ``__self__``
        is the middleware instance. We walk the nodes and reach the instance
        via a bounded recursive descent through the node's runnable wrappers.

        Returns:
            The ``SummarizationMiddleware`` instance.

        Raises:
            RuntimeError: When no ``SummarizationMiddleware`` is present — a
                broken deep-agent build (see DESIGN_STARTUP_QUERY_OPTIONS.md
                §4.4). The caller cannot recover except by fixing the graph.
        """
        from deepagents.middleware.summarization import SummarizationMiddleware

        agent = self.agent
        if agent is None:
            msg = "Cannot resolve SummarizationMiddleware: agent not initialised."
            raise RuntimeError(msg)

        def _descend(obj: Any, depth: int, visited: set[int]) -> Any:
            if obj is None or depth > 6 or id(obj) in visited:
                return None
            visited.add(id(obj))
            if isinstance(obj, SummarizationMiddleware):
                return obj
            for attr in (
                "bound",
                "func",
                "afunc",
                "runnable",
                "_func",
                "node",
                "steps",
                "first",
                "__self__",
            ):
                child = getattr(obj, attr, None)
                if isinstance(child, (list, tuple)):
                    for item in child:
                        found = _descend(item, depth + 1, visited)
                        if found is not None:
                            return found
                elif child is not None:
                    found = _descend(child, depth + 1, visited)
                    if found is not None:
                        return found
            return None

        nodes = getattr(agent, "nodes", {}) or {}
        for node in nodes.values():
            found = _descend(node, 0, set())
            if found is not None:
                return found

        msg = (
            "No SummarizationMiddleware found in the compiled agent graph; "
            "cannot apply a startup pseudo-summary (--sum-query). This is a "
            "broken deep-agent build."
        )
        raise RuntimeError(msg)

    async def _run_startup_or_send_query(
        self,
        sum_query: str | None,
        query: str | None,
        *,
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
    ) -> None:
        """Shared offload→wipe→invoke sequence for startup and idle paths.

        This is the SINGLE implementation of the startup-query semantics,
        invoked both from ``_auto_run_startup_query`` (startup / restart) and
        from the extended ``exposed_invoke_agent`` (idle send-through), so the
        two routes cannot drift (DESIGN_STARTUP_QUERY_OPTIONS.md §4.5).

        Sequence when ``sum_query`` is set:
          1. Read ``current_messages`` from the checkpointer for
             ``thread_id == self.name`` via ``aget_state``.
          2. ``apply_pseudo_summary(self.name, sum_query, current_messages)``
             — offload to ``<agent>.jsonl`` (explicit thread_id, Blocker-1);
             pure producer, no wipe. Raises ``HistoryOffloadError`` on write
             failure BEFORE any wipe (atomicity).
          3. Wipe + install the summary-shape message via ``aupdate_state``
             with ``RemoveMessage(REMOVE_ALL_MESSAGES)`` (Blocker-2).
        Then a single ``_handle_invoke`` appends the plain ``query`` (if any)
        onto the freshly-installed state — one turn, two messages.

        Concurrency (§4.7): the ENTIRE offload→wipe→invoke sequence runs under
        ``_invoke_lock`` as ONE critical section, so it is serialised against
        any concurrent normal invoke. This matters for the idle send-through
        path (``exposed_invoke_agent`` on a LIVE agent): without the lock, the
        ``aget_state`` read could observe state a concurrent turn is mutating
        (stale read) and the ``aupdate_state`` wipe could clobber messages a
        concurrent turn just appended (lost update). Because ``_handle_invoke``
        ALSO acquires ``_invoke_lock`` (non-reentrant ``asyncio.Lock``), we
        must NOT call it here — we hold the lock ourselves and call
        ``_handle_invoke_owned`` directly (the already-factored lock-free body,
        agent_process.py:_handle_invoke_owned). Startup/restart has no
        concurrency, but sharing the one locked path keeps startup and idle
        identical (no drift).

        Args:
            sum_query: The pseudo-summary text, or ``None``.
            query: The plain query text, or ``None``.
            connection / notify_callback / approval_callback: forwarded to
                ``_handle_invoke_owned`` (all optional; startup passes ``None``).
        """
        from langchain_core.messages import HumanMessage, RemoveMessage
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        config: RunnableConfig = {"configurable": {"thread_id": self.name}}

        # Acquire the per-agent turn gate ONCE for the whole sequence. Mirror
        # the lazy-init used by _handle_invoke so focused tests that build
        # AgentProcess via __new__ still work.
        lock = getattr(self, "_invoke_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._invoke_lock = lock

        async with lock:
            if sum_query is not None:
                if self.agent is None:
                    msg = (
                        "Cannot run a startup sum-query: the agent graph is not "
                        "initialised."
                    )
                    raise RuntimeError(msg)
                sm = self._find_summarization_middleware()
                # current_messages source: the checkpointer for this thread.
                state = await self.agent.aget_state(config)
                current_messages = (
                    list(state.values.get("messages", []))
                    if state and getattr(state, "values", None)
                    else []
                )
                # Offload (explicit thread_id → <agent>.jsonl). Pure producer.
                new_msgs, _file_path = await sm.apply_pseudo_summary(
                    self.name, sum_query, current_messages
                )
                # Caller-owned single wipe: replace the whole message channel
                # with exactly the summary-shape message.
                await self.agent.aupdate_state(
                    config,
                    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_msgs]},
                )

            # Contract (DESIGN §4.6): when BOTH sum_query and query are given,
            # they are TWO SEQUENTIAL user requests — "first comes the
            # sum-query request, then immediately the query request". The
            # agent first
            # EXECUTES the sum-query (a full turn on the freshly-installed
            # summary-shape state, producing an AI response), and only THEN
            # executes the plain query as the next user message on top of that
            # result. NOT a single turn with [summary-shape, query] (the old
            # behaviour executed only the query and never ran the sum-query as
            # its own request — a related internal bug).
            #
            # Each turn is a distinct _handle_invoke_owned call. We call the
            # _owned variant (NOT _handle_invoke) because we already hold
            # _invoke_lock — the lock is non-reentrant. Both turns run under the
            # one lock we acquired, so the whole offload→wipe→sum-turn→query-turn
            # sequence stays a single serialised critical section.

            async def _invoke_turn(message_list: list[Any]) -> None:
                await self._handle_invoke_owned(
                    {
                        "message_list": message_list,
                        "stream": True,
                        "reply_back": False,
                    },
                    connection=connection,
                    notify_callback=notify_callback,
                    approval_callback=approval_callback,
                )

            # Turn 1 — execute the sum-query. The checkpointer already holds
            # exactly [summary-shape] (installed above), so an empty message_list
            # re-drives the graph on it: the agent responds to the summary-shape
            # message as its first user request.
            if sum_query is not None:
                await _invoke_turn([])

            # Turn 2 — execute the plain query as the next user message. When
            # sum_query ran first, this appends the query on top of turn 1's
            # result (summary-shape + AI response). When only query was given
            # (no sum_query), this is the sole turn and matches today's
            # single-query behaviour exactly.
            if query is not None:
                await _invoke_turn(
                    [HumanMessage(content=query, id=f"user-{uuid.uuid4().hex}")]
                )

    async def reindex_history_owned(self) -> dict[str, int]:
        from deep_worker_cli.summaries import get_summaries_dir

        lock = getattr(self, "_invoke_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._invoke_lock = lock

        async with lock:
            summaries_dir = get_summaries_dir(self.name)
            if not summaries_dir.exists():
                raise RuntimeError(
                    f"no summaries directory for agent '{self.name}' "
                    f"({summaries_dir})"
                )
            stats = await asyncio.to_thread(_clear_then_index_history, summaries_dir)
            return {
                "files": stats.files_processed,
                "lines": stats.lines_read,
                "chunks": stats.chunks_indexed,
                "errors": stats.errors,
            }

    async def request_summarization(
        self,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        from langchain_core.messages import RemoveMessage
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        if self.agent is None:
            raise RuntimeError("Cannot summarize: agent graph is not initialised.")
        lock = getattr(self, "_invoke_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._invoke_lock = lock
        config: RunnableConfig = {"configurable": {"thread_id": self.name}}
        async with lock:
            state = await self.agent.aget_state(config)
            current_messages = (
                list(state.values.get("messages", []))
                if state and getattr(state, "values", None)
                else []
            )
            middleware = self._find_summarization_middleware()
            result = await middleware.apply_requested_summary(
                self.name,
                current_messages,
                prompt,
            )
            replacement = result.replacement_messages
            if replacement is not None:
                await self.agent.aupdate_state(
                    config,
                    {
                        "messages": [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES),
                            *replacement,
                        ]
                    },
                )
            return {
                "summarized": bool(result.summarized),
                "history_path": result.history_path,
                "messages_before": len(current_messages),
                "messages_after": (
                    len(replacement)
                    if replacement is not None
                    else len(current_messages)
                ),
            }

    async def _auto_run_startup_query(self) -> None:
        """Self-initiate a startup query right after start()/restart.

        Symmetric to ``_auto_continue_after_restart``: if ``--sum-query`` /
        ``-q`` was supplied on this launch, the agent immediately begins
        executing it instead of sitting idle. One-shot: the pending fields
        are cleared here before running so a later restart never replays them.

        Offload failures during the sum-query path become ``HistoryOffloadError``
        which propagates through ``_handle_invoke``'s catch-all and transitions
        the agent to ``AgentStatus.ERROR`` (fail-loud). Any other unexpected
        error is swallowed and logged so a bad startup query never crashes the
        process — the user can always send a new message manually.
        """
        sum_query = self._pending_startup_sum_query
        query = self._pending_startup_query
        summarize = getattr(self, "_pending_summarize", False) is True
        summary_prompt = getattr(self, "_pending_summary_prompt", None)
        # Clear immediately (one-shot invariant).
        self._pending_startup_sum_query = None
        self._pending_startup_query = None
        self._pending_summarize = False
        self._pending_summary_prompt = None

        if summarize:
            await self.request_summarization(summary_prompt)
            return

        if sum_query is None and query is None:
            return

        try:
            _logger.info(
                f"[STARTUP-QUERY] Auto-running startup query for '{self.name}' "
                f"(sum_query={'yes' if sum_query else 'no'}, "
                f"query={'yes' if query else 'no'})"
            )
            await self._run_startup_or_send_query(
                sum_query,
                query,
                connection=None,
                notify_callback=None,
                approval_callback=None,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                f"[STARTUP-QUERY] Auto-run startup query failed: {exc!r}",
                exc_info=True,
            )

    @staticmethod
    def _log_startup_task_result(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:  # noqa: BLE001
            _logger.exception("Startup one-shot task failed")

    async def _auto_continue_after_restart(self) -> None:
        """Self-initiate an invoke right after a restart.

        After ``restart_self`` triggers a real process restart, the new
        process must NOT sit in idle waiting for the user to type
        something. The restart-notice that was just loaded into
        ``self._pending_restart_notice`` must be delivered to the model
        immediately so it can resume work.

        **Ordering invariant** (now satisfied by prewarm, not polling):
        ``start()`` calls ``_prewarm_client_manager`` immediately after
        ``create_cli_agent`` returns, which puts the FINAL-tool-inventory
        ``ProxyMCPServer`` into the ``ClientManager`` cache before this
        coroutine ever runs. The first ``ainvoke`` therefore hits the
        cache and inherits the correct tools — no waiting, no polling.

        Earlier versions of this method awaited
        ``_wait_for_mcp_tools_ready`` (a 30s polling barrier). That
        helper has been REMOVED — it returned True on the first non-
        empty ``tools_metadata``, which was the 7-tool ``web_tools``
        subset on restart, not the full 23-tool middleware-augmented
        inventory. See the related internal incident analysis (not included
        here). DO NOT reintroduce
        polling here; if the ordering ever breaks again, fix the
        prewarm in ``start()``, not this entry point.

        Errors here are swallowed and logged — failing the auto-continue
        must never crash the agent. The user can always send a new
        message manually if anything goes wrong.
        """
        try:
            _logger.info(
                f"[RESTART] Auto-continuing agent '{self.name}' "
                f"after restart (pending notice present)"
            )
            await self._handle_invoke(
                {
                    "message": (
                        "(automatic continuation after restart — "
                        "respond according to the SYSTEM notice above)"
                    ),
                    "stream": True,
                    "reply_back": False,
                },
                connection=None,
                notify_callback=None,
                approval_callback=None,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                f"[RESTART] Auto-continue after restart failed: {exc!r}"
            )

    async def _start_external_mcp_pool(self) -> Any:
        """Build & start the ExternalMCPClientPool for this agent.

        Layers (lowest → highest):
            1. ``~/.deepagents/mcp.json``
            2. ``<working_dir>/.deepagents/mcp.json``  (project)
            3. ``~/.deepagents/<agent>/mcp.json``      (per-agent)
            4. CLI ``--mcp-config`` / ``--mcp-server`` overrides

        Returns the started pool, or ``None`` if pool creation failed (in
        which case the agent boots without external MCP — degraded but
        functional, per design §3.4).
        """
        try:
            from deep_worker_cli.external_mcp.pool import (
                ExternalMCPClientPool,
            )
            from deep_worker_cli.external_mcp.config import UnresolvedEnvError
        except Exception as e:  # pragma: no cover — mcp SDK absent
            _logger.info(f"external MCP disabled (import failed): {e}")
            return None

        from pathlib import Path

        home = Path.home()
        user_path = home / ".deepagents" / "mcp.json"
        agent_path = home / ".deepagents" / self.name / "mcp.json"
        working_dir = getattr(self.config, "working_dir", None)
        project_path = None
        if working_dir:
            from deep_worker_cli.project_utils import find_project_root

            start_path = Path(working_dir)
            project_root = find_project_root(start_path) or start_path
            project_path = project_root / ".deepagents" / "mcp.json"
        cli_configs = list(getattr(self, "mcp_config_paths", []) or [])
        cli_servers = list(getattr(self, "mcp_server_inline", []) or [])
        try:
            pool = await ExternalMCPClientPool.create(
                user_path=user_path,
                project_path=project_path,
                agent_path=agent_path,
                cli_configs=cli_configs,
                cli_servers=cli_servers,
            )
            _logger.info(
                f"external MCP pool ready for {self.name}: "
                f"{len(pool.tools)} adapter tool(s)"
            )
            return pool
        except UnresolvedEnvError:
            raise
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                f"external MCP pool failed to start for {self.name}: {e}",
                exc_info=True,
            )
            return None

    async def cleanup(self) -> None:
        """Clean up resources, including checkpointer.

        This method should be called on agent shutdown to properly close
        database connections and release resources.
        """
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            except Exception as e:
                _logger.info(f"Warning: Error closing checkpointer: {e}")
            finally:
                self._checkpointer_ctx = None

        # External MCP pool — properly tear down stdio clients and child
        # processes. See docs/DESIGN_EXTERNAL_MCP_TOOLS.md §3.4.
        if self._external_mcp_pool is not None:
            try:
                await self._external_mcp_pool.stop()
            except Exception as e:  # noqa: BLE001
                _logger.info(f"Warning: Error stopping external MCP pool: {e}")
            finally:
                self._external_mcp_pool = None

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

        # 2.1 Stop A2A server (if running)
        if self._a2a_handle is not None:
            try:
                await self._a2a_handle.shutdown()
            except Exception as e:  # noqa: BLE001 — best-effort shutdown
                print(f"[A2A] Error stopping A2A server: {e}")
            finally:
                self._a2a_handle = None

        # 2.2 Stop the durable-inbox drain. Inbound acceptance already stopped
        # (RPyC server closed above), so no new records can arrive. The flag is
        # set BEFORE cancelling so the drain can distinguish shutdown from a
        # user interrupt without depending on timing; the drain cancels and
        # awaits its active child on the way out, leaving no orphan turn.
        # MUST run before cleanup() closes the checkpointer.
        # See DESIGN_DURABLE_INTER_AGENT_DELIVERY.md §3.3.4.
        self._shutting_down = True
        await self.stop_inbox_drain()

        # 3. Persist state
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.SHUTDOWN)

        # 4. Cleanup checkpointer
        await self.cleanup()

        # 5. Unregister
        await self._unregister()

        # Note: the legacy deferred-restart mechanism (spawning
        # deep-worker --restart from inside shutdown) was removed.
        # ``restart_self`` now spawns the restart subprocess synchronously
        # at call time, so by the time we reach shutdown the new process
        # is already running independently.

        # 7. Cleanup files
        await self._remove_pid_file()

        print(f"Agent '{self.name}' stopped")

    async def _maybe_start_a2a_server(self) -> None:
        """Start the optional A2A protocol server (RC-6/RC-7).

        Gated on ``A2AConfig.from_env(...).enabled``. A missing optional
        ``deepworker_a2a`` package or ``a2a-sdk`` dependency, or any startup
        error, is logged and swallowed — A2A must never crash the agent. On
        success ``self._a2a_handle`` is set so ``_register`` can persist
        ``a2a_port``/``a2a_url`` atomically with the rest of the registration.
        """
        try:
            from deepworker_a2a.config import A2AConfig
            from deepworker_a2a.server import maybe_start_a2a_server
        except Exception:  # noqa: BLE001 — optional dependency / package
            _logger.info("deepworker_a2a not importable; A2A disabled")
            return

        try:
            config = A2AConfig.from_env(self.name)
        except Exception:  # noqa: BLE001 — bad env config must not be fatal
            _logger.exception("Invalid A2A config from env; A2A disabled")
            return

        if not config.enabled:
            return

        self._a2a_handle = await maybe_start_a2a_server(
            self, config, agent_name=self.name
        )
        if self._a2a_handle is not None:
            print(f"{_ts()} A2A server: {self._a2a_handle.url}")

    async def _register(self) -> None:
        """Register agent in ProcessRegistry."""
        if self.started_at is None:
            self.started_at = datetime.now()

        # RPyC port is required for communication
        if self.rpyc_port is None:
            raise RuntimeError("RPyC port must be configured - RPyC is required for agent communication")

        # A2A endpoint (RC-6): written atomically with the rest of the
        # registration so a peer never sees the agent without its A2A URL.
        a2a_port: int | None = None
        a2a_url: str | None = None
        if self._a2a_handle is not None:
            a2a_url = getattr(self._a2a_handle, "url", None)
            a2a_port = getattr(self._a2a_handle, "port", None)

        process_info = AgentProcessInfo(
            name=self.name,
            pid=os.getpid(),
            pid_file=self.pid_file,
            working_dir=str(self.config.working_dir),
            started_at=self.started_at,
            rpyc_port=self.rpyc_port,  # RPyC port (REQUIRED for communication)
            a2a_port=a2a_port,
            a2a_url=a2a_url,
        )
        self.registry.register(process_info)

    # ================================================================
    # Durable inter-agent inbox
    # See docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md (signed off, r6) and
    # a related internal incident analysis (not included here).
    # ================================================================

    def bring_up_inbox(self) -> int:
        """Production startup wiring for the durable inbox — ONE entry point.

        Performs, in order:

        1. eager spool creation + ``.tmp-*`` cleanup + ``next_seq`` priming
           (§3.2.1), which also arms the startup edge if records survived a
           previous life;
        2. creation of the single long-lived drain consumer (§3.3.4).

        This exists as a named hook so that an end-to-end test can exercise
        the REAL startup path instead of re-implementing it. A test that calls
        ``init_inbox_spool()`` and ``start_inbox_drain()`` by hand would stay
        green even if ``start()`` stopped calling them — the wiring would be
        unproven. Everything here must be safe to call once the agent graph
        and checkpointer are ready; it is idempotent with respect to the drain
        task.

        Returns the number of pending records recovered from a previous life.

        A broken spool must not stop the agent from starting: ``OSError`` is
        logged and swallowed here. The counter is then left UNPRIMED, and
        ``publish_inbox_message`` retries the prime inside its lock before
        allocating — so a sender either gets a correctly-ordered record or an
        explicit ``success: false``, never a silently misordered seq 0.
        """
        recovered = 0
        try:
            recovered = self.init_inbox_spool()
        except OSError as exc:
            _logger.exception("[INBOX] could not initialise spool: %s", exc)
        self.start_inbox_drain()
        return recovered

    def init_inbox_spool(self) -> int:
        """Create the spool eagerly and prime the seq counter (§3.2.1).

        Called during startup BEFORE any enqueue can happen, so the per-record
        fsync chain in ``inbox_spool.publish`` is sufficient on its own and
        there is no create-race between concurrent first enqueues.

        Priming is ALL-OR-NOTHING: ``_inbox_seq_primed`` is set only after a
        full successful scan. A partially-failed prime must never leave the
        counter at its ``0`` default while records with higher seqs survive on
        disk — that would publish a new arrival *ahead* of them and break the
        ``next_seq = max(pending ∪ quarantined) + 1`` contract. Any ``OSError``
        propagates with the flag left False so the enqueue path can retry or
        refuse loudly.

        Idempotent and safe to call twice (once before the RPyC server starts,
        once from :meth:`bring_up_inbox`). Once primed it will NOT rescan or
        reset a live counter, since by then the server may already have
        allocated seqs.

        Returns the number of pending records recovered from a previous life.
        """
        from deep_worker_cli.process import inbox_spool

        inbox_spool.ensure_spool(self.data_dir)
        if not self._inbox_seq_primed:
            inbox_spool.cleanup_tmp(self.data_dir)
            # Assign only after the scan succeeds — an exception here must
            # leave both the counter and the flag untouched.
            next_seq = inbox_spool.next_seq(self.data_dir)
            self._inbox_next_seq = next_seq
            self._inbox_seq_primed = True
        pending = inbox_spool.pending_records(self.data_dir)
        if pending:
            # Startup edge (§3.4): anything durable from before the restart is
            # processed now, without waiting for a new arrival.
            self._inbox_event.set()
        return len(pending)

    async def publish_inbox_message(self, message: str, reply_back: bool) -> str:
        """Durably publish an inbound message and wake the drain.

        This is the honest-ack primitive: it returns only after the record is
        fsynced, so the caller may acknowledge the send. Ordering is strict —
        publish durably, THEN set the event, THEN let the caller ack.

        FAIL CLOSED on an unprimed counter: if the startup scan failed (it is
        logged and swallowed so a broken spool cannot stop the agent from
        booting), we prime here, inside the lock, BEFORE allocating. If that
        still fails the exception propagates and the sender is told
        ``success: false``. Allocating against the unprimed default would
        publish this arrival as seq 0 while records with higher seqs survive
        on disk — silently reordering it ahead of them while reporting
        success, which is precisely the class of dishonest ack this whole
        change removes.

        Returns the record's ``delivery_id``.

        Raises:
            OSError: the spool could not be primed or written.
        """
        from deep_worker_cli.process import inbox_spool

        record = inbox_spool.build_record(message, reply_back)
        # Allocate+publish under a dedicated short-lived lock (§3.1.1). This is
        # NOT _invoke_lock: it is held for microseconds and never across a turn.
        async with self._inbox_seq_lock:
            if not self._inbox_seq_primed:
                # Late prime; propagates on failure (no allocation happens).
                self.init_inbox_spool()
            seq = self._inbox_next_seq
            self._inbox_next_seq = seq + 1
            inbox_spool.publish(self.data_dir, record, seq)
        # Strictly after durability.
        self._inbox_event.set()
        return str(record["delivery_id"])

    def set_inbox_notify(self, notify_callback: Any) -> None:
        """Inject the subscriber-fanout callback (§3.3.3).

        Assignment ONLY — this never creates a task, so it is safe to call at
        any time, including long after the drain started. The drain re-reads
        ``_inbox_notify`` on every record, so a late injection is picked up by
        the next delivery without restarting anything.
        """
        self._inbox_notify = notify_callback

    def start_inbox_drain(self) -> Any:
        """Start the single long-lived drain consumer (§3.3.4).

        Guarded: a second call is a no-op. Takes no callback argument — that is
        ``set_inbox_notify``'s job — so this guard can never silently discard a
        callback.
        """
        if self._inbox_drain_task is not None and not self._inbox_drain_task.done():
            return self._inbox_drain_task
        self._inbox_drain_task = asyncio.create_task(self._inbox_drain_loop())
        return self._inbox_drain_task

    async def _inbox_drain_loop(self) -> None:
        """Consume the spool. Event-driven; no polling, no timed wakeups.

        Protocol (§3.4.1) — ``clear()`` strictly BEFORE reading the directory::

            await event.wait(); event.clear(); drain snapshot

        There is deliberately NO self-rearm on pending entries: combined with
        "a failed record stays on disk" that would spin forever on a
        deterministically failing record (§3.3.1a). Re-arming is solely the
        producer's job, plus the startup edge.
        """
        while True:
            await self._inbox_event.wait()
            self._inbox_event.clear()
            try:
                await self._drain_inbox_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - defensive per-pass guard
                # An error in the scan/snapshot machinery itself must not kill
                # the only consumer while durable records are pending. No
                # re-arm here either — otherwise a reproducible exception is
                # the same hot loop we just removed.
                _logger.exception("[INBOX] drain pass failed")

    async def _drain_inbox_once(self) -> None:
        """One pass over a snapshot of the spool.

        Each record is attempted AT MOST ONCE per wake (§3.3.1a). A retained
        failure is not re-read within the pass and does not re-arm the event;
        it is retried on the next arrival or at startup.
        """
        from deep_worker_cli.process import inbox_spool

        for path in inbox_spool.pending_records(self.data_dir):
            record = inbox_spool.read_record(path)
            if record is None:
                inbox_spool.quarantine(self.data_dir, path)
                continue
            await self._deliver_inbox_record(path, record)

    async def _deliver_inbox_record(
        self, path: Path, record: dict[str, Any]
    ) -> None:
        """Run one queued message as a turn, then apply the delete boundary.

        The turn runs in its own SHORT-LIVED CHILD TASK calling the PUBLIC
        ``_handle_invoke`` (§3.3.0b). This is load-bearing:

        * ``_handle_invoke_owned`` sets ``current_worker_task =
          asyncio.current_task()``, and ``exposed_interrupt_agent`` cancels
          exactly that task. Running the turn inline would make the drain
          itself the interrupt target, so a routine user interrupt would kill
          the only consumer permanently — a silent processing outage.
        * ``_handle_invoke`` (public) acquires ``_invoke_lock`` itself, so the
          drain holds no lock and there is no recursion/deadlock.
        """
        delivery_id = record.get("delivery_id")
        started = time.monotonic()
        child = asyncio.create_task(self._handle_invoke(
            {
                "command": "invoke",
                "message": record["message"],
                "stream": True,
                "reply_back": record.get("reply_back", True),
                "inter_agent_delivery": True,
                "delivery_id": delivery_id,
            },
            connection=None,
            # Re-read per record so a late set_inbox_notify is honoured.
            notify_callback=self._inbox_notify,
            approval_callback=None,
        ))
        try:
            result = await child
        except asyncio.CancelledError:
            current = asyncio.current_task()
            drain_cancelled = self._shutting_down or (
                current is not None and current.cancelling() > 0
            )
            if drain_cancelled:
                # Shutdown: cancel and await the child so no orphan turn
                # survives into checkpointer teardown, then propagate.
                child.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await child
                raise
            # User interrupt cancelled the TURN, not us. Absorb it, retain the
            # record, stay alive (§3.3.0b).
            _logger.info(
                "[INBOX] delivery %s interrupted; record retained", delivery_id
            )
            return
        except Exception:  # noqa: BLE001 - per-record containment (§3.3.4)
            # One broken record must not strand the rest of the batch.
            _logger.exception(
                "[INBOX] delivery %s failed; record retained", delivery_id
            )
            return

        # DELETE BOUNDARY — reached only on a normal return of the await above.
        # Never in a `finally`: a cancelled or failed turn must keep its record.
        if isinstance(result, dict) and result.get("success"):
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best effort
                _logger.warning("[INBOX] could not unlink delivered %s", path)
            _logger.info(
                "[INBOX] delivered id=%s arrival=%s in %.3fs",
                delivery_id, record.get("ts"), time.monotonic() - started,
            )
        else:
            _logger.warning(
                "[INBOX] delivery %s did not succeed; record retained for retry",
                delivery_id,
            )

    async def stop_inbox_drain(self) -> None:
        """Cancel and await the drain before the checkpointer closes (§3.3.4).

        ``_shutting_down`` is set by the caller BEFORE cancelling so the drain
        can tell shutdown from a user interrupt without depending on timing.
        """
        task = self._inbox_drain_task
        if task is None:
            return
        self._inbox_drain_task = None
        if task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def cancel_current_worker_task(self) -> bool:
        """Interrupt the currently-running agent turn, if any (RC-1).

        Thin, in-process wrapper around the same mechanism used by
        ``AgentService.exposed_interrupt_agent`` (``is_running`` +
        ``current_worker_task.cancel()``) — but with NO RPyC/netref involved,
        so the A2A executor's ``cancel()`` can call it directly on the local
        event loop.

        Returns:
            True if there was an active, not-yet-finished turn that we asked to
            cancel; False if no turn was running (nothing to interrupt).
        """
        if not getattr(self, "is_running", False):
            return False
        task = getattr(self, "current_worker_task", None)
        if task is None or task.done():
            return False
        task.cancel()
        return True

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
        forever still serving RPyC (stop_agent then timed out / kill -9 needed).
        Setting the event is what actually unblocks shutdown.
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
                # — the agent keeps working.
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
        # call (see the related internal incident analysis, not included
        # here). Keep ONLY
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
            import sqlite3
            import aiosqlite
            if isinstance(e, (aiosqlite.DatabaseError, sqlite3.DatabaseError)):
                import traceback
                _logger.info(f"[GET_HISTORY] Database corruption detected for agent "
                    f"'{self.name}': {e}")
                _logger.info(f"[GET_HISTORY] Run 'deep-worker --repair {self.name}' or "
                    f"delete the agent's threads.db file to recover.")
                _logger.info(f"[GET_HISTORY] Traceback:\n{traceback.format_exc()}")
                return {
                    "success": False,
                    "error": f"Database corrupted: {e}",
                }
            import traceback
            _logger.info(f"[GET_HISTORY] Exception: {e}")
            _logger.info(f"[GET_HISTORY] Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to retrieve history: {str(e)}",
            }

    async def _handle_get_system_prompt(self) -> dict[str, Any]:
        """Return the INVARIANT (startup) system prompt the agent was built with.

        Mode #4 of the Full Dialog verbosity combo (§9 of the Full Dialog
        design doc).

        The invariant prompt is the provider-AGNOSTIC role/instruction text
        passed to ``create_deep_agent(system_prompt=...)``. It is captured at
        build time into ``self._invariant_system_prompt`` so the reported value
        is EXACTLY what the agent was constructed with. If the attribute is
        missing (e.g. a legacy process that predates this feature), recompute
        it deterministically via ``get_system_prompt`` — it is a pure function
        of agent configuration.
        """
        provider = getattr(getattr(self, "config", None), "provider", None)
        model = getattr(getattr(self, "config", None), "model", None)
        system_prompt = getattr(self, "_invariant_system_prompt", None)
        if not isinstance(system_prompt, str) or not system_prompt:
            try:
                from pathlib import Path

                from deep_worker_cli.agent import get_system_prompt
                working_dir = getattr(
                    getattr(self, "config", None), "working_dir", None
                )
                project_memory_dir = Path(working_dir) if working_dir else None
                system_prompt = get_system_prompt(
                    assistant_id=self.name,
                    sandbox_type=None,
                    project_memory_dir=project_memory_dir,
                    provider=getattr(
                        getattr(self, "config", None), "provider", None
                    ),
                )
            except Exception as e:  # noqa: BLE001
                return {"success": False, "error": f"system prompt unavailable: {e}"}
        return {
            "success": True,
            "system_prompt": system_prompt,
            "provider": provider,
            "model": model,
        }

    async def _handle_get_tools(self) -> dict[str, Any]:
        """Return the agent's available tool list (name + description).

        Mode #5 of the Full Dialog verbosity combo (§9). The tool inventory is
        read from the compiled LangGraph agent via the existing
        ``_extract_final_tool_metadata`` helper — the SAME inventory the model
        binds on every turn (single source of truth). The list MAY depend on
        the active provider. Each metadata entry is reduced to name +
        description; the JSON input schema is intentionally omitted from this
        view. Metadata entries may be dicts or ToolMetadata-like objects.
        """
        provider = getattr(getattr(self, "config", None), "provider", None)
        try:
            metadata = type(self)._extract_final_tool_metadata(self.agent)
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"tools unavailable: {e}"}

        tools: list[dict[str, str]] = []
        for m in metadata or []:
            if isinstance(m, dict):
                name = m.get("name", "?")
                desc = m.get("description", "")
            else:
                name = getattr(m, "name", "?")
                desc = getattr(m, "description", "")
            tools.append(
                {
                    "name": str(name),
                    "description": str(desc) if desc is not None else "",
                }
            )
        return {"success": True, "provider": provider, "tools": tools}

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
            # Specific handling for database corruption — module-level
            # imports avoid UnboundLocalError (sqlite3/aiosqlite are
            # already imported at the top of this file).
            if isinstance(e, (sqlite3.DatabaseError, __import__("aiosqlite").DatabaseError)):
                logger.warning(
                    "Database corruption detected in checkpointer DB for "
                    "agent '%s': %s. Run 'deep-worker --repair %s' or delete "
                    "the agent's threads.db file to recover.",
                    self.name, e, self.name,
                )
                return {"success": False, "error": f"Database corrupted: {e}"}
            logger.error(f"Error getting last response: {e}")
            return {"success": False, "error": f"Failed to get last response: {str(e)}"}

    # ------------------------------------------------------------------
    #    Failover scheduling (§7.1 of DESIGN_PROVIDER_FAILOVER)
    # ------------------------------------------------------------------
    async def _schedule_failover(self, provider: str, marker_text: str) -> None:
        """Record ``provider`` failure in availability tracker and, if
        a next available provider exists in the failover chain, schedule
        a restart via ``deep-worker --restart <name> --provider <next>
        --no-update-preferred``.

        Fire-and-forget: runs as a background task so the caller's error
        response is not delayed.

        Args:
            provider: The provider that hit the usage limit
                (e.g. ``"claude-code"``, ``"codex"``).
            marker_text: The raw error marker from the provider's error
                event, which may contain a human-readable reset time.
        """
        from deep_worker_cli.runtime.availability_tracker import (  # noqa: PLC0415
            AvailabilityTracker,
            parse_limit_reset,
        )
        from deep_worker_cli.runtime.provider_chain import (  # noqa: PLC0415
            PROVIDER_FAILOVER_CHAIN,
        )

        try:
            reset = parse_limit_reset(provider, marker_text)
            tracker = AvailabilityTracker()
            was_available = tracker.record_failure(
                provider=provider,
                marker_text=marker_text,
                agent_name=self.name,
                available_after_utc=reset.available_after_utc,
                available_after_source=reset.source,
                available_after_note=reset.note,
            )

            if provider == "deepseek":
                # Deepseek is terminal — never failover.
                return

            # Find current position in failover chain
            try:
                current_idx = PROVIDER_FAILOVER_CHAIN.index(provider)
            except ValueError:
                # Unknown provider — don't attempt failover.
                return

            # Advance to the next available provider
            for candidate in PROVIDER_FAILOVER_CHAIN[current_idx + 1:]:
                if tracker.is_available(candidate):
                    _logger.info(f"[FAILOVER] Agent '{self.name}' failed on {provider}; "
                        f"scheduling restart with {candidate} …")
                    # Write the restart-notice BEFORE spawning so the new
                    # process consumes it on startup and
                    # ``_auto_continue_after_restart`` CONTINUES the pending
                    # task instead of going idle (a related internal incident
                    # analysis (not included here)). Same notice format as
                    # ``restart_self``.
                    try:
                        from deep_worker_cli.tools import write_restart_notice  # noqa: PLC0415
                        write_restart_notice(
                            self.data_dir,
                            cause="failover",
                            message=(
                                f"Agent '{self.name}' hit a usage limit on "
                                f"provider '{provider}' and was automatically "
                                f"restarted on '{candidate}'. Continue the "
                                f"task you were working on."
                            ),
                        )
                    except OSError as notice_exc:
                        _logger.info(
                            f"[FAILOVER] Could not write restart notice for "
                            f"'{self.name}': {notice_exc!r} (agent will restart "
                            f"but may sit idle instead of auto-continuing)"
                        )
                    proc = await asyncio.create_subprocess_exec(
                        "deep-worker",
                        "--restart",
                        self.name,
                        "--provider",
                        candidate,
                        "--no-update-preferred",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    # Detach — do NOT await. The restart will kill this process.
                    asyncio.ensure_future(proc.wait())
                    return

            _logger.info(f"[FAILOVER] Agent '{self.name}' failed on {provider}; "
                f"no available provider in the failover chain — "
                f"agent stays on {provider}.")

        except Exception as exc:
            _logger.info(f"[FAILOVER] Error scheduling failover for '{self.name}' "
                f"(provider={provider}): {exc!r}")

    async def _append_recoverable_error_to_history(
        self,
        *,
        exc_name: str,
        exc_repr: str,
        input_message_id: str | None,
    ) -> None:
        """Append a SystemMessage describing a recoverable invoke failure.

        Implements the No-Agent-Error-State Policy (docs/POLICIES.md §1):
        an exception that does NOT belong to FATAL_EXCEPTION_TYPES is
        converted into a SystemMessage so the model sees it on the next
        turn and the conversation can simply continue.

        Uses the agent's existing checkpointer via ``aupdate_state`` so
        the new message is persisted exactly like any other graph state
        update — no manual file IO, no new threads, no polling.
        """
        from langchain_core.messages import SystemMessage

        if self.agent is None:
            # Cannot persist without a graph — caller already settled to
            # IDLE; the user will simply see the notify_callback payload.
            return

        content = (
            f"[runtime] The previous turn failed with an internal error "
            f"({exc_name}: {exc_repr}).  The error was caught by the "
            f"agent runtime so the conversation can continue.  Reply to "
            f"the user's last request, or wait for them to send the next "
            f"message."
        )
        ref = input_message_id or "unknown"
        sys_msg = SystemMessage(
            content=content,
            additional_kwargs={
                "recoverable_error": True,
                "exc_name": exc_name,
                "input_message_id": ref,
            },
        )

        # Use the thread_id we already maintain for the agent.
        config = {"configurable": {"thread_id": getattr(self, "thread_id", self.name)}}
        try:
            await self.agent.aupdate_state(config, {"messages": [sys_msg]})
        except Exception:
            # We never re-raise from the recoverable path — see policy.
            _logger.exception(
                "[INVOKE] aupdate_state failed for recoverable error "
                "(agent=%s, exc_name=%s)",
                self.name,
                exc_name,
            )

    async def _handle_invoke(
        self,
        command: dict[str, Any],
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
        thread_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Serialize complete top-level turns for this agent thread.

        ``thread_id_override`` (optional): when provided, this string is used as
        the LangGraph checkpointer ``thread_id`` instead of the default
        ``self.name``. Used by the A2A executor to isolate each A2A
        ``context_id`` into its own conversation history (see
        design documentation §4.1). ``None`` (default) preserves the existing
        behaviour exactly, so all current callers are unaffected.
        """
        lock = getattr(self, "_invoke_lock", None)
        if lock is None:
            # Some focused tests construct AgentProcess via __new__. Production
            # instances always create the lock in __init__.
            lock = asyncio.Lock()
            self._invoke_lock = lock
        async with lock:
            return await self._handle_invoke_owned(
                command,
                connection=connection,
                notify_callback=notify_callback,
                approval_callback=approval_callback,
                thread_id_override=thread_id_override,
            )

    async def _handle_invoke_owned(
        self,
        command: dict[str, Any],
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
        thread_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Handle one invocation while holding the per-agent turn gate.

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
        # message_list: pre-built list of messages for a single invoke. Used by
        # the startup-query path (_run_startup_or_send_query) to feed
        # [summary-shape, query] as one turn. An EMPTY list is valid — it means
        # "drive the graph from the already-checkpointed state" (sum-query-only,
        # where the summary-shape message was installed via aupdate_state).
        # See DESIGN_STARTUP_QUERY_OPTIONS.md §4.6.
        message_list = command.get("message_list")
        uses_message_list = message_list is not None
        stream = command.get("stream", True)  # Default to streaming to match deepagents
        inter_agent_delivery = bool(command.get("inter_agent_delivery", False))
        reply_back = command.get("reply_back", True)  # For inter-agent messaging
        # Stable id of the durable spool record this turn came from, when any.
        delivery_id = command.get("delivery_id")
        input_message_id: str | None = None
        last_error_messages: list[Any] = []

        if not uses_message_list and not message:
            return {
                "success": False,
                "error": "No message provided",
            }

        # Derive a display/label string for notify_callback / TaskManager /
        # initial_request when running the message_list path. Prefer the last
        # message's content (the plain query); fall back to a synthetic marker
        # when the list is empty (sum-query-only start).
        if uses_message_list:
            if message_list:
                _last = message_list[-1]
                _last_content = getattr(_last, "content", None)
                message = _last_content if isinstance(_last_content, str) else str(_last_content)
            else:
                message = "(starting from summarized context)"

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
                    extra_kwargs: dict[str, Any] = {}
                    if inter_agent_delivery:
                        extra_kwargs = {
                            "inter_agent_delivery": True,
                            "reply_back": reply_back,
                        }
                        # Stable delivery identity, so a duplicate produced by
                        # a crash between checkpoint and unlink can be linked
                        # back to its original spool record (DESIGN §3.3.1).
                        if delivery_id is not None:
                            extra_kwargs["delivery_id"] = delivery_id
                    return HumanMessage(
                        content=content,
                        id=input_message_id,
                        additional_kwargs=extra_kwargs,
                    )

                def _build_invoke_messages(content: str) -> list:
                    """Build the messages list for this invoke.

                    If a restart-notice is pending (set on startup by
                    ``_load_and_consume_restart_notice``), prepend a
                    SystemMessage informing the model of the restart.
                    The notice is single-shot: it is cleared after
                    being emitted so subsequent invokes do not
                    re-deliver it.

                    When the command supplied a ``message_list`` (startup
                    query path), those messages are used verbatim as the base
                    instead of a single freshly-built HumanMessage. An empty
                    ``message_list`` yields ``[]`` (a no-op for add_messages),
                    driving the graph from the already-checkpointed state.
                    See DESIGN_STARTUP_QUERY_OPTIONS.md §4.6.
                    """
                    base = list(message_list) if uses_message_list else [_next_user_message(content)]
                    # Defensive: tests may construct AgentProcess without
                    # going through __init__ (MagicMock); fall back to
                    # "no pending notice" so the absence of the attribute
                    # never breaks the invoke path.
                    notice = getattr(self, "_pending_restart_notice", None)
                    if notice is None:
                        return base
                    sys_msg = self._format_restart_notice_message(notice)
                    self._pending_restart_notice = None
                    return [sys_msg, *base]

                # Create status tracking callback
                status_callback = StatusTrackingCallback(session_manager=self.session_manager)

                # Configure thread_id for checkpointer. Defaults to the agent
                # name (one shared history); an A2A turn passes
                # thread_id_override to isolate its context_id into a separate
                # conversation history (A2A design documentation §4.1).
                _thread_id = thread_id_override or self.name
                config: RunnableConfig = {
                    "configurable": {"thread_id": _thread_id},
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
                            _logger.info(f"[STREAM] TaskManager iteration {iteration}")

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
                            {"messages": _build_invoke_messages(current_message)},
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
                                _logger.info(f"[STREAM] Skipping TaskManager: outgoing inter-agent message")
                                break

                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=full_response,
                                send_to_subordinate_callback=None
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                _logger.info(f"[STREAM] TaskManager evaluation complete (done/unable)")
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
                                    _logger.info(f"[STREAM] Warning: Failed to notify subscribers of TaskManager prompt: {e}")

                            # Continue with corrective prompt
                            _logger.info(f"[STREAM] TaskManager requesting corrections: {corrective_prompt[:100]}...")
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

                    # Note: the legacy deferred-restart mechanism (flag
                    # file polled here) was removed. ``restart_self`` is
                    # now synchronous — it spawns ``deep-worker --restart``
                    # at call time, which SIGTERMs this process. No
                    # post-turn flag detection is needed.

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
                            _logger.info(f"[INVOKE] TaskManager iteration {iteration}")

                            # Invoke agent
                            result = await self.agent.ainvoke(
                                {"messages": _build_invoke_messages(current_message)},
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
                                _logger.info(f"[INVOKE] Skipping TaskManager: outgoing inter-agent message")
                                break

                            # Send to TaskManager for evaluation
                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=content,
                                send_to_subordinate_callback=None  # Not used in loop approach
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                _logger.info(f"[INVOKE] TaskManager evaluation complete (done/unable)")
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
                            _logger.info(f"[INVOKE] TaskManager requesting corrections: {corrective_prompt[:100]}...")
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
                                    _logger.info(f"[INVOKE_DEBUG] log_token_usage: rm={rm}")
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
                            # "forever working" bug).
                            # No-Agent-Error-State Policy: an empty model
                            # response is recoverable — append a hint and
                            # settle to IDLE so the user can just type
                            # "continue".  See docs/POLICIES.md §1.
                            if self.session_manager:
                                _log_error_context(
                                    error="No messages returned from agent",
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                    recoverable=True,
                                )
                                try:
                                    await self._append_recoverable_error_to_history(
                                        exc_name="EmptyModelResponse",
                                        exc_repr="The model returned no messages.",
                                        input_message_id=input_message_id,
                                    )
                                except Exception as hist_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to persist empty-response hint: %s",
                                        hist_err,
                                    )
                                await self.session_manager.set_status(AgentStatus.IDLE)
                            if notify_callback is not None:
                                try:
                                    await notify_callback({
                                        "type": "system",
                                        "role": "system",
                                        "content": (
                                            "The model returned an empty response; "
                                            "send 'continue' to retry."
                                        ),
                                    })
                                except Exception as notify_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to notify empty response: %s",
                                        notify_err,
                                    )
                            return {
                                "success": False,
                                "recovered": True,
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
                            {"messages": _build_invoke_messages(message)},
                            config
                        )
                        _logger.info(f"[INVOKE] Initial ainvoke: Got result with {len(result.get('messages', []))} messages")

                        while iteration < max_iterations:
                            iteration += 1

                            messages = result.get("messages", [])
                            last_error_messages = list(messages) if isinstance(messages, list) else []
                            if not messages:
                                _logger.info(f"[INVOKE] No messages in result")
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

                                _logger.info(f"[INVOKE] Iteration {iteration}: Broadcasting {len(new_messages)} new messages (total={len(messages)}, last_broadcasted={last_broadcasted})")

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

                            _logger.info(f"[INVOKE] Iteration {iteration}: content='{content[:100] if content else '(empty)'}', role={role}, has_tool_calls={has_tool_calls}, has_interrupt={has_interrupt}")

                            # If we have real content and it's from assistant, this is final response.
                            # CRITICAL: Must check role in ('assistant', 'ai') to avoid treating tool results as final response!
                            # LangGraph uses 'ai' for AIMessage.type, while dicts may use 'assistant'
                            # NOTE: do NOT require ``not has_tool_calls`` — the agent may produce text
                            # thinking alongside tool calls (issue #51). The text IS a response; tool
                            # calls without interrupt are handled below (break + continue to the
                            # no-final-content check which sets final_content and returns success).
                            if content and role in ('assistant', 'ai'):
                                final_content = content
                                _logger.info(f"[INVOKE] Got final response: {content[:100]}...")
                                break

                            # If there's an interrupt, handle it properly:
                            # 1. Get approval decision (via callback or auto-approve)
                            # 2. Resume with Command(resume=...) — do NOT re-send HumanMessage
                            if has_interrupt:
                                interrupts = result["__interrupt__"]
                                _logger.info(f"[INVOKE] Interrupt detected: {len(interrupts)} pending approvals")

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
                                    _logger.info(f"[INVOKE] Auto-approving {len(interrupts)} interrupts (no approval callback)")

                                # Resume graph with decisions (NOT with HumanMessage!)
                                from langgraph.types import Command
                                resume_input = Command(resume=hitl_response)
                                result = await self.agent.ainvoke(resume_input, config)
                                _logger.info(f"[INVOKE] Resumed after interrupt, got {len(result.get('messages', []))} messages")

                                # Re-evaluate the new result (loop continues from evaluation)
                                continue

                            # If we have tool calls but no interrupt, they should have been
                            # executed by ainvoke already. Break to avoid re-sending.
                            if has_tool_calls:
                                _logger.info(f"[INVOKE] Tool calls detected without interrupt — ainvoke should have handled them")
                                break

                            # If we got here with empty content and no tool calls,
                            # the agent finished but has nothing to say (shouldn't happen normally)
                            if not content:
                                _logger.info(f"[INVOKE] Empty response with no tool calls at iteration {iteration}")
                                break

                        if iteration >= max_iterations:
                            _logger.warning(
                                "[INVOKE] Max iterations reached without final response"
                            )
                            # No-Agent-Error-State Policy: hitting the
                            # iteration cap is recoverable — append a
                            # hint and settle IDLE so the user can say
                            # "continue".  See docs/POLICIES.md §1.
                            if self.session_manager:
                                _log_error_context(
                                    error="Agent exceeded maximum iterations without producing final response",
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                    recoverable=True,
                                )
                                try:
                                    await self._append_recoverable_error_to_history(
                                        exc_name="MaxIterationsExceeded",
                                        exc_repr=(
                                            "The agent hit its per-turn iteration cap "
                                            "without producing a final response."
                                        ),
                                        input_message_id=input_message_id,
                                    )
                                except Exception as hist_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to persist max-iterations hint: %s",
                                        hist_err,
                                    )
                                await self.session_manager.set_status(AgentStatus.IDLE)
                            if notify_callback is not None:
                                try:
                                    await notify_callback({
                                        "type": "system",
                                        "role": "system",
                                        "content": (
                                            "The agent hit its iteration cap. "
                                            "Send 'continue' to resume."
                                        ),
                                    })
                                except Exception as notify_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to notify max-iterations: %s",
                                        notify_err,
                                    )
                            return {
                                "success": False,
                                "recovered": True,
                                "error": "Agent exceeded maximum iterations without producing final response",
                            }

                        if not final_content:
                            # Check if the last message had tool calls — the agent DID produce
                            # a response (tool requests) but no text content (issue #51).
                            _has_unhandled_tool_calls = False
                            _tool_call_count = 0
                            if last_error_messages:
                                _last = last_error_messages[-1]
                                if isinstance(_last, dict):
                                    _tc = _last.get("tool_calls", [])
                                else:
                                    _tc = getattr(_last, "tool_calls", []) or []
                                if _tc:
                                    _has_unhandled_tool_calls = True
                                    _tool_call_count = len(_tc)
                                    _logger.info(f"[INVOKE] No text content but last message has "
                                        f"{_tool_call_count} tool calls (type={_message_attr(_last, 'type')})")

                            if _has_unhandled_tool_calls:
                                # The agent produced tool calls but no text — more accurate than
                                # "did not produce a response" (issue #51).
                                _err = f"Agent produced {_tool_call_count} tool call(s) but no text response"
                            else:
                                _err = "Agent did not produce a response"

                            _logger.warning(
                                "[INVOKE] No final content found after %d iterations",
                                iteration,
                            )
                            # No-Agent-Error-State Policy: missing text
                            # content is recoverable — settle IDLE and
                            # surface a SystemMessage so the next user
                            # message ("continue") proceeds normally.
                            if self.session_manager:
                                _log_error_context(
                                    error=_err,
                                    input_message_id=input_message_id,
                                    messages=last_error_messages,
                                    recoverable=True,
                                )
                                try:
                                    await self._append_recoverable_error_to_history(
                                        exc_name="NoFinalContent",
                                        exc_repr=_err,
                                        input_message_id=input_message_id,
                                    )
                                except Exception as hist_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to persist no-final-content hint: %s",
                                        hist_err,
                                    )
                                await self.session_manager.set_status(AgentStatus.IDLE)
                            if notify_callback is not None:
                                try:
                                    await notify_callback({
                                        "type": "system",
                                        "role": "system",
                                        "content": _err,
                                    })
                                except Exception as notify_err:
                                    _logger.warning(
                                        "[INVOKE] Failed to notify no-final-content: %s",
                                        notify_err,
                                    )
                            return {
                                "success": False,
                                "recovered": True,
                                "error": _err,
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
                # STRICT FAILURE: Agent MUST be initialized, NO SILENT FALLBACKS.
                # This is genuinely fatal — the catch-all below will recognise
                # AgentNotInitialisedError via FATAL_EXCEPTION_TYPES and set
                # AgentStatus.ERROR.  See docs/POLICIES.md §1.
                from deep_worker_cli.exceptions import AgentNotInitialisedError

                error_msg = (
                    "CRITICAL: Agent not initialized! Cannot process requests.\n"
                    "This indicates a fatal initialization error that must be fixed.\n"
                    f"Agent state: {self.agent}\n"
                    "Check agent process logs for initialization errors."
                )
                _logger.error("[INVOKE] %s", error_msg)
                raise AgentNotInitialisedError(error_msg)

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
            _logger.info("[INVOKE] Agent task cancelled (interrupted)")

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
            # No-Agent-Error-State Policy (docs/POLICIES.md §1):
            #   * Fatal exception (usage limit, corrupted checkpoint,
            #     unrecoverable auth, not-initialised) -> AgentStatus.ERROR.
            #   * Everything else is recoverable -> append a SystemMessage
            #     to history, leave the agent IDLE so the user can simply
            #     type "continue".
            #
            # NEVER use print() here — see docs/POLICIES.md §1.4 / design
            # §4.6.  ``stderr`` blocking under load (BlockingIOError) is the
            # exact failure mode this catch-all used to amplify.
            from deep_worker_cli.exceptions import is_fatal_exception

            _logger.exception("[INVOKE] Unhandled exception in agent invocation")
            is_fatal = is_fatal_exception(e)
            exc_name = type(e).__name__

            # Provider-specific fatal handling (usage limits): preserve the
            # exact behaviour from before this refactor — clean user-facing
            # message, notify TUI, set_error, schedule_failover.
            if exc_name == "ClaudeUsageLimitError":
                limit_msg = f"Claude usage limit reached — {str(e)}"
                _logger.info("[INVOKE] %s", limit_msg)
                if notify_callback is not None:
                    try:
                        await notify_callback({
                            "type": "error",
                            "role": "system",
                            "content": limit_msg,
                        })
                    except Exception as notify_err:
                        _logger.warning(
                            "[INVOKE] Failed to notify limit to TUI: %s",
                            notify_err,
                        )
                if self.session_manager:
                    _log_error_context(
                        error=limit_msg,
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                        recoverable=False,
                    )
                    await self.session_manager.set_error(limit_msg)
                await self._schedule_failover("claude-code", str(e))
                return {
                    "success": False,
                    "error": limit_msg,
                }

            if exc_name == "CodexUsageLimitError":
                limit_msg = f"Codex usage limit reached — {str(e)}"
                _logger.info("[INVOKE] %s", limit_msg)
                if notify_callback is not None:
                    try:
                        await notify_callback({
                            "type": "error",
                            "role": "system",
                            "content": limit_msg,
                        })
                    except Exception as notify_err:
                        _logger.warning(
                            "[INVOKE] Failed to notify limit to TUI: %s",
                            notify_err,
                        )
                if self.session_manager:
                    _log_error_context(
                        error=limit_msg,
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                        recoverable=False,
                    )
                    await self.session_manager.set_error(limit_msg)
                await self._schedule_failover("codex", str(e))
                return {
                    "success": False,
                    "error": limit_msg,
                }

            if exc_name == "GrokUsageLimitError":
                limit_msg = f"Grok usage limit reached — {str(e)}"
                _logger.info("[INVOKE] %s", limit_msg)
                if notify_callback is not None:
                    try:
                        await notify_callback({
                            "type": "error",
                            "role": "system",
                            "content": limit_msg,
                        })
                    except Exception as notify_err:
                        _logger.warning(
                            "[INVOKE] Failed to notify limit to TUI: %s",
                            notify_err,
                        )
                if self.session_manager:
                    _log_error_context(
                        error=limit_msg,
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                        recoverable=False,
                    )
                    await self.session_manager.set_error(limit_msg)
                await self._schedule_failover("grok", str(e))
                return {
                    "success": False,
                    "error": limit_msg,
                }

            if is_fatal:
                # Non-provider-specific fatal — CheckpointCorruptedError,
                # ProviderAuthFatalError, AgentNotInitialisedError.
                fatal_msg = f"{exc_name}: {str(e)}"
                if notify_callback is not None:
                    try:
                        await notify_callback({
                            "type": "error",
                            "role": "system",
                            "content": fatal_msg,
                        })
                    except Exception as notify_err:
                        _logger.warning(
                            "[INVOKE] Failed to notify fatal to TUI: %s",
                            notify_err,
                        )
                if self.session_manager:
                    _log_error_context(
                        error=fatal_msg,
                        input_message_id=input_message_id,
                        messages=last_error_messages,
                        recoverable=False,
                    )
                    await self.session_manager.set_error(fatal_msg)
                return {
                    "success": False,
                    "error": fatal_msg,
                }

            # ---------- Recoverable path ----------
            # Convert the exception into a SystemMessage so the model sees
            # it on the next turn, transition to IDLE, notify the UI, and
            # persist a structured log entry.
            #
            # No-Silent-API-Errors Policy (docs/POLICIES.md §9): a recoverable
            # *provider/API* error (e.g. a non-terminal grok ACP error, a codex
            # protocol hiccup) must be surfaced to the user as a VISIBLE,
            # PERSISTENT error — notify type="error" carrying the real API text
            # so the web pins it in the error panel — even though the agent
            # stays IDLE. A generic (non-API) recoverable error keeps the
            # benign type="system" "please type continue" note so we don't cry
            # wolf on the persistent panel.
            from deep_worker_cli.exceptions import is_provider_api_error

            is_api_error = is_provider_api_error(e)
            recoverable_msg = (
                f"Internal error during agent invocation "
                f"({exc_name}: {str(e)}).  The agent runtime caught the "
                f"exception and kept the conversation alive — please type "
                f"'continue' or correct your request to proceed."
            )
            if is_api_error:
                # Real, user-facing provider/API error text. Persisted in the
                # web error panel (type="error"), NOT a fleeting toast.
                notify_type = "error"
                notify_content = (
                    f"Provider API error ({exc_name}): {str(e)}.  The agent "
                    f"recovered and stayed available — type 'continue' or send "
                    f"a new message to retry."
                )
            else:
                notify_type = "system"
                notify_content = recoverable_msg
            if notify_callback is not None:
                try:
                    await notify_callback({
                        "type": notify_type,
                        "role": "system",
                        "content": notify_content,
                    })
                except Exception as notify_err:
                    _logger.warning(
                        "[INVOKE] Failed to notify recoverable error to TUI: %s",
                        notify_err,
                    )

            try:
                await self._append_recoverable_error_to_history(
                    exc_name=exc_name,
                    exc_repr=str(e),
                    input_message_id=input_message_id,
                )
            except Exception as hist_err:
                # We must not raise from the catch-all itself — the policy
                # is "no ERROR for anything we can recover from".  Just
                # log and continue settling to IDLE.
                _logger.warning(
                    "[INVOKE] Failed to append recoverable error to history: %s",
                    hist_err,
                )

            if self.session_manager:
                _log_error_context(
                    error=f"{exc_name}: {str(e)}",
                    input_message_id=input_message_id,
                    messages=last_error_messages,
                    recoverable=True,
                )
                await self.session_manager.set_status(AgentStatus.IDLE)

            return {
                "success": False,
                "recovered": True,
                # For a provider/API error, the return value carries the SAME
                # real API text the UI notification did (POLICIES.md §9), so a
                # programmatic consumer of result["error"] sees the real cause,
                # not the generic "type continue" wrapper. Generic recoverable
                # errors keep the wrapper.
                "error": notify_content if is_api_error else recoverable_msg,
            }

        finally:
            # Only the active owner may clear shared execution state. The
            # identity check also protects focused callers that bypass the
            # serialization wrapper in tests.
            if self.current_worker_task is asyncio.current_task():
                self.is_running = False
                self.current_worker_task = None

