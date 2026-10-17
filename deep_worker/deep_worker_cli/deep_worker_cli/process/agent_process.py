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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    # Annotation-only: the runtime import stays local (see _init_task_manager)
    # so importing this module does not drag in the TaskManager stack. The
    # package-boundary test asserts that boundary.
    from deep_worker_cli.task_manager import TaskManager

# Timestamp helper for log messages (shared format with the web app)
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

def _ts() -> str:
    return datetime.now(UTC).strftime(_TS_FORMAT)


def _get_git_revision(tree_root: str | None) -> tuple[str, bool | None]:
    """Return ``(commit_sha, dirty)`` for the git tree at *tree_root*.

    This is the revision an agent actually ran, captured fresh at every
    process start. Agents run concurrently in different worktrees on
    different commits, so a report that names only the agent is ambiguous
    about which code produced it.

    ``dirty`` is tri-state on purpose:

    * ``False`` — the checkout is exactly *commit_sha*.
    * ``True``  — tracked modifications, staged changes or untracked files
      are present, so *commit_sha* does NOT describe the running code.
    * ``None``  — we could not determine it (no git, no repo, git failed).
      Never ``False``: that would assert a clean tree we never observed.

    Untracked files count as dirty. Under the normal workflow (fix on dev,
    fast-forward to master, run agents from master) a production tree has
    nothing uncommitted, so ``True`` is a real signal that something wrote
    into the checkout — not background noise.

    Never raises: any failure degrades to ``("unknown", None)``.

    ACCEPTED ASSUMPTION (ruff S603/S607): ``git`` is invoked by name and
    resolved via PATH, matching ``_get_git_commit_info`` below and the rest
    of this codebase. Both arguments are fixed literals and *tree_root* is
    passed via ``-C`` rather than interpolated into a shell, so the only
    residual risk is a hostile PATH — which, for a process that already
    executes the agent's own tooling, is not a boundary this function can
    meaningfully defend. Pinning an absolute git path would break the
    supported layouts (system git, pyenv shims, container images).
    """
    if not tree_root:
        return "unknown", None
    try:
        completed = subprocess.run(
            ["git", "-C", tree_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        sha = completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        sha = ""
    if not sha:
        # No usable commit means no usable dirty verdict either: reporting
        # a status against an unknown revision would be meaningless.
        return "unknown", None
    try:
        # --porcelain is the stable, machine-readable contract; any output at
        # all means the tree deviates from HEAD. Untracked files included.
        completed = subprocess.run(
            ["git", "-C", tree_root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = (
            bool(completed.stdout.strip()) if completed.returncode == 0 else None
        )
    except Exception:
        dirty = None
    return sha, dirty


def get_reportable_revision(tree_root: str | None) -> tuple[str | None, bool | None]:
    """``_get_git_revision`` normalised for a feedback record.

    "unknown" is the log line's placeholder, not a value a consumer should
    have to recognise; on the record it becomes ``None``, matching
    ``dirty=None``. Every caller that stamps provenance onto reports goes
    through here so the mapping cannot drift between call sites.
    """
    sha, dirty = _get_git_revision(tree_root)
    return (None if sha == "unknown" else sha), dirty


def get_code_tree_revision() -> tuple[str | None, bool | None]:
    """Revision of the deep_worker CODE that is running, for foreground modes.

    Deliberately NOT derived from the current working directory. An installed
    ``dwcli`` is normally invoked from inside some unrelated project repo, so
    cwd would yield that project's SHA: a well-formed but false claim about
    which deep_worker code produced the report — worse than reporting null.

    The source location is resolved from this module instead; ``git -C``
    walks up to the enclosing repository root on its own. Running from a
    wheel or an install with no ``.git`` therefore honestly yields
    ``(None, None)`` rather than a fabricated revision.

    Daemonized agents do NOT use this: they capture ``tree_root`` from their
    launch config, which is the worktree that agent was actually started from.
    """
    return get_reportable_revision(str(Path(__file__).resolve().parent))


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
# RPYC_ASYNC — LOCAL SOURCES FIRST, INSTALLED PACKAGE AS FALLBACK
# ============================================================================
# rpyc_async provides the AsyncioServer-based protocol:
#   - rpyc_async.core.async_connect.async_connect  (async client)
#   - Connection.aclose() / wait_closed()          (async close)
# A local rpyc_async source checkout takes priority over any installed
# distribution; without one we fall back to `pip install rpyc-async`.
# Owner decision 2026-10-11. The import below must stay BELOW this one, which
# is what puts the checkout on sys.path. See rpyc_async_source for why the
# sources have to win (the published release lacks `async_call`) and for the
# correction of the "PyPI only (D5)" claim that used to sit here.
# ============================================================================

import deep_worker_cli.rpyc_async_source  # noqa: F401  (import for side effect)
from rpyc_async.core.protocol import DEFAULT_CONFIG
from rpyc_async.utils.async_server import AsyncioServer  # AsyncioServer from rpyc_async!

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.cancellation import is_external_cancellation, root_cause_text
from deep_worker_cli.middleware.tool_completion import (
    TurnEndingTool,
    _parse_turn_ending_tool_specs,
)
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus

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
        from deep_worker_cli.storage_paths import agent_root
        state_path = agent_root(agent_name, create=False) / "grok_session_state.json"
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
        from deep_worker_cli.storage_paths import agent_root
        state_path = agent_root(agent_name, create=False) / "codex_session_state.json"
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
    with ``config.model is None`` (unpinned), the observation is stored separately
    as ``resolved_model`` and must NOT overwrite ``model`` — doing so defeats the
    operator rule "no explicit --model ⇒ provider default". See
    ``DESIGN_FIX_CROSS_PROVIDER_MODEL_POISON.md`` (incident 2026-06-29):
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
        update_resolved = getattr(session_manager, "update_resolved_model", None)
        if callable(update_resolved):
            await update_resolved(model_name, getattr(config, "provider", None))
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
        turn_ending_tool_specs: list[str] | None = None,
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
            turn_ending_tool_specs: Raw ``--turn-ending-tool`` strings of
                the form ``<name>`` or ``<name>:<field>``. Parsed and
                validated SYNCHRONOUSLY here so a malformed or colliding
                declaration fails the construction outright instead of
                surfacing inside ``start()``'s init ``try`` as an
                ERROR-mode agent. See
                ``docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`` §3.5.
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

        # Turn-ending tool declarations are parsed HERE, at the boundary,
        # before a port is bound, a checkpointer opened or the registry
        # touched. Deferring this to start() would put it inside the init
        # ``try`` that converts any exception into AgentStatus.ERROR while
        # leaving the agent registered — a config typo must not produce a
        # ghost process. The parsed list is what start() hands to
        # create_cli_agent(); the raw strings are kept for logging.
        specs_raw = list(turn_ending_tool_specs or [])
        self._turn_ending_tools: list[TurnEndingTool] = (
            _parse_turn_ending_tool_specs(specs_raw)
        )
        self._turn_ending_tool_specs_raw = specs_raw

        # Same boundary, same reason: a contradictory tool selection must
        # fail before anything is bound, not inside start()'s swallowing
        # try. ``config.tools`` was validated by AgentConfig already; this
        # re-parse is what start() hands to create_cli_agent().
        from deep_worker_cli.tool_selection import parse_tool_selection

        self._tool_selection = parse_tool_selection(getattr(config, "tools", None))

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
        # The checkpointer itself, kept so history stays readable when the
        # graph failed to build. Reading history only ever needs the
        # checkpoint — binding it to ``self.agent`` made a broken subsystem
        # (e.g. an unreachable external MCP) hide a conversation that is
        # sitting intact on disk. See ``_load_checkpoint_messages``.
        self._checkpointer: Any = None
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
        from deep_worker_cli.storage_paths import (
            CURRENT_STORAGE_VERSION,
            _ensure_agent_layout,
            agent_root,
        )
        # Lazy migration on agent startup (DESIGN_AGENT_DBS §4.4). This
        # is the single point where we walk a legacy layout to the new
        # ``agents/<name>/`` layout. Idempotent — already-migrated
        # agents skip cleanly.
        if data_dir is None:
            _ensure_agent_layout(name)
        self.data_dir = data_dir or agent_root(name)
        from deep_worker_cli.storage_paths import (
            detect_storage_version,
            ensure_private_dir,
        )
        self.storage_version = detect_storage_version(name)
        assert self.storage_version == CURRENT_STORAGE_VERSION, (
            f"agent {name!r} storage_version {self.storage_version} "
            f"!= current {CURRENT_STORAGE_VERSION}"
        )

        ensure_private_dir(self.data_dir)

        # Components
        self.registry = registry or ProcessRegistry()
        self.agent = None  # Will be initialized in start()
        self.backend = None  # Will be initialized in start()
        self.session_manager: SessionManager | None = None
        self.task_manager = None  # Will be initialized in start() if enabled
        # Token-stats recorder (DESIGN_AGENT_TOKEN_STATS.md §5.1). Lazily
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
        # Git revision of the tree this process runs, filled in at startup.
        # Defaults assert nothing: an agent whose startup path did not run
        # must report unknown provenance rather than a fabricated clean tree.
        self._startup_commit: str | None = None
        self._startup_dirty: bool | None = None
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
        # delivery_ids already delivered AND durably ledgered. Loaded from disk
        # at startup so suppression survives a restart.
        self._inbox_delivered: set[str] = set()
        # Foreign filenames already warned about, so a file we deliberately
        # refuse to move does not spam a WARNING on every arrival forever.
        self._inbox_foreign_warned: set[str] = set()
        # delivery_ids the checkpointer has CONFIRMED as committed to the
        # context. The delete boundary reads this and nothing else: a turn
        # returning success is not proof the message was recorded.
        self._inbox_committed: set[str] = set()
        # Content fingerprints of messages PROVEN to be in this agent's
        # context. Populated only where a commit is proven, and consulted at
        # publish to refuse an automatic resend of something already received.
        self._inbox_received: set[str] = set()
        # False when the saver cannot commit (InMemorySaver, --memory-only).
        # Then no commit proof can ever arrive, so the drain falls back to
        # the turn result rather than retaining every record forever.
        self._inbox_commit_signal_available: bool = False
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

    @property
    def _reported_commit(self) -> str | None:
        """Startup SHA as it should appear on a feedback record.

        Applies the same "unknown" -> None mapping as
        ``get_reportable_revision`` to the value captured at startup, so the
        main agent and the TaskManager can never disagree about the revision.
        """
        if self._startup_commit == "unknown":
            return None
        return self._startup_commit

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

    def _active_task_manager(self) -> "TaskManager | None":
        """The TaskManager that should drive this turn, or ``None``.

        Two conditions, and both are load-bearing:

        * the manager exists at all (``launch_config.task_manager``), and
        * the operator has not switched it off (``set_task_manager``).

        The evaluation loops used to test only the first. A disabled manager
        then still entered the loop — harmlessly as far as its own methods were
        concerned, since ``init_task`` and ``on_subordinate_response`` both
        return early when disabled, but visibly to the operator, who saw
        ``TaskManager iteration`` lines and decisions continuing after turning
        it off (rg1_worker, 2026-10-10).

        Kept as one accessor rather than repeating the conjunction at the four
        call sites so the two conditions cannot drift apart again.

        Returns the manager instead of a bool so callers can bind it to a local
        and keep the type narrowing that the old ``if self.task_manager:`` gate
        gave them for free. A bool predicate type-checks as a gate but leaves
        ``self.task_manager`` as ``TaskManager | None`` at the call, which is
        both a type error and a real race: the attribute can be reassigned
        between the check and the use.
        """
        task_manager = self.task_manager
        if task_manager is None or task_manager.disabled:
            return None
        return task_manager

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
           (2026-05-19). With the timeout, each dispatched
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
        is_status_event = _payload.get("type") != "model_context_changed"
        payload = self._current_status_snapshot() if is_status_event else _payload
        payload_str = _json.dumps(payload, default=str)

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
                # "stuck at idle while actually working" bug
                # (2026-05-20). A live web re-subscribes on its next
                # connect_to_agent, so eviction is safe.
                #
                # Evict by IDENTITY: ``set.discard(netref)`` would call
                # ``netref.__hash__`` → a sync RPyC round-trip over the
                # already-dead channel → EOFError, which on the invoke path
                # escaped and forced the agent to ERROR (2026-05-22).
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
        if is_status_event:
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
        - ``availability_tracker.is_restore_eligible(preferred_provider)``
          is True (stricter than ``is_available`` — active limit blocks restore)
        - No dedicated ``_restore_scheduled`` flag is already set (one per reopen).

        There is no provider-name special case here. A ``preferred_provider ==
        "deepseek"`` skip used to encode "deepseek is terminal", but
        terminality is now a chain position, not a name (owner directive
        2026-09-03). Keeping the skip would strand an agent that failed over
        INTO deepseek from a chain like ``["or-glm-52", "deepseek"]``: it could
        never return to a deepseek preference. Nothing is lost by removing it —
        an agent already on its preferred provider returns at the
        ``preferred_provider == current_provider`` guard above.

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
        rpyc_async/docs/DESIGN_INBOUND_BACKPRESSURE.md.
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
                # __hash__ crosses the dead wire → EOFError,
                # 2026-05-22).
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
        appends to ``<data_dir>/token_stats.jsonl`` (DESIGN_AGENT_TOKEN_STATS.md
        §4.2). The same recorder instance is shared with the TaskManager so
        both write to one file, tagged ``kind="agent"`` / ``kind="tm"``.
        """
        if self._token_stats_recorder_obj is None:
            from deep_worker_cli.stats.token_stats import TokenUsageRecorder

            self._token_stats_recorder_obj = TokenUsageRecorder(
                self.data_dir, on_record=self._dispatch_status_event
            )
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
            # Capture the revision ONCE per process start, here, before the
            # agent is built. It is deliberately NOT read from launch_config:
            # that dict is persisted and replayed across restarts, and a
            # restart is usually done precisely to pick up new code, so a
            # stored SHA would describe the previous process.
            self._startup_commit, self._startup_dirty = _get_git_revision(tree)
            pairs = [
                ("agent", self.name),
                ("provider", str(lc.get("provider", "unknown"))),
                ("model", str(lc.get("model", "unknown"))),
                ("tree", tree),
                ("commit", commit),
                ("dirty", str(self._startup_dirty)),
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
            # The revision captured above is stamped here — at step 3, BEFORE
            # the agent is built at step 4 and before any activity. It must
            # not depend on the agent ever filing a report: consumers read
            # session_metadata.json to learn which code an agent runs.
            self.session_manager = SessionManager(
                data_dir=self.data_dir,
                launch_config=self.launch_config,
                notifier=self._dispatch_status_event,
                commit=self._reported_commit,
                dirty=self._startup_dirty,
            )

            # 4. Initialize agent with create_cli_agent
            try:
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

                # Load environment BEFORE importing config: config.py builds
                # Settings at module level and needs OPENAI_MODEL,
                # OPENAI_BASE_URL, provider API keys, etc. already set.
                #
                # Sources, highest precedence first (env_source_paths):
                #   the product tree's .env and env.d,
                # then the ancestor's
                # upstream ancestor's directory, still read so existing
                # installs keep working but never allowed to shadow the
                # product tree. Within a tree the main .env stays
                # authoritative and env.d drop-ins only fill blanks
                # (DESIGN_EXTERNAL_MCP_TOOLS.md §7). The 0600 permission
                # gate on drop-ins is enforced by load_env_d.
                from deep_worker_cli.process.env_loader import (
                    env_source_paths,
                    load_env_sources,
                )

                applied = load_env_sources()
                if applied:
                    sources = ", ".join(
                        str(p) for p in env_source_paths() if p.exists()
                    )
                    msg = (
                        f"{_ts()} [AGENT_INIT] Loaded {len(applied)} variable(s) "
                        f"from [{sources}]: {sorted(applied.keys())}\n"
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
                # for BOTH the production tree and
                # the developer tree. Hard-coding
                # the prod path here used to defeat the --dev launcher
                # dispatch: even after the bash launcher correctly re-execs
                # into the dev tree, this insertion would prepend the prod
                # ``deepagents`` and shadow the dev one. See
                # docs/DESIGN_DEV_FLAG.md.
                #
                # __file__ here =
                #   <TREE>/deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py
                # parent x4 =
                #   <TREE>/deep_worker
                # ================================================================
                deepagents_path = (
                    PathLib(__file__).resolve().parent.parent.parent.parent
                    / "deepagents"
                )
                if deepagents_path.exists() and str(deepagents_path) not in sys.path:
                    sys.path.insert(0, str(deepagents_path))

                # Now import AFTER environment is loaded and deepagents is in path
                from deep_worker_cli.agent import create_cli_agent
                from deep_worker_cli.config import WebSearchPolicy, create_model
                from deep_worker_cli import tools as cli_tools
                from deep_worker_cli.tools import (
                    http_request,
                    fetch_url,
                    make_send_message_to_agent_tool,
                    read_image,
                    resolve_web_search_tool,
                )

                # Create model using configuration from AgentConfig
                # Pass provider and model explicitly from config.
                # Pass user=self.name for KV-cache isolation so the main
                # agent and its TaskManager (:tm) use separate user_ids
                # and do not evict each other's DeepSeek prefix-cache.
                # kind="agent" + factories wire the token-stats recorder
                # (DESIGN_AGENT_TOKEN_STATS.md §5.1).
                from deep_worker_cli.stats.callback import (
                    make_stats_callback_factory,
                    make_usage_event_callback_factory,
                )

                model = create_model(
                    provider=self.config.provider,
                    model_name=self.config.model,
                    user=self.name,
                    kind="agent",
                    auto_approve=self.skip_permissions,
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
                # model resolution) can reach it. After incident
                # 2026-06-13 (#2), the MCP-tools-ready barrier that
                # used to live here was replaced by a deterministic
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
                        # _meta.modelId. See agent_process incident
                        # test_grok 2026-06-18.
                        persisted_model = _resolve_persisted_grok_display_model(self.name)
                        if persisted_model:
                            await _on_resolved_model(persisted_model, persist=False)

                restart_tool_factory = getattr(cli_tools, "make_restart_agent_tool")
                restart_self_tool_factory = getattr(cli_tools, "make_restart_self_tool")

                # Prepare web tools
                # - http_request and fetch_url work without API keys
                # - web_search (delegation) uses Claude Code SDK for universal
                #   web search; resolved lazily under the tri-state policy
                #   (auto/enabled/disabled).
                # - send_message_to_agent: inter-agent messaging; sender name
                #   is baked into the factory from self.name so the LLM
                #   cannot forge it. See docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md
                web_tools: list[Any] = [
                    http_request,
                    fetch_url,
                    read_image,
                    restart_tool_factory(self.name),
                    restart_self_tool_factory(self.name),
                    make_send_message_to_agent_tool(self.name),
                ]

                # Universal web-search under the agent's tri-state policy.
                # ``enabled`` raises CapabilityError when the SDK is missing;
                # ``auto`` logs once and skips; ``disabled`` never resolves.
                _web_search_policy = WebSearchPolicy(
                    getattr(self.config, "web_search_policy", "auto")
                )
                _web_search_avail = resolve_web_search_tool(_web_search_policy)
                if _web_search_avail.tool is not None:
                    web_tools.append(_web_search_avail.tool)

                # A2A outbound tools (a2a_*): only registered when A2A is
                # enabled for this agent. Lets the LLM initiate A2A protocol
                # calls to other A2A-compliant agents. Import/build failures
                # are non-fatal (A2A is optional) — the agent keeps its other
                # tools. See deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md.
                try:
                    from deep_worker.providers.deepworker_a2a.config import A2AConfig as _A2AConfig

                    if _A2AConfig.from_env(self.name).enabled:
                        from deep_worker.providers.deepworker_a2a.tools import make_a2a_tools

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
                    # Resolve the vendored provider through provider_paths
                    # (source root on sys.path in source mode; no-op in
                    # installed mode). See deep_worker_cli/provider_paths.py.
                    from deep_worker_cli.provider_paths import ensure_provider_importable
                    from deep_worker_cli.exceptions import CapabilityError

                    resolution = ensure_provider_importable(
                        "deep_worker.providers.codex_provider"
                    )
                    if resolution.unavailable:
                        raise CapabilityError(
                            capability="codex",
                            remedy=resolution.remedy or resolution.unavailable,
                        )
                    from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS
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
                    # Resolve the vendored provider through provider_paths
                    # (source root on sys.path in source mode; no-op in
                    # installed mode). See deep_worker_cli/provider_paths.py.
                    from deep_worker_cli.provider_paths import ensure_provider_importable
                    from deep_worker_cli.exceptions import CapabilityError

                    resolution = ensure_provider_importable(
                        "deep_worker.providers.grok_provider"
                    )
                    if resolution.unavailable:
                        raise CapabilityError(
                            capability="grok",
                            remedy=resolution.remedy or resolution.unavailable,
                        )
                    from deep_worker.providers.grok_provider.builtin_tool_passthrough import (
                        build_passthrough_tools,
                    )
                    web_tools.extend(build_passthrough_tools())

                # Every OTHER ACP provider: same channel-B recording, selected
                # by CAPABILITY rather than by provider name. The model already
                # carries the profile, and the profile already declares which
                # built-ins the CLI runs locally, so nothing here needs to know
                # who the provider is.
                #
                # Without this, claude-code lost every native tool result
                # (rg1_worker, 2026-09-28). Its profile declares
                # ``{Bash, Read, Write, Edit, Glob, Grep, ...}`` and its chat
                # model surfaces those names as real ``AIMessage.tool_calls``,
                # but no matching BaseTool was ever bound — only grok and codex
                # had a branch. LangGraph's ToolNode then took the unknown-name
                # path and wrote ``ToolMessage(status="error", "Error: Bash is
                # not a valid tool...")`` WITHOUT running anything, while the
                # CLI had in fact executed the command. 3439 checkpointed
                # refusals on top of successful work in that agent alone, with
                # the real output persisted nowhere.
                # ``else``: grok is handled above by its own pinned entry point
                # (it must keep the ``GrokBuiltin`` name, a public contract), so
                # it is excluded here. codex is NOT excluded by name: it reaches
                # this branch and binds nothing because it declares no ACP
                # built-ins — its native tools came from ``NATIVE_TOOLS``.
                else:
                    from deep_worker.providers.acp.builtin_tool_passthrough import (
                        build_passthrough_tools_for_profile,
                    )

                    # ``_profile`` FIRST, and it is load-bearing.
                    #
                    # ``AcpChatModel`` exposes the profile as a ``@property``
                    # while ALSO inheriting ``profile`` as a pydantic field.
                    # Pydantic strips the property off the class and the
                    # instance ``__dict__`` shadows it, so
                    # ``getattr(model, "profile")`` hands back the raw
                    # ``property`` OBJECT — not the profile. It is not None, so
                    # the guard below passed, and
                    # ``build_passthrough_tools_for_profile`` then read no
                    # ``known_builtins`` off it and returned ``()``. Result:
                    # ZERO recorders bound for every ACP provider, and ToolNode
                    # went back to fabricating "Error: Bash is not a valid
                    # tool" without running anything — the very defect the
                    # capability wiring was added to kill (rg1_worker,
                    # 2026-09-29: 122 Bash / 61 Edit / 43 Read refusals from a
                    # POST-fix process). ``summarization.py`` already carries
                    # this same workaround for the same reason.
                    #
                    # NB: ``profile`` is also a field on langchain's
                    # BaseChatModel, so every model HAS the attribute — it is
                    # the VALUE that discriminates. Non-ACP providers leave it
                    # at its ``None`` default, and even a populated upstream
                    # ``ModelProfile`` (a plain dict, no ``known_builtins``)
                    # yields ``()``.
                    _profile = getattr(model, "_profile", None) or getattr(
                        model, "profile", None
                    )
                    if _profile is not None:
                        web_tools.extend(
                            build_passthrough_tools_for_profile(_profile)
                        )

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

                # Keep the checkpointer reachable independently of the graph.
                # It is created BEFORE the parts of init that can fail (the
                # external MCP pool, the model backend), so when they do, the
                # conversation is still readable straight from here.
                self._checkpointer = checkpointer

                # ── Self-restart delivery barrier ──
                # The barrier fires when a tool result becomes part of the
                # model's context, i.e. when the checkpointer commits it.
                # This is the one persistence path shared by EVERY provider,
                # so restart_self needs no transport hook and no per-provider
                # gate. See docs/DESIGN_DELIVERY_BARRIER_ON_CHECKPOINT.md.
                #
                # An InMemorySaver (--memory-only, tests) has no commit and
                # therefore no delivery signal; restart_self stays unarmed
                # and reports the failure rather than promising a restart
                # that cannot be confirmed.
                if checkpointer is not None and hasattr(
                    checkpointer, "add_commit_listener"
                ):
                    from deep_worker_cli.process.self_restart_barrier import (
                        fire_armed_self_restart,
                        mark_self_restart_delivery_available,
                    )

                    checkpointer.add_commit_listener(fire_armed_self_restart)
                    mark_self_restart_delivery_available()

                # ── Inbox delete boundary ──
                # Same criterion, different subject: a spool record may be
                # dropped only once the commit proves its message reached
                # the context. Without this signal the drain falls back to
                # "the turn returned success", which a death between the
                # return and the commit turns into a lost message.
                if checkpointer is not None and hasattr(
                    checkpointer, "add_delivery_commit_listener"
                ):
                    checkpointer.add_delivery_commit_listener(
                        self._on_delivery_committed
                    )
                    self._inbox_commit_signal_available = True

                # Bring up external MCP servers (see
                # docs/DESIGN_EXTERNAL_MCP_TOOLS.md) BEFORE create_cli_agent,
                # so their tools enter the graph from the start and land in
                # _extract_final_tool_metadata (so Full Dialog
                # verbosity=tools shows them, and the provider prewarm
                # sees the correct final set).
                self._external_mcp_pool = await self._start_external_mcp_pool()

                # Create the agent with auto-approve for testing
                # Provider-max validation runs HERE (before create_cli_agent
                # mounts the guard middleware) so any auto-clamp is visible
                # to the operator in the boot log and the persisted
                # launch_config is the clamped value. See
                # docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md §3.4.
                from deep_worker_cli.context_limit.provider_max import (
                    resolve_provider_max,
                )

                _requested_limit = (self.launch_config or {}).get("context_limit")
                _provider = self.config.provider
                _model_name = self.config.model
                # Resolve the provider max via API lookup. For family
                # aliases (claude-code / claude-code-sdk) this calls
                # the ACP session/new picker to find the resolved
                # dated id, then looks up in the LangChain SDK
                # profile data. For other providers/models it does a
                # direct SDK lookup. Returns ``None`` on any
                # failure — per operator contract 2026-08-19: NO
                # assumption-based fallback.
                _resolved_max = await resolve_provider_max(
                    _provider, _model_name
                )
                # Default: pass through unchanged (silent passthrough).
                _clamped_limit = _requested_limit
                if (
                    _resolved_max is not None
                    and _requested_limit is not None
                    and _requested_limit > _resolved_max
                ):
                    print(
                        f"[context-limit] WARN: -l {_requested_limit:,} "
                        f"exceeds provider cap "
                        f"({_provider}/{_model_name} = {_resolved_max:,}). "
                        f"Auto-clamping to provider max.",
                        file=sys.stderr,
                    )
                    _clamped_limit = _resolved_max
                    if self.launch_config is not None:
                        # Persist the clamped value so the agent does
                        # not re-warn on every restart (and so the
                        # ``-l <N>`` the operator sees in
                        # ``deep-worker --info`` matches what the
                        # guard actually enforces).
                        self.launch_config["context_limit"] = (
                            _clamped_limit
                        )

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
                    # Already parsed and validated in __init__ — never
                    # re-parsed here, so bad input can never reach this
                    # swallowing try block.
                    turn_ending_tools=self._turn_ending_tools,
                    context_limit=_clamped_limit,
                    hot_cache_ttl=(self.launch_config or {}).get("hot_cache_ttl"),
                    summary_keep=(self.launch_config or {}).get("summary_keep"),
                    provider=self.config.provider,
                    commit=self._reported_commit,
                    dirty=self._startup_dirty,
                    tool_selection=self._tool_selection,
                )

                # Capture the INVARIANT (startup) system prompt so the
                # Full Dialog "system_prompt" introspection mode (§9 of
                # DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md)
                # reports EXACTLY what the agent was built with. We pass no
                # ``system_prompt`` override above, so ``create_cli_agent``
                # resolves it via ``get_system_prompt`` with these same args —
                # recomputing here yields the identical provider-agnostic text.
                # The tool selection is one of those args: omitting it would
                # report the FULL prompt for a restricted agent, i.e. exactly
                # the text the agent was NOT built with.
                try:
                    from deep_worker_cli.agent import get_system_prompt
                    self._invariant_system_prompt = get_system_prompt(
                        assistant_id=self.name,
                        sandbox_type=None,
                        project_memory_dir=Path(self.config.working_dir),
                        provider=getattr(self.config, "provider", None),
                        tool_selection=self._tool_selection,
                    )
                except Exception:  # noqa: BLE001 — never block start()
                    self._invariant_system_prompt = None

                # ════════════════════════════════════════════════════════
                # PREWARM ClientManager with the FINAL tool inventory.
                # ════════════════════════════════════════════════════════
                # Incident 2026-06-13 (#2) root-cause fix. The compiled
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
                #   - claude_code_sdk_provider/client_manager.py
                #     (rotate-on-mismatch — defence in depth).
                # Both halves of the fix MUST coexist.
                _final_metadata = self._extract_final_tool_metadata(
                    self.agent, provider=getattr(self.config, "provider", None)
                )

                # Report CLI-declared turn-ending tools against the FINAL
                # inventory. Earlier would give false misses (external MCP
                # may still be connecting); this is the first point where
                # the graph's tool set is known and complete.
                self._log_turn_ending_tool_diagnostics(_final_metadata)

                _client_manager = getattr(model, "_client_manager", None)
                if _client_manager is not None and _final_metadata:
                    await self._prewarm_client_manager(
                        client_manager=_client_manager,
                        thread_id=self.name,
                        tools_metadata=_final_metadata,
                        working_dir=self.config.working_dir,
                    )

            except BaseException as e:
                # ``BaseException``, not ``Exception``: initialization can
                # fail with a ``CancelledError`` that arrives from a FOREIGN
                # anyio scope (the mcp SDK cancels its own TaskGroup when the
                # HTTP server is unreachable). Such a cancellation is an
                # ordinary subsystem failure, but ``except Exception`` did not
                # catch it: the process died HERE, before reaching RPyC
                # (step 5) and registry registration (step 6). As a result the
                # agent was missing from the web UI and its history could not
                # be read — a related internal incident (2026-08-29).
                #
                # A cancellation initiated from OUTSIDE must not be absorbed:
                # otherwise a normal shutdown would stop stopping the process.
                # An external one is recognised by ``Task.cancelling() > 0``
                # (plus KeyboardInterrupt / SystemExit), see
                # ``deep_worker_cli.cancellation``.
                if is_external_cancellation(e):
                    raise
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
                        auto_approve=self.skip_permissions,
                        commit=self._reported_commit,
                        dirty=self._startup_dirty,
                        # Replay the operator's off switch from the previous
                        # process. Seeded at construction rather than toggled
                        # after start() so no turn can slip through enabled in
                        # between. Default False when it was never set.
                        disabled=bool(
                            getattr(
                                self.session_manager,
                                "task_manager_disabled",
                                False,
                            )
                        ),
                    )
                    if self.task_manager.disabled:
                        _logger.info(
                            "[AGENT_INIT] TaskManager restored as DISABLED "
                            "(operator switch persisted from an earlier process)"
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

        except BaseException as e:
            print(f"{_ts()} Failed to start agent: {e}")
            await self.shutdown()
            # ``shutdown()`` exits on the ``if not self.running`` guard for
            # any failure BEFORE step 7 — but the pid file is written in
            # step 1. Without this, the pid of an already-dead process stayed
            # on disk: we remove it ourselves, since the normal path never
            # reached the "cleanup files" step.
            if not self.running:
                with contextlib.suppress(Exception):
                    await self._remove_pid_file()
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

    # How many near-miss names a ``turn_ending_tool_not_found`` line lists.
    _TURN_ENDING_CANDIDATE_LIMIT = 8

    @staticmethod
    def _tool_metadata_name(entry: Any) -> str:
        """Read a tool name out of one ``_extract_final_tool_metadata`` entry.

        The helper returns plain dicts for non-claude providers and
        ``ToolMetadata``-shaped objects for the claude-code-sdk probe, so
        both shapes are accepted here.
        """
        if isinstance(entry, dict):
            return str(entry.get("name", ""))
        return str(getattr(entry, "name", ""))

    def _log_turn_ending_tool_diagnostics(self, metadata: list[Any]) -> None:
        """Report whether each declared turn-ending tool actually exists.

        Emits ONE machine-greppable line per declared spec, always —
        including the healthy case. A launcher that greps only for the
        warning cannot distinguish "the tool was found" from "the flag
        never reached this code", which is precisely the ambiguity the
        consumer asked to remove::

            turn_ending_tool_registered name=<tool>
            turn_ending_tool_not_found name=<tool> candidates=[...]

        A miss is a WARNING, never fatal: an external MCP endpoint may
        come up after the agent, and the declaration stays registered
        with the middleware either way — if the tool appears later it
        matches by name as usual. Turning "the optimisation did not
        apply this run" into "the run did not start" would be worse.

        Args:
            metadata: the value ``_extract_final_tool_metadata`` returned,
                i.e. the FINAL tool inventory of the compiled graph. This
                is the only inventory source; there is no second lookup
                that could disagree with it.
        """
        if not self._turn_ending_tools:
            return

        known = [self._tool_metadata_name(m) for m in metadata]
        known_set = set(known)

        for spec in self._turn_ending_tools:
            if spec.tool_name in known_set:
                _logger.info(
                    "turn_ending_tool_registered name=%s", spec.tool_name
                )
                continue
            _logger.warning(
                "turn_ending_tool_not_found name=%s candidates=%r",
                spec.tool_name,
                self._turn_ending_tool_candidates(spec.tool_name, known),
            )

    @classmethod
    def _turn_ending_tool_candidates(
        cls, missing: str, known: list[str]
    ) -> list[str]:
        """Best near-misses for a declared-but-absent tool name.

        Ranked by ``(-similarity, name)``: a total order, so the same
        inventory always yields the same list and tests can pin exact
        contents rather than "contains".
        """
        import difflib

        ranked = sorted(
            known,
            key=lambda name: (
                -difflib.SequenceMatcher(None, missing, name).ratio(),
                name,
            ),
        )
        return ranked[: cls._TURN_ENDING_CANDIDATE_LIMIT]

    @staticmethod
    def _extract_final_tool_metadata(agent: Any, provider: str | None = None) -> list[Any]:
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
        failure mode of incident 2026-06-13 (#2): the first non-empty
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
        #
        # Structural invariant: non-Claude
        # paths MUST NEVER import ``deep_worker.providers.claude_code_sdk_provider``
        # — a wheel without the SDK fork / claude CLI must still start API
        # providers. The guard on ``provider == "claude-code-sdk"`` happens BEFORE
        # the import, and the Claude path only surfaces a classified
        # missing-capability (actionable) or re-raises internal defects.
        tools_list = list(tools_by_name.values())
        if not tools_list:
            return []
        if provider != "claude-code-sdk":
            # Non-Claude provider: no ClaudeCodeChatModel probe — return the
            # raw tool names. The prewarm proxy only ever runs for the
            # claude-code-sdk provider (model has no _client_manager otherwise),
            # and _handle_get_tools only needs names/descriptions.
            return [
                {
                    "name": getattr(t, "name", "?"),
                    "description": getattr(t, "description", ""),
                    "input_schema": getattr(t, "args", {}),
                }
                for t in tools_list
            ]
        from deep_worker_cli.provider_paths import ensure_provider_importable
        from deep_worker_cli.exceptions import CapabilityError

        resolution = ensure_provider_importable(
            "deep_worker.providers.claude_code_sdk_provider"
        )
        if resolution.unavailable:
            raise CapabilityError(
                capability="claude-code-sdk",
                remedy=resolution.remedy or resolution.unavailable,
            )
        from deep_worker.providers.claude_code_sdk_provider.chat_model import ClaudeCodeChatModel  # noqa: PLC0415

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
        ⚠ WHY THIS EXISTS — INCIDENT 2026-06-13 (#2) ROOT CAUSE FIX.
        ════════════════════════════════════════════════════════════════
        ``ClientManager`` caches one ``(ClaudeSDKClient, ProxyMCPServer)``
        pair per ``thread_id``. The cached ``ProxyMCPServer`` is built
        from whatever ``tools_metadata`` was passed on the FIRST
        ``get_or_create_client`` call. After incident-2026-06-13 (#2)
        we also rotate-on-mismatch (see ``ClientManager``), but the
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
        is determinism. See incident 2026-06-13 (#2) post-mortem.
        """
        if not tools_metadata:
            # Nothing to prewarm — likely a tool-less agent or extraction
            # gave up. Leave the cache slot empty; first real turn fills
            # it normally.
            return
        # ════════════════════════════════════════════════════════════════
        # ⚠ PREWARM MUST NOT OPEN THE CLAUDE CLI SESSION — INCIDENT
        # 2026-06-15 (mj1_init forgot its history after restart).
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
        # inventory — incident 2026-06-13 #2) WITHOUT opening the CLI
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

    def _summarization_middleware_or_none(self) -> Any:
        """``_find_summarization_middleware`` without the raise.

        That method is fail-loud by design: a missing middleware breaks the
        pseudo-summary contract and must not pass silently. Observability is the
        opposite case — summarization can legitimately be absent (disabled, or a
        focused test building the process via ``__new__``), and a turn must never
        fail because there was no incident to report.
        """
        try:
            return self._find_summarization_middleware()
        except Exception:  # noqa: BLE001 - absence is normal here
            return None

    def _reset_turn_compaction_report(self) -> None:
        """Drop any mid-turn compaction report left over from a previous turn."""
        sm = self._summarization_middleware_or_none()
        if sm is not None:
            sm.last_turn_compaction = None

    def _turn_compaction_report(self) -> dict[str, Any] | None:
        """The compaction report for the turn that just ran, if there was one.

        Returns ``None`` on a normal turn. The consumer's contract (agreed with
        repo_graph, REQ-90) is that an ABSENT key means "no compaction": a report
        on every turn would mark every document of every run as an incident.
        """
        sm = self._summarization_middleware_or_none()
        if sm is None:
            return None
        report = getattr(sm, "last_turn_compaction", None)
        return report if isinstance(report, dict) else None

    def _compaction_is_observable(self) -> bool:
        """Whether a mid-turn compaction could be DETECTED on this provider.

        False means "we cannot tell either way", which is NOT "it did not
        happen". A session-stateful provider keeps the conversation on its own
        side (CLI subprocess: ACP, codex, claude-code, grok), so it can compact
        that history without telling us: no protocol carries the fact. Verified
        against the protocols rather than inferred from an empty grep — ACP's
        ``HANDLED_SESSION_UPDATE_KINDS`` has no compaction kind and unknown kinds
        are logged-and-ignored; codex never engages ``thread/compact/start``
        because it drives ``codex exec``; claude-code has no notion of it.

        Keyed on the session-stateful classification, NOT on a fresh list of
        provider names: that classification IS this property (history lives on
        their side => they can compact it unseen), and
        ``SESSION_STATEFUL_PROVIDERS`` is the single source of truth for it. A
        second hand-maintained list here would drift the first time a provider
        is added, and drift silently in the direction of claiming observability
        we do not have.
        """
        try:
            from deep_worker_cli.context_limit.hot_cache import (
                SESSION_STATEFUL_PROVIDERS,
            )
        except Exception:  # noqa: BLE001
            return True
        provider = getattr(getattr(self, "config", None), "provider", None)
        # Not a real string => not a provider identity we can reason about; the
        # in-process path is the observable one, so do not claim a blind spot.
        if not isinstance(provider, str) or not provider.strip():
            return True
        return provider.strip().lower() not in SESSION_STATEFUL_PROVIDERS

    def _with_compaction_flag(self, response: dict[str, Any]) -> dict[str, Any]:
        """Stamp the compaction keys onto a turn reply.

        Top level, beside ``success``/``recovered`` — NOT inside ``response``,
        whose ``content`` the consumer reads as text. One wrapper for all three
        reply sites (plain, streaming, TaskManager): a flag delivered on one path
        out of three is an incident that reproduces every other run.

        TWO KEYS, TWO MEANINGS — agreed with rg1_worker (877726c), and they must
        never be conflated in either direction:

        * ``context_compacted`` / ``context_summarized`` — a per-turn FACT: a
          real size-based compaction fired during THIS turn. The consumer treats
          it as an incident and re-extracts the document.
        * ``context_compaction_observable: false`` — a STANDING property of the
          provider, true of every turn alike. The consumer logs it loudly and
          does NOT fail the document; read as an incident it would fail every
          document of every run forever.

        Hence the standing key is deliberately NOT expressed as
        ``context_summarized: {observable: false}``: that shape is the per-turn
        report, and sending it on every turn is exactly the permanent-outage
        failure mode.
        """
        report = self._turn_compaction_report()
        if report is not None:
            response["context_compacted"] = True
            response["context_summarized"] = report
        if not self._compaction_is_observable():
            response["context_compaction_observable"] = False
        return response

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

                # Reset provider-side session state that mirrors the
                # LangGraph channel (ACP: sent_count / session_id /
                # context gauge). Without this the persisted sent_count
                # from the previous document (or the previous restart)
                # exceeds the freshly wiped state, and the very next
                # ainvoke burns on the ``chat_model.py:571`` fail-loud
                # guard (rg1_worker 2026-08-22 report: alternating
                # burn/ok per document on the repo_graph extractor).
                #
                # Capability-based: API-only models do not carry session
                # state and simply do not implement the port — the call is
                # skipped without complaint.
                #
                # ``implements_summarizer_port`` is the single definition of
                # "implements the port" (it checks the TYPE, so a plain
                # MagicMock placeholder does not read as a live port and
                # break sibling tests, and it requires BOTH halves). This
                # used to be an inline ``getattr`` probe, which silently
                # skipped any provider that misspelled or omitted a method —
                # how claude-code-sdk went without ``areset_after_summary``
                # unnoticed until 2026-08-24.
                model = getattr(self, "_model", None)
                from deep_worker_cli.context_limit.summarizer_port import (
                    implements_summarizer_port,
                )

                # ``model is not None`` is redundant at runtime
                # (``implements_summarizer_port(None)`` is False) but not to
                # the type checker, which sees ``Any | None`` and cannot infer
                # non-None from an opaque predicate.
                if model is not None and implements_summarizer_port(model):
                    await model.areset_after_summary(thread_id=self.name)

            # Contract (DESIGN §4.6): when BOTH sum_query and query are given,
            # they are TWO SEQUENTIAL user requests — "first the sum-query
            # request runs, and then the query request runs right after". The agent first
            # EXECUTES the sum-query (a full turn on the freshly-installed
            # summary-shape state, producing an AI response), and only THEN
            # executes the plain query as the next user message on top of that
            # result. NOT a single turn with [summary-shape, query] (the old
            # behaviour executed only the query and never ran the sum-query as
            # its own request — bug 2026-07-04).
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
            from deep_worker_cli.semantic_memory.client import SemanticMemoryClient

            reply = await SemanticMemoryClient().index_history(
                summaries_dir, clear_first=True
            )
            return {
                "files": reply.files_indexed,
                "lines": reply.lines_read,
                "chunks": reply.chunks_added,
                "errors": reply.errors,
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
            # A requested summarization IS a model turn — a cold path dumps the
            # history into a fresh session and can run for tens of seconds. It
            # used to leave the agent reading IDLE for that whole window, so
            # the web UI badge said "idle" while the agent was busy and nothing
            # discouraged the operator from sending a message into it (owner
            # report, 2026-08-24). Mirrors `_handle_invoke` (RUNNING at :3915).
            #
            # Inside the lock on purpose: `_invoke_lock` serialises this against
            # a real turn, so we can never clear a status that another turn owns.
            #
            # NOT routed through `is_running` / `current_worker_task`: those
            # drive the Interrupt control, and a summarization is not
            # user-interruptible — claiming otherwise would offer a cancel that
            # tears the history apart mid-replacement.
            # ``getattr``: the method must degrade to "no status reporting"
            # rather than AttributeError-out when ``session_manager`` was never
            # assigned (``AgentProcess.__new__`` in tests). Same defensive
            # shape as ``_invoke_lock`` a few lines above.
            session_manager = getattr(self, "session_manager", None)
            if session_manager:
                # See the twin call in ``_run_agent``: a starting turn
                # clears the previous turn's recoverable-error marker, and
                # only a starting turn does (§4.4). The mid-turn RUNNING
                # transitions in ``status_callback`` must leave it alone.
                await session_manager.clear_error()
                await session_manager.set_status(AgentStatus.RUNNING)
            # Terminal transition owned exactly once. Success -> IDLE (below,
            # right before the return). Failure -> ERROR for the fatal class
            # (POLICIES §12: automatic summarization is required to be "fatal
            # and loud, never a summary"; landing silently in IDLE on
            # ``SummarizationFailedError`` would recreate the very
            # silent-failure surface the requested path must not offer),
            # IDLE for non-fatal so a transient hiccup is not sticky. RUNNING
            # is always cleared. No ``finally`` — that would double-set the
            # status for the success branch and race the ERROR branch.
            try:
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
            except BaseException as exc:
                from deep_worker_cli.exceptions import (  # noqa: PLC0415
                    is_fatal_exception,
                )

                if session_manager:
                    if isinstance(exc, Exception) and is_fatal_exception(exc):
                        try:
                            await session_manager.set_error(
                                f"Summarization failed: {exc}"
                            )
                        except Exception:  # noqa: BLE001 — best-effort
                            await session_manager.set_status(AgentStatus.IDLE)
                    else:
                        await session_manager.set_status(AgentStatus.IDLE)
                raise

            if session_manager:
                await session_manager.set_status(AgentStatus.IDLE)
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
        inventory. See incident 2026-06-13 (#2). DO NOT reintroduce
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
            1. user-level ``mcp.json`` in the product home directory
            2. ``<working_dir>/.deep_worker/mcp.json``        (project)
            3. ``agents/<agent>/mcp.json`` in the product home    (per-agent)
            4. CLI ``--mcp-config`` / ``--mcp-server`` overrides

        Returns the started pool, or ``None`` if the mcp SDK is absent
        altogether (external MCP simply not available in this install).

        A CONFIGURED server that fails to start is NOT silent degradation:
        it raises ``ExternalMCPStartupError``, the init handler catches it
        and puts the agent into ``error`` state with a readable message.
        The agent still finishes booting, so it stays visible in the web UI
        and its history remains readable.
        """
        try:
            from deep_worker_cli.external_mcp.pool import (
                ExternalMCPClientPool,
            )
            from deep_worker_cli.external_mcp.config import (
                ExternalMCPStartupError,
                UnresolvedEnvError,
            )
        except Exception as e:  # pragma: no cover — mcp SDK absent
            _logger.info(f"external MCP disabled (import failed): {e}")
            return None

        from pathlib import Path

        from deep_worker_cli.external_mcp.config_store import scope_to_path

        # POLICIES §16: paths come from the single resolver, never rebuilt
        # from a literal here.
        user_path = scope_to_path("user")
        agent_path = scope_to_path(f"agent:{self.name}")
        working_dir = getattr(self.config, "working_dir", None)
        project_path = None
        if working_dir:
            from deep_worker_cli.project_utils import find_project_root

            start_path = Path(working_dir)
            project_root = find_project_root(start_path) or start_path
            project_path = scope_to_path("project", project_root=project_root)
        cli_configs = list(getattr(self, "mcp_config_paths", []) or [])
        cli_servers = list(getattr(self, "mcp_server_inline", []) or [])
        try:
            pool = await ExternalMCPClientPool.create(
                user_path=user_path,
                project_path=project_path,
                agent_path=agent_path,
                cli_configs=cli_configs,
                cli_servers=cli_servers,
                working_dir=working_dir,
            )
            _logger.info(
                f"external MCP pool ready for {self.name}: "
                f"{len(pool.tools)} adapter tool(s)"
            )
            return pool
        except UnresolvedEnvError:
            raise
        except BaseException as e:
            # No silent degradation: an external MCP failure MUST REACH the
            # initialization handler and bring the agent up in the ERROR
            # state with readable text. The former `return None` brought the
            # agent up silently, with no trace in the web UI — exactly what
            # the "silent failures are forbidden" policy prohibits.
            #
            # ``BaseException`` — because an unreachable HTTP server arrives
            # here as a ``CancelledError`` from the mcp SDK's anyio scope
            # (a related internal incident, 2026-08-29). An external
            # cancellation is re-raised as is — it must not be rewritten into
            # a "subsystem error".
            if is_external_cancellation(e):
                raise
            _logger.warning(
                f"external MCP pool failed to start for {self.name}: {e}",
                exc_info=True,
            )
            raise ExternalMCPStartupError(
                f"external MCP startup failed ({root_cause_text(e)}). "
                f"Servers: {self._describe_mcp_servers(user_path, project_path, agent_path, cli_configs, cli_servers, working_dir)}"
            ) from e

    @staticmethod
    def _describe_mcp_servers(
        user_path: Path | None,
        project_path: Path | None,
        agent_path: Path | None,
        cli_configs: list[Path] | None = None,
        cli_servers: list[str] | None = None,
        working_dir: Path | str | None = None,
    ) -> str:
        """``name -> url|command`` for each server in the merged config.

        Needed so the web UI message names the SPECIFIC server and address —
        otherwise the owner sees "MCP failed to start" and still has to go to
        the logs. Best-effort by itself: if the config is already unreadable,
        this must not replace the original cause of the failure.
        """
        try:
            from deep_worker_cli.external_mcp.config import load_mcp_config

            merged = load_mcp_config(
                user_path=user_path,
                project_path=project_path,
                agent_path=agent_path,
                cli_configs=list(cli_configs or []),
                cli_servers=list(cli_servers or []),
                working_dir=working_dir,
            )
        except Exception as e:  # noqa: BLE001 — the description matters less than the cause
            return f"<config unreadable: {type(e).__name__}: {e}>"
        if not merged:
            return "<none configured>"
        return ", ".join(
            f"{name} -> {spec.url or spec.command or '?'}"
            for name, spec in sorted(merged.items())
        )

    async def cleanup(self) -> None:
        """Clean up resources, including checkpointer.

        This method should be called on agent shutdown to properly close
        database connections and release resources.
        """
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            except Exception as e:
                # Write to stderr AND logger. cleanup() runs during shutdown,
                # when the logging pipeline may already be partially torn down;
                # stderr is the last-resort visible channel. Tests assert
                # visibility via capsys.readouterr().err.
                print(f"Warning: Error closing checkpointer: {e}", file=sys.stderr)
                _logger.info(f"Warning: Error closing checkpointer: {e}")
            finally:
                self._checkpointer_ctx = None

        # External MCP pool — properly tear down stdio clients and child
        # processes. See docs/DESIGN_EXTERNAL_MCP_TOOLS.md §3.4.
        if self._external_mcp_pool is not None:
            try:
                await self._external_mcp_pool.stop()
            except Exception as e:  # noqa: BLE001
                # Same rationale as the checkpointer branch above: shutdown
                # path, dual channel for maximal visibility.
                print(f"Warning: Error stopping external MCP pool: {e}", file=sys.stderr)
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

        # 3. Persist state.
        # ERROR is a diagnostic terminal state — set by set_error() when the
        # agent could not initialise (missing MCP env, unresolved provider,
        # etc.) or when a turn failed fatally. It carries error_message that
        # the Web UI renders as a prominent banner. Blindly overwriting it
        # with SHUTDOWN here would hide the reason the agent died and violate
        # the "errors must be VISIBLE and PERSISTENT" contract enforced by
        # TestInitErrorVisibility (see also No-Agent-Error-State policy).
        # SHUTDOWN is meaningful only as a transition from a live state.
        if self.session_manager and self.session_manager.status != AgentStatus.ERROR:
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
            from deep_worker.providers.deepworker_a2a.config import A2AConfig
            from deep_worker.providers.deepworker_a2a.server import maybe_start_a2a_server
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

        Priming is ALL-OR-NOTHING across the ENTIRE method, not merely the seq
        scan: ``_inbox_seq_primed`` is set at the very end, once the ledger
        load, sweep and compaction have also succeeded. A partially-failed
        prime must never leave the counter at its ``0`` default while records
        with higher seqs survive on disk — that would publish a new arrival
        *ahead* of them and break the ``next_seq = max(pending ∪ quarantined)
        + 1`` contract — and must never leave the process marked primed on a
        spool whose ledger we refused to trust. Any ``OSError`` propagates with
        the flag left False so the enqueue path can retry or refuse loudly.

        Idempotent and safe to call twice (once before the RPyC server starts,
        once from :meth:`bring_up_inbox`). Once primed it will NOT rescan or
        reset a live counter, since by then the server may already have
        allocated seqs.

        Returns the number of pending records recovered from a previous life.
        """
        from deep_worker_cli.process import inbox_spool

        inbox_spool.ensure_spool(self.data_dir)
        already_primed = self._inbox_seq_primed
        next_seq = self._inbox_next_seq
        if not already_primed:
            inbox_spool.cleanup_tmp(self.data_dir)
            # Computed here, but NOT published to self yet — see the
            # all-or-nothing assignment at the end of this method.
            next_seq = inbox_spool.next_seq(self.data_dir)
        self._inbox_delivered = inbox_spool.load_delivered(self.data_dir)
        # Survives restart on purpose: "already in context" is a property of
        # the persisted conversation, not of this process. Loading it here is
        # what stops a resend from being re-admitted just because the receiver
        # was restarted (observed live: the same message re-delivered across
        # several PIDs, each time as a brand-new delivery_id).
        self._inbox_received = inbox_spool.load_received(self.data_dir)
        self._sweep_inbox()
        # Compact AFTER the sweep: the sweep is what determines which records
        # are still pending, and only those ids can suppress anything.
        #
        # The INTERSECTION with the ledger is load-bearing, not decoration.
        # Writing every pending id would mark records that were NEVER DELIVERED
        # as delivered: this process might still run them (its in-memory set is
        # separate), but a crash before that turn leaves the next startup
        # sweeping them away unlaunched — a silent loss, exactly what this
        # subsystem exists to prevent. Only ids ALREADY in the ledger, i.e.
        # already durably delivered, may be kept.
        pending_ids = {
            (inbox_spool.read_record(p) or {}).get("delivery_id")
            for p in inbox_spool.pending_records(self.data_dir)
        }
        inbox_spool.compact_ledger(
            self.data_dir,
            self._inbox_delivered & {i for i in pending_ids if isinstance(i, str)},
        )
        pending = inbox_spool.pending_records(self.data_dir)
        # ALL-OR-NOTHING, and only HERE. Priming must cover the whole startup
        # chain — cleanup_tmp, next_seq, load_delivered, sweep, compact, pending
        # scan — not just the seq scan. Setting the flag earlier meant a raise
        # from load_delivered (which now fails closed on a symlinked or foreign
        # ledger) still left the process "primed": bring_up_inbox logs and
        # swallows that OSError, and publish_inbox_message would then skip its
        # late-prime retry and publish into a spool that was never swept. The
        # sender would be ack'd on a spool we refused to validate.
        self._inbox_next_seq = next_seq
        self._inbox_seq_primed = True
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

        # NO-RESEND RULE. A message is SENT the moment it is in the recipient's
        # inbox — that is what the sender is acknowledged for — so from that
        # moment a second send of the same message is forbidden. The ban is
        # therefore armed on PUBLISH, not on delivery into the context: waiting
        # for the turn would leave a window in which the message is already
        # queued, already acknowledged, and still resendable.
        #
        # The receiver-side ledger cannot enforce this, because every send mints
        # a fresh delivery_id and a resend is indistinguishable from new mail by
        # identity alone — that is precisely how the same review verdict entered
        # one context five times. Identity must come from CONTENT.
        #
        # Refusal is HARD and visible: the sender gets success:false. A silent
        # drop would hide a broken resender behind an apparent success, and a
        # component that resends automatically must be seen, not accommodated.
        fingerprint = inbox_spool.content_fingerprint(message, reply_back)
        if fingerprint in self._inbox_received:
            _logger.warning(
                "[INBOX] REFUSED resend: this message was already sent to this "
                "agent (fingerprint=%s). Automatic resends are forbidden.",
                fingerprint,
            )
            raise inbox_spool.InboxDuplicateMessageError(
                f"This exact message was already sent to '{self.name}'. "
                "Automatic resends of an already-sent message are forbidden; "
                "if this was a retry, the original delivery already succeeded."
            )

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
            # Arm the ban INSIDE the lock and only after publish() returned,
            # i.e. once the record is fsynced and the sender is entitled to its
            # ack. Recording it before the write would ban a message that was
            # never actually queued if publish() then raised; recording it
            # outside the lock would let two concurrent sends of the same text
            # both pass the check above and both enqueue.
            try:
                inbox_spool.append_received(self.data_dir, fingerprint)
            except OSError:
                # The message IS queued, so the ban must hold regardless; an
                # in-memory record still stops resends for this process's life.
                _logger.warning(
                    "[INBOX] could not persist the no-resend fingerprint for a "
                    "published message; the ban holds in memory but would not "
                    "survive a restart",
                )
            self._inbox_received.add(fingerprint)
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

    def _sweep_inbox(self) -> None:
        """Startup sweep: drop already-delivered records, retire expired ones.

        This is what disarms a record whose ``unlink`` was lost — the failure
        observed live on a production agent, where two 13-day-old records sat pending
        while their ``delivery_id``s were already in the agent's own history.
        Neither is delivered again: the ledgered one is unlinked silently (its
        turn already ran), the aged-out one is retired LOUDLY to ``expired/``.
        """
        from deep_worker_cli.process import inbox_spool

        for path in inbox_spool.pending_records(self.data_dir):
            record = inbox_spool.read_record(path)
            if record is None:
                continue  # left for the drain to quarantine
            delivery_id = record.get("delivery_id")
            if isinstance(delivery_id, str) and delivery_id in self._inbox_delivered:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - best effort
                    _logger.warning("[INBOX] could not unlink delivered %s", path)
                _logger.info(
                    "[INBOX] suppressed already-delivered record id=%s "
                    "(its unlink was lost)", delivery_id,
                )
                continue
            if inbox_spool.record_age_seconds(record) > inbox_spool.RECORD_TTL:
                inbox_spool.retire(self.data_dir, path, record, "ttl")

    def _report_foreign_entries(self) -> None:
        """WARN once per filename per process about non-record files.

        Left in place deliberately: moving another component's file is how you
        lose someone else's data. Warned once (not once per drain pass) so a
        file we refuse to move cannot spam the log on every arrival forever.
        """
        from deep_worker_cli.process import inbox_spool

        for path in inbox_spool.foreign_entries(self.data_dir):
            if path.name in self._inbox_foreign_warned:
                continue
            self._inbox_foreign_warned.add(path.name)
            try:
                mode = oct(path.stat().st_mode & 0o777)
            except OSError:  # pragma: no cover - best effort
                mode = "?"
            _logger.warning(
                "[INBOX] foreign file in spool: %s mode=%s — not a delivery "
                "record, ignored and left in place", path.name, mode,
            )

    async def _drain_inbox_once(self) -> None:
        """One pass over a snapshot of the spool.

        Each record is attempted AT MOST ONCE per wake (§3.3.1a). A retained
        failure is not re-read within the pass and does not re-arm the event;
        it is retried on the next arrival or at startup.

        ORDER: never-attempted records first, then previously-attempted ones,
        each in ``seq`` order. Walking plain ``seq`` order meant a retained
        record was always attempted BEFORE the fresh message that woke the
        drain — which is how a stale verdict kept arriving ahead of current
        traffic. Partitioning on ``attempted`` (not on the failure counter) is
        load-bearing: a user interrupt does not touch the failure counter, so
        keying on that would leave an interrupted record "fresh" forever.
        """
        from deep_worker_cli.process import inbox_spool

        self._report_foreign_entries()

        fresh: list[tuple[Path, dict[str, Any]]] = []
        retained: list[tuple[Path, dict[str, Any]]] = []
        for path in inbox_spool.pending_records(self.data_dir):
            record = inbox_spool.read_record(path)
            if record is None:
                inbox_spool.quarantine(self.data_dir, path)
                continue
            delivery_id = record.get("delivery_id")
            if isinstance(delivery_id, str) and delivery_id in self._inbox_delivered:
                # Its turn already ran; the unlink was lost. Suppress, do not
                # replay — this is the whole point of the ledger.
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - best effort
                    _logger.warning("[INBOX] could not unlink delivered %s", path)
                _logger.info(
                    "[INBOX] suppressed duplicate delivery id=%s", delivery_id
                )
                continue
            if inbox_spool.record_age_seconds(record) > inbox_spool.RECORD_TTL:
                inbox_spool.retire(self.data_dir, path, record, "ttl")
                continue
            if record.get("attempted"):
                retained.append((path, record))
            else:
                fresh.append((path, record))

        for path, record in fresh + retained:
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
        from deep_worker_cli.process import inbox_spool

        delivery_id = record.get("delivery_id")
        started = time.monotonic()
        age = inbox_spool.record_age_seconds(record)
        message = record["message"]
        if age > inbox_spool.STALE_MARK:
            # Marked in the BODY, not just the log: only this lets the
            # recipient model discount content that may be obsolete.
            message = (
                f"[STALE MESSAGE — queued {inbox_spool._format_age(age)} ago, "
                f"delivered late; may be obsolete]\n{message}"
            )
        child = asyncio.create_task(self._handle_invoke(
            {
                "command": "invoke",
                "message": message,
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
            # record, stay alive (§3.3.0b). It IS marked attempted — so it stops
            # overtaking fresh mail — but is NOT charged toward retirement: a
            # human interrupt is not the message's fault, and charging it would
            # let three interrupts silently expire a message nobody read.
            self._mark_attempt(path, record, failed=False)
            _logger.info(
                "[INBOX] delivery %s interrupted; record retained", delivery_id
            )
            return
        except Exception:  # noqa: BLE001 - per-record containment (§3.3.4)
            # One broken record must not strand the rest of the batch.
            self._mark_attempt(path, record, failed=True)
            _logger.exception(
                "[INBOX] delivery %s failed; record retained", delivery_id
            )
            return

        # DELETE BOUNDARY — reached only on a normal return of the await above.
        # Never in a `finally`: a cancelled or failed turn must keep its record.
        #
        # A successful return is NECESSARY but NOT SUFFICIENT. Delivery means
        # the message was recorded in the model's context, which is the
        # checkpoint commit; the turn can return between the last write and
        # that commit, and unlinking there loses a message the sender was
        # already told had been accepted.
        if (
            isinstance(result, dict)
            and result.get("success")
            and not self._delivery_is_committed(delivery_id)
        ):
            self._mark_attempt(path, record, failed=False)
            _logger.info(
                "[INBOX] delivery %s returned success but is not yet in a "
                "committed checkpoint; record retained until it is",
                delivery_id,
            )
            return

        if isinstance(result, dict) and result.get("success"):
            # LEDGER BEFORE UNLINK. A crash between the two leaves the record
            # pending with its id already durable, so the next pass suppresses
            # it instead of replaying it. Reversing this order re-creates the
            # unbounded-replay defect; skipping the ledger's directory fsync on
            # first create turns it into silent LOSS (§3.1.1).
            try:
                if isinstance(delivery_id, str):
                    inbox_spool.append_delivered(self.data_dir, delivery_id)
                    self._inbox_delivered.add(delivery_id)
            except OSError:
                # Could not make the delivery durable: keep the record. One
                # visible duplicate beats a silently lost message.
                _logger.exception(
                    "[INBOX] could not record delivery %s in the dedup ledger; "
                    "record retained (it may be delivered again)", delivery_id,
                )
                return
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best effort
                _logger.warning("[INBOX] could not unlink delivered %s", path)
            _logger.info(
                "[INBOX] delivered id=%s arrival=%s in %.3fs",
                delivery_id, record.get("ts"), time.monotonic() - started,
            )
        else:
            self._mark_attempt(path, record, failed=True)
            _logger.warning(
                "[INBOX] delivery %s did not succeed; record retained for retry",
                delivery_id,
            )

    def _on_delivery_committed(self, delivery_ids: frozenset[str]) -> None:
        """Checkpointer callback: these messages are now durably in context.

        Runs on the commit path, so it only records the fact — the drain
        applies it. Kept deliberately trivial: anything heavier here would
        run inside the persistence path.
        """
        self._inbox_committed.update(delivery_ids)

    def _delivery_is_committed(self, delivery_id: Any) -> bool:
        """Whether the delete boundary may drop this record.

        The proof is a checkpoint commit carrying this ``delivery_id``.
        When the saver cannot commit at all (InMemorySaver / --memory-only)
        no such proof can ever arrive; retaining every record forever would
        replay the whole spool on each wake, so in that configuration the
        turn result stays the boundary — the pre-existing behaviour, now
        confined to the case that genuinely has no better signal.
        """
        if not self._inbox_commit_signal_available:
            return True
        return isinstance(delivery_id, str) and delivery_id in self._inbox_committed

    def _mark_attempt(
        self, path: Path, record: dict[str, Any], *, failed: bool
    ) -> None:
        """Persist the two counters, then retire the record if it is spent.

        ``attempted`` is set on every outcome (ordering); ``failure_attempts``
        only on a genuine failure (retirement). Both live IN the record so they
        survive a restart — which is the entire point, since the replay this
        prevents is itself a cross-restart phenomenon.
        """
        from deep_worker_cli.process import inbox_spool

        record["attempted"] = True
        if failed:
            record["failure_attempts"] = int(record.get("failure_attempts", 0)) + 1
        try:
            inbox_spool.update_record(path, record)
        except OSError:  # pragma: no cover - best effort
            _logger.warning("[INBOX] could not update counters on %s", path)
            return
        if int(record.get("failure_attempts", 0)) >= inbox_spool.MAX_FAILURES:
            inbox_spool.retire(self.data_dir, path, record, "max_failures")

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
                # — the agent keeps working. 2026-05-22.
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

    async def _checkpoint_config(self) -> Any:
        """``RunnableConfig`` addressing this agent's thread in the checkpoint."""
        from langchain_core.runnables import RunnableConfig

        config: RunnableConfig = {
            "configurable": {"thread_id": self.name},
            "metadata": {
                "assistant_id": self.name,
                "agent_name": self.name,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        }
        return config

    async def _load_checkpoint_messages(self) -> list[Any] | None:
        """Messages for this thread, readable WITHOUT a working graph.

        Reading history needs the checkpoint and nothing else. Routing it
        through ``self.agent.aget_state`` tied it to a fully built graph, so
        an agent that failed to initialise answered "Agent not initialized"
        even though its conversation sat intact on disk — the web UI then
        showed a 500 (Full Dialog) or, worse, a silently empty history.

        Prefers the graph when there is one: it owns the same checkpointer and
        keeps whatever state logic it layers on top. Falls back to reading the
        checkpointer directly, which is exactly what the graph would do.

        Returns ``None`` when there is no way to read at all (no graph AND no
        checkpointer, e.g. ``--memory-only`` that never got built) — the
        caller reports that as a failure rather than as an empty history,
        because "cannot read" and "nothing to read" must not look alike.
        """
        config = await self._checkpoint_config()

        if self.agent is not None:
            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return []
            return list(state.values.get("messages", []))

        checkpointer = self._checkpointer
        if checkpointer is None:
            return None

        tup = await checkpointer.aget_tuple(config)
        if not tup or not tup.checkpoint:
            return []
        channel_values = tup.checkpoint.get("channel_values", {}) or {}
        return list(channel_values.get("messages", []))

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
        # call (2026-05-23; first seen 2026-05-08, see a related internal
        # incident analysis (not included here)). Keep ONLY
        # bounded breadcrumbs (sizes/counts, never content), and at DEBUG so a
        # polled hot path is silent by default. NEVER interpolate ``state``,
        # ``self.agent``, ``config``, or any message into a log record.
        logger.debug("[GET_HISTORY] start (limit=%s, offset=%s)", limit, offset)

        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

            # History lives in the checkpoint, so a broken graph must not hide
            # it. Do NOT log the messages — see the log-storm guard above;
            # only a bounded COUNT is logged below.
            messages = await self._load_checkpoint_messages()
            if messages is None:
                logger.error("[GET_HISTORY] no checkpointer and no graph")
                return {
                    "success": False,
                    "error": (
                        "History unavailable: agent has neither an initialised "
                        "graph nor a checkpointer to read from"
                    ),
                }
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

        Mode #4 of the Full Dialog verbosity combo (§9 of
        DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md).

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
                    tool_selection=getattr(self, "_tool_selection", None),
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
        ``_extract_final_tool_metadata`` helper. The list MAY depend on the
        active provider. Each metadata entry is reduced to name + description;
        the JSON input schema is intentionally omitted from this view.
        Metadata entries may be dicts or ToolMetadata-like objects.

        That helper reads ``graph.nodes["tools"]``, which holds the FULL
        registry — tool selection is enforced a layer above it, on
        ``request.tools`` in ``ToolSelectionMiddleware.wrap_model_call``, and
        the graph keeps every tool on purpose so ``ToolNode`` can refuse a
        denied call by name. So the raw inventory is NOT what the model is
        offered, and reporting it unfiltered told an operator that an agent
        launched with ``--tools '*,project_memory'`` had all 23 tools when it
        could call one (observed 2026-08-30 on agent ``tsel_probe``).

        Applying the selection here is the same choice
        ``_handle_get_system_prompt`` already makes: an introspection view
        answers "what can this agent do", so it must reflect the restriction
        rather than the registry behind it.
        """
        provider = getattr(getattr(self, "config", None), "provider", None)
        try:
            metadata = type(self)._extract_final_tool_metadata(
                self.agent, provider=provider
            )
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"tools unavailable: {e}"}

        # ``getattr`` because a process predating tool selection has no such
        # attribute; ``apply`` is a no-op for the empty (default) selection,
        # so the unrestricted path is untouched.
        selection = getattr(self, "_tool_selection", None)
        if selection is not None:
            metadata = selection.apply(metadata or [])

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

        try:
            from langchain_core.messages import AIMessage

            # Same rule as get_history: the last response lives in the
            # checkpoint, so a failed init must not hide it.
            messages = await self._load_checkpoint_messages()
            if messages is None:
                return {
                    "success": False,
                    "error": (
                        "Last response unavailable: agent has neither an "
                        "initialised graph nor a checkpointer to read from"
                    ),
                }

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
    @staticmethod
    def _provider_is_usable(provider: str) -> bool:
        """Whether restarting onto ``provider`` could actually authenticate.

        Config-driven providers carry their key in an env var named by
        ``providers.json``; if that variable is unset, a restart would spawn an
        agent that cannot talk to anything. Checking here — at failover time
        rather than at config-load time — is deliberate: a missing key must not
        stop the CLI from starting, but it must stop us from sending a working
        agent to a dead endpoint (owner directive 2026-09-03: don't send it
        to a dead address).

        Built-in providers are assumed usable: their credentials live in the
        vendor CLI's own store (``claude``, ``codex``), which we cannot inspect
        and must not second-guess.
        """
        try:
            from deep_worker_cli.runtime.provider_registry import (  # noqa: PLC0415
                load_provider_specs,
            )

            for spec in load_provider_specs():
                if spec.name == provider:
                    return spec.has_credentials()
        except Exception as exc:  # noqa: BLE001 — config trouble must not block failover
            _logger.info(
                f"[FAILOVER] Could not check credentials for {provider!r}: "
                f"{exc!r} — treating it as usable"
            )
        return True

    async def _schedule_failover(self, provider: str, marker_text: str) -> None:
        """Record ``provider`` failure in availability tracker and, if
        a next available provider exists in the failover chain, schedule
        a restart via ``deep-worker --restart <name> --provider <next>
        --no-update-preferred``.

        Fire-and-forget: runs as a background task so the caller's error
        response is not delayed.

        Args:
            provider: The provider that hit the usage limit
                (e.g. ``"claude-code-sdk"``, ``"codex"``).
            marker_text: The raw error marker from the provider's error
                event, which may contain a human-readable reset time.
        """
        from deep_worker_cli.runtime.availability_tracker import (  # noqa: PLC0415
            AvailabilityTracker,
            parse_limit_reset,
        )
        from deep_worker_cli.runtime.failover_chains import (  # noqa: PLC0415
            get_chain,
            resolve_step_model,
        )
        from deep_worker_cli.runtime.provider_chain import (  # noqa: PLC0415
            failover_candidates,
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

            # No provider is terminal by name any more: "terminal" means
            # "last step of this agent's chain", and the chain is operator
            # config (owner directive 2026-09-03). An agent whose provider is
            # the final step simply gets no candidates below.
            chain_name = ""
            if isinstance(self.launch_config, dict):
                chain_name = self.launch_config.get("failover_chain") or ""
            current_model = None
            if isinstance(self.launch_config, dict):
                current_model = self.launch_config.get("model")

            # Advance to the next available step, skipping same-account
            # siblings (a usage cap is account-level — retrying the same
            # subscription fails again; DESIGN_ACP_PROVIDER_LAYER.md §7).
            chain = get_chain(chain_name)
            for step in failover_candidates(provider, chain=chain, model=current_model):
                candidate = step.provider
                if tracker.is_available(candidate) and self._provider_is_usable(candidate):
                    step_model = resolve_step_model(step)
                    _logger.info(f"[FAILOVER] Agent '{self.name}' failed on {provider}; "
                        f"scheduling restart with {candidate}"
                        f"{f' (model={step_model})' if step_model else ''} "
                        f"[chain={chain.name}] …")
                    # Write the restart-notice BEFORE spawning so the new
                    # process consumes it on startup and
                    # ``_auto_continue_after_restart`` CONTINUES the pending
                    # task instead of going idle (incident 2026-06-17).
                    # Same notice format as ``restart_self``.
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
                    failover_cmd = [
                        "deep-worker",
                        "--restart",
                        self.name,
                        "--provider",
                        candidate,
                        "--no-update-preferred",
                    ]
                    # A step may pin a model (exact id or family alias). It
                    # wins over the provider default for this position —
                    # that is the point of writing it in the chain.
                    if step_model:
                        failover_cmd += ["--model", step_model]
                    proc = await asyncio.create_subprocess_exec(
                        *failover_cmd,
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
        deepworker_a2a §4.1). ``None`` (default) preserves the existing
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
        # REQ-90: clear the mid-turn compaction report BEFORE anything can
        # return, so this turn can never inherit the previous turn's incident.
        # Without the reset, one compaction would mark every later document of
        # the run as compacted too.
        self._reset_turn_compaction_report()
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
            # A new turn supersedes the previous turn's recoverable-error
            # marker (docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §4.4);
            # without this the marker set on the cap path would survive
            # every later success and the web would show it forever.
            #
            # Deliberately NOT done at the ``status_callback`` RUNNING
            # transitions: those are mid-turn returns to RUNNING after a
            # tool finishes, so clearing there would erase the marker
            # inside the very turn that produced it.
            await self.session_manager.clear_error()
            await self.session_manager.set_status(AgentStatus.RUNNING)
            # The counter existed but nothing in production ever advanced it,
            # so every consumer read a permanent 0. This is the one place a
            # user-visible turn begins, which is exactly what the field
            # claims to count. ``increment_message_count`` also re-stamps
            # ``last_activity``, matching the ``set_status`` above.
            await self.session_manager.increment_message_count()

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
                # conversation history (deepworker_a2a §4.1).
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

                    # Initialize TaskManager if enabled (see
                    # _active_task_manager: the off switch gates the loop,
                    # not just the methods it calls).
                    active_task_manager = self._active_task_manager()
                    if active_task_manager is not None:
                        await active_task_manager.init_task(
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
                        if self._active_task_manager() is not None:
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
                        active_task_manager = self._active_task_manager()
                        if active_task_manager is not None:
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

                            corrective_prompt = await active_task_manager.on_subordinate_response(
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
                    return self._with_compaction_flag({
                        "success": True,
                        "response": {
                            "role": "assistant",
                            "content": full_response,
                        },
                    })
                else:
                    # Non-streaming mode: wait for complete response
                    # If TaskManager enabled, wrap in evaluation loop.
                    #
                    # The gate asks whether the manager is ACTIVE, not merely
                    # whether the object exists. Gating on existence alone let a
                    # manager the operator had switched off keep driving the
                    # loop: init_task and on_subordinate_response each return
                    # early when disabled, so the loop span and broke
                    # immediately, but it still logged "TaskManager iteration"
                    # and reached the middleware (rg1_worker, 2026-10-10 — 14
                    # iterations after the disable, with no enable in between).
                    active_task_manager = self._active_task_manager()
                    if active_task_manager is not None:
                        # Initialize task with user's message
                        await active_task_manager.init_task(
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
                            corrective_prompt = await active_task_manager.on_subordinate_response(
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

                            return self._with_compaction_flag({
                                "success": True,
                                "response": {
                                    "role": role,
                                    "content": content,
                                },
                            })
                        else:
                            # Terminal failure: settle status so the agent
                            # is not left dangling at RUNNING forever (the
                            # 2026-05-20 "forever working" bug).
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
                                await self.session_manager.set_status(
                                    AgentStatus.IDLE,
                                    error_message=(
                                        "EmptyModelResponse: The model "
                                        "returned no messages."
                                    ),
                                )
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
                        # A registered turn-ending tool
                        # (deep_worker_cli/middleware/tool_completion.py)
                        # can end the LangGraph turn immediately after a
                        # successful tool call with NO subsequent AIMessage.
                        # The tail of ``state['messages']`` is therefore the
                        # successful ``ToolMessage`` itself — that IS the
                        # completed turn. When ``_handle_invoke`` sees such
                        # a tail it sets this flag, breaks, and returns
                        # success with empty content (so the client does not
                        # see the misleading "Agent did not produce a
                        # response" error surface reported by rg1_worker
                        # 2026-08-22 for repo_graph's ``submit_*`` tools).
                        _terminal_tool_success = False

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

                            # Turn-ending tool: a trailing successful
                            # ToolMessage is a legitimate turn end (see
                            # ToolCompletionMiddleware). We inspect the tail
                            # explicitly rather than relying on the loop's
                            # ``role``/``content`` derivation, because a
                            # ToolMessage has role='tool' with a non-empty
                            # JSON content string — which falls through
                            # every other branch and loops until the cap.
                            #
                            # We check ONLY ``last_msg``. The "sole trailing
                            # ToolMessage" requirement is enforced upstream by
                            # ToolCompletionMiddleware
                            # (``_trailing_tool_message_run_length(...) != 1``),
                            # which refuses to end the turn on a parallel
                            # tool-call batch. A second gate here would
                            # duplicate that logic; ``ainvoke`` returning a
                            # ToolMessage tail already means the graph ended.
                            #
                            # FAIL-CLOSED status check: LangChain defaults
                            # ``ToolMessage.status`` to ``'success'`` for
                            # producer convenience, but on this consumer
                            # seam we require an EXPLICIT success signal.
                            # ``getattr(..., 'status', None) == 'success'``
                            # rejects a missing / ``None`` / ``'error'``
                            # status; only the LangChain literal
                            # ``'success'`` counts. A tool that forgets to
                            # mark itself failed does NOT get to claim
                            # success by default.
                            if role == "tool":
                                if isinstance(last_msg, dict):
                                    tail_status = last_msg.get("status")
                                else:
                                    tail_status = getattr(
                                        last_msg, "status", None
                                    )
                                if tail_status == "success":
                                    _terminal_tool_success = True
                                    _logger.info(
                                        "[INVOKE] Turn ended by a successful "
                                        "tool tail (name=%r); no further model "
                                        "response expected",
                                        (
                                            last_msg.get("name")
                                            if isinstance(last_msg, dict)
                                            else getattr(
                                                last_msg, "name", None
                                            )
                                        ),
                                    )
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
                                await self.session_manager.set_status(
                                    AgentStatus.IDLE,
                                    error_message=(
                                        "MaxIterationsExceeded: The agent hit "
                                        "its per-turn iteration cap without "
                                        "producing a final response."
                                    ),
                                )
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

                        if not final_content and not _terminal_tool_success:
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
                                await self.session_manager.set_status(
                                    AgentStatus.IDLE,
                                    error_message=f"NoFinalContent: {_err}",
                                )
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
                        # Skip the "assistant" broadcast on a turn-ending
                        # tool: there IS no assistant text — emitting an
                        # empty one manufactures a ghost message identical
                        # to the one ToolCompletionMiddleware refuses to
                        # inject (see its ``echo_field=None`` branch).
                        if notify_callback is not None and not _terminal_tool_success:
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

                        return self._with_compaction_flag({
                            "success": True,
                            "response": {
                                "role": "assistant",
                                "content": final_content,
                            },
                        })
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
            from deep_worker_cli.exceptions import (
                is_fatal_exception,
                is_rate_limit_error as _is_rate_limit_error,
            )

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
                await self._schedule_failover("claude-code-sdk", str(e))
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

            # LEGACY. Live grok no longer reaches this branch: since it moved
            # onto the shared ACP client (2026-08-24) its usage cap raises
            # ``AcpUsageLimitError``, handled below with the provider taken
            # from config. Kept because ``grok_provider.limit_errors`` still
            # defines the class and a stale checkpoint / in-flight error can
            # still carry it; routing it to failover("grok") stays correct.
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

            if exc_name == "AcpUsageLimitError":
                # The exception class names the TRANSPORT (the shared ACP
                # client), not the ACCOUNT that hit the cap. Every CLI on that
                # client — claude-code, grok, and any future ACP provider —
                # raises this one class, so the failover target and the
                # availability record MUST come from ``self.config.provider``.
                #
                # Hardcoding "claude-code" here was correct only while ACP had
                # a single provider. When grok moved off its private fork onto
                # the shared client (2026-08-24) it stopped raising
                # ``GrokUsageLimitError`` and started raising this class, so a
                # grok cap would have (1) marked claude-code — a provider that
                # never failed — unavailable in the tracker, and (2) failed
                # over to ``failover_candidates("claude-code")`` =
                # [codex, grok, deepseek], i.e. possibly straight back to the
                # provider that had just capped, instead of
                # ``failover_candidates("grok")`` = [deepseek].
                #
                # Same principle as the config-driven branch below: identity
                # from config, never from the error's class name or text.
                acp_provider = (
                    getattr(getattr(self, "config", None), "provider", None)
                    or "claude-code"
                )
                limit_msg = f"ACP usage limit reached — {str(e)}"
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
                await self._schedule_failover(acp_provider, str(e))
                return {
                    "success": False,
                    "error": limit_msg,
                }

            # Config-driven providers (providers.json, OpenAI dialect) have no
            # typed usage-limit class of their own: their cap arrives as a
            # generic 429. Without this branch the cap was never recorded, so
            # the tracker stayed empty and no failover fired — two agents sat
            # stalled on a provider whose cap had already lifted
            # (incident 2026-08-17).
            #
            # The provider identity comes from ``self.config.provider``, NEVER
            # from the error text: config-driven providers are added by editing
            # providers.json, so sniffing for a vendor name would mis-attribute
            # every provider added after this code was written.
            if _is_rate_limit_error(e):
                from deep_worker_cli.runtime.provider_chain import (  # noqa: PLC0415
                    SUPPORTED_PROVIDERS,
                )

                active_provider = getattr(
                    getattr(self, "config", None), "provider", None
                )
                if active_provider and active_provider not in SUPPORTED_PROVIDERS:
                    limit_msg = f"{active_provider} usage limit reached — {str(e)}"
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
                    await self._schedule_failover(active_provider, str(e))
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
                # IDLE per POLICIES §1, but not silently: the marker is
                # what makes the failure visible in the web UI, which
                # reads status rather than the error log (§4).
                await self.session_manager.set_status(
                    AgentStatus.IDLE,
                    error_message=f"{exc_name}: {str(e)}",
                )

            return {
                "success": False,
                "recovered": True,
                # For a provider/API error, the return value carries the SAME
                # real API text the UI notification did (POLICIES.md §9), so a
                # programmatic consumer of result["error"] sees the real cause,
                # not the generic "type continue" wrapper (review nit
                # 2026-07-04). Generic recoverable errors keep the wrapper.
                "error": notify_content if is_api_error else recoverable_msg,
            }

        finally:
            # Only the active owner may clear shared execution state. The
            # identity check also protects focused callers that bypass the
            # serialization wrapper in tests.
            if self.current_worker_task is asyncio.current_task():
                self.is_running = False
                self.current_worker_task = None

                # End-of-turn handling for the self-restart barrier.
                #
                # Reaching here with a restart still armed means the
                # ``restart_self`` result was never committed to the
                # checkpoint, so it never became part of the model's
                # context — provider error, cancelled turn, crash before
                # the commit.
                #
                # ⚠️ We DISARM; we do not spawn. An earlier revision fired
                # the driver here on the theory that "the turn is over, so
                # nothing is in flight". That does not follow: the turn can
                # end precisely BECAUSE the result never got out, and the
                # transcript may already hold a dangling ``tool_use``.
                # Stopping the process at that point is what makes the CLI
                # fabricate a "user-rejected" denial — the defect this
                # whole mechanism exists to prevent. An agent that visibly
                # failed to restart is recoverable; a fabricated denial
                # replayed into the next process is not.
                try:
                    from deep_worker_cli.process.self_restart_barrier import (  # noqa: PLC0415
                        disarm_undelivered_self_restart,
                    )

                    disarm_undelivered_self_restart()
                except Exception:  # noqa: BLE001 - never break turn teardown
                    _logger.exception(
                        "Self-restart disarm raised during turn teardown"
                    )
