"""CodexChatModel — LangChain BaseChatModel that drives Codex CLI.

This module is the public surface of the Codex provider. Same shape as
``claude_code_sdk_provider/chat_model.py:ClaudeCodeChatModel`` — only
``ainvoke`` is real; ``_generate`` raises (async-only); LangChain
streaming methods are not implemented.

See ``docs/DESIGN_CODEX_PROVIDER.md`` for the design rationale.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Awaitable, Callable, Sequence

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, PrivateAttr

from .client_state import get_global_client_state
from .custom_types import ToolMetadata
from .exec_runner import (
    AgentMessageItem,
    CodexExecRunner,
    CodexProtocolError,
    ExecArgs,
    ExecHandle,
    GenericItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from .proxy_mcp_server import get_global_proxy_mcp_server


# Codex built-in deny-list (Q5). Codex CLI 0.130 disables native tools via
# per-feature flags (``--disable <feature>``), NOT via the older
# ``--config disabled_tools=[...]`` channel — that channel is silently
# ignored (verified live via ``RUST_LOG=codex_api=trace``: the
# ``response.create`` websocket payload's ``tools=[...]`` list was
# identical with or without it).
#
# Mapping verified live against ``codex features list`` on cli 0.130.0:
#
#   shell_tool, unified_exec → exec_command, write_stdin
#   multi_agent              → spawn_agent, send_input, resume_agent,
#                              wait_agent, close_agent
#   image_generation         → image_generation
#   tool_search              → tool_search
#   apps                     → mcp__codex_apps__*
#   plugins                  → plugin-contributed tools (we don't use any)
#
# Native tools that CANNOT be removed via any config/feature key on
# 0.130 and therefore remain in every API request:
#
#   apply_patch     — always-injected ``custom`` tool (no flag).
#                     Recorded into LangGraph via the native-passthrough
#                     path (see codex_provider.event_pump and
#                     codex_provider.native_tools).
#   update_plan     — Codex's TodoWrite analog. Harmless side-effect-free
#                     planning surface; we accept the noise.
#   view_image      — non-mutating; we ship our own ``read_image`` tool
#                     for the same purpose. Harmless.
#   request_user_input
#                   — elicitation surface; Codex would prompt the user via
#                     its own UI, which is not reachable in our headless
#                     ``codex exec`` topology. Harmless.
#   list_mcp_resources,
#   list_mcp_resource_templates,
#   read_mcp_resource
#                   — read-only MCP introspection. Harmless.
DEFAULT_DISABLED_FEATURES: tuple[str, ...] = (
    # Shell / exec — must be off; otherwise the model picks ``exec_command``
    # in preference to our ``execute`` MCP proxy.
    "shell_tool",
    "unified_exec",
    # Native multi-agent family — Codex's own subagent spawn would not
    # become a LangGraph ``Task`` middleware call.
    "multi_agent",
    # Native code-mode / image / search / plugins / apps — none of these
    # have analogs in deep-worker history, and the model should not have
    # the option to use them. ``apps`` covers ``mcp__codex_apps__github``
    # (the connector apps surface), ``plugins`` covers plugin-contributed
    # tools, ``image_generation`` and ``tool_search`` cover their own tools.
    "image_generation",
    "tool_search",
    "apps",
    "plugins",
)

# Top-level Codex config keys (non-feature) that we override to disable
# additional native tools. Each pair is ``(key, value_as_TOML)``. We quote
# string values here so the resulting ``-c key=value`` argv parses as TOML
# on the Codex side.
#
# ``web_search="disabled"`` is the canonical case — Codex 0.130 moved
# ``web_search`` out of ``[features]`` into a top-level string config
# (``"live"`` | ``"cached"`` | ``"disabled"``). Passing ``--disable web_search``
# on 0.130 triggers a deprecation error AND leaves the tool active; the
# top-level override is the only working channel.
DEFAULT_CONFIG_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("web_search", '"disabled"'),
)


_logger = logging.getLogger(__name__)

# Native item types already reported as unknown, so the "new codex tool"
# notice is logged ONCE per type per process instead of once per call.
_SEEN_UNKNOWN_NATIVE_TYPES: set[str] = set()


# NOTE: The legacy ``BLOCKED_NATIVE_ITEM_TYPES`` / ``_raise_if_blocked_
# native_item`` defense-in-depth scheme was removed on 2026-05-18. It
# fired a ``RuntimeError`` whenever Codex emitted a native tool event,
# which aborted the turn and surfaced as ``status="error"`` — bad UX
# for the user and architecturally wrong (we never want to drop a real
# Codex action on the floor). The replacement is the native-passthrough
# scheme: native tool calls are recorded as honest LangGraph
# ``AIMessage.tool_calls`` + ``ToolMessage`` pairs via the
# ``codex_provider.event_pump`` + ``codex_provider.native_tools``
# machinery. See ``docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md``.


def _stamp_provider_context(
    message: AIMessage, used: Any, messages: list[Any] | None = None
) -> AIMessage:
    """Stamp the PROVIDER's own context reading onto the outgoing message.

    The guard and the summarizer decide on this number and on nothing else
    (owner directive 2026-08-25: provider data only, no estimates, no
    unapproved fallback). A provider that returns an unstamped message
    leaves them with UNKNOWN — honest, but unguarded. Best effort: a stamp
    must never break a turn.
    """
    try:
        if isinstance(used, bool) or not isinstance(used, int) or used <= 0:
            return message
        from deep_worker_cli.context_limit.observed_context import (  # noqa: PLC0415
            annotate_context_observation,
            first_state_message_id,
        )

        annotate_context_observation(
            message,
            used=used,
            size=None,
            first_id=first_state_message_id(messages or []),
        )
    except Exception:  # noqa: BLE001 — never break a turn over a stamp
        pass
    return message


def _get_codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def _parse_rollout_ts(raw: Any) -> "datetime | None":
    """Parse a rollout ``timestamp`` into an aware datetime, or ``None``."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def last_token_usage_from_rollout(
    session_uuid: str,
    emit_ts: str,
    *,
    codex_home: Path | None = None,
) -> dict[str, Any] | None:
    """This turn's OWN token usage, read from the Codex rollout (§2.1-ter E).

    ``turn.completed.usage`` carries the *process* TOTAL whenever one
    ``codex exec`` spans several model calls, which inflated a live record
    to ``prompt=3736695`` against an actual spend of ``113457`` (32.9x).
    The per-turn datum exists in the rollout as
    ``token_count.info.last_token_usage``.

    Correlation is on the ``task_complete`` boundary, NOT on recency: the
    rollout is append-only for the whole session across many exec
    processes, so "the latest ``token_count``" belongs to a *later* turn
    (live: ``114460`` instead of ``113457`` — wrong by a small, plausible
    margin that passes every magnitude rail). We take the last
    ``task_complete`` at-or-before ``emit_ts`` and return the
    ``token_count`` immediately preceding it.

    FAIL-CLOSED (mandatory floor): returns ``None`` — never the TOTAL blob
    — when the file is missing/unreadable, or no ``task_complete`` at or
    before ``emit_ts`` exists (e.g. ``--ephemeral``, a pruned session, or a
    turn still in flight). Callers must record ``no_usage`` instead of
    falling back to ``turn.completed.usage``.
    """
    if not session_uuid:
        return None
    emit_dt = _parse_rollout_ts(emit_ts)
    if emit_dt is None:
        return None

    sessions_dir = (codex_home or _get_codex_home()) / "sessions"
    if not sessions_dir.exists():
        return None
    try:
        # Glob by uuid — the path is date-sharded and a turn can cross
        # midnight, so today's date must never be constructed.
        candidates = sorted(
            sessions_dir.rglob(f"*{session_uuid}.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None

    for candidate in candidates:
        pending: dict[str, Any] | None = None
        found: dict[str, Any] | None = None
        try:
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = entry.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    kind = payload.get("type")
                    if kind == "token_count":
                        info = payload.get("info")
                        if isinstance(info, dict):
                            last = info.get("last_token_usage")
                            if isinstance(last, dict):
                                pending = last
                        continue
                    if kind != "task_complete":
                        continue
                    ts = _parse_rollout_ts(entry.get("timestamp"))
                    if ts is None or ts > emit_dt:
                        # A terminal after our emit belongs to a later turn.
                        continue
                    if pending is not None:
                        found = pending
                    pending = None
        except OSError:
            continue
        if found is not None:
            return dict(found)
    return None


def resolve_model_from_session_log(
    thread_id: str,
    *,
    codex_home: Path | None = None,
) -> str | None:
    if not thread_id:
        return None
    sessions_dir = (codex_home or _get_codex_home()) / "sessions"
    if not sessions_dir.exists():
        return None
    try:
        candidates = sorted(
            sessions_dir.rglob(f"*{thread_id}.jsonl"),
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
                    if event.get("type") != "turn_context":
                        continue
                    payload = event.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    model = payload.get("model")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
        except OSError:
            continue
    return None


DEEP_WORKER_TOOL_CONTRACT = """\
Deep-worker tool contract:
- Use only the deep-worker MCP proxy tools (prefix `mcp__deepworker_*__`)
  for file, shell, web, task, memory, and agent operations.
- For shell commands use the deep-worker `execute` tool — NEVER the
  native `exec_command`/`shell` tools. Deep-worker has disabled them for
  this session.
- For file edits use deep-worker's `read_file`/`write_file`/`edit_file`
  — prefer them over the native `apply_patch` tool. (`apply_patch` may
  appear in your tools list because Codex CLI hard-codes it; deep-worker
  records every native call into history but cannot gate it with HITL.)
- For web search / fetch use deep-worker's `web_search_delegation_tool`
  / `fetch_url` — NEVER Codex's native `web_search`. Deep-worker has
  disabled it for this session.
- **CRITICAL — inter-agent messaging (`mcp__deepworker_*__send_message_to_agent`).**
  When you see an incoming user-role message whose content starts with
  `<sender-name> -> <your-name> :> …`, it was sent by another deep-worker
  agent, NOT by the human user. You MUST reply using
  `mcp__deepworker_*__send_message_to_agent(<sender-name>, <your reply>)` —
  NOT a free-text turn. This is the ONLY way to respond to a peer.
  A free-text reply is delivered to the human user as if it were your
  turn, silently breaking the peer conversation. The fire-and-forget
  semantics mean a successful call IS your final response for this turn —
  do not add preamble or trailing narration. The peer message may
  surface as a user-role block prefixed with `System:>` (or similar);
  the required action is invariant regardless of the transport detail.
"""


class CodexChatModel(BaseChatModel):
    """LangChain chat-model wrapper around Codex CLI.

    Provider name in deep-worker: ``"codex"``. Construction is cheap and
    side-effect-free; the proxy MCP server is started lazily on the
    first ``ainvoke`` (or on the first ``bind_tools`` if it carries
    tools).
    """

    # Pydantic config identical to claude-code-sdk:
    # - arbitrary_types_allowed for non-Pydantic privates
    # - extra='allow' so LangChain can pass through model/max_tokens/etc.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _client_state: Any = PrivateAttr(default=None)
    _exec_runner: Any = PrivateAttr(default=None)
    _proxy_mcp: Any = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    _working_dir: str | None = PrivateAttr(default=None)
    _model_name: str | None = PrivateAttr(default=None)
    _resolved_model_name: str | None = PrivateAttr(default=None)
    _resolved_model_callback: Callable[[str], Awaitable[None] | None] | None = PrivateAttr(default=None)
    _last_reported_model_name: str | None = PrivateAttr(default=None)
    # Token-stats usage-event callback (DESIGN_AGENT_TOKEN_STATS.md §5.2):
    # async callable installed by create_model via set_usage_event_callback.
    _usage_event_callback: Any = PrivateAttr(default=None)
    _disabled_features: tuple[str, ...] = PrivateAttr(default=DEFAULT_DISABLED_FEATURES)
    _config_overrides: tuple[tuple[str, str], ...] = PrivateAttr(
        default=DEFAULT_CONFIG_OVERRIDES,
    )
    _sandbox_mode: str = PrivateAttr(default="workspace-write")
    _approval_policy: str = PrivateAttr(default="never")
    # deep-worker's ``-sp``; fail-closed default, same as every other
    # provider (owner directive 2026-08-25 on a single permission policy).
    _auto_approve: bool = PrivateAttr(default=False)
    _proxy_started: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------
    # Pydantic schema overrides — same surgery as claude_code_sdk_provider:
    # LangGraph pickles models during checkpointing, and BaseChatModel
    # carries callable fields that can't be JSON-serialised. Strip them.
    # ------------------------------------------------------------------
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        try:
            parent_raw: Any = handler(core_schema)
            parent: dict[str, Any] = parent_raw if isinstance(parent_raw, dict) else {
                "type": "object", "properties": {}
            }
        except Exception:
            parent = {"type": "object", "properties": {}}
        props_raw: Any = parent.get("properties", {}) or {}
        props: dict[str, Any] = props_raw if isinstance(props_raw, dict) else {}
        parent["properties"] = {
            k: v for k, v in props.items()
            if k not in {"custom_get_token_ids", "callbacks", "rate_limiter"}
        }
        parent["title"] = cls.__name__
        parent["description"] = "Codex chat model for LangChain"
        return parent

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude: set[str] = {"custom_get_token_ids", "callbacks", "rate_limiter"}
        user_exclude = kwargs.get("exclude")
        if isinstance(user_exclude, set):
            kwargs["exclude"] = exclude | user_exclude
        elif user_exclude is None:
            kwargs["exclude"] = exclude
        return super().model_dump(**kwargs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        sandbox_mode: str = "workspace-write",
        approval_policy: str = "never",
        auto_approve: bool = False,
        disabled_features: Sequence[str] | None = None,
        config_overrides: Sequence[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseChatModel has a Callable field that breaks JSON-schema
        # generation; set it to None up front.
        kwargs["custom_get_token_ids"] = None
        super().__init__(**kwargs)

        self._client_state = get_global_client_state()
        self._exec_runner = CodexExecRunner()
        self._tools_metadata = []
        self._model_name = model
        self._resolved_model_name = model
        self._sandbox_mode = sandbox_mode
        self._approval_policy = approval_policy
        self._auto_approve = auto_approve
        if disabled_features is not None:
            self._disabled_features = tuple(disabled_features)
        if config_overrides is not None:
            self._config_overrides = tuple(config_overrides)

    @property
    def _llm_type(self) -> str:
        return "codex"

    @property
    def resolved_model_name(self) -> str | None:
        return self._resolved_model_name

    def set_resolved_model_callback(
        self,
        callback: Callable[[str], Awaitable[None] | None],
    ) -> None:
        self._resolved_model_callback = callback

    async def _record_resolved_model(self, model: str | None) -> None:
        if not model or model == self._last_reported_model_name:
            return
        self._resolved_model_name = model
        self._last_reported_model_name = model
        if self._resolved_model_callback is None:
            return
        result = self._resolved_model_callback(model)
        if inspect.isawaitable(result):
            await result

    def set_usage_event_callback(self, callback: Any) -> None:
        """Install the async usage-event callback (§5.2).

        ``create_model`` injects the token-stats state-machine callback here
        (custom providers never receive a LangChain callback handler — the
        event callback is the single usage source, MAJOR-2).
        """
        self._usage_event_callback = callback

    async def _emit_usage_event(
        self,
        *,
        event: str,
        status: str,
        usage: dict[str, Any] | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        context_used: int | None = None,
        context_size: int | None = None,
    ) -> None:
        """Report a terminal Codex event to the stats callback (§5.2).

        Best-effort: the callback is awaited; failures are logged and never
        break the provider turn (except ``asyncio.CancelledError``, which
        propagates). ``TurnCompletedEvent.usage`` is already a plain dict —
        copied defensively before handing over (MAJOR-4).
        """
        cb = getattr(self, "_usage_event_callback", None)
        if cb is None:
            return
        try:
            if usage is not None:
                # The handed-over ``turn.completed.usage`` is the exec
                # process TOTAL, not this turn's spend (§2.1-ter). Override
                # it with the rollout's ``last_token_usage``; fail CLOSED to
                # ``no_usage`` rather than persist the inflated blob.
                emit_ts = (
                    datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z")
                )
                usage = last_token_usage_from_rollout(session_id or "", emit_ts)
            if isinstance(usage, dict):
                usage = dict(usage)
            await cb(
                event=event,
                status=status,
                usage=usage,
                model=getattr(self, "_resolved_model_name", None) or "",
                session_id=session_id,
                turn_id=turn_id,
                context_used=context_used,
                context_size=context_size,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort
            _logger.exception("[token-stats] codex usage event callback failed")

    # ------------------------------------------------------------------
    # Live context gauge (mid-turn)
    # ------------------------------------------------------------------
    async def _emit_context_snapshot(
        self, used: int, total: int | None, *, session_id: str | None
    ) -> None:
        """Record one mid-turn context reading.

        ``turn_id=None`` is REQUIRED, not incidental: codex records are
        de-duplicated on ``session_id|turn_id``
        (token_stats.UsageEventCallback._event_key), so reusing the turn id
        would make every reading after the first vanish as a duplicate —
        exactly the frozen gauge this exists to fix. ``total`` is carried
        for the record only; it is the cumulative spend across model calls
        and must never be presented as a context size.
        """
        _ = total
        await self._emit_usage_event(
            event="terminal",
            status="ok",
            usage=None,
            session_id=session_id,
            turn_id=None,
            context_used=used,
        )

    async def _start_context_watch(self, state: Any) -> None:
        """Watch this session's rollout so the gauge moves per model call.

        No-op (silently) when the session id is not known yet, when the
        rollout is absent (``--ephemeral``, pruned sessions), or when the
        watcher cannot start: the turn is never affected by telemetry.
        """
        try:
            session_uuid = getattr(state, "codex_session_uuid", None)
            if not session_uuid:
                return
            existing = getattr(state, "context_watcher", None)
            if existing is not None:
                return
            from .context_watch import (  # noqa: PLC0415 — keep import cost off the hot path
                RolloutContextWatcher,
                rollout_path_for,
            )

            path = rollout_path_for(str(session_uuid), _get_codex_home())
            if path is None:
                return

            async def _on_context(used: int, total: int | None) -> None:
                # Keep the latest reading so the turn's AIMessage carries it:
                # the guard and the summarizer read that stamp, never an
                # estimate.
                state.last_provider_context = used
                await self._emit_context_snapshot(
                    used, total, session_id=str(session_uuid)
                )

            watcher = RolloutContextWatcher(path=path, on_context=_on_context)
            await watcher.start()
            state.context_watcher = watcher
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort telemetry
            _logger.debug(
                "codex_provider: live context watch unavailable", exc_info=True
            )

    async def _stop_context_watch(self, state: Any) -> None:
        """Release the watcher and its inotify handle. Idempotent."""
        watcher = getattr(state, "context_watcher", None)
        if watcher is None:
            return
        state.context_watcher = None
        try:
            await watcher.stop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _logger.debug(
                "codex_provider: context watcher stop failed", exc_info=True
            )

    # ------------------------------------------------------------------
    # Native tool mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _native_tool_call_for(
        *, item_id: str, item_type: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Map a codex ``GenericItem`` to a LangGraph tool_call, or ``None``
        when the item is purely informational.

        A type we know keeps its own name (there is a bound tool for it). A
        type we do NOT know is recorded through the generic recorder with
        its real type preserved in the args — codex ships tools faster than
        this package is updated, and the alternative (the previous
        behaviour) was to drop the call silently, leaving history claiming
        the agent never did the work.

        The first sighting of each unknown type is logged at INFO: new
        native tooling is expected traffic, not an alarm, but it must be
        visible — see docs/BACKLOG.md and the 2026-08-25 owner directive.
        """
        from .event_pump import (  # noqa: PLC0415 — avoid import cycle
            NATIVE_TOOL_ITEM_TYPES,
            is_recordable_tool_item,
        )
        from .native_tools import (  # noqa: PLC0415
            GENERIC_NATIVE_TOOL_NAME,
            ORIGINAL_TYPE_ARG_KEY,
        )

        if not is_recordable_tool_item(item_type):
            return None
        if item_type in NATIVE_TOOL_ITEM_TYPES:
            return {
                "id": item_id,
                "name": item_type,
                "args": args,
                "type": "tool_call",
            }
        if item_type not in _SEEN_UNKNOWN_NATIVE_TYPES:
            _SEEN_UNKNOWN_NATIVE_TYPES.add(item_type)
            _logger.info(
                "codex_provider: native tool %r is not known to this "
                "provider version — recording it via %r so it stays in "
                "history. Add it to NATIVE_TOOL_ITEM_TYPES for a first-class "
                "entry, or to INFORMATIONAL_ITEM_TYPES if it should be "
                "ignored.",
                item_type,
                GENERIC_NATIVE_TOOL_NAME,
            )
        return {
            "id": item_id,
            "name": GENERIC_NATIVE_TOOL_NAME,
            "args": {**args, ORIGINAL_TYPE_ARG_KEY: item_type},
            "type": "tool_call",
        }

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "CodexChatModel":
        """Extract MCP-compatible metadata from LangChain tools and bind it.

        Returns self with the metadata populated (LangChain pattern
        allows returning self or a clone; claude-code-sdk returns a clone,
        we return self to avoid duplicating the proxy MCP server).
        The proxy MCP server is materialised lazily on first ainvoke
        so that bind_tools stays side-effect free.
        """
        tools_metadata: list[ToolMetadata] = []
        for item in tools:
            if not isinstance(item, BaseTool):
                continue
            schema = self._extract_tool_schema(item)
            tools_metadata.append({
                "name": item.name,
                "description": item.description or "",
                "input_schema": schema,
            })
        self._tools_metadata = tools_metadata
        # Drop any previously-started proxy — its tool set is stale.
        # Actual restart happens lazily on next ainvoke.
        return self

    @staticmethod
    def _extract_tool_schema(tool: BaseTool) -> dict[str, Any]:
        """Same algorithm as claude_code_sdk_provider.chat_model.bind_tools."""
        schema: dict[str, Any] = {}
        if tool.args_schema:
            try:
                if hasattr(tool.args_schema, "model_json_schema"):
                    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
                elif hasattr(tool.args_schema, "schema"):
                    schema = tool.args_schema.schema()  # type: ignore[union-attr,deprecated]
            except Exception:
                schema = {}

        if not schema.get("properties"):
            func = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
            if func:
                try:
                    sig = inspect.signature(func)
                    props: dict[str, Any] = {}
                    required: list[str] = []
                    for pname, param in sig.parameters.items():
                        if pname in ("runtime", "self", "kwargs"):
                            continue
                        ann = param.annotation
                        if ann is int:
                            ptype = "integer"
                        elif ann is bool:
                            ptype = "boolean"
                        elif ann is float:
                            ptype = "number"
                        else:
                            ptype = "string"
                        prop: dict[str, Any] = {
                            "type": ptype,
                            "title": pname.replace("_", " ").title(),
                        }
                        # ``Annotated[...]`` carries metadata at ``ann.__metadata__``.
                        # pyright sees ``ann`` as ``type[int]`` / ``type[bool]`` etc.
                        # in the early branches above, where ``__metadata__`` is
                        # not a member. Read it through a runtime ``getattr`` so
                        # the type checker doesn't infer a member access on
                        # plain types.
                        meta_tuple: Any = getattr(ann, "__metadata__", ())
                        for meta in meta_tuple:
                            if isinstance(meta, str):
                                prop["description"] = meta
                                break
                        if param.default != inspect.Parameter.empty:
                            prop["default"] = param.default
                        else:
                            required.append(pname)
                        props[pname] = prop
                    if props:
                        schema = {
                            "type": "object",
                            "properties": props,
                            "required": required,
                        }
                except (ValueError, TypeError):
                    pass
        return schema

    # ------------------------------------------------------------------
    # SummarizerPort (deep_worker_cli.context_limit.summarizer_port).
    #
    # ``codex`` is in ``SESSION_STATEFUL_PROVIDERS``: Codex owns the
    # transcript (``CODEX_HOME/sessions/<uuid>.jsonl``) and we send only
    # ``messages[sent_count:]`` each turn. The middleware therefore REFUSES
    # the generic identical-prefix fallback for this provider — re-invoking
    # the agent thread would replay the whole history into a fresh Codex
    # session, the exact dump the port exists to avoid.
    #
    # Until this port existed that refusal made EVERY summarization on codex
    # deterministically fatal: agent_a died with
    # ``SummarizationFailedError: summarization port unavailable on a
    # session-stateful provider (CodexChatModel)`` on 2026-08-23, and — being
    # past its context limit — re-died on each following turn.
    #
    #   hot  — resume the live rollout in a SEPARATE, throwaway ``codex exec``
    #          and send only the instruction. Codex reloads its own transcript
    #          from disk, so the history is never re-sent by us and never
    #          re-tokenized on the wire.
    #   cold — no resume: a fresh one-shot ``codex exec`` carrying the
    #          instruction plus the already-trimmed message dump.
    #   both — NEVER touch the agent's live session. The summary turn takes no
    #          tools and no proxy MCP server, and its own session uuid is
    #          discarded, so it cannot perturb ``sent_count`` or the rollout
    #          the next real turn resumes.
    #
    # Note on ``codex app-server``: its ``thread/compact/start`` would let
    # Codex compact its own context server-side, and ``thread/fork`` with
    # ``ephemeral: true`` would give an even cheaper cold path. That is the
    # better long-term transport, but this provider drives ``codex exec``, and
    # the port is implemented on the transport we actually run.
    # ------------------------------------------------------------------

    async def _arun_summary_turn(
        self, instruction: str, resume_uuid: str | None
    ) -> tuple[str, dict[str, Any] | None]:
        """Run ONE isolated ``codex exec`` and return ``(text, usage)``.

        Deliberately spawns its own handle rather than reusing
        ``_spawn_turn``: that method records the handle on
        ``state.current_exec`` and passes the proxy MCP server, both of which
        would entangle the summary with the agent's live turn.
        """
        args = ExecArgs(
            prompt=instruction,
            session_uuid=resume_uuid,
            working_directory=self._working_dir,
            sandbox_mode=self._sandbox_mode,
            approval_policy=self._approval_policy,
            # getattr, not attribute access: a half-constructed model
            # (``__new__`` without ``__init__`` — the shape several
            # unit tests build) must fall back to the SAFE value, never
            # silently inherit bypass.
            auto_approve=getattr(self, "_auto_approve", False),
            model=self._model_name,
            disabled_features=self._disabled_features,
            config_overrides=self._config_overrides,
            # No mcp_server_name/port: the summary turn must not be able to
            # call deep-worker tools.
        )
        handle = await self._exec_runner.spawn(args)
        text = ""
        usage: dict[str, Any] | None = None
        try:
            while True:
                event = await handle.ainvoke_queue.get()
                if isinstance(event, ItemCompletedEvent):
                    if isinstance(event.item, AgentMessageItem):
                        text = event.item.text
                elif isinstance(event, TurnCompletedEvent):
                    usage = event.usage
                    break
                elif isinstance(event, TurnFailedEvent):
                    raise RuntimeError(
                        f"codex summary turn failed: {event.error_message}"
                    )
        finally:
            with _suppress_errors():
                await handle.aclose()
        return text.strip(), usage

    async def asummarize(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        instruction: str,
        decision: Any,
    ) -> Any:
        """SummarizerPort: hot rollout-resume or cold one-shot session.

        Raises on failure and on an empty summary — never returns an error
        string. The caller is about to delete the history this summary
        replaces, so a soft failure would destroy the conversation silently
        (POLICIES §12).
        """
        from deep_worker_cli.context_limit.hot_cache import (  # noqa: PLC0415
            SummaryResult,
        )

        hot = bool(getattr(decision, "hot", False))
        resume_uuid: str | None = None
        if hot:
            state = self._client_state.get_or_create(thread_id)
            resume_uuid = getattr(state, "codex_session_uuid", None)
            if not resume_uuid:
                # Nothing to resume — the caller's "hot" belief is stale
                # (fresh process, or the session was already reset). Fall back
                # to cold rather than silently summarizing an empty session.
                hot = False

        if hot:
            prompt = instruction
        else:
            from deepagents.middleware.message_timestamps import (  # noqa: PLC0415
                MonotonicUtcClock,
                format_messages_with_timestamps,
            )

            dump = format_messages_with_timestamps(
                list(messages), MonotonicUtcClock()
            )
            prompt = f"{instruction}\n\n{dump}"

        text, usage = await self._arun_summary_turn(prompt, resume_uuid)
        if not text:
            raise RuntimeError(
                "codex summarization returned an empty summary "
                f"(path={'hot' if hot else 'cold'}, thread={thread_id!r})"
            )
        return SummaryResult(
            text=text,
            path="hot" if hot else "cold",
            provider_session_id=resume_uuid,
            usage=usage,
        )

    async def areset_after_summary(self, *, thread_id: str) -> None:
        """SummarizerPort: drop the pre-summary Codex session for ``thread_id``.

        ``record_summarization`` clears ``codex_session_uuid``, zeroes
        ``sent_count`` and persists both, so the next turn starts a fresh
        rollout and mirrors the new summary-shaped history from index 0. The
        persist matters: without it a ``--restart`` immediately after a
        summarization would reload the stale pre-summary uuid.

        Idempotent — it creates an empty slot when none exists.
        """
        self._client_state.record_summarization(thread_id)

    # ------------------------------------------------------------------
    # Async-only — _generate raises, _agenerate not provided, ainvoke is real.
    # ------------------------------------------------------------------
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "CodexChatModel only supports async operations. Use ainvoke()."
        )

    async def ainvoke(  # type: ignore[override]
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Drive one Codex turn (or resume one mid-turn after a tool call)."""
        from langchain_core.prompt_values import PromptValue
        from langchain_core.runnables.config import ensure_config

        # 1. Normalize input
        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)  # type: ignore[arg-type]

        # 2. Mandatory agent_thread_id (the deep-worker contract).
        config = ensure_config(config)
        cfg = config.get("configurable") or {}
        if "thread_id" not in cfg:
            raise ValueError(
                "CodexChatModel requires explicit thread_id in config — "
                "this is deep-worker's agent name. Each caller (main "
                "agent, TaskManager, subagent) must own its thread_id. "
                "Implicit defaults are forbidden."
            )
        agent_thread_id: str = cfg["thread_id"]

        # Token-stats: mark the invocation start (no record is written here —
        # the record closes on the terminal TurnCompletedEvent/TurnFailedEvent,
        # §5.2). Best-effort.
        _cb = getattr(self, "_usage_event_callback", None)
        if _cb is not None:
            try:
                await _cb(event="invocation_start")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                _logger.exception("[token-stats] invocation_start callback failed")

        # 3. Per-agent_thread_id state
        state = self._client_state.get_or_create(agent_thread_id)
        # Bind context stamps to the history they measure (first_id).
        state.last_request_messages = list(messages)

        # 4. Summarization detection — drop Codex session, keep agent_thread_id.
        if state.sent_count > len(messages):
            self._client_state.record_summarization(agent_thread_id)
            # current_exec, if any, was for the OLD session — drop it too
            if state.current_exec is not None:
                with _suppress_errors():
                    await state.current_exec.aclose()
            state.current_exec = None
            # record_summarization mutated state.codex_session_uuid and
            # state.sent_count in place; re-fetch for clarity:
            state = self._client_state.get_or_create(agent_thread_id)

        # 5. System prompt (first SystemMessage in messages)
        #
        # ⚠️ Must extract *plain text*, not ``str(msg.content)``. LangGraph
        # middleware (deepagents memory/skills/etc.) routinely wraps the
        # system prompt in ``SystemMessage(content=[{'type':'text',
        # 'text': ...}, ...])`` via ``request.override``. ``str([...])``
        # would emit a Python repr like
        # ``[{'type': 'text', 'text': '...'}, ...]`` into our
        # AGENTS.override.md, which Codex tokenises as literal noise
        # instead of instructions.
        system_prompt: str | None = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = _flatten_message_content_to_text(msg.content)
                break

        # 6. Ensure proxy MCP server is started (lazy, idempotent)
        await self._ensure_proxy_started()

        # 7. Tail delta only — incremental mirror, never full rebuild.
        #
        # ToolMessages from LangGraph carry the codex_native marker
        # when they came from the native-tool passthrough path. Those
        # must NOT be pushed back to Codex via the proxy MCP — Codex
        # ran the tool itself and is not blocked on us. Proxy MCP
        # ToolMessages, by contrast, must unblock the parked HTTP
        # handler so Codex can continue.
        new_messages = messages[state.sent_count:]
        user_text_parts: list[str] = []
        tool_resolved = False
        saw_native_resume = False
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                is_native = bool(
                    (getattr(msg, "additional_kwargs", None) or {}).get(
                        "codex_native", False
                    )
                )
                if is_native:
                    saw_native_resume = True
                else:
                    self._proxy_mcp.resolve_call(
                        msg.tool_call_id,
                        {"content": _build_mcp_content(msg.content)},
                    )
                    tool_resolved = True
            elif isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                user_text_parts.append(content)
            elif isinstance(msg, AIMessage):
                # already in Codex's session — don't re-send
                pass
            elif isinstance(msg, SystemMessage):
                # forwarded via base-instructions, not as a user message
                pass

        # 8. Update sent_count BEFORE awaiting response (mirror claude-code-sdk)
        # and PERSIST it so a restart resumes from the right cursor instead
        # of replaying the whole history into an empty session.
        # See docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.3.
        state.sent_count = len(messages)
        self._client_state.persist(agent_thread_id)

        # 9. Mid-turn (resuming an in-flight Codex turn) vs new turn
        #    (spawn fresh codex exec). Either a proxy MCP ToolMessage
        #    (tool_resolved) or a native ToolMessage (saw_native_resume)
        #    means Codex is still running inside the same exec — keep
        #    reading the same handle's queue.
        is_mid_turn = (
            (tool_resolved or saw_native_resume)
            and state.current_exec is not None
            and not user_text_parts
        )
        if is_mid_turn:
            # Resuming an in-flight exec. No new spawn / no session_uuid
            # decision here, so the resume fallback does NOT apply — keep
            # the existing error semantics.
            handle = state.current_exec
            return await self._consume_turn(state, agent_thread_id, handle)

        # New turn → spawn a fresh ``codex exec``. If there's an old
        # current_exec from a previous interrupted turn (e.g. a main user
        # message arrived before tool resolution), close it cleanly to avoid
        # an orphaned subprocess.
        if state.current_exec is not None:
            with _suppress_errors():
                await state.current_exec.aclose()
            state.current_exec = None

        prompt = "\n\n".join(user_text_parts) if user_text_parts else ""

        # Resume fallback (DESIGN §6.1): we are about to spawn with whatever
        # ``codex_session_uuid`` is cached. If that uuid points at a stale /
        # missing Codex session (e.g. CODEX_HOME wiped, session pruned),
        # ``codex exec resume <uuid>`` fails — but, unlike claude-code-sdk's
        # synchronous ``connect()``, the failure surfaces ASYNCHRONOUSLY in
        # ``_consume_turn`` as a TurnFailedEvent / END_OF_STREAM pump_error.
        # When we attempted a resume and it failed, fall back ONCE to a fresh
        # session (session_uuid=None) and replay the FULL history so the agent
        # comes back working instead of going to ERROR status.
        spawned_with_resume = state.codex_session_uuid is not None
        handle = await self._spawn_turn(state, prompt, system_prompt)
        try:
            return await self._consume_turn(state, agent_thread_id, handle)
        except (RuntimeError, CodexProtocolError) as exc:
            if not spawned_with_resume:
                # A genuine fresh-session failure — nothing to retry.
                raise
            # The resume of a stale/missing Codex session failed. Reset to a
            # fresh session and replay the full history once.
            sys.stderr.write(
                f"[codex_provider] resume of {state.codex_session_uuid!r} "
                f"failed ({exc!r}); starting a fresh session and replaying "
                f"history for thread {agent_thread_id!r}.\n"
            )
            sys.stderr.flush()
            with _suppress_errors():
                if state.current_exec is not None:
                    await state.current_exec.aclose()
            state.current_exec = None
            state.codex_session_uuid = None
            # Force a full-history replay: every HumanMessage is re-sent.
            state.sent_count = 0
            self._client_state.persist(agent_thread_id)

            fresh_prompt = "\n\n".join(
                m.content if isinstance(m.content, str) else str(m.content)
                for m in messages
                if isinstance(m, HumanMessage)
            )
            # Mirror the normal post-spawn cursor update + persist.
            state.sent_count = len(messages)
            self._client_state.persist(agent_thread_id)
            handle = await self._spawn_turn(state, fresh_prompt, system_prompt)
            return await self._consume_turn(state, agent_thread_id, handle)

    async def _spawn_turn(
        self,
        state: Any,
        prompt: str,
        system_prompt: str | None,
    ) -> "ExecHandle":
        """Build ``ExecArgs`` from the current state + prompt and spawn one
        ``codex exec``. Records the handle on ``state.current_exec``.

        ``ExecArgs.session_uuid`` comes from ``state.codex_session_uuid`` —
        when set, Codex resumes the existing session
        (``codex exec resume <uuid>``); when ``None``, a fresh session starts.
        """
        # Per-agent MCP isolation (learned from a related internal
        # incident): pass THIS
        # agent's own proxy server into the ExecArgs so it is injected
        # per-invocation via ``--config mcp_servers.<name>.url``, and
        # ``_build_argv`` also adds ``--ignore-user-config`` so the shared
        # ~/.codex/config.toml (with other agents' stale servers) is NOT
        # loaded. We deliberately do NOT register globally via
        # ``codex mcp add`` — see ``_ensure_proxy_started``.
        args = ExecArgs(
            prompt=prompt,
            session_uuid=state.codex_session_uuid,
            working_directory=self._working_dir,
            sandbox_mode=self._sandbox_mode,
            approval_policy=self._approval_policy,
            # getattr, not attribute access: a half-constructed model
            # (``__new__`` without ``__init__`` — the shape several
            # unit tests build) must fall back to the SAFE value, never
            # silently inherit bypass.
            auto_approve=getattr(self, "_auto_approve", False),
            model=self._model_name,
            base_instructions=_augment_base_instructions(system_prompt),
            disabled_features=self._disabled_features,
            config_overrides=self._config_overrides,
            mcp_server_name=(
                self._proxy_mcp.server_name
                if self._proxy_mcp is not None else None
            ),
            mcp_server_port=(
                self._proxy_mcp.port
                if self._proxy_mcp is not None else None
            ),
        )
        handle = await self._exec_runner.spawn(
            args,
            proxy_mcp_server_name=(
                self._proxy_mcp.server_name
                if self._proxy_mcp is not None else None
            ),
        )
        state.current_exec = handle
        return handle

    async def _consume_turn(
        self,
        state: Any,
        agent_thread_id: str,
        handle: "ExecHandle",
    ) -> AIMessage:
        """Consume events from the pump's queue — event-driven, no polling.

        The pump runs in the background, draining ``handle.events`` once and
        routing each event to either ``ainvoke_queue`` (us) or
        ``native_futures`` (the matching tool body). Native item.completed
        events never reach this loop — the tool body awaits the matching
        Future.

        Raises on a failed turn (TurnFailedEvent / END_OF_STREAM pump_error);
        the fresh-turn caller may catch that to drive the resume fallback.
        """
        # Local import — keep module-load cost low and avoid a circular
        # dependency between chat_model and event_pump.
        from .event_pump import (
            END_OF_STREAM,
            native_args,
        )

        # Live context gauge. TWO call sites, deliberately: a RESUMED
        # session already knows its uuid and need not emit ``thread.started``
        # at all, so hanging the start on that event alone left every
        # resumed turn — i.e. nearly every turn of a working agent —
        # without a gauge. ``_start_context_watch`` is idempotent, so the
        # pair is safe; the other site covers a fresh session, whose uuid
        # only exists once codex announces it.
        await self._start_context_watch(state)

        messages_for_stamp = list(
            getattr(state, "last_request_messages", None) or []
        )
        final_text = ""
        pending_tool_calls: list[dict[str, Any]] = []
        turn_done = False
        # Token-stats (§5.2): the id of the last AgentMessageItem before the
        # terminal event — part of the stable dedup key ``codex|<thread_id>|<id>``.
        last_agent_message_id: str | None = None
        try:
            while True:
                event = await handle.ainvoke_queue.get()
                # Live gauge, THIRD attempt site — and the one that actually
                # lands. Codex creates the rollout a beat AFTER announcing
                # ``thread.started``, so the two sites above routinely probe
                # a file that does not exist yet, resolve None and return
                # silently: measured 2026-08-26, a whole turn with zero
                # readings and an unstamped message, i.e. UNKNOWN for the
                # guard and the summarizer. Retrying per event costs one
                # glob only while the watch is still unset — the call
                # returns immediately once it is.
                if getattr(state, "context_watcher", None) is None:
                    await self._start_context_watch(state)
                # End-of-stream sentinel — pump task is done (process
                # exited or was cancelled). Stop consuming.
                if event is END_OF_STREAM:
                    # If the pump finished because the Codex stream
                    # raised (e.g. an API 400 for an unsupported model),
                    # it recorded the cause on ``handle.pump_error``.
                    # Re-raise it here so the caller sees a real error
                    # instead of an empty assistant message — the latter
                    # silently stalls the agent (observed live).
                    pump_error = getattr(handle, "pump_error", None)
                    if pump_error is not None:
                        raise pump_error
                    turn_done = True
                    break
                if isinstance(event, ThreadStartedEvent):
                    if state.codex_session_uuid is None:
                        state.codex_session_uuid = event.thread_id
                        # Persist the freshly-learned uuid as soon as Codex
                        # assigns it, so even a crash mid-turn leaves a
                        # resumable uuid on disk (DESIGN §3.3).
                        self._client_state.persist(agent_thread_id)
                    # The rollout path is keyed by this uuid, so the live
                    # context gauge can only start once it is known.
                    await self._start_context_watch(state)
                    if self._model_name is None:
                        await self._record_resolved_model(
                            resolve_model_from_session_log(event.thread_id)
                        )
                elif isinstance(event, ItemStartedEvent):
                    it = event.item
                    if (
                        isinstance(it, McpToolCallItem)
                        and self._proxy_mcp is not None
                        and it.server == self._proxy_mcp.server_name
                        and it.status == "in_progress"
                    ):
                        # Proxy MCP tool starting — event-driven wait for the
                        # HTTP handler's parked Future, BOUNDED by the SINGLE
                        # source of truth (variant A; no independent 30s timer).
                        # On timeout: NON-error in_progress, never raise, never
                        # hang (POLICIES.md §7.1). See
                        # DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.
                        try:
                            pending_tool_calls = await self._proxy_mcp.wait_for_pending_calls(
                                tool_name=it.tool,
                                arguments=it.arguments,
                            )
                        except asyncio.TimeoutError:
                            # No pending call ever registered → no work was
                            # parked. The prior in_progress notice advertised
                            # wait_for_job/kill_job under it.id or it.tool
                            # (neither is a canonical job_N) — a false
                            # promise the model could not honour. Emit the
                            # honest not_started notice; the turn still ends
                            # cleanly (no phantom tool_call).
                            # See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.
                            from deepagents.middleware.bounded_tool_call import (  # noqa: PLC0415
                                build_dispatch_timeout_notice,
                                resolve_tool_wait_seconds,
                            )
                            final_text = build_dispatch_timeout_notice(
                                tool_name=it.tool,
                                wait_seconds=resolve_tool_wait_seconds(),
                            )
                            turn_done = True
                            break
                        if pending_tool_calls:
                            break
                    elif isinstance(it, GenericItem):
                        # Native tool starting. The pump has already
                        # created native_futures[it.id]; emit the
                        # AIMessage.tool_call so LangGraph routes into
                        # ToolNode, which invokes the matching async
                        # tool body in ``codex_provider.native_tools``.
                        # That body parks on the future and wakes up
                        # event-driven when the pump processes the
                        # matching item.completed.
                        #
                        # An item type we do not know is recorded through
                        # the generic recorder rather than dropped —
                        # ``_native_tool_call_for`` returns None ONLY for
                        # the explicitly informational types.
                        tc = self._native_tool_call_for(
                            item_id=it.id,
                            item_type=it.type,
                            args=native_args(it),
                        )
                        if tc is not None:
                            pending_tool_calls.append(tc)
                            break
                elif isinstance(event, ItemCompletedEvent):
                    it = event.item
                    if isinstance(it, AgentMessageItem):
                        final_text = it.text
                        last_agent_message_id = it.id
                    # McpToolCallItem completion was filtered by the
                    # pump; native item.completed was consumed into
                    # native_futures by the pump and never reaches
                    # here.
                elif isinstance(event, TurnCompletedEvent):
                    # Token-stats: terminal event with per-turn usage
                    # (§5.2, branch A). One record per observed terminal event.
                    await self._emit_usage_event(
                        event="terminal",
                        status="ok",
                        usage=event.usage,
                        session_id=state.codex_session_uuid,
                        turn_id=last_agent_message_id,
                    )
                    turn_done = True
                    break
                elif isinstance(event, TurnFailedEvent):
                    # Token-stats: terminal failure → status="failed" (§5.2).
                    await self._emit_usage_event(
                        event="terminal",
                        status="failed",
                        usage=None,
                        session_id=state.codex_session_uuid,
                        turn_id=last_agent_message_id,
                    )
                    turn_done = True
                    raise RuntimeError(f"Codex turn failed: {event.error_message}")
        finally:
            # Always — a cancelled or failed turn must not leak the watcher
            # task or its inotify handle. The terminal record written by
            # turn.completed remains the authoritative per-turn number.
            await self._stop_context_watch(state)
            if turn_done:
                with _suppress_errors():
                    await handle.aclose()
                state.current_exec = None

        # The context codex reported for THIS turn — the same number the
        # status bar shows, read from the rollout the live gauge follows.
        # Without it the guard/summarizer would treat the turn as UNKNOWN.
        used = getattr(state, "last_provider_context", None)
        if pending_tool_calls:
            return _stamp_provider_context(
                AIMessage(content=final_text, tool_calls=pending_tool_calls),
                used,
                messages_for_stamp,
            )
        return _stamp_provider_context(
            AIMessage(content=final_text), used, messages_for_stamp
        )

    # ------------------------------------------------------------------
    # Proxy MCP lifecycle
    # ------------------------------------------------------------------
    async def _ensure_proxy_started(self) -> None:
        # Process-global singleton (one proxy MCP server per agent
        # process). Tests may inject a custom ``_proxy_mcp`` — don't
        # clobber that.
        #
        # Skip materialisation entirely if we have no tools to expose:
        # Codex caches ``tools/list`` per MCP-server registration, so
        # if we register an empty server and *later* call ``bind_tools``
        # with tools, Codex won't know about them. By deferring
        # registration until we actually have tools, the first
        # ``codex exec`` after ``bind_tools`` sees the right list.
        if self._proxy_mcp is None:
            if not self._tools_metadata:
                return
            self._proxy_mcp = get_global_proxy_mcp_server(self._tools_metadata)
        else:
            if hasattr(self._proxy_mcp, "update_tools"):
                try:
                    self._proxy_mcp.update_tools(self._tools_metadata)
                except Exception:
                    pass
        if not self._proxy_started:
            await self._proxy_mcp.start()
            # NOTE: we deliberately do NOT register the server into the
            # global ``~/.codex/config.toml`` (no ``codex mcp add``).
            # That global list is shared by every codex process on the
            # host, never gets cleaned, and made each ``codex exec``
            # connect to dozens of dead ``deepworker_<pid>`` ports —
            # 120 s hang per stale server (observed in a related
            # internal incident). Instead the server is injected per-invocation
            # via ``ExecArgs.mcp_server_name``/``mcp_server_port`` (see
            # the ExecArgs construction in ``ainvoke`` and
            # ``CodexExecRunner._build_argv``). Agents stay fully
            # independent — there is no sanctioned global MCP list.
            self._proxy_started = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_mcp_content(content: Any) -> list[dict[str, Any]]:
    """Convert a LangChain ToolMessage content into MCP content blocks.

    Same shape as ``claude_code_sdk_provider.chat_model._build_mcp_content``.
    """
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks: list[dict[str, Any]] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            blocks.append({"type": "text", "text": str(b.get("text", ""))})
        else:
            blocks.append({"type": "text", "text": str(b)})
    return blocks


def _flatten_message_content_to_text(content: Any) -> str:
    """Collapse a LangChain message ``content`` field to a plain string.

    LangChain's ``BaseMessage.content`` is ``str | list[str | dict]``.
    Middleware that uses ``request.override(system_message=
    SystemMessage(content=[...]))`` produces the list form. ``str(list)``
    yields a Python repr that is useless to an LLM; we must instead pull
    out the text payload of every text block (and discard non-text blocks
    such as images / tool_use / anything else, since the consumer in this
    module is Codex's ``AGENTS.override.md`` file which is plain markdown).

    The flattener is intentionally permissive:

    - ``str`` content is returned unchanged.
    - ``list`` items that are themselves plain strings are kept verbatim
      (LangChain tolerates that shape).
    - ``dict`` items with ``type == "text"`` contribute their ``text`` value.
    - All other items are silently skipped (no exception, no garbage in
      the output).

    Text segments are joined with a single ``"\\n\\n"`` so a multi-block
    prompt looks like consecutive markdown paragraphs.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        # Unknown shape — degrade gracefully to ``str()`` so the prompt
        # still carries *something* readable.
        return str(content)
    parts: list[str] = []
    for item in content:  # type: ignore[reportUnknownVariableType]
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            text_val = item.get("text", "")
            if isinstance(text_val, str):
                parts.append(text_val)
            else:
                parts.append(str(text_val))
        # else: image / tool_use / unknown — skip (cannot render to text).
    return "\n\n".join(parts)


def _augment_base_instructions(system_prompt: str | None) -> str:
    if not system_prompt:
        return DEEP_WORKER_TOOL_CONTRACT
    if DEEP_WORKER_TOOL_CONTRACT in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{DEEP_WORKER_TOOL_CONTRACT}"


class _suppress_errors:
    """Context manager that swallows any exception during cleanup paths.

    Used in the few places where we must drain a generator / kill a
    subprocess and any exception there is strictly worse than swallowing
    it (the agent must continue serving other turns).
    """

    def __enter__(self) -> "_suppress_errors":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return True  # suppress

    async def __aenter__(self) -> "_suppress_errors":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return True


# NOTE: ``_register_proxy_with_codex`` (which ran ``codex mcp add`` to
# write the proxy server into the GLOBAL ``~/.codex/config.toml``) was
# REMOVED on 2026-05-20. That global, never-cleaned list accumulated one
# ``[mcp_servers.deepworker_<pid>]`` entry per agent ever launched, and
# every ``codex exec`` then tried to connect to all of them — each dead
# port hanging for the 120 s proxy timeout (observed live: "hangs 10 min then
# works"). The proxy server is now injected per-invocation via
# ``ExecArgs.mcp_server_name`` / ``mcp_server_port`` +
# ``--ignore-user-config`` (see ``CodexExecRunner._build_argv``). DO NOT
# re-introduce a global ``codex mcp add`` — agents must stay independent.
