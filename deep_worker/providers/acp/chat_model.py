"""AcpChatModel — LangChain BaseChatModel that drives any ACP-speaking CLI.

Public surface of the shared ACP provider layer. Same shape as
``codex_provider/chat_model.py`` / ``claude_code_sdk_provider`` / ``grok_provider``
— only ``ainvoke`` is real; ``_generate`` raises (async-only); LangChain
streaming methods are not implemented.

The CLI is driven over the Agent Client Protocol (ACP) v1 on stdio (see
``acp_client.py``). One long-lived subprocess per agent process; one
``session_id`` per ``agent_thread_id``; one ``session/prompt`` per turn.
Tools flow through two channels, both recorded in LangGraph history:

  * Channel A — deep-worker MCP tools via the in-agent proxy MCP server
    (``proxy_mcp_server.py``). The CLI dispatches them through its MCP
    dispatch built-in; the handler parks a Future resolved when the
    ToolMessage returns.
  * Channel B — the CLI's local built-ins (Bash/Shell/Read/Edit/...),
    observed on the ACP stream and recorded via ``builtin_tool_passthrough.py``.

Per-CLI configuration (command, vendor prefix, built-ins, permission policy,
session-load support, usage-limit markers) comes from :class:`AcpProfile`.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import mimetypes
import os
from datetime import UTC, datetime
from pathlib import Path
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

from . import acp_types
from .acp_client import AcpSinkOverflowError
from .arg_compat import normalize_tool_args
from .builtin_tool_passthrough import (
    BUILTIN_MARKER,
    META_KEY,
    PLAN_TOOL_NAME,
    is_builtin_passthrough_tool,
    resolve_builtin_future,
)
from .client_state import get_global_client_state
from .custom_types import (
    CALL_ID_MAP_MAX_ENTRIES,
    FORWARDED_EXTRA_KEYS_MAX,
    NOTIF_QUEUE_MAX_ITEMS,
    PROMPT_IDLE_TIMEOUT_DEFAULT,
    PROMPT_IDLE_TIMEOUT_ENV,
    QUEUED_TEXT_MAX_BYTES,
    QUEUED_TEXT_MAX_CHUNKS,
    QUEUED_TOOL_CALLS_MAX,
    REFINE_PENDING_MAX,
    ToolMetadata,
)
from .profile import AcpProfile
from .proxy_mcp_server import get_global_proxy_mcp_server

# Observed-context contract (deep_worker_cli side owns the arithmetic; the
# provider only stamps). module-level import is safe: session_state.py
# already imports deep_worker_cli at module load.
from deep_worker_cli.context_limit.observed_context import (
    annotate_context_observation,
    first_state_message_id,
)

logger = logging.getLogger(__name__)

# How long ``_correlate_proxy_tool_calls`` waits for the MCP handler a CLI
# tool_call should correspond to. Two budgets, picked per call from whether
# our proxy has EVER served a ``tools/call`` in this process — see the
# comment at the call site (a related internal incident analysis, not included here; latency section).
PROXY_CORRELATION_TIMEOUT_SEC = 10.0
PROXY_CORRELATION_PROBE_SEC = 0.5

# How many times a summarize prompt is re-issued when the CLI answers with
# tool_calls instead of prose (DESIGN §5.2a-3). Bounded so a model that
# insists on tools cannot spin: after this we surface "empty summary" and
# the caller demotes hot → cold as before.
_SUMMARY_TOOL_RETRIES = 2


def _required_args_present(schema: dict[str, Any] | None, args: dict) -> bool:
    """Whether ``args`` satisfies the REQUIRED properties of ``schema``.

    claude-agent-acp streams ``rawInput`` PROGRESSIVELY: a single tool call
    produces several refine updates, each carrying a strictly larger argument
    set (live-verified on 0.68.0 — ``{}`` → ``{"file_path"}`` →
    ``{"file_path", "content"}``). Dispatching on the first non-empty refine
    runs the tool with a half-streamed payload, which fails schema validation
    and leaves the CLI waiting for a result that never comes.

    A missing/!object schema, or one with no ``required`` list, is treated as
    satisfied — we must never hold a call hostage to a schema we don't
    understand (fail-open here means "behave as before", not "run empty":
    the empty-rawInput guard still applies). A malformed ``required`` entry
    (any element that is not a str) is treated the same way: the whole
    schema is untrusted, so releasing the call preserves the old behaviour
    rather than wedging on a well-typed subset we happen to recognise.
    """
    if not isinstance(schema, dict):
        return True
    required = schema.get("required")
    if not isinstance(required, list) or not required:
        return True
    if not all(isinstance(key, str) for key in required):
        # Malformed schema — fail OPEN, as documented above.
        return True
    return all(key in args and args[key] is not None for key in required)


def _prompt_idle_timeout() -> float:
    """Seconds of TOTAL silence before an in-flight turn is declared wedged.

    Read from the environment on every call (not cached at import) so an
    operator can tune a running deployment's restart without editing code.
    A non-numeric or non-positive value falls back to the default rather
    than disabling the bound — an unbounded wait is the very defect this
    guards against, so it must not be reachable by a typo.
    """
    raw = os.environ.get(PROMPT_IDLE_TIMEOUT_ENV)
    if raw is None:
        return PROMPT_IDLE_TIMEOUT_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "acp_provider: %s=%r is not a number — using default %.0fs",
            PROMPT_IDLE_TIMEOUT_ENV,
            raw,
            PROMPT_IDLE_TIMEOUT_DEFAULT,
        )
        return PROMPT_IDLE_TIMEOUT_DEFAULT
    if value <= 0:
        logger.warning(
            "acp_provider: %s=%r is not positive — refusing to disable the "
            "idle bound; using default %.0fs",
            PROMPT_IDLE_TIMEOUT_ENV,
            raw,
            PROMPT_IDLE_TIMEOUT_DEFAULT,
        )
        return PROMPT_IDLE_TIMEOUT_DEFAULT
    return value


class AcpRefusalError(RuntimeError):
    """Raised when an ACP turn ends with stopReason == "refusal"."""


class AcpPromptIdleTimeoutError(RuntimeError):
    """An in-flight ``session/prompt`` produced NO output for the idle window.

    Distinct from a crash (``AcpClientError``, EOF): the CLI process is
    alive, it simply stopped saying anything. Raised instead of waiting
    forever — see ``PROMPT_IDLE_TIMEOUT_DEFAULT``.
    """


class AcpTurnCancelledError(RuntimeError):
    """Raised for an ACP cancelled stop reason, not local task cancellation."""


class AcpSummarizationResetRequired(RuntimeError):
    """The 4b safety net fired: session state was reset mid-request.

    Raised INSTEAD of continuing the request. Continuing would open a fresh
    CLI session and use the entire visible history as its first prompt —
    the full-history dump the summarizer port exists to avoid (incident
    2026-08-19). The reset is already persisted when this is raised, so the
    next turn starts from clean state.
    """


def _contract_call_name(
    name: str, profile: AcpProfile, server_name: str | None
) -> str:
    """The provider-correct CALL name for bound tool ``name``.

    Names of MCP tools are MANGLED differently per CLI
    (per related internal incident analyses, not included here): grok 1.0.5 lists them as
    ``<server>__<tool>``, claude-agent-acp as ``mcp__<server>__<tool>``.
    The contract must name the form the CLI actually resolves; without a
    live ``server_name`` it degrades to the bare name (better than a
    guessed-stale PID)."""
    if not server_name:
        return name
    if profile.mcp_dispatch_style == "mcp_name_prefix":
        return f"mcp__{server_name}__{name}"
    return f"{server_name}__{name}"


def _build_tool_contract(
    tool_names: list[str],
    profile: AcpProfile,
    server_name: str | None = None,
) -> str:
    """Build the <system>-block tool guidance for the tools ACTUALLY bound.

    ACP CLIs may route MCP tool calls through one of two mechanisms
    (``profile.mcp_dispatch_style``): grok wraps them in a dispatch
    built-in (``use_tool`` on 1.0.5+, ``CallMcpTool`` before), while
    claude-agent-acp surfaces them as ordinary ``tool_call`` updates
    whose name carries the MCP prefix (``mcp__<server>__<tool>``).
    Either way the tool NAME the model must select is the MANGLED
    provider-specific form rendered by :func:`_contract_call_name` — a
    bare name is a native built-in at best and "No such tool" at worst.
    The contract is DYNAMIC and TRUTHFUL: it names exactly the bound
    deep-worker tools under their callable forms and explicitly forbids
    hunting for their schema on disk.

    When ``send_message_to_agent`` is among the bound tools we ALSO
    append a CRITICAL emphasis block (with the provider-correct
    invocation form rendered by ``proxy_tool_hints``) so the model has
    the inter-agent policy in front of it on EVERY turn — peer messages
    may surface as user-role blocks with ``System:>`` prefixes (or
    similar) on ACP providers, and the model must NOT mistake them for a
    user turn.

    Returns an empty string when no deep-worker tools are bound."""
    if not tool_names:
        return ""
    bound = set(tool_names)
    listed = ", ".join(
        f"`{_contract_call_name(n, profile, server_name)}`"
        for n in tool_names
    )
    lines = [
        "Deep-worker tool contract:",
        "- You have these deep-worker MCP tools available THIS TURN, under "
        f"exactly these names in your tool list: {listed}.",
        "- These are deep-worker MCP tools, NOT your native built-ins. "
        "Select them from your tool list under the exact names above — the "
        "CLI routes the call to the deep-worker MCP server automatically. "
        "Do NOT invent shorter names, and do NOT search the filesystem for "
        "their schema or definition — you already have everything needed "
        "to call them.",
    ]
    if profile.known_builtins:
        builtin_names = ", ".join(
            f"`{n}`" for n in sorted(profile.known_builtins)
        )
        lines.append(
            "- Use your local built-ins only for operations with no "
            "deep-worker equivalent. Both channels are recorded in history; "
            "this is a preference, not a hard restriction. "
            f"Known local built-ins: {builtin_names}."
        )
    if "send_message_to_agent" in bound:
        from deep_worker_cli.prompts.proxy_tool_hints import (  # noqa: PLC0415 — local import keeps the ACP layer importable without the CLI package at module load
            critical_send_message_to_agent_block,
        )

        lines.append("")
        lines.append(
            critical_send_message_to_agent_block(
                profile.provider, server_name=server_name
            )
        )
    return "\n".join(lines)


def _unshadow_profile_property(cls: type) -> None:
    """Undo pydantic's capture of our ``profile`` property as a field default.

    ``BaseChatModel`` declares ``profile: ModelProfile | None`` as a pydantic
    field. When a subclass defines ``profile`` as a ``@property``, pydantic's
    metaclass does not see a descriptor overriding a field — it sees a *value*
    assigned to an inherited annotated name, and stores the ``property`` object
    itself as that field's default, removing it from the class dict. Every
    instance then carries the raw ``property`` under ``__dict__["profile"]``,
    so ``model.profile`` yields a ``property`` object instead of the profile.

    That is not a hypothetical: it silently disabled built-in tool passthrough
    for every ACP provider, because the ``profile is not None`` gate passed on
    a ``property`` while ``known_builtins`` was never found.

    This restores the property as a real class-level descriptor and drops the
    field, so ``model.profile`` honours the LangChain contract. It must run for
    every subclass too — pydantic rebuilds ``model_fields`` per class, so each
    one re-captures the inherited property unless cleared again.
    """
    field = cls.model_fields.get("profile")
    if field is None or not isinstance(field.default, property):
        return
    prop = field.default
    del cls.model_fields["profile"]
    if not isinstance(cls.__dict__.get("profile"), property):
        setattr(cls, "profile", prop)
    cls.model_rebuild(force=True)


class AcpChatModel(BaseChatModel):
    """LangChain chat-model wrapper around an ACP-speaking CLI.

    Construction is cheap and side-effect-free; the proxy MCP server and ACP
    client are started lazily on the first ``ainvoke``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _client_state: Any = PrivateAttr(default=None)
    _acp_client: Any = PrivateAttr(default=None)  # AcpClient (one per process)
    _proxy_mcp: Any = PrivateAttr(default=None)
    _profile: AcpProfile = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    _working_dir: str | None = PrivateAttr(default=None)
    _model_name: str | None = PrivateAttr(default=None)
    _resolved_model_name: str | None = PrivateAttr(default=None)
    _resolved_model_callback: Callable[[str], Awaitable[None] | None] | None = (
        PrivateAttr(default=None)
    )
    _last_reported_model_name: str | None = PrivateAttr(default=None)
    _proxy_started: bool = PrivateAttr(default=False)
    _acp_started: bool = PrivateAttr(default=False)
    _usage_event_callback: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Pydantic schema overrides — same surgery as codex/claude-code-sdk/grok.
    # ------------------------------------------------------------------
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        try:
            parent_raw: Any = handler(core_schema)
            parent: dict[str, Any] = (
                parent_raw
                if isinstance(parent_raw, dict)
                else {"type": "object", "properties": {}}
            )
        except Exception:
            parent = {"type": "object", "properties": {}}
        props_raw: Any = parent.get("properties", {}) or {}
        props: dict[str, Any] = props_raw if isinstance(props_raw, dict) else {}
        parent["properties"] = {
            k: v
            for k, v in props.items()
            if k not in {"custom_get_token_ids", "callbacks", "rate_limiter"}
        }
        parent["title"] = cls.__name__
        parent["description"] = "ACP chat model for LangChain"
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
        profile: AcpProfile,
        model: str | None = None,
        max_tokens: int | None = None,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["custom_get_token_ids"] = None
        super().__init__(**kwargs)
        self._profile = profile
        self._client_state = get_global_client_state()
        self._client_state.set_profile(profile.provider, profile.fingerprint())
        self._tools_metadata = []
        self._model_name = model
        self._resolved_model_name = model
        self._working_dir = working_dir

    @property
    def profile(self) -> AcpProfile:
        return self._profile

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        _unshadow_profile_property(cls)

    @property
    def _llm_type(self) -> str:
        return "acp"

    @property
    def resolved_model_name(self) -> str | None:
        return self._resolved_model_name

    def set_resolved_model_callback(
        self, callback: Callable[[str], Awaitable[None] | None]
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
        """Install the async usage-event callback (§5.2)."""
        self._usage_event_callback = callback

    async def _emit_usage_event(
        self,
        *,
        event: str,
        status: str,
        usage: dict[str, Any] | None = None,
        session_id: str | None = None,
        context_used: int | None = None,
        context_size: int | None = None,
    ) -> None:
        """Report the per-turn usage to the stats callback (§5.2)."""
        cb = getattr(self, "_usage_event_callback", None)
        if cb is None:
            return
        try:
            await cb(
                event=event,
                status=status,
                usage=usage,
                model=getattr(self, "_resolved_model_name", None) or "",
                session_id=session_id,
                context_used=context_used,
                context_size=context_size,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception("[token-stats] acp usage event callback failed")

    # ------------------------------------------------------------------
    # Tools — extract MCP metadata (same algorithm as codex/claude-code-sdk/grok).
    # ------------------------------------------------------------------
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "AcpChatModel":
        # CRITICAL: built-in passthrough tools (channel B) MUST NOT be
        # published as MCP tools via proxy_mcp_server (channel A). They exist
        # ONLY for LangGraph's ToolNode to route the surfaced tool_call back
        # to a body that awaits the reader-resolved Future — they don't have a
        # remote endpoint. Filter them out here.
        tools_metadata: list[ToolMetadata] = []
        for item in tools:
            if not isinstance(item, BaseTool):
                continue
            if is_builtin_passthrough_tool(item):
                continue  # channel B — bound for ToolNode, not for proxy MCP
            schema = self._extract_tool_schema(item)
            tools_metadata.append(
                {
                    "name": item.name,
                    "description": item.description or "",
                    "input_schema": schema,
                }
            )
        self._tools_metadata = tools_metadata
        return self

    def bind(self, **kwargs: Any) -> Any:
        """Text-only turn: LangGraph calls ``bind()`` without ``bind_tools``.

        When middleware strips all tools (e.g. ``InterAgentReplyMiddleware``
        on a ``reply_back=False`` inter-agent delivery), tag this binding so
        ``ainvoke`` omits MCP for exactly one call.
        """
        return super().bind(_acp_text_only_turn=True, **kwargs)

    @staticmethod
    def _extract_tool_schema(tool: BaseTool) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        if tool.args_schema:
            try:
                if hasattr(tool.args_schema, "model_json_schema"):
                    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
                elif hasattr(tool.args_schema, "schema"):
                    schema = tool.args_schema.schema()  # type: ignore[union-attr]
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
                        if pname in ("runtime", "self", "kwargs", "config"):
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
    # Async-only
    # ------------------------------------------------------------------
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "AcpChatModel only supports async operations. Use ainvoke()."
        )

    # ------------------------------------------------------------------
    # Lazy startup of the proxy MCP + ACP client.
    # ------------------------------------------------------------------
    async def _ensure_proxy_started(self) -> None:
        if self._proxy_mcp is None:
            self._proxy_mcp = get_global_proxy_mcp_server(self._tools_metadata)
        elif hasattr(self._proxy_mcp, "update_tools"):
            with contextlib.suppress(Exception):
                self._proxy_mcp.update_tools(self._tools_metadata)
        if not self._proxy_started:
            await self._proxy_mcp.start()
            self._proxy_started = True

    async def _ensure_acp_started(self) -> None:
        if self._acp_client is None:
            # SHARED client (review item 6): the main agent model and the
            # TaskManager model resolve to ONE AcpClient / CLI subprocess per
            # profile, keyed by (provider, cmd). Each thread still owns a
            # distinct ACP session_id over the same stdio channel.
            from .acp_client import get_global_acp_client  # noqa: PLC0415

            self._acp_client = get_global_acp_client(
                self._profile, self._client_state
            )
        # Startup is driven by the SHARED client's own liveness/handshake
        # state (review item 5/6): after an EOF the client resets
        # `_proc`/`_initialized` in place, so `ensure_started()` transparently
        # re-spawns and re-handshakes on the NEXT invocation of this same
        # model — a stale per-model `_acp_started` flag must not wedge a
        # recovered client.
        await self._acp_client.ensure_started()
        self._acp_started = True

    # ------------------------------------------------------------------
    # ainvoke
    # ------------------------------------------------------------------
    async def ainvoke(  # type: ignore[override]
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        from langchain_core.prompt_values import PromptValue
        from langchain_core.runnables.config import ensure_config

        # 1. Normalize input.
        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)  # type: ignore[arg-type]

        # 2. Mandatory agent_thread_id.
        config = ensure_config(config)
        cfg = config.get("configurable") or {}
        if "thread_id" not in cfg:
            raise ValueError(
                "AcpChatModel requires explicit thread_id in config — this "
                "is deep-worker's agent name. Each caller (main agent, "
                "TaskManager, subagent) must own its thread_id. Implicit "
                "defaults are forbidden."
            )
        agent_thread_id: str = cfg["thread_id"]

        # Token-stats: mark the invocation start. Best-effort.
        _cb = getattr(self, "_usage_event_callback", None)
        if _cb is not None:
            try:
                await _cb(event="invocation_start")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("[token-stats] invocation_start callback failed")

        # 3. Per-thread state.
        state = self._client_state.get_or_create(agent_thread_id)

        # 4. LangGraph factory prepends ``request.system_message`` ahead of
        # checkpoint messages. ``sent_count`` tracks STATE messages only.
        state_messages = messages
        if messages and isinstance(messages[0], SystemMessage):
            state_messages = messages[1:]

        # 4b. Safety net only. Summarization MUST go through SummarizerPort
        # (asummarize / areset_after_summary). If sent_count still exceeds
        # state length, areset was skipped — log loud and recover WITHOUT
        # treating this ainvoke as a dump-into-new-session (incident 2026-08-19).
        if state.sent_count > len(state_messages):
            logger.error(
                "acp: sent_count=%s > len(state_messages)=%s on thread %s "
                "without areset_after_summary; resetting and refusing to "
                "re-prompt (a fresh session here would re-send the whole "
                "visible history — incident 2026-08-19)",
                state.sent_count,
                len(state_messages),
                agent_thread_id,
            )
            await self._adopt_pump(state)
            self._discard_pending_prompt(state, "summarization")
            self._client_state.record_summarization(agent_thread_id)
            # STOP here. Continuing would fall through to session/new plus a
            # first prompt built from every HumanMessage in ``state_messages``
            # — exactly the dump this net exists to prevent. The reset is
            # persisted, so the NEXT turn starts clean from a correct state.
            raise AcpSummarizationResetRequired(
                f"ACP session state was reset on thread {agent_thread_id} "
                "because summarization did not go through the summarizer "
                "port. The next turn will start from a clean session."
            )

        # 4c. Validity key for the observed-context stamp on every AIMessage
        # of this request (see ``_stamp_context_observation``). Set AFTER 4b
        # so it lands on the state object that survives a summarization
        # reset.
        state.context_first_id = first_state_message_id(messages)

        # 5. System prompt (plain text — factory-prepended SystemMessage).
        system_prompt: str | None = None
        if messages and isinstance(messages[0], SystemMessage):
            system_prompt = _flatten_message_content_to_text(messages[0].content)

        text_only_turn = bool(kwargs.pop("_acp_text_only_turn", False))
        state.text_only_turn_active = text_only_turn

        # 6. Lazy start of proxy MCP + ACP client. Text-only turns use a
        # fresh session without MCP — skip touching the proxy listener so
        # in-flight HTTP handlers cannot wedge the agent event loop.
        if not text_only_turn:
            await self._ensure_proxy_started()
        await self._ensure_acp_started()

        # 7. Tail delta only (state messages — not the prepended system).
        new_messages = state_messages[state.sent_count :]
        user_text_parts: list[str] = []
        tool_resolved = False
        saw_builtin_resume = False

        # 7b. LangGraph may append ToolMessage / echo AIMessage after a
        # mid-turn tool_call return without another ainvoke. sent_count then
        # lags behind state. On a fresh turn (no in-flight session/prompt)
        # send only the trailing HumanMessage delta so the CLI is not re-fed
        # stale tool rows.
        # ``session_live_in_process`` gates this: the re-slice is only safe
        # for a session the CLI is already carrying. On the FIRST turn after
        # a restart — or after a switch from another provider, where no ACP
        # session state exists at all — ``sent_count`` seeds to 0 and the
        # guard below is trivially true, so the re-slice would throw away the
        # agent's ENTIRE history and hand the brand-new session only the
        # newest line. That is the amnesia observed on a live agent
        # (178 messages collapsed to 1). Cases (a)/(b) in step 10 own the
        # first turn's payload; 7b must not pre-empt them.
        if (
            not tool_resolved
            and not saw_builtin_resume
            and state.session_live_in_process
            and state.pending_session_prompt_future is None
            and state.sent_count < len(state_messages)
        ):
            tail_human_idx: int | None = None
            for i in range(len(state_messages) - 1, state.sent_count - 1, -1):
                if isinstance(state_messages[i], HumanMessage):
                    tail_human_idx = i
                    break
            if tail_human_idx is not None and tail_human_idx >= state.sent_count:
                new_messages = state_messages[tail_human_idx:]

        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                is_builtin = bool(
                    (getattr(msg, "additional_kwargs", None) or {}).get(
                        BUILTIN_MARKER, False
                    )
                )
                if is_builtin:
                    saw_builtin_resume = True
                else:
                    proxy_call_id = state.proxy_call_id_by_acp_id.pop(
                        msg.tool_call_id, msg.tool_call_id
                    )
                    self._proxy_mcp.resolve_call(
                        proxy_call_id,
                        {"content": _build_mcp_content(msg.content)},
                        tool_name=msg.name,
                    )
                    tool_resolved = True
            elif isinstance(msg, HumanMessage):
                user_text_parts.append(
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
            # AIMessage — already in the CLI session. SystemMessage routing is
            # handled in step 11 via ``_collect_runtime_system_chunks``.

        # 7c. Text-only reply turns spin a fresh session with no prior CLI
        # history — include every HumanMessage from state so the model sees
        # the original user ask AND the inter-agent delivery.
        if text_only_turn:
            user_text_parts = [
                m.content if isinstance(m.content, str) else str(m.content)
                for m in state_messages
                if isinstance(m, HumanMessage)
            ]

        # 8. Persist sent_count BEFORE awaiting response (mirror codex).
        state.sent_count = len(state_messages)
        self._client_state.persist(agent_thread_id)

        # 9. Mid-turn resume vs new turn. A built-in-only resume MUST take
        # this branch too, else we'd fire a second concurrent session/prompt
        # (ACP violation). _consume_turn re-uses the same current_notif_queue
        # without a new subscription.
        #
        # EXCEPTION: when the tail carries a fresh HumanMessage (arrived
        # after a turn-ending tool_call — send_message_to_agent,
        # submit_relations, any turn-ender), taking the mid-turn resume
        # would return the tail of the OLD turn as this ainvoke's answer
        # and silently drop ``user_text_parts``. sent_count is already
        # persisted (step 8), so the swallowed prompt is permanently
        # marked "sent". Instead: drain the old turn to completion
        # (session_cancel the still-open response, clear queued state),
        # then fall through to step 10 (case c: session already live) so
        # step 11 builds a proper second session/prompt with the new
        # user text. (Observed in 2/2 live captures + repo_graph extractor.)
        if (tool_resolved or saw_builtin_resume) and (
            state.pending_session_prompt_future is not None
        ):
            if user_text_parts and not text_only_turn:
                # Cancel the abandoned turn so the CLI is not left waiting
                # on a tool result LangGraph already handled on its side.
                # Mirrors _prompt_for_summary retry (~955): cancel then
                # adopt pump then discard so the next session_prompt does
                # not collide with a live one.
                with contextlib.suppress(Exception):
                    await self._acp_client.session_cancel(state.session_id)
                await self._adopt_pump(state)
                self._discard_pending_prompt(state, "midturn-newq")
                self._client_state.clear_queued_turn_state(state)
                # Fall through to step 10 / 11: a new session/prompt on
                # the LIVE session carrying user_text_parts.
            else:
                await self._adopt_pump(state)
                message = await self._consume_turn(state, agent_thread_id)
                return await self._flush_live_context(state, message)

        # 10. New turn → ensure a session exists AND is live in THIS CLI
        # subprocess. Three cases:
        #   (a) no session id           → session/new
        #   (b) persisted id, not live  → session/load (restart resume, only
        #       when the profile supports it)
        #   (c) id already live here    → use directly
        mcp_servers: list[dict[str, Any]] = []
        if self._tools_metadata and not text_only_turn:
            proxy_url = await self._proxy_mcp.url_async()
            mcp_servers = [
                {
                    "type": "http",
                    "name": self._proxy_mcp.server_name,
                    "url": proxy_url,
                    "headers": [],
                }
            ]
        if state.session_id is None:
            # (a) Brand-new session.
            session_id = await self._acp_client.session_new(
                cwd=self._working_dir or ".",
                mcp_servers=mcp_servers,
            )
            state.session_id = session_id
            state.session_live_in_process = True
            self._client_state.register_session(session_id, agent_thread_id)
            self._client_state.persist(agent_thread_id)
            if hasattr(self._acp_client, "wait_session_ready"):
                await self._acp_client.wait_session_ready(session_id)
            is_first_prompt_in_session = True
        elif not state.session_live_in_process:
            # (b) A session id loaded from disk after a restart. The NEW
            # subprocess does not know this id — prompting it directly is
            # silently dropped (empty turn). Re-attach with session/load when
            # the profile supports it; otherwise start fresh.
            loaded = False
            if self._profile.can_load_session():
                try:
                    await self._acp_client.session_load(
                        state.session_id,
                        cwd=self._working_dir or ".",
                        mcp_servers=mcp_servers,
                    )
                    if hasattr(self._acp_client, "wait_session_ready"):
                        await self._acp_client.wait_session_ready(
                            state.session_id
                        )
                    # The CLI may still have a stale pending prompt from the
                    # previous lifecycle (session/load restores it). Cancel it
                    # so the next session/prompt doesn't get a "cancelled"
                    # response.
                    with contextlib.suppress(Exception):
                        await self._acp_client.session_cancel(
                            state.session_id
                        )
                    loaded = True
                except Exception:  # noqa: BLE001 — stale/un-loadable → fresh
                    logger.warning(
                        "acp: session/load failed for %s (thread %s); starting "
                        "a fresh session and re-sending the full prompt",
                        state.session_id, agent_thread_id,
                    )
                    # Usually the id died with a previous process (nothing to
                    # reap, the close just reports "Session not found"), but a
                    # load can also be REJECTED by a live CLI that still holds
                    # the subprocess. Only an explicit close covers that case.
                    with contextlib.suppress(Exception):
                        await self._acp_client.session_close(state.session_id)
                    self._client_state._evict_session(state.session_id)
            if not loaded:
                session_id = await self._acp_client.session_new(
                    cwd=self._working_dir or ".",
                    mcp_servers=mcp_servers,
                )
                state.session_id = session_id
                if hasattr(self._acp_client, "wait_session_ready"):
                    await self._acp_client.wait_session_ready(session_id)
                # The fresh session knows NOTHING. Re-send the entire visible
                # history's user text as this turn's prompt and re-deliver the
                # system block, exactly like a brand-new session.
                user_text_parts = [
                    m.content if isinstance(m.content, str) else str(m.content)
                    for m in state_messages
                    if isinstance(m, HumanMessage)
                ]
            state.session_live_in_process = True
            self._client_state.register_session(
                state.session_id, agent_thread_id
            )
            self._client_state.persist(agent_thread_id)
            is_first_prompt_in_session = not loaded
        else:
            # (c) Session already live in this process.
            is_first_prompt_in_session = False

        # 10b. Text-only turns on a LIVE session: cancel any in-flight prompt
        # and detach proxy MCP via ``session/load`` with ``mcp_servers=[]``.
        if (
            text_only_turn
            and state.session_live_in_process
            and state.session_id is not None
        ):
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(state.session_id)
            if self._tools_metadata and not state.mcp_detached_for_text_only:
                await self._acp_client.session_load(
                    state.session_id,
                    cwd=self._working_dir or ".",
                    mcp_servers=[],
                )
                if hasattr(self._acp_client, "wait_session_ready"):
                    await self._acp_client.wait_session_ready(
                        state.session_id
                    )
                with contextlib.suppress(Exception):
                    await self._acp_client.session_cancel(
                        state.session_id
                    )
                state.mcp_detached_for_text_only = True
        elif (
            not text_only_turn
            and self._tools_metadata
            and state.mcp_detached_for_text_only
            and state.session_live_in_process
            and state.session_id is not None
        ):
            proxy_url = await self._proxy_mcp.url_async()
            await self._acp_client.session_load(
                state.session_id,
                cwd=self._working_dir or ".",
                mcp_servers=[
                    {
                        "type": "http",
                        "name": self._proxy_mcp.server_name,
                        "url": proxy_url,
                        "headers": [],
                    }
                ],
            )
            if hasattr(self._acp_client, "wait_session_ready"):
                await self._acp_client.wait_session_ready(state.session_id)
            state.mcp_detached_for_text_only = False

        # 11. Build prompt blocks — first prompt prepends the <system> block.
        runtime_chunks, new_first_chars, new_extra_keys = (
            _collect_runtime_system_chunks(
                messages,
                new_messages,
                is_first_prompt_in_session=is_first_prompt_in_session,
                forwarded_first_system_chars=state.forwarded_first_system_chars,
                forwarded_extra_system_keys=state.forwarded_extra_system_keys,
            )
        )
        state.forwarded_first_system_chars = new_first_chars
        state.forwarded_extra_system_keys = new_extra_keys

        user_text = "\n\n".join(user_text_parts) if user_text_parts else ""
        if runtime_chunks:
            user_text = (
                "\n\n".join(runtime_chunks + [user_text])
                if user_text
                else "\n\n".join(runtime_chunks)
            )
        system_text = self._system_block_for_prompt(
            state,
            is_first_prompt_in_session=is_first_prompt_in_session,
            system_prompt=system_prompt,
        )
        if system_text is not None:
            text = (
                f"<system>\n{system_text}\n</system>\n\n"
                f"{user_text}"
            )
        else:
            text = user_text
        blocks = [{"type": "text", "text": text}]

        # 12. Fire session/prompt + consume until response / tool_calls.
        state.current_notif_queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        state.pending_session_prompt_future = self._acp_client.session_prompt(
            session_id=state.session_id,
            prompt=blocks,
            notification_sink=state.current_notif_queue,
        )
        try:
            message = await self._consume_turn(state, agent_thread_id)
            return await self._flush_live_context(state, message)
        finally:
            state.text_only_turn_active = False

    async def _proxy_mcp_servers(self, *, include_proxy: bool) -> list[dict[str, Any]]:
        if not include_proxy or not self._tools_metadata:
            return []
        proxy = getattr(self, "_proxy_mcp", None)
        if proxy is None:
            return []
        proxy_url = await proxy.url_async()
        return [
            {
                "type": "http",
                "name": proxy.server_name,
                "url": proxy_url,
                "headers": [],
            }
        ]

    async def _prompt_for_summary(
        self,
        state: Any,
        thread_id: str,
        text: str,
        *,
        path: str,
    ) -> Any:
        from deep_worker_cli.context_limit.hot_cache import SummaryResult

        # The CLI may answer a summarize prompt with tool_calls instead of
        # prose. That yields empty content, and treating it as "empty
        # summary" would demote a HOT summarization to a COLD one — a full
        # history dump (DESIGN §5.5 hot-miss cost) triggered by nothing more
        # than the model reaching for a tool. So: tell it tools are
        # unavailable and re-prompt, up to _SUMMARY_TOOL_RETRIES times
        # (DESIGN §5.2a-3).
        prompt_text = text
        message = None
        content = ""
        for attempt in range(_SUMMARY_TOOL_RETRIES + 1):
            state.current_notif_queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
            state.pending_session_prompt_future = self._acp_client.session_prompt(
                session_id=state.session_id,
                prompt=[{"type": "text", "text": prompt_text}],
                notification_sink=state.current_notif_queue,
            )
            message = await self._consume_turn(state, thread_id)
            raw = message.content
            content = raw if isinstance(raw, str) else str(raw)
            if str(content).strip():
                break
            if not getattr(message, "tool_calls", None):
                break
            # Tool calls were requested: cancel them, restate the constraint.
            logger.warning(
                "acp summarize: CLI requested tools (%s) on attempt %d/%d — "
                "re-prompting with tools declared unavailable",
                [tc.get("name") for tc in message.tool_calls],
                attempt + 1,
                _SUMMARY_TOOL_RETRIES + 1,
            )
            # The turn is still OPEN on the CLI side: it is waiting for the
            # tool results it just asked for. Issuing the next session/prompt
            # without cancelling would collide with that live turn. Same
            # cancel → adopt pump → discard future sequence _summarize_hot
            # uses for a mid-turn re-entry (reviewer B3-1, 2026-08-21).
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(state.session_id)
            await self._adopt_pump(state)
            self._discard_pending_prompt(state, "summarize-tool-retry")
            self._client_state.clear_queued_turn_state(state)
            prompt_text = (
                "Tools are unavailable during summarization. Do not call any "
                "tool. Reply with the summary text only.\n\n" + text
            )
        if not str(content).strip():
            raise RuntimeError("empty summary")
        # ``deep_worker_usage`` is written by ProvenanceMiddleware, which
        # only wraps GRAPH model calls — ``asummarize`` runs outside the
        # graph, so the summary AIMessage never carries it. Fall back to the
        # live gauge the CLI reported during this very prompt, otherwise
        # every SummaryResult.usage would be None (reviewer note 3e).
        usage = (message.response_metadata or {}).get("deep_worker_usage")
        if not isinstance(usage, dict):
            snap = getattr(state, "usage_snapshot", None)
            if isinstance(snap, dict) and isinstance(snap.get("used"), int):
                usage = {
                    "context_used": snap["used"],
                    "context_size": snap.get("size") or None,
                    "source": "acp_usage_update",
                }
        return SummaryResult(
            text=str(content).strip(),
            path=path,
            provider_session_id=state.session_id,
            usage=usage if isinstance(usage, dict) else None,
        )

    async def _summarize_hot(self, thread_id: str, instruction: str) -> Any:
        state = self._client_state.get_or_create(thread_id)
        if not state.session_id:
            raise RuntimeError("hot summarization requires session_id")
        logger.info(
            "acp summarize_hot thread=%s session=%s live=%s pending=%s",
            thread_id,
            state.session_id,
            state.session_live_in_process,
            state.pending_session_prompt_future is not None,
        )
        if not state.session_live_in_process:
            if not self._profile.can_load_session():
                raise RuntimeError("load_failed")
            mcp_servers = await self._proxy_mcp_servers(include_proxy=True)
            await self._acp_client.session_load(
                state.session_id,
                cwd=self._working_dir or ".",
                mcp_servers=mcp_servers,
            )
            if hasattr(self._acp_client, "wait_session_ready"):
                await self._acp_client.wait_session_ready(state.session_id)
            state.session_live_in_process = True
        if state.pending_session_prompt_future is not None:
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(state.session_id)
            await self._adopt_pump(state)
            self._discard_pending_prompt(state, "hot-summarize")
        return await self._prompt_for_summary(
            state, thread_id, instruction, path="hot"
        )

    async def _summarize_cold(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        instruction: str,
    ) -> Any:
        from deepagents.middleware.message_timestamps import (
            MonotonicUtcClock,
            format_messages_with_timestamps,
        )

        ephemeral = f"{thread_id}:summarizer"
        agent_state = self._client_state.get(thread_id)
        logger.info(
            "acp summarize_cold thread=%s agent_session=%s live=%s",
            thread_id,
            getattr(agent_state, "session_id", None),
            getattr(agent_state, "session_live_in_process", None),
        )
        st = self._client_state.get_or_create(ephemeral)
        sid = await self._acp_client.session_new(
            cwd=self._working_dir or ".",
            mcp_servers=[],
        )
        st.session_id = sid
        st.session_live_in_process = True
        self._client_state.register_session(sid, ephemeral)
        if hasattr(self._acp_client, "wait_session_ready"):
            await self._acp_client.wait_session_ready(sid)
        dump = format_messages_with_timestamps(list(messages), MonotonicUtcClock())
        try:
            return await self._prompt_for_summary(
                st, ephemeral, f"{instruction}\n\n{dump}", path="cold"
            )
        finally:
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(sid)
            # This session exists ONLY for this dump. Cancel interrupts the
            # turn but leaves the CLI subprocess (and its MCP servers) alive,
            # so without an explicit close every cold summarization leaks one
            # whole SDK process for the lifetime of the adapter.
            with contextlib.suppress(Exception):
                await self._acp_client.session_close(sid)
            self._client_state._evict_session(sid)
            self._client_state._states.pop(ephemeral, None)

    async def asummarize(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        instruction: str,
        decision: Any,
    ) -> Any:
        """SummarizerPort: hot in-session prompt or cold ephemeral dump session."""
        await self._ensure_acp_started()
        if getattr(decision, "hot", False):
            return await self._summarize_hot(thread_id, instruction)
        return await self._summarize_cold(thread_id, messages, instruction)

    async def areset_after_summary(self, *, thread_id: str) -> None:
        state = self._client_state.get(thread_id)
        doomed: str | None = None
        if state is not None:
            await self._adopt_pump(state)
            self._discard_pending_prompt(state, "summarization")
            # Capture BEFORE record_summarization nulls it: the post-summary
            # reset abandons this session (the next turn opens a fresh one),
            # and an abandoned id we never closed is a leaked subprocess.
            doomed = state.session_id
        self._client_state.record_summarization(thread_id)
        if doomed is not None:
            with contextlib.suppress(Exception):
                await self._acp_client.session_close(doomed)

    def _stamp_context_observation(
        self, state: Any, message: AIMessage, meta: "dict[str, Any] | None" = None
    ) -> AIMessage:
        """Stamp the provider's context reading onto an outgoing AIMessage.

        TWO sources, both the provider's own count (POLICIES §13 — nothing
        is ever estimated here):

        1. ``state.usage_snapshot`` — the CLI's live gauge from
           ``usage_update`` (used/size). Preferred when present: it measures
           the whole session the CLI holds, including its own prompt, tool
           schemas and thinking blocks. claude-agent-acp sends it.
        2. ``meta.inputTokens`` — the prompt the model actually read, from
           the ``session/prompt`` reply's ``_meta``.

        The second source is not a nicety: **grok 1.0.5 sends no
        ``usage_update`` at all** (live capture 2026-08-25 — a full turn
        yields only user_message_chunk / agent_thought_chunk /
        agent_message_chunk / available_commands_update). Every grok agent
        in this deployment therefore had ZERO context stamps, so the guard
        and the summarizer ran on UNKNOWN and an agent launched with
        ``-l 250k`` was completely unguarded. ``_meta.inputTokens`` is the
        same fact from the same CLI: measured over three consecutive turns
        it grew 14144 → 14190 → 14335, matching grok's own
        ``inference_done.prompt_tokens`` exactly and exceeding
        ``cachedReadTokens`` (i.e. it counts the cached prefix, which is
        what a context size means).

        No source → NO stamp. The size is then UNKNOWN and stays unknown.
        """
        used: int | None = None
        size: int | None = None
        snap = getattr(state, "usage_snapshot", None)
        if isinstance(snap, dict):
            candidate = snap.get("used")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                used = candidate
                size = snap.get("size")
        if used is None:
            # The live gauge is fresher than anything a turn-boundary reply
            # can carry: it is updated after every model call.
            candidate = getattr(state, "live_context_used", None)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                used = candidate
        if used is None and isinstance(meta, dict):
            candidate = meta.get("inputTokens")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                used = candidate
        if used is None:
            return message
        annotate_context_observation(
            message,
            used=used,
            size=size if isinstance(size, int) and size > 0 else None,
            first_id=getattr(state, "context_first_id", None),
        )
        state.last_context_used = used
        state.last_response_at = datetime.now(UTC).isoformat(timespec="microseconds")
        # Persist NOW, not at the next turn's persist(): a --restart between
        # this turn and the next would otherwise drop the gauge and leave
        # ``extra_floor`` (DESIGN §6) blind on the first before_model.
        with contextlib.suppress(Exception):
            self._client_state.persist_state(state)
        return message

    async def _publish_pending_context(self, state: Any) -> None:
        """Publish a newly arrived context reading. One source, one moment.

        The CLI reports its context DURING the turn, once per model call.
        Holding those readings until the turn boundary is what made the web
        UI sit frozen and then jump, and — worse — what let two numbers
        exist at once: on 2026-08-26 ``acp_session_state.json`` climbed
        98163 -> 159414 -> 167831 for a live claude-code agent while
        ``token_stats.jsonl``, the only file the UI reads, still held 286194
        written by the PREVIOUS process on a DIFFERENT provider. The UI
        showed that stale number next to the live provider and model.

        So publishing is not a separate step that can be skipped: accepting
        a reading, updating the gauge the guard and the summarizer read, and
        writing the row the UI reads are one operation.

        Idempotent — a value already published is not news and is not
        rewritten. Best-effort: telemetry never breaks a turn, and a failed
        write must not strand the reading, so the gauge is advanced first.
        """
        used = getattr(state, "live_context_used", None)
        if not isinstance(used, int) or isinstance(used, bool) or used <= 0:
            return
        if getattr(state, "last_context_used", None) == used:
            return
        state.last_context_used = used
        try:
            await self._emit_context_snapshot(
                used, session_id=str(getattr(state, "session_id", None) or "")
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a gauge must never break a turn
            logger.debug(
                "acp_provider: publishing a context reading failed",
                exc_info=True,
            )

    async def _consume_turn(self, state: Any, agent_thread_id: str) -> AIMessage:
        """Consume session/update notifications until either pending
        tool_call(s) need LangGraph routing (return AIMessage(tool_calls),
        keeping the in-flight future for the next ainvoke) or the
        session/prompt response arrives (return AIMessage(content)).

        On re-entry from a mid-turn ainvoke this method PREPENDS the
        pump-accumulated ``state.queued_text`` and ``state.queued_tool_calls``.
        If the pump already observed the response future resolving with a
        terminal stopReason it stashed the result in ``state.queued_response``
        so we short-circuit without waiting for ``response_fut`` (already
        done).

        The live context gauge is started here: the session id exists by
        now, and a turn may run a dozen model calls before returning, all
        of which must be visible (owner directive 2026-08-25).
        """
        await self._start_live_context_watch(state)
        # _BoundedTextAcc, not a plain list: _append_bounded_text is called
        # once per agent_message_chunk, and re-summing the buffer there made a
        # long turn quadratic (the burst defect's root cause — see the class).
        final_text_parts: list[str] = _BoundedTextAcc(state.queued_text)
        pending_tool_calls: list[dict] = list(state.queued_tool_calls)
        state.queued_text = _BoundedTextAcc()
        state.queued_tool_calls = []
        queued_response = state.queued_response
        state.queued_response = None

        notif_q: asyncio.Queue = state.current_notif_queue
        response_fut: asyncio.Future = state.pending_session_prompt_future
        cleared_normally = False
        get_task: asyncio.Task | None = None
        try:
            # If a previous pump already saw response_fut resolve, finalize
            # immediately. ``queued_response`` is the raw JSON-RPC reply.
            if queued_response is not None:
                state.pending_session_prompt_future = None
                cleared_normally = True
                return await self._finalize_response(
                    queued_response, final_text_parts, state
                )

            # If the pump already queued tool_calls AND the response future
            # is still in flight, hand them back to LangGraph before reading
            # more notifications — the model needs to run them next.
            if pending_tool_calls:
                if state.text_only_turn_active:
                    logger.warning(
                        "acp text-only turn surfaced tool_calls %r — "
                        "dropping to avoid hung ToolNode",
                        [tc.get("name") for tc in pending_tool_calls],
                    )
                    with contextlib.suppress(Exception):
                        await self._acp_client.session_cancel(
                            state.session_id
                        )
                    state.pending_session_prompt_future = None
                    cleared_normally = True
                    return self._stamp_context_observation(
                        state, AIMessage(content="".join(final_text_parts))
                    )
                await self._correlate_proxy_tool_calls(
                    state, pending_tool_calls
                )
                cleared_normally = True
                # Re-arm the drain BEFORE handing the calls back. This is the
                # same mid-turn pause as the in-loop branch below (:~1486) and
                # it needs the same pump: the response future stays in flight,
                # so the sink stays registered in the client and the reader
                # keeps routing session/update into the SAME queue (created
                # once per turn at ~:982, not per ainvoke — capacity 1000 is a
                # per-TURN budget).
                #
                # Without this, the path was a drainer-less hole: _adopt_pump
                # cancelled the previous pump on re-entry (~:794/:801) and is a
                # silent no-op when none exists, so a long tool dispatched here
                # (wait_for_job, 10-20 min) ran with NOBODY reading the queue
                # until acp_client's sink.full() failed the whole turn —
                # observed in a live session:
                # AcpSinkOverflowError surfacing at response_fut.result().
                self._start_pump(state, agent_thread_id)
                return self._stamp_context_observation(
                    state,
                    AIMessage(
                        content="".join(final_text_parts),
                        tool_calls=pending_tool_calls,
                    ),
                )

            while True:
                get_task = asyncio.ensure_future(notif_q.get())
                done, _pending = await asyncio.wait(
                    {get_task, response_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=_prompt_idle_timeout(),
                )
                if not done:
                    # IDLE TIMEOUT (round 8, defect #4): neither a
                    # notification nor the response arrived for the whole
                    # window. Only a CRASH (EOF) was handled before, so
                    # "alive but silent" — a wedged CLI, or a reader that
                    # died — hung the agent FOREVER with nothing logged.
                    # The timer resets on every notification, so reaching
                    # here means TOTAL silence: treat the channel as wedged,
                    # cancel the turn (the finally block does it) and fail
                    # loudly instead of hanging.
                    get_task.cancel()
                    logger.error(
                        "acp_provider: no session/update and no response for "
                        "%.0fs on session %s — treating the turn as wedged "
                        "(tune with %s)",
                        _prompt_idle_timeout(),
                        state.session_id,
                        PROMPT_IDLE_TIMEOUT_ENV,
                    )
                    raise AcpPromptIdleTimeoutError(
                        f"ACP turn produced no output for "
                        f"{_prompt_idle_timeout():.0f}s (session "
                        f"{state.session_id}). The CLI is alive but silent; "
                        f"the turn was cancelled. Tune with "
                        f"{PROMPT_IDLE_TIMEOUT_ENV}."
                    )
                if response_fut in done:
                    # CRITICAL: the CLI streams all of its
                    # agent_message_chunk text and THEN resolves the response
                    # in the SAME event loop iteration, so by the time we get
                    # here the notif queue can still hold the chunks that make
                    # up the answer. Finalizing straight away returned an
                    # EMPTY AIMessage. So first absorb the get_task we already
                    # launched, then drain whatever else is queued, applying
                    # every update — only THEN finalize.
                    if get_task in done and not get_task.cancelled():
                        with contextlib.suppress(asyncio.CancelledError):
                            self._apply_update(
                                state, agent_thread_id, get_task.result(),
                                final_text_parts, pending_tool_calls,
                            )
                    else:
                        get_task.cancel()
                    await self._drain_until_stream_exhausted(
                        state, agent_thread_id, notif_q,
                        final_text_parts, pending_tool_calls,
                    )
                    await self._publish_pending_context(state)
                    # ``final_text_parts``/``pending_tool_calls`` are frame
                    # locals, so a raising ``.result()`` would discard every
                    # update we just drained along with the frame — the whole
                    # turn's work, silently. Hand them to the state first so
                    # the error path can still surface them (owner directive:
                    # a harness error must not destroy completed work).
                    self._preserve_turn_work(
                        state, final_text_parts, pending_tool_calls
                    )
                    response = response_fut.result()
                    # ``.result()`` did not raise: this turn is delivered, and
                    # the parking above has done its job. Release it — the
                    # buffer is per-``agent_thread_id`` and OUTLIVES the turn,
                    # so anything still parked here is what the next turn in
                    # this branch picks up at :1320 and prepends to its own
                    # text. That is not a leak of bytes but of MEANING: the
                    # reply grew by one whole turn per turn (task-manager agent,
                    # 23→27 verdict blocks in a single AIMessage, seams glued
                    # mid-line because the parts are joined with no separator),
                    # and the TaskManager re-read its own earlier verdicts as
                    # part of the new one.
                    #
                    # Only the SUCCESS path clears. The raising path must keep
                    # the work parked — that is the whole point of preserving
                    # it — and a mid-turn tool pause returns earlier, before
                    # the parking, because its text belongs to a turn that has
                    # not ended yet.
                    self._release_preserved_turn_work(state)
                    state.pending_session_prompt_future = None
                    cleared_normally = True
                    if pending_tool_calls:
                        if state.text_only_turn_active:
                            logger.warning(
                                "acp text-only turn end-of-turn tool_calls %r "
                                "— dropping",
                                [tc.get("name") for tc in pending_tool_calls],
                            )
                            pending_tool_calls.clear()
                        else:
                            # The CLI streamed a tool_call AND resolved the
                            # prompt in the same tick. _finalize_response
                            # builds an AIMessage from text only, so falling
                            # through here would silently drop the call and
                            # the tool would never run. Surface it exactly
                            # like the mid-turn branch — but the turn is
                            # already over, so no pump and no pending future
                            # to keep alive.
                            await self._correlate_proxy_tool_calls(
                                state, pending_tool_calls
                            )
                            return self._stamp_context_observation(
                                state,
                                AIMessage(
                                    content="".join(final_text_parts),
                                    tool_calls=pending_tool_calls,
                                ),
                            )
                    return await self._finalize_response(
                        response, final_text_parts, state
                    )

                notif = get_task.result()
                self._apply_update(
                    state,
                    agent_thread_id,
                    notif,
                    final_text_parts,
                    pending_tool_calls,
                )
                await self._publish_pending_context(state)

                if pending_tool_calls:
                    if state.text_only_turn_active:
                        logger.warning(
                            "acp text-only turn mid-stream tool_calls %r — "
                            "dropping",
                            [tc.get("name") for tc in pending_tool_calls],
                        )
                        pending_tool_calls.clear()
                        continue
                    # Mid-turn pause: give LangGraph the tool_calls. KEEP the
                    # response future — the next ainvoke re-enters
                    # _consume_turn. Spawn the pump so the queue keeps being
                    # drained while ToolNode runs the passthrough body.
                    await self._correlate_proxy_tool_calls(
                        state, pending_tool_calls
                    )
                    cleared_normally = True
                    self._start_pump(state, agent_thread_id)
                    return self._stamp_context_observation(
                        state,
                        AIMessage(
                            content="".join(final_text_parts),
                            tool_calls=pending_tool_calls,
                        ),
                    )
        finally:
            # Always — a cancelled or failed turn must not leak the gauge
            # task or its inotify handle.
            await self._stop_live_context_watch(state)
            if (
                not cleared_normally
                and state.pending_session_prompt_future is not None
            ):
                fut = state.pending_session_prompt_future
                state.pending_session_prompt_future = None
                if not fut.done():
                    fut.cancel()
                # Stop the pump too — abnormal exit means we're abandoning
                # the turn and the pump must not keep racing on a dead queue.
                self._cancel_pump(state)
                with contextlib.suppress(Exception):
                    await self._acp_client.session_cancel(state.session_id)
            # Never leave this iteration's get_task registered as a waiting
            # getter: the next reader (pump after mid-turn pause, or a future
            # _consume_turn) would have its put_nowait data stolen by a dead
            # task. Cancelling a done task is a no-op.
            if get_task is not None and not get_task.done():
                with contextlib.suppress(Exception):
                    get_task.cancel()
            state.refine_pending_tool_calls.clear()
            # INVARIANT (do not downgrade to ``assert``: -O strips it in
            # exactly the production run we care about). Leaving this method
            # with the response future still in flight means the sink stays
            # registered in the client and session/update keeps landing in
            # the bounded per-turn queue — so SOMETHING must be reading it.
            # The only legitimate reader at this point is the pump.
            #
            # Violating this does not break the turn here; it creates the
            # condition for a death 10-20 minutes later at an unrelated
            # ``response_fut.result()``, which is why the drainer-less hole
            # above (:~1376) cost a full debugging session to localise. Log
            # loudly instead of raising: the turn is still serviceable, and a
            # new third path may be legitimate — but it will NAME itself here
            # rather than being reconstructed from a stack trace.
            fut_alive = state.pending_session_prompt_future
            if (
                fut_alive is not None
                and not fut_alive.done()
                and state.pump_task is None
            ):
                logger.error(
                    "acp_provider: leaving _consume_turn with the prompt "
                    "future in flight and NO pump on session %s (thread %s) "
                    "— the notification queue has no reader and will "
                    "overflow; this is a drainer-less turn, not a stall",
                    state.session_id,
                    agent_thread_id,
                )

    # ------------------------------------------------------------------
    # Shared update handling (used by both _consume_turn and _drain_pump)
    # ------------------------------------------------------------------
    def _apply_update(
        self,
        state: Any,
        agent_thread_id: str,
        notif: dict,
        text_acc: list[str],
        tool_calls_acc: list[dict],
    ) -> None:
        """Apply ONE session/update notification to the accumulators.

        Pure data manipulation — no awaits, no Future creation other than the
        ``builtin_passthrough_futures`` entry registered by ``_map_tool_call``
        (which is event-loop-safe). Used by both the foreground consumer
        (``_consume_turn``) and the background pump (``_drain_pump``) so the
        semantics are identical regardless of who saw the update first.
        """
        upd = (notif.get("params") or {}).get("update") or {}
        kind = acp_types.session_update_kind(upd)
        if kind == "agent_message_chunk":
            chunk = (upd.get("content") or {}).get("text", "")
            _append_bounded_text(text_acc, chunk)
        elif kind == "agent_thought_chunk":
            return  # reasoning chunks — never surfaced
        elif kind == "tool_call":
            tc = self._map_tool_call(state, upd)
            if tc is not None:
                self._append_tool_call_bounded(state, tool_calls_acc, tc)
            elif (
                self._profile.mcp_dispatch_style == "mcp_name_prefix"
                and upd.get("toolCallId")
            ):
                # claude-agent-acp surfaces tool_call with rawInput={}
                # (pending) and streams the real arguments across one or MORE
                # follow-up tool_call_update events WITHOUT status (refines).
                # Hold the call back until a refine carries a COMPLETE
                # argument set (or the turn ends).
                # BOUNDED (reject-new): if the set is full we drop the hold —
                # the follow-up refine update then takes the plain path and
                # the tool simply never runs (fail-closed, no hang).
                if len(state.refine_pending_tool_calls) < REFINE_PENDING_MAX:
                    state.refine_pending_tool_calls.add(upd["toolCallId"])
                else:
                    logger.error(
                        "acp_provider: refine_pending_tool_calls at capacity "
                        "(%d) — dropping refine hold for %r (reject-new)",
                        REFINE_PENDING_MAX,
                        upd["toolCallId"],
                    )
        elif kind == "tool_call_update":
            tcid = upd.get("toolCallId")
            if (
                tcid
                and tcid in state.refine_pending_tool_calls
                and upd.get("status") is None
            ):
                # Refine event: more args just streamed for a held-back call.
                # rawInput arrives PROGRESSIVELY, so this may still be an
                # INCOMPLETE snapshot — _map_tool_call returns None until the
                # required args are all present. Keep the hold in that case so
                # a later refine can complete it; releasing early is exactly
                # what dispatched write_file without its `content`.
                tc = self._map_tool_call(state, upd)
                if tc is not None:
                    state.refine_pending_tool_calls.discard(tcid)
                    self._append_tool_call_bounded(state, tool_calls_acc, tc)
            else:
                # Terminal (or non-held) update. A terminal status for a
                # never-refined pending call releases the hold (the turn is
                # done regardless of whether args ever streamed).
                if tcid and upd.get("status") in ("completed", "failed", "cancelled"):
                    state.refine_pending_tool_calls.discard(tcid)
                self._resolve_builtin_update(agent_thread_id, upd)
        elif kind == "usage_update":
            # Context gauge (token-stats design notes): used/size at
            # the top level of the update per ACP spec, with a content{}-dict
            # fallback. Keeps the LATEST snapshot of the current turn.
            gauge: dict[str, Any] = (
                upd.get("content")
                if isinstance(upd.get("content"), dict)
                else upd
            )
            used = gauge.get("used")
            size = gauge.get("size")
            if isinstance(used, (int, float)) or isinstance(size, (int, float)):
                state.usage_snapshot = {
                    "used": int(used) if isinstance(used, (int, float)) else 0,
                    "size": int(size) if isinstance(size, (int, float)) else 0,
                }
            # A ``usage_update`` IS a model call finishing. Record it here,
            # in the one place a reading enters the model, so the number the
            # operator sees can never be a different number from the one the
            # guard and the summarizer act on. ``_apply_update`` is sync by
            # contract, so the value is parked and the async consumers
            # publish it on the next line of their loop.
            if isinstance(used, (int, float)) and not isinstance(used, bool) and used > 0:
                state.live_context_used = int(used)
        elif kind == "plan":
            tc = _plan_to_tool_call(upd)
            if tc is not None:
                # A plan is a COMPLETED FACT: the CLI sends the entries and
                # never follows up with ``tool_call_update status=completed``.
                # So the recorder's Future has to be resolved HERE, at emission
                # time — parking it the way a real built-in parks would hang
                # the ToolNode body forever, since nothing would ever resolve
                # it. Without a resolved Future the tool_call would instead
                # reach the ToolNode with no bound tool and LangGraph would
                # fabricate ``"update_plan is not a valid tool"`` into history
                # (observed in a live session) — the same defect this records
                # around, for a name our own layer mints.
                self._preresolve_plan_result(state, tc)
                self._append_tool_call_bounded(state, tool_calls_acc, tc)
        # other kinds — silently ignored

    def _preresolve_plan_result(self, state: Any, tc: dict) -> None:
        """Store the ALREADY-KNOWN result for a synthetic ``update_plan``.

        For a real built-in the reader resolves a Future on
        ``tool_call_update``; for a plan there is no such follow-up — the plan
        IS the result — so the payload is known now and stored now. The
        passthrough body pops it before parking, so that call never blocks.

        Stored as a plain dict rather than a resolved Future on purpose:
        creating a Future needs an event loop, and ``_apply_update`` is sync by
        contract and legitimately runs with none (proven by
        tests/acp_provider/test_chat_model_refine.py, which raised
        ``RuntimeError: no running event loop``); a Future bound to one loop
        would also break when awaited on another.

        Bounded like the call-id map: at capacity the entry is skipped and
        logged rather than growing the dict without limit. Skipping is
        ACCEPTABLE here — not consequence-free — because the failure is loud
        and bounded: the caller still records the tool_call itself, so the
        ToolNode finds the bound recorder either way, and the body then raises
        ``RuntimeError`` (the turn fails visibly) instead of parking forever.
        Dropping the tool_call instead would hang LangGraph.
        """
        tcid = tc.get("id")
        if not isinstance(tcid, str) or not tcid:
            return
        if tcid in state.builtin_passthrough_results:
            return
        if len(state.builtin_passthrough_results) >= CALL_ID_MAP_MAX_ENTRIES:
            logger.error(
                "acp_provider: builtin_passthrough_results at capacity (%d) — "
                "not recording plan result %r (reject-new)",
                CALL_ID_MAP_MAX_ENTRIES,
                tcid,
            )
            return
        entries = (tc.get("args") or {}).get("entries")
        state.builtin_passthrough_results[tcid] = {
            "text": _render_plan_entries(entries),
            "is_error": False,
            "metadata": {},
            "meta": {"entries": entries} if entries is not None else {},
        }

    def _append_tool_call_bounded(
        self, state: Any, acc: list[dict], tc: dict
    ) -> None:
        """Append ``tc`` to the tool-call accumulator with a hard cap.

        Overflow → FAIL-LOUD: the active turn's response future is failed with
        ``AcpSinkOverflowError`` so ``_consume_turn`` raises a terminal error
        instead of silently losing a tool_call (which would hang LangGraph's
        ToolNode waiting for a ToolMessage that never arrives).
        """
        if len(acc) >= QUEUED_TOOL_CALLS_MAX:
            logger.error(
                "acp_provider: tool_call accumulator at capacity (%d) — "
                "failing the active turn (fail-loud)",
                QUEUED_TOOL_CALLS_MAX,
            )
            self._fail_turn(
                state,
                AcpSinkOverflowError(
                    f"ACP tool_call accumulator overflowed (capacity "
                    f"{QUEUED_TOOL_CALLS_MAX})"
                ),
            )
            return
        acc.append(tc)

    async def _drain_until_stream_exhausted(
        self,
        state: Any,
        agent_thread_id: str,
        notif_q: asyncio.Queue,
        text_acc: list[str],
        tool_calls: list[dict],
    ) -> None:
        """Apply every queued update, then every update still in the reader.

        ``while not notif_q.empty()`` is NOT a sufficient end-of-stream test.
        The reader must be interruptible (otherwise a buffered burst can never
        drain — measured: zero yields, so the consumer never runs and the sink
        overflows), and the instant it became interruptible an empty queue
        stopped meaning "the stream is exhausted": it can equally mean "the
        reader is holding the next frame", e.g. parked in a backpressured
        ``await sink.put()``.

        That is exactly how the tail used to be lost. The two-source wait in
        ``_consume_turn`` resolves a tie in favour of the response future
        (checked before the queue branch), so with the reader parked the
        consumer finalized while the tail was still unread inside the
        ``StreamReader`` — measured 4802 of 5001 frames read at the break, and
        LOST = tail (50 of 50, 200 of 200).

        The correct test is BOTH: the queue is empty AND the reader reports it
        holds no frame. Both are events, so this needs no timeout and no polling;
        a CLI that keeps streaming simply keeps the loop going, which is the
        behaviour the premise above already relies on.

        ``reader_is_idle``/``wait_reader_idle`` are best-effort: a client that
        does not expose them (a stub in a unit test) degrades to the old
        queue-only drain rather than raising.
        """
        client = self._acp_client
        reader_is_idle = getattr(client, "reader_is_idle", None)
        wait_reader_idle = getattr(client, "wait_reader_idle", None)
        while True:
            while not notif_q.empty():
                self._apply_update(
                    state, agent_thread_id, notif_q.get_nowait(),
                    text_acc, tool_calls,
                )
            if not callable(reader_is_idle) or not callable(wait_reader_idle):
                return
            if reader_is_idle():
                return
            # The reader holds a frame. Wait for it to hand that frame over and
            # park again — then re-check the queue, because the frame it was
            # holding has just landed there.
            await wait_reader_idle()

    def _preserve_turn_work(
        self,
        state: Any,
        text_acc: list[str],
        tool_calls: list[dict],
    ) -> None:
        """Move frame-local turn work back onto ``state`` before a raise.

        ``_consume_turn`` accumulates text and tool_calls in LOCALS. Any
        exception between the drain loop and the return — notably
        ``response_fut.result()`` re-raising ``AcpSinkOverflowError`` — unwinds
        the frame and takes the entire turn's accumulated output with it, which
        is exactly the "a harness error destroyed the work" failure the owner
        ruled out. Parking the work on the state keeps it recoverable.

        Idempotent-by-construction: entries already on the state are not
        duplicated, because ``_consume_turn`` MOVED them out (``state.queued_*``
        were emptied at entry) before this is called.

        Bounded: reuses the same ``_append_bounded_text`` /
        ``_append_tool_call_bounded`` caps as the pump, so preserving work can
        never make the buffers grow without limit.

        Consequently preservation is BEST-EFFORT, not complete: a turn that
        already sits at ``QUEUED_TOOL_CALLS_MAX`` (or the text cap) still sheds
        the excess here — ``_append_tool_call_bounded`` logs, fails the turn and
        drops that tool_call. That is the accepted trade for the overflow path
        (the turn is terminally failed anyway, and previously ALL of the work
        was lost), but do not read "bounded" as "everything is saved".
        """
        # Upgrade in place: a plain list here (dataclass default, or a restored
        # checkpoint) would make every _append_bounded_text below re-sum the
        # whole buffer, which is the quadratic path this turn may already have
        # thousands of chunks in. The caps are still enforced per chunk.
        if not isinstance(state.queued_text, _BoundedTextAcc):
            state.queued_text = _BoundedTextAcc(state.queued_text)
        for chunk in text_acc:
            _append_bounded_text(state.queued_text, chunk)
        for tc in tool_calls:
            self._append_tool_call_bounded(state, state.queued_tool_calls, tc)

    def _release_preserved_turn_work(self, state: Any) -> None:
        """Drop the turn work parked by ``_preserve_turn_work`` once delivered.

        The counterpart to the parking, and the reason it is safe: the parking
        exists only so a raising ``response_fut.result()`` cannot destroy a
        completed turn's output. Once ``.result()`` has returned, the turn IS
        the caller's and the copy on the state has no reader left.

        Keeping it is not merely untidy. ``state`` is per-``agent_thread_id``
        and outlives the turn, and the next turn opens with
        ``_BoundedTextAcc(state.queued_text)`` — so a leftover buffer is
        silently prepended to the next reply, which then parks the sum, and so
        on. Measured on a task-manager agent: one AIMessage per evaluation
        carrying 23, 24, 25, 26, 27 ``STATUS:`` blocks (36086 → 40871 chars),
        every seam glued mid-line (``...MESSAGE: NONESTATUS: done``) because
        the parts are concatenated without a separator.

        Clearing in place rather than rebinding: ``_preserve_turn_work`` may
        have upgraded the buffer to ``_BoundedTextAcc``, and the pump holds the
        same object, so rebinding would leave a writer pointing at the old
        list. ``_BoundedTextAcc.clear`` resets the incremental total with it.
        """
        state.queued_text.clear()
        state.queued_tool_calls.clear()

    def _fail_turn(self, state: Any, exc: BaseException) -> None:
        """Fail the in-flight ``session/prompt`` response future with ``exc``.

        The consumer (``_consume_turn`` / ``_drain_pump``) races that future
        against the notification queue, so an exception set here propagates
        out of the next await — the turn ends with a visible error instead of
        hanging. Idempotent and safe from both the foreground consumer and the
        background pump.
        """
        fut = state.pending_session_prompt_future
        if fut is not None and not fut.done():
            fut.set_exception(exc)

    async def _finalize_response(
        self,
        response: dict,
        text_acc: list[str],
        state: Any | None = None,
    ) -> AIMessage:
        """Common terminal path: report modelId, raise on cancelled/refusal,
        emit the token-stats context snapshot, return the assembled AIMessage.
        Shared between the natural turn-end in ``_consume_turn`` and the
        pump-prefetched response path.

        ``session/prompt`` resolves its future with the RAW reply and bypasses
        ``_call``'s error check, so an error reply reaches here as
        ``{"id", "error"}`` with no ``result``. Detect it FIRST and raise
        (usage/billing cap -> failover, anything else -> hard error) instead
        of silently returning an empty AIMessage."""
        from .acp_client import raise_for_error_reply  # noqa: PLC0415

        if isinstance(response, dict):
            raise_for_error_reply(
                "session/prompt",
                response,
                extra_limit_markers=self._profile.extra_limit_markers,
            )
            result = response.get("result", {})
        else:
            result = {}
        await self._maybe_report_model(result)
        stop_reason = result.get("stopReason")
        if stop_reason == "cancelled":
            raise AcpTurnCancelledError("ACP turn cancelled")
        if stop_reason == "refusal":
            raise AcpRefusalError("ACP refused to continue")
        # Token-stats: two independent observations arrive at the turn
        # boundary and BOTH matter to the UI:
        #   * ``_meta`` on the session/prompt reply — per-turn token
        #     usage (inputTokens/outputTokens/cachedReadTokens/totalTokens).
        #     Not comparable with ``-l`` — surfaced as the UI "Last turn" row.
        #   * ``state.usage_snapshot`` from the last ``usage_update`` — the
        #     CLI's live context gauge (used/size). Comparable with ``-l`` —
        #     surfaced as the UI "Context" field.
        # Previously the two paths were mutually exclusive: if _meta had
        # totalTokens we emitted a tokens-only record and DROPPED the
        # snapshot; ``latest_context_by_kind`` then fell back to
        # ``total_tokens`` and the UI painted a per-turn number in the
        # Context field (observed: "2265.4k vs -l 400k").
        # Emit both together in ONE hybrid record and reset the accumulator.
        # Stamp the live context gauge on the final AIMessage BEFORE the
        # turn-boundary reset below clears ``usage_snapshot``.
        final_message = AIMessage(content="".join(text_acc))
        if state is not None:
            self._stamp_context_observation(
                state, final_message, meta=result.get("_meta")
            )
        if state is not None:
            meta = result.get("_meta") or {}
            meta_usage: dict[str, Any] | None = None
            inp = meta.get("inputTokens")
            out = meta.get("outputTokens")
            if isinstance(inp, (int, float)) or isinstance(out, (int, float)):
                meta_usage = {
                    "input_tokens": int(inp or 0),
                    "output_tokens": int(out or 0),
                    "cached_input_tokens": (
                        int(meta["cachedReadTokens"])
                        if isinstance(meta.get("cachedReadTokens"), (int, float))
                        else None
                    ),
                    "total_tokens": (
                        int(meta["totalTokens"])
                        if isinstance(meta.get("totalTokens"), (int, float))
                        else None
                    ),
                }
            # Hybrid emit: when we have BOTH a snapshot (usage_update
            # gauge) AND per-turn usage totals (_meta.*Tokens), fire them
            # together so the recorder persists ONE hybrid row — the UI's
            # ``Context`` field then reads the honest snapshot (comparable
            # to ``-l``) while ``total_tokens`` stays available for the
            # ``Last turn`` UI row.  Previously the two paths were
            # mutually exclusive and the snapshot lost to totalTokens,
            # so the UI showed the per-turn total (e.g. 2265.4k tokens)
            # as the current Context even when ``-l 400k`` was set —
            # (observed in a live session).
            snapshot = getattr(state, "usage_snapshot", None)
            snap_used = (
                snapshot.get("used") if isinstance(snapshot, dict) else None
            )
            snap_size = (
                snapshot.get("size") if isinstance(snapshot, dict) else None
            )
            if meta_usage is not None or snapshot is not None:
                await self._emit_usage_event(
                    event="terminal",
                    status="ok",
                    usage=meta_usage,
                    session_id=getattr(state, "session_id", None),
                    context_used=snap_used,
                    context_size=snap_size,
                )
                state.usage_snapshot = None
        return final_message

    # ------------------------------------------------------------------
    # Live context gauge (per model call)
    # ------------------------------------------------------------------
    async def _start_live_context_watch(self, state: Any) -> None:
        """Follow the CLI's own telemetry so the context tracks every model
        call, not just the turn boundary.

        No-op when the profile declares no source (the CLI reports on the
        wire), when the session id is unknown, or when anything fails: a
        gauge must never affect the turn.
        """
        try:
            source = getattr(self._profile, "live_context_source", None)
            if not source:
                return
            session_id = getattr(state, "session_id", None)
            if not session_id:
                return
            if getattr(state, "live_context_watcher", None) is not None:
                return
            from deep_worker_cli.context_limit.live_gauge import (  # noqa: PLC0415
                LiveContextWatcher,
            )

            from .context_watch import build_live_source  # noqa: PLC0415

            built = build_live_source(str(source), str(session_id))
            if built is None:
                return
            path, parse = built

            watcher = LiveContextWatcher(
                path=path,
                parse=parse,
                on_context=self._live_context_callback(state, str(session_id)),
            )
            await watcher.start()
            state.live_context_watcher = watcher
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort telemetry
            logger.debug("acp_provider: live context gauge unavailable", exc_info=True)

    def _live_context_callback(self, state: Any, session_id: str) -> Any:
        """Build the per-reading callback the live watcher drives.

        Named rather than inline because it carries a contract worth
        testing: every model call must move BOTH gauges.

        * ``live_context_used`` is what the outgoing message is stamped
          with, i.e. what a later turn can anchor to;
        * ``last_context_used`` is what ``client_state`` persists and what
          BOTH the summarizer's ``token_counter`` floor and the context
          guard read. The summarizer runs ``abefore_model`` — before every
          model call inside the turn — so leaving this one to the end of
          the turn makes both of them decide on a number that is a whole
          turn old. On grok a turn is up to 98 model calls.

        Persisting to disk stays at the turn boundary
        (``_stamp_context_observation``): the readers above live in this
        process and read memory; the file only has to survive a restart.
        """

        async def _on_context(used: int, _total: int | None) -> None:
            state.live_context_used = used
            state.last_context_used = used
            await self._emit_context_snapshot(used, session_id=session_id)

        return _on_context

    async def _flush_live_context(self, state: Any, message: Any) -> Any:
        """Drain the gauge and stamp ``message`` with the freshest reading.

        Ordering matters and cost a live miss: the watcher's own flush runs
        in the turn's ``finally``, i.e. AFTER the message has been built and
        stamped, so every turn published the PREVIOUS turn's number and a
        short turn published none at all. Draining here — before the
        message leaves the provider — makes the stamp carry the last model
        call of THIS turn.

        Best-effort: a gauge must never affect the turn.
        """
        watcher = getattr(state, "live_context_watcher", None)
        if watcher is not None:
            try:
                await watcher._drain()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.debug("acp_provider: live gauge flush failed", exc_info=True)
        used = getattr(state, "live_context_used", None)
        if not (isinstance(used, int) and not isinstance(used, bool) and used > 0):
            return message
        # Do not overwrite a stamp that already reports this same reading.
        # ``_finalize_response`` stamps from ``usage_snapshot``, which also
        # carries the context WINDOW; by the time we get here that snapshot
        # has been consumed, so re-stamping the identical number would keep
        # ``used`` and silently drop ``size`` to None.
        from deep_worker_cli.context_limit.observed_context import (  # noqa: PLC0415
            context_observation,
        )

        existing = context_observation(message)
        if isinstance(existing, dict) and existing.get("used") == used:
            return message
        self._stamp_context_observation(state, message)
        return message

    async def _stop_live_context_watch(self, state: Any) -> None:
        """Release the gauge and its inotify handle. Idempotent."""
        watcher = getattr(state, "live_context_watcher", None)
        if watcher is None:
            return
        state.live_context_watcher = None
        try:
            await watcher.stop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("acp_provider: live gauge stop failed", exc_info=True)

    async def _emit_context_snapshot(self, used: int, *, session_id: str) -> None:
        """Record one mid-turn context reading.

        The record carries no ``turn_id``: records are de-duplicated per
        turn, and reusing the id would swallow every reading after the
        first — the frozen gauge this exists to fix.
        """
        await self._emit_usage_event(
            event="terminal",
            status="ok",
            usage=None,
            session_id=session_id,
            context_used=used,
        )

    # ------------------------------------------------------------------
    # Mid-turn pump
    # ------------------------------------------------------------------
    def _start_pump(self, state: Any, agent_thread_id: str) -> None:
        """Spawn the background drain task. Idempotent — a second call while
        the previous pump is still alive is a programming bug; we cancel the
        old task to keep the queue single-consumer."""
        self._cancel_pump(state)
        state.pump_task = asyncio.create_task(
            self._drain_pump(state, agent_thread_id),
            name=f"acp-pump-{agent_thread_id}",
        )

    def _cancel_pump(self, state: Any) -> None:
        """Cancel any live pump task. Safe to call multiple times / when no
        pump is running."""
        task = state.pump_task
        if task is None:
            return
        state.pump_task = None
        if not task.done():
            task.cancel()

    async def _adopt_pump(self, state: Any) -> None:
        """Called at the start of a mid-turn re-entry. Cancels the pump and
        waits for it to actually finish so the accumulators
        (``queued_text``/``queued_tool_calls``/``queued_response``) reach a
        stable state before ``_consume_turn`` reads them. The cancellation
        is synchronous from the pump's point of view — it never observes a
        partial update."""
        task = state.pump_task
        if task is None:
            return
        state.pump_task = None
        if task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _drain_pump(self, state: Any, agent_thread_id: str) -> None:
        """Background drain task spawned at the mid-turn pause.

        While LangGraph's ToolNode is running the channel-B passthrough body,
        SOMEONE has to keep reading ``state.current_notif_queue`` — otherwise
        the reader pushes ``tool_call_update status=completed`` events that
        nobody observes, the passthrough Future never resolves, and the
        ToolNode body hangs forever.

        The pump:
          - resolves built-in Futures by calling ``_apply_update`` (which
            internally calls ``_resolve_builtin_update`` on
            ``tool_call_update``);
          - accumulates ``agent_message_chunk`` text into ``state.queued_text``
            (BOUNDED — see helpers);
          - queues any fresh ``tool_call`` events into
            ``state.queued_tool_calls`` so the next ainvoke surfaces them;
          - stashes the response future's reply in ``state.queued_response``
            if the CLI ends the turn while ToolNode is still running.

        Exits cleanly on cancellation (the next ainvoke calls ``_adopt_pump``)
        or when ``pending_session_prompt_future`` resolves. Never raises out —
        any exception during update handling is logged and the pump keeps
        running so a single malformed event can't kill the turn.
        """
        notif_q: asyncio.Queue = state.current_notif_queue
        response_fut: asyncio.Future = state.pending_session_prompt_future
        if notif_q is None or response_fut is None:
            return
        get_task: asyncio.Task | None = None
        try:
            while True:
                get_task = asyncio.ensure_future(notif_q.get())
                done, _pending = await asyncio.wait(
                    {get_task, response_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=_prompt_idle_timeout(),
                )
                if not done:
                    # IDLE TIMEOUT (round 8, defect #4) — pump side.
                    #
                    # The pump runs in the BACKGROUND while ToolNode executes,
                    # so there is no caller to raise at: failing the response
                    # future is what surfaces the problem, because the next
                    # _consume_turn awaits exactly that future. Without this,
                    # a channel that went silent mid-tool-run left the pump
                    # spinning on a dead queue and the following turn waiting
                    # on a future nobody would ever resolve.
                    get_task.cancel()
                    logger.error(
                        "acp pump: no session/update and no response for "
                        "%.0fs on session %s — failing the turn (tune with %s)",
                        _prompt_idle_timeout(),
                        state.session_id,
                        PROMPT_IDLE_TIMEOUT_ENV,
                    )
                    if not response_fut.done():
                        response_fut.set_exception(
                            AcpPromptIdleTimeoutError(
                                f"ACP turn produced no output for "
                                f"{_prompt_idle_timeout():.0f}s (session "
                                f"{state.session_id}) while tools were "
                                f"running. Tune with "
                                f"{PROMPT_IDLE_TIMEOUT_ENV}."
                            )
                        )
                    return
                if response_fut in done:
                    # Same drain-before-finalize discipline as _consume_turn:
                    # absorb the in-flight get_task plus any remaining queued
                    # updates so trailing agent_message_chunk text is not lost
                    # when the CLI resolves the response in the same loop tick.
                    if get_task in done and not get_task.cancelled():
                        with contextlib.suppress(Exception):
                            self._apply_update(
                                state, agent_thread_id, get_task.result(),
                                state.queued_text, state.queued_tool_calls,
                            )
                    else:
                        get_task.cancel()
                    # Same end-of-stream test as _consume_turn: queue empty AND
                    # the reader holding nothing. A queue-only drain here would
                    # park the tail on the state short of its last frames.
                    with contextlib.suppress(Exception):
                        await self._drain_until_stream_exhausted(
                            state, agent_thread_id, notif_q,
                            state.queued_text, state.queued_tool_calls,
                        )
                    await self._publish_pending_context(state)
                    with contextlib.suppress(Exception):
                        state.queued_response = response_fut.result()
                    return
                notif = get_task.result()
                try:
                    self._apply_update(
                        state,
                        agent_thread_id,
                        notif,
                        state.queued_text,
                        state.queued_tool_calls,
                    )
                    await self._publish_pending_context(state)
                except Exception:  # noqa: BLE001 — pump must survive one bad event
                    logger.exception("acp pump: failed to apply update")
        except asyncio.CancelledError:
            raise
        finally:
            # A leftover get_task (created this iteration, then cancelled or
            # returned early) would stay registered on the queue as a waiting
            # getter and STEAL the next put_nowait — the data would vanish
            # into a dead task. Cancel it so the queue delivers to the next
            # real reader (_consume_turn after _adopt_pump).
            if get_task is not None and not get_task.done():
                with contextlib.suppress(Exception):
                    get_task.cancel()

    # ------------------------------------------------------------------
    # tool_call mapping
    # ------------------------------------------------------------------
    def _bound_tool_names(self) -> set[str]:
        return {t["name"] for t in self._tools_metadata}

    def _input_schema_for(self, tool_name: str) -> dict[str, Any] | None:
        """JSON Schema of a bound deep-worker tool, or None if not bound."""
        for meta in self._tools_metadata:
            if meta.get("name") == tool_name:
                schema = meta.get("input_schema")
                return schema if isinstance(schema, dict) else None
        return None

    def _tool_name_from(self, upd: dict) -> str | None:
        """Extract the CLI's real tool name per the profile's extraction style."""
        return acp_types.tool_name_from_update(
            upd, self._profile.tool_name_from_update
        )

    def _system_block_for_prompt(
        self,
        state: Any,
        *,
        is_first_prompt_in_session: bool,
        system_prompt: str | None,
    ) -> str | None:
        """Text for this prompt's leading ``<system>`` block, or ``None``.

        Two triggers:

        * first prompt of a NEW session — full system prompt + tool
          contract (the pre-existing behaviour);
        * first prompt of THIS PROCESS on a RESUMED session — tool contract
          ONLY. The proxy MCP server name embeds the PID, so the contract a
          previous process delivered names DEAD mangled tools; without
          re-delivery the model keeps calling stale names forever
          (a related internal incident analysis, not included here).
        """
        parts: list[str] = []
        if is_first_prompt_in_session and system_prompt:
            parts.append(system_prompt)
        need_contract = is_first_prompt_in_session or not getattr(
            state, "tool_contract_sent_in_process", True
        )
        if need_contract and self._tools_metadata:
            contract = _build_tool_contract(
                [tool["name"] for tool in self._tools_metadata],
                self._profile,
                server_name=getattr(self._proxy_mcp, "server_name", None),
            )
            if contract:
                parts.append(contract)
                state.tool_contract_sent_in_process = True
        if not parts:
            return None
        return "\n\n".join(parts)

    def _proxy_tool_from_update(self, upd: dict) -> tuple[str, dict] | None:
        """Return (name, args) when this update is a channel-A proxy tool."""
        if self._proxy_mcp is None:
            return None
        raw_input = upd.get("rawInput") or {}
        style = self._profile.mcp_dispatch_style
        if acp_types.is_proxy_mcp_dispatch(
            upd, self._proxy_mcp.server_name, style
        ):
            if style == "mcp_name_prefix":
                # claude-agent-acp: name carries mcp__<server>__<tool>; args
                # are the rawInput directly (no dispatch wrapper).
                name = self._tool_name_from(upd) or upd.get("title")
                inner = acp_types.mcp_prefixed_tool_name(
                    name, self._proxy_mcp.server_name
                )
                if inner:
                    return inner, raw_input
                return None
            # dispatch_builtin (grok): both wrapper forms —
            # CallMcpTool{server,toolName,arguments} (≤1.0.4) and
            # use_tool{tool_name,tool_input} (1.0.5+) — are unwrapped by the
            # shared extractor (see a related internal incident analysis, not included here).
            return acp_types.proxy_dispatch_inner(
                upd, self._proxy_mcp.server_name
            )
        if style == "dispatch_builtin":
            # For dispatch-builtin CLIs a plain-title tool_call is ALWAYS a
            # local built-in: MCP dispatch travels through the wrapper.
            # grok 1.0.5 named its natives after our tools (read_file, grep,
            # …), so the plain-name fallback below would route the CLI's
            # LOCAL call into OUR bound tool — a false 10s MCP-handler
            # correlation per call and double execution with the CLI's
            # arguments (see a related internal incident analysis, not included here).
            return None
        name = self._tool_name_from(upd)
        if name and name in self._bound_tool_names():
            return name, raw_input
        return None

    @staticmethod
    def _discard_pending_prompt(state: Any, why: str) -> None:
        """Drop an in-flight ``session/prompt`` future that no longer applies.

        Used when the session it belongs to is being abandoned (currently:
        summarization). Leaving it set is what lets step 9 mistake a fresh
        turn for a mid-turn resume and swallow the user's message.

        Cancels rather than resolves: nobody is entitled to a result from a
        session that is going away. Safe on ``None``/already-done.
        """
        fut = getattr(state, "pending_session_prompt_future", None)
        if fut is None:
            return
        state.pending_session_prompt_future = None
        if not fut.done():
            fut.cancel()
            logger.info(
                "acp_provider: cancelled in-flight prompt future (%s)", why
            )

    def resolve_pending_tool(
        self, thread_id: str, tool_call_id: str, content: Any
    ) -> None:
        """Resolve a pending proxy-MCP tool Future for ``thread_id`` NOW.

        ROUND 8 DEFECT #2. The normal path resolves a tool's Future in the
        send loop of the NEXT ``ainvoke`` (the ``ToolMessage`` branch). For a
        TURN-ENDING tool like ``send_message_to_agent`` that next ainvoke
        never happens in the same graph run — ``ToolCompletionMiddleware``
        jumps to the graph end first — so the MCP handler stayed parked and
        the CLI was handed a FALSE "still running… job_N" notice 120s later.
        The model then tried to chase the phantom job, parking more calls,
        and the turn only unwedged on the next external impulse (observed:
        15:41 → 18:28).

        ``claude-code-sdk`` already solved this with
        ``InterAgentDeliveryResolveMiddleware``, which is duck-typed on this
        exact method — it was simply never registered for ACP models because
        ``AcpChatModel`` did not provide it. Providing it wires ACP into the
        existing, proven seam instead of inventing a second mechanism.

        Best-effort and idempotent: unknown thread, unknown id, or an
        already-resolved Future are all safe no-ops (``resolve_call`` guards
        on presence and ``future.done()``). Never raises into the caller.
        """
        try:
            if self._proxy_mcp is None:
                return
            state = self._client_state.get(thread_id)
            # The ACP id the CLI knows is mapped to the handler's synthetic
            # id at correlation time; without the hop resolve_call would be a
            # no-op and we would be back to the 120s false timeout.
            proxy_call_id = tool_call_id
            if state is not None:
                proxy_call_id = state.proxy_call_id_by_acp_id.pop(
                    tool_call_id, tool_call_id
                )
            self._proxy_mcp.resolve_call(proxy_call_id, content)
        except Exception:  # noqa: BLE001 — must never break a tool turn
            logger.debug(
                "acp_provider: resolve_pending_tool skipped for %r",
                tool_call_id,
                exc_info=True,
            )

    async def _correlate_proxy_tool_calls(
        self, state: Any, tool_calls: list[dict]
    ) -> None:
        """Map ACP toolCallIds to proxy handler call_ids before LangGraph runs.

        The MCP handler parks under a synthetic id; LangGraph later resolves
        via ToolMessage.tool_call_id (the ACP id). Without this map,
        resolve_call is a no-op and the CLI hangs until the job-handle
        timeout. BOUNDED: the correlation map is capacity-capped; overflow is
        rejected (reject-new, DESIGN §4.5) so a runaway stream cannot grow
        the map unboundedly.

        ALSO adopts the handler's arguments as authoritative — see
        :meth:`_adopt_authoritative_args`. The ACP ``rawInput`` we mapped from
        is a progressive snapshot and can be missing OPTIONAL arguments; the
        MCP ``tools/call`` payload is complete by construction.
        """
        if self._proxy_mcp is None:
            return
        for tc in tool_calls:
            tcid = tc.get("id")
            name = tc.get("name")
            args = tc.get("args") or {}
            if not isinstance(tcid, str) or not tcid or not name:
                continue
            if tcid in state.proxy_call_id_by_acp_id:
                continue
            if name not in self._bound_tool_names():
                continue
            if len(state.proxy_call_id_by_acp_id) >= CALL_ID_MAP_MAX_ENTRIES:
                logger.error(
                    "acp_provider: proxy_call_id_by_acp_id at capacity "
                    "(%d) — rejecting new correlation for %r (reject-new)",
                    CALL_ID_MAP_MAX_ENTRIES,
                    name,
                )
                continue
            # ADAPTIVE BUDGET (measured 2026-08-24, grok 1.0.5 live): some
            # CLIs LIST our tools over MCP but DISPATCH them on the ACP
            # tool_call channel and return the result the same way — our
            # proxy then serves ZERO ``tools/call`` for the whole session and
            # there is never a handler to correlate. Waiting the full budget
            # for one cost a flat ~10s on EVERY deep-worker tool call (the
            # tool still ran via the ToolNode, so it was pure dead latency).
            # A CLI that does use the MCP execution path parks the handler
            # essentially simultaneously with the ACP notification, so the
            # short probe is ample until the first call is actually served.
            served = getattr(self._proxy_mcp, "served_tool_call", True)
            budget = (
                PROXY_CORRELATION_TIMEOUT_SEC
                if served
                else PROXY_CORRELATION_PROBE_SEC
            )
            try:
                # Match by tool name only — ACP tool_call args and the MCP
                # tools/call payload can differ (key order, coercion).
                pending = await self._proxy_mcp.wait_for_pending_calls(
                    tool_name=name,
                    arguments=None,
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                if served:
                    logger.warning(
                        "acp_provider: no pending MCP handler for %r "
                        "(toolCallId=%s) within %.1fs",
                        name,
                        tcid,
                        budget,
                    )
                else:
                    # Expected traffic on an ACP-executing CLI — warning on it
                    # buried real problems under 69 warnings in three turns.
                    logger.debug(
                        "acp_provider: %r dispatched over ACP, no MCP handler "
                        "to correlate (toolCallId=%s)",
                        name,
                        tcid,
                    )
                continue
            if pending:
                state.proxy_call_id_by_acp_id[tcid] = pending[0]["id"]
                self._adopt_authoritative_args(tc, pending[0])

    @staticmethod
    def _adopt_authoritative_args(tc: dict, pending: dict) -> None:
        """Replace a tool_call's args with the MCP handler's payload.

        ROUND 8 DEFECT. ``rawInput`` streams PROGRESSIVELY across refine
        updates, and :meth:`_map_tool_call` releases a held call as soon as
        the REQUIRED arguments are present. Every OPTIONAL argument that had
        not streamed by that moment was silently dropped — live-reproduced as
        ``read_file(file_path=..., offset=100, limit=3)`` returning lines
        1-100 instead of 101-103, with the outcome depending on nothing but
        the key ORDER the model happened to emit. The failure is silent:
        the tool runs, just with fewer arguments than the model asked for.

        The MCP ``tools/call`` payload does NOT stream — the CLI sends it once
        the arguments are complete — so the parked handler is the authoritative
        source. Adopt it whenever it is a strict superset of what we mapped.

        Deliberately conservative: adopt ONLY when every key we already hold
        is present with an equal value in the handler payload. If the two
        disagree (different values for a shared key, or the handler is missing
        one), we are not looking at the same logical call — correlation is by
        tool NAME only, so a second concurrent call to the same tool could
        have parked first. In that case keep the mapped args rather than
        splice two different calls together.
        """
        handler_args = pending.get("args")
        if not isinstance(handler_args, dict):
            return
        mapped = tc.get("args") or {}
        if not isinstance(mapped, dict):
            return
        if handler_args == mapped:
            return
        for key, value in mapped.items():
            if key not in handler_args or handler_args[key] != value:
                logger.debug(
                    "acp_provider: not adopting handler args for %r — "
                    "mapped key %r disagrees (concurrent call to the same "
                    "tool?)",
                    tc.get("name"),
                    key,
                )
                return
        added = sorted(set(handler_args) - set(mapped))
        tc["args"] = dict(handler_args)
        if added:
            logger.info(
                "acp_provider: adopted authoritative MCP args for %r — "
                "recovered %d argument(s) lost to progressive rawInput: %s",
                tc.get("name"),
                len(added),
                ", ".join(added),
            )

    def _map_tool_call(self, state: Any, upd: dict) -> dict | None:
        """Map a CLI ``tool_call`` update to a LangChain tool_call dict.

        CASE A (proxy MCP): dispatch built-in OR the CLI calling a bound
        deep-worker tool by plain name. Surface the inner/plain name + args.
        ``_correlate_proxy_tool_calls`` maps ACP ids to handler ids before
        LangGraph runs; ToolMessage resolution unblocks the MCP handler.

        CASE B (built-in): register a passthrough Future on the per-thread
        state keyed by toolCallId and surface the built-in name + rawInput.
        """
        tcid = upd.get("toolCallId")
        if not tcid:
            return None
        title = upd.get("title")
        raw_input = upd.get("rawInput") or {}

        # claude-agent-acp streams args: the first tool_call carries
        # rawInput={} (pending); the real arguments arrive in a follow-up
        # tool_call_update without status (refine). Never map a pending
        # call to empty args — that would run the tool with no arguments.
        if (
            self._profile.mcp_dispatch_style == "mcp_name_prefix"
            and not raw_input
            and upd.get("status") == "pending"
        ):
            return None

        proxy_tool = self._proxy_tool_from_update(upd)
        if proxy_tool is not None:
            name, args = proxy_tool
            if name:
                # Narrow arg-compat: rewrite Cursor-style keys / coerce
                # human-readable values into the shape our tool schemas expect.
                # CASE A only — built-in passthrough (CASE B) is verbatim.
                args = normalize_tool_args(name, args)
                # rawInput streams progressively (see _required_args_present):
                # an intermediate refine can carry SOME required args but not
                # all. Dispatching then invokes the tool with a half-streamed
                # payload — it fails validation and the CLI waits forever for
                # a result. Hold the call until the schema is satisfied; the
                # next refine (or the turn end) decides its fate.
                if not _required_args_present(
                    self._input_schema_for(name), args
                ):
                    return None
                return {
                    "id": tcid,
                    "name": name,
                    "args": args,
                    "type": "tool_call",
                }

        # Built-in (CASE B). Register the passthrough Future for the reader.
        if tcid not in state.builtin_passthrough_futures:
            if len(state.builtin_passthrough_futures) >= CALL_ID_MAP_MAX_ENTRIES:
                logger.error(
                    "acp_provider: builtin_passthrough_futures at capacity "
                    "(%d) — rejecting new built-in call %r (reject-new)",
                    CALL_ID_MAP_MAX_ENTRIES,
                    title,
                )
                return None
            state.builtin_passthrough_futures[tcid] = (
                asyncio.get_running_loop().create_future()
            )
        real_name = self._tool_name_from(upd) or title
        if real_name in self._profile.known_builtins:
            surfaced_name = real_name
            surfaced_args = raw_input
        elif real_name in self._profile.colliding_builtins:
            # A CLI-native built-in whose name is IDENTICAL to a deep-worker
            # tool (grok 1.0.5: read_file, grep, …). Recorded via the
            # GENERIC recorder — surfacing it under its own name would hit
            # OUR same-named tool in the ToolNode. Expected traffic: no
            # "unknown built-in" warning (see a related internal incident analysis, not included here).
            surfaced_name = self._generic_builtin_name()
            surfaced_args = {
                **raw_input,
                self._profile.original_title_arg_key: real_name,
            }
        elif real_name:
            server_name = getattr(self._proxy_mcp, "server_name", None)
            if server_name and server_name in repr(raw_input):
                # The built-in is unknown, yet its payload references OUR
                # proxy MCP server: this is how the NEXT dispatch-wrapper
                # rename will look (CallMcpTool -> use_tool happened
                # silently on 1.0.5). Contract drift must be loud.
                logger.error(
                    "acp_provider: unknown built-in %r references our proxy "
                    "MCP server %s — probable MCP dispatch contract drift; "
                    "the wrapped deep-worker tool call is NOT being "
                    "executed. Update acp_types.proxy_dispatch_inner for "
                    "the new wrapper shape "
                    "(a related internal incident analysis, not included here).",
                    real_name,
                    server_name,
                )
            else:
                logger.warning(
                    "acp_provider: unknown built-in %r — routing via %s",
                    real_name,
                    self._generic_builtin_name(),
                )
            surfaced_name = self._generic_builtin_name()
            surfaced_args = {
                **raw_input,
                self._profile.original_title_arg_key: real_name,
            }
        else:
            surfaced_name = self._generic_builtin_name()
            surfaced_args = raw_input
        return {
            "id": tcid,
            "name": surfaced_name,
            "args": surfaced_args,
            "type": "tool_call",
        }

    def _generic_builtin_name(self) -> str:
        return (
            self._profile.generic_builtin_name
            or f"{self._profile.provider}Builtin".replace("-", "_").replace("_", "")
        )

    def _resolve_builtin_update(self, agent_thread_id: str, upd: dict) -> None:
        """On a built-in ``tool_call_update`` terminal status, resolve the
        passthrough Future so the ToolNode body unblocks."""
        tcid = upd.get("toolCallId")
        if not tcid:
            return
        status = upd.get("status")
        if status not in ("completed", "failed", "cancelled"):
            return
        payload = _render_completed_payload(upd, status)
        resolve_builtin_future(self._client_state, agent_thread_id, tcid, payload)

    async def _maybe_report_model(self, result: dict) -> None:
        meta = result.get("_meta") if isinstance(result, dict) else None
        if isinstance(meta, dict):
            model_id = meta.get("modelId")
            if isinstance(model_id, str):
                await self._record_resolved_model(model_id)


# ``__pydantic_init_subclass__`` only fires for subclasses, so clear the
# shadow on the base class itself here.
_unshadow_profile_property(AcpChatModel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_mcp_content(content: Any) -> list[dict[str, Any]]:
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
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n\n".join(parts)


_RUNTIME_SYSTEM_PREFIX = "System:> "


def _collect_runtime_system_chunks(
    messages: list[BaseMessage],
    new_messages: list[BaseMessage],
    *,
    is_first_prompt_in_session: bool,
    forwarded_first_system_chars: int,
    forwarded_extra_system_keys: set[str],
) -> tuple[list[str], int, set[str]]:
    """Collect ``System:>`` lines for runtime system content not yet sent.

    ACP CLIs accept a ``<system>`` block only on the first ``session/prompt``
    of a session. Later middleware cognitive hints (InterAgentReply,
    ProjectMemory, …) MUST ride on the same user prompt as the HumanMessage
    they refer to — mirroring ``claude_code_sdk_provider/chat_model.py``.

    Two paths:

    1. **new_messages scan** — any SystemMessage in the send delta *after* the
       conversation's first SystemMessage becomes a ``System:>`` line.
    2. **First-system append delta** — when middleware appends to the existing
       first SystemMessage, only the suffix since ``forwarded_first_system_chars``
       is forwarded.

    Returns ``(chunks, new_forwarded_first_system_chars,
    new_forwarded_extra_system_keys)``.
    """
    chunks: list[str] = []
    new_first_chars = forwarded_first_system_chars
    new_extra_keys = set(forwarded_extra_system_keys)

    # Path 1 — scan the send delta only.
    first_system_seen_before = any(
        isinstance(m, SystemMessage) for m in messages if m not in new_messages
    )
    first_system_consumed = first_system_seen_before
    for msg in new_messages:
        if isinstance(msg, SystemMessage):
            if not first_system_consumed:
                first_system_consumed = True
            else:
                text = _flatten_message_content_to_text(msg.content)
                key = getattr(msg, "id", None) or f"extra:{id(msg)}:{text[:80]}"
                if key not in new_extra_keys and text.strip():
                    # BOUNDED dedup set (reject-new): past the cap the
                    # SystemMessage text still rides the prompt, only the key
                    # stops being deduplicated (worst case: a duplicate
                    # System:> line on a later turn — visible, not fatal).
                    if len(new_extra_keys) >= FORWARDED_EXTRA_KEYS_MAX:
                        logger.warning(
                            "acp_provider: forwarded_extra_system_keys at "
                            "capacity (%d) — not deduplicating key %r",
                            FORWARDED_EXTRA_KEYS_MAX,
                            key,
                        )
                    else:
                        chunks.append(f"{_RUNTIME_SYSTEM_PREFIX}{text}")
                        new_extra_keys.add(key)

    # Path 2 — suffix appended to the first SystemMessage (not in new_messages).
    first_sys_index: int | None = None
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            first_sys_index = i
            break

    if first_sys_index is not None:
        first_text = _flatten_message_content_to_text(
            messages[first_sys_index].content
        )
        if is_first_prompt_in_session:
            new_first_chars = len(first_text)
        elif len(first_text) > forwarded_first_system_chars:
            delta = first_text[forwarded_first_system_chars:].strip()
            if delta:
                prefixed = f"{_RUNTIME_SYSTEM_PREFIX}{delta}"
                if prefixed not in chunks:
                    chunks.append(prefixed)
            new_first_chars = len(first_text)

    return chunks, new_first_chars, new_extra_keys


def _render_completed_payload(upd: dict, status: str) -> dict[str, Any]:
    """Render a terminal built-in ``tool_call_update`` into the passthrough
    payload ``{"text", "is_error", "metadata"}``.

    CLIs put result text in ``content`` (a list of content blocks) and
    structured metadata (exit_code, current_dir, duration, output file) in
    ``rawOutput``.
    """
    text_parts: list[str] = []
    for block in upd.get("content") or []:
        if isinstance(block, dict):
            inner = block.get("content")
            if isinstance(inner, dict) and "text" in inner:
                text_parts.append(str(inner.get("text", "")))
            elif "text" in block:
                text_parts.append(str(block.get("text", "")))
    raw_output = upd.get("rawOutput")
    metadata = raw_output if isinstance(raw_output, dict) else {}
    acp_meta = dict(metadata)
    artifact_path = next(
        (
            metadata[key]
            for key in ("path", "file_path")
            if isinstance(metadata.get(key), str)
            and Path(metadata[key]).is_file()
        ),
        None,
    )
    if artifact_path is not None:
        mime = metadata.get("mime") or mimetypes.guess_type(artifact_path)[0]
        acp_meta["artifact_path"] = artifact_path
        if mime:
            acp_meta["mime"] = mime
        raw_type = str(metadata.get("type", "")).lower()
        acp_meta["kind"] = (
            "video"
            if (isinstance(mime, str) and mime.startswith("video/"))
            or "video" in raw_type
            else "image"
        )
    return {
        "text": "".join(text_parts) if text_parts else (
            f"Tool {status}" if status != "completed" else ""
        ),
        "is_error": status in ("failed", "cancelled"),
        "metadata": metadata,
        "meta": acp_meta,
    }


def _render_plan_entries(entries: Any) -> str:
    """Render plan entries as the recorder's ToolMessage text.

    The plan is the RESULT — there is nothing else to report — so an empty or
    malformed payload still yields a non-empty line rather than a blank
    ToolMessage that reads like a tool that did nothing.
    """
    if not isinstance(entries, list) or not entries:
        return "Plan updated (no entries)."
    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            content = entry.get("content") or entry.get("title") or ""
            status = entry.get("status") or ""
            text = str(content).strip()
            if not text:
                continue
            lines.append(f"[{status}] {text}" if status else f"- {text}")
        else:
            text = str(entry).strip()
            if text:
                lines.append(f"- {text}")
    if not lines:
        return "Plan updated (no entries)."
    return "Plan updated:\n" + "\n".join(lines)


def _plan_to_tool_call(upd: dict) -> dict | None:
    """Surface a ``plan`` update as an ``update_plan`` tool_call so the plan
    lands in LangGraph history (codex update_plan analog)."""
    tcid = upd.get("toolCallId") or "plan"
    entries = upd.get("entries")
    return {
        "id": tcid,
        "name": PLAN_TOOL_NAME,
        "args": {"entries": entries} if entries is not None else {},
        "type": "tool_call",
    }


class _BoundedTextAcc(list):
    """A ``list[str]`` that knows its own total length in O(1).

    ``_append_bounded_text`` used to re-derive the byte total with
    ``sum(len(c) for c in acc)`` on EVERY notification, making a turn's text
    accumulation quadratic. Measured on the real function with 200-byte chunks:
    the per-notification cost rose from 2.4 us at 0 accumulated chunks to
    300.7 us at 8000 (125x), and that ``sum()`` was 97-101% of the call while
    the ``append`` itself cost 0.15 us. Total cost grew 4.08x for a 2x input.

    That is the burst defect's root cause, not merely a slow path: the reader
    routes a notification for ~3.4 us, so from ~1000 accumulated chunks on, the
    consumer is an order of magnitude slower than the arrival rate and the
    1000-slot sink fills. The same arithmetic is the "a trivial tool call hangs"
    symptom — the drain holds the event loop with no await, so an unrelated
    coroutine measured 1441 ms of latency at 8000 notifications and 3.49 s of
    solid CPU at the 10000-chunk cap.

    Subclassing ``list`` keeps every existing reader working unchanged
    (``"".join(acc)``, ``list(acc)``, indexing, iteration) while the total is
    carried incrementally. ``_total`` is advisory: it is only ever used to
    short-circuit the cap check, and ``total_len`` falls back to a real sum for
    a plain list that was built elsewhere.

    Only ``append`` (via ``_append_bounded_text``) and ``clear`` maintain the
    total. Any OTHER inherited mutator — ``pop``, ``__delitem__``,
    ``extend`` — would leave ``_total`` too high and silently tighten the byte
    cap; measured before ``clear`` was overridden, an emptied buffer still
    reported its old 11 bytes. None of those are used on this type today; add
    an override here rather than a caller-side fixup if that changes.
    """

    __slots__ = ("_total",)

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._total = sum(len(c) for c in self)

    def clear(self) -> None:
        """Empty the buffer AND the carried total (``list.clear`` keeps it)."""
        super().clear()
        self._total = 0


def _text_acc_total(acc: list[str]) -> int:
    """Byte total of ``acc`` — O(1) for ``_BoundedTextAcc``, O(n) otherwise.

    The fallback exists because ``state.queued_text`` can be a plain list built
    by older code paths (or by a checkpoint restore), and a wrong total is worse
    than a slow one: it would silently move the cap.
    """
    total = getattr(acc, "_total", None)
    if total is None:
        return sum(len(c) for c in acc)
    return total


def _append_bounded_text(acc: list[str], chunk: str) -> None:
    """Append one agent_message_chunk to a partial text buffer, honoring the
    bounded-state caps (DESIGN §4.5): chunk count AND total bytes. On either
    overflow the chunk is dropped with a loud error log — the buffer can never
    grow without bound, and the drop is observable (not silent).

    O(1) per call: the byte total is carried by :class:`_BoundedTextAcc` rather
    than re-summed. See that class for the measurements behind this.
    """
    if not chunk:
        return
    if len(acc) >= QUEUED_TEXT_MAX_CHUNKS:
        logger.error(
            "acp_provider: partial text buffer at chunk cap (%d) — "
            "dropping chunk (bounded backpressure)",
            QUEUED_TEXT_MAX_CHUNKS,
        )
        return
    total = _text_acc_total(acc)
    if total + len(chunk) > QUEUED_TEXT_MAX_BYTES:
        logger.error(
            "acp_provider: partial text buffer at byte cap (%d) — "
            "dropping chunk (bounded backpressure)",
            QUEUED_TEXT_MAX_BYTES,
        )
        return
    acc.append(chunk)
    if isinstance(acc, _BoundedTextAcc):
        acc._total += len(chunk)


# Re-export the meta key so callers can attach meta to ToolMessages.
__all__ = ["AcpChatModel", "AcpRefusalError", "META_KEY"]
