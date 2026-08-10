"""GrokChatModel — LangChain BaseChatModel that drives the xAI Grok CLI.

Public surface of the grok provider. Same shape as
``codex_provider/chat_model.py`` / ``claude_code_provider`` — only
``ainvoke`` is real; ``_generate`` raises (async-only); LangChain
streaming methods are not implemented.

The CLI is driven over the Agent Client Protocol (ACP) v1 on
``grok agent stdio`` (see ``acp_client.py``). One long-lived subprocess
per agent process; one ``grok_session_id`` per ``agent_thread_id``; one
``session/prompt`` per turn. Tools flow through two channels, both
recorded in LangGraph history:

  * Channel A — deep-worker MCP tools via the in-agent proxy MCP server
    (``proxy_mcp_server.py``). grok dispatches them through ``CallMcpTool``;
    the handler parks a Future resolved when the ToolMessage returns.
  * Channel B — grok's local built-ins (Shell/Read/Edit/...), observed on
    the ACP stream and recorded via ``builtin_tool_passthrough.py``.

See docs/DESIGN_GROK_PROVIDER.md §8-§11A.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import mimetypes
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

from grok_provider import acp_types
from grok_provider.arg_compat import normalize_tool_args
from grok_provider.builtin_tool_passthrough import (
    GROK_BUILTIN_MARKER,
    GROK_META_KEY,
    resolve_builtin_future,
)
from grok_provider.client_state import get_global_client_state
from grok_provider.custom_types import ToolMetadata
from grok_provider.proxy_mcp_server import get_global_proxy_mcp_server

logger = logging.getLogger(__name__)


class GrokRefusalError(RuntimeError):
    """Raised when a grok turn ends with stopReason == "refusal"."""


class GrokTurnCancelledError(RuntimeError):
    """Raised for an ACP cancelled stop reason, not local task cancellation."""


# Equivalence map: grok built-in -> preferred deep-worker MCP tool. Used to
# build the per-turn tool contract, but ONLY for entries whose deep-worker
# tool is actually bound this turn (so we never advertise a tool grok can't
# see — that mismatch made grok 0.2.54 go hunting for the schema on disk).
_BUILTIN_EQUIVALENTS = [
    ("Shell", "execute"),
    ("Read", "read_file"),
    ("Write", "write_file"),
    ("Edit", "edit_file"),
    ("StrReplace", "edit_file"),
    ("Glob", "glob"),
    ("Grep", "grep"),
    ("WebSearch", "fetch_url"),
    ("Task", "task"),
]


def _build_tool_contract(tool_names: list[str]) -> str:
    """Build the <system>-block tool guidance for the tools ACTUALLY bound
    this turn.

    grok 0.2.54 does NOT expose a string ``mcp__deepworker_*__`` prefix to the
    model: the model calls a deep-worker tool by its plain name and the grok
    CLI itself routes the call to our MCP server
    (``deep_worker_proxy_<pid>``) via its ``CallMcpTool`` built-in. IMPORTANT:
    this does NOT make them grok's NATIVE built-ins (Shell/Read/Write/…) — the
    deep-worker tools are MCP tools, reachable only through that MCP channel.
    (See a related internal incident analysis, not included here, and
    docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md §3.3/§8.)

    The previous static contract claimed a literal prefix AND listed
    equivalents (Shell->execute, …) regardless of whether those tools were
    bound. When the user bound a tool whose name didn't match the advertised
    prefix/list (e.g. a plain ``get_magic_word``), grok failed to recognise it
    as "one of the contract tools" and went introspecting the on-disk MCP
    schema cache (Glob/Grep/Read over ``~/.grok/.../mcps/.../tools/*.json``)
    instead of just calling it — burning turns and never converging.

    The contract is now DYNAMIC and TRUTHFUL: it names exactly the bound
    deep-worker tools, tells grok they are MCP tools callable directly by name
    (CLI routes them to our MCP server — they are NOT native built-ins), and
    explicitly forbids hunting for their schema on disk. Returns an empty
    string when no deep-worker tools are bound (nothing to advertise)."""
    if not tool_names:
        return ""
    bound = set(tool_names)
    lines = [
        "Deep-worker tool contract:",
        "- You have these deep-worker MCP tools available THIS TURN, callable "
        "directly by name: " + ", ".join(f"`{n}`" for n in tool_names) + ".",
        "- These are deep-worker MCP tools, NOT your native built-ins "
        "(Shell/Read/Write/Edit/Glob/Grep/WebSearch/Task). Call them directly "
        "by name — the CLI routes the call to the deep-worker MCP server "
        "automatically. Do NOT search the filesystem for their schema or "
        "definition (do not Glob/Grep/Read the `mcps/` directory) — you "
        "already have everything needed to call them.",
    ]
    equiv = [
        (b, d) for (b, d) in _BUILTIN_EQUIVALENTS if d in bound
    ]
    if equiv:
        lines.append(
            "- Prefer these over the equivalent Grok built-in (they route "
            "through deep-worker's logging, retry and HITL middleware):"
        )
        lines.extend(f"  - {b} -> `{d}`" for (b, d) in equiv)
    lines.append(
        "- Use Grok built-ins only for operations with no deep-worker "
        "equivalent. Both channels are recorded in history; this is a "
        "preference, not a hard restriction."
    )
    imagine_names = ", ".join(
        f"`{n}`" for n in sorted(acp_types.IMAGINE_GROK_BUILTINS)
    )
    lines.extend([
        "",
        "Imagine (subscription, Grok built-ins):",
        f"- These Grok built-ins are available when you need visuals: {imagine_names}.",
        "- Call them directly. Do NOT search the filesystem for their schema.",
        "- For charts/diagrams with exact numbers, prefer code (HTML/SVG), "
        "not image generation built-ins.",
    ])
    return "\n".join(lines)


class GrokChatModel(BaseChatModel):
    """LangChain chat-model wrapper around the grok CLI (ACP).

    Provider name in deep-worker: ``"grok"``. Construction is cheap and
    side-effect-free; the proxy MCP server and ACP client are started
    lazily on the first ``ainvoke``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _client_state: Any = PrivateAttr(default=None)
    _acp_client: Any = PrivateAttr(default=None)  # GrokAcpClient (one per process)
    _proxy_mcp: Any = PrivateAttr(default=None)
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
    # Token-stats usage-event callback (DESIGN_AGENT_TOKEN_STATS.md §5.2):
    # async callable installed by create_model via set_usage_event_callback.
    _usage_event_callback: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Pydantic schema overrides — same surgery as codex/claude-code:
    # LangGraph pickles models during checkpointing; strip callable fields.
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
        parent["description"] = "Grok chat model for LangChain"
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
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["custom_get_token_ids"] = None
        super().__init__(**kwargs)
        self._client_state = get_global_client_state()
        self._tools_metadata = []
        self._model_name = model
        self._resolved_model_name = model
        self._working_dir = working_dir

    @property
    def _llm_type(self) -> str:
        return "grok"

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
        context_used: int | None = None,
        context_size: int | None = None,
    ) -> None:
        """Report the per-turn usage to the stats callback (§5.2).

        Best-effort: the callback is awaited; failures are logged and never
        break the provider turn (except ``asyncio.CancelledError``, which
        propagates).

        grok (live-verified 0.2.73, DESIGN §2.2 live-probe): per-turn token
        usage arrives in the ``session/prompt`` RESULT ``_meta``
        (``inputTokens``/``outputTokens``/``cachedReadTokens``/``totalTokens``)
        — a plain token record like codex. A ``usage_update``-derived context
        gauge (``context_used``/``context_size``) is the fallback when no
        ``_meta`` usage was seen. ``event_key`` is always null (no stable
        provider id, MINOR v3).
        """
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
            logger.exception("[token-stats] grok usage event callback failed")

    # ------------------------------------------------------------------
    # Tools — extract MCP metadata (same algorithm as codex/claude-code).
    # ------------------------------------------------------------------
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "GrokChatModel":
        # CRITICAL: built-in passthrough tools (channel B) MUST NOT be
        # published as MCP tools via proxy_mcp_server (channel A). They exist
        # ONLY for LangGraph's ToolNode to route the surfaced tool_call back
        # to a body that awaits the reader-resolved Future — they don't have a
        # remote endpoint. If we publish e.g. ``Shell`` in our proxy MCP
        # ``tools/list``, grok sees a deepworker MCP tool with the same name
        # as its native ``Shell`` and dispatches the model's "Shell" call
        # through CallMcpTool, which parks a Future on proxy_mcp_server that
        # the built-in path will never resolve → 120s job-handle notice and a
        # stuck turn (test_grok incident 2026-06-18). Filter them out here.
        from grok_provider.builtin_tool_passthrough import (  # noqa: PLC0415
            is_grok_builtin_passthrough_tool,
        )

        tools_metadata: list[ToolMetadata] = []
        for item in tools:
            if not isinstance(item, BaseTool):
                continue
            if is_grok_builtin_passthrough_tool(item):
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
        on a ``reply_back=False`` inter-agent delivery), the agent factory
        binds with ``request.model.bind(**settings)`` — leaving
        ``_tools_metadata`` from the original ``bind_tools`` intact.  That
        would still register the proxy MCP server with grok and invite
        spurious tool calls / hung turns.  Tag this binding so ``ainvoke``
        omits MCP for exactly one call.
        """
        return super().bind(_grok_text_only_turn=True, **kwargs)

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
            "GrokChatModel only supports async operations. Use ainvoke()."
        )

    # ------------------------------------------------------------------
    # Lazy startup of the proxy MCP + ACP client.
    # ------------------------------------------------------------------
    async def _ensure_proxy_started(self) -> None:
        # The proxy MCP server is ALWAYS required, even when no deep-worker
        # MCP tools are bound: every ``session/new`` references it as grok's
        # MCP endpoint (DESIGN §10 step 10). A server exposing zero tools is
        # valid — grok simply gets an empty tools/list. Returning early when
        # ``_tools_metadata`` was empty left ``_proxy_mcp`` as None and made
        # the very next line of ainvoke (``self._proxy_mcp.url_async()``)
        # raise AttributeError on any tool-less turn — caught by the live
        # integration tests 2026-06-18 (hello-world / memory / isolation).
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
            from grok_provider.acp_client import GrokAcpClient

            self._acp_client = GrokAcpClient(client_state=self._client_state)
        if not self._acp_started:
            await self._acp_client.start()
            await self._acp_client.initialize()
            await self._acp_client.authenticate()
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
                "GrokChatModel requires explicit thread_id in config — this "
                "is deep-worker's agent name. Each caller (main agent, "
                "TaskManager, subagent) must own its thread_id. Implicit "
                "defaults are forbidden."
            )
        agent_thread_id: str = cfg["thread_id"]

        # Token-stats: mark the invocation start (no record is written here —
        # the record closes at the turn boundary in _finalize_response, §5.2).
        # Best-effort.
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

        # 4b. Summarization detection — drop grok session, keep agent_thread_id.
        if state.sent_count > len(state_messages):
            # Cancel any live mid-turn pump first (fix #2 2026-06-18): the
            # turn it was draining belongs to a now-dropped session; leaving
            # it racing on a dead queue would leak the task and could resolve
            # stale built-in Futures. record_summarization is sync and cannot
            # await, so the cancellation lives here on the caller side.
            await self._adopt_pump(state)
            self._client_state.record_summarization(agent_thread_id)
            state = self._client_state.get_or_create(agent_thread_id)

        # 5. System prompt (plain text — factory-prepended SystemMessage).
        system_prompt: str | None = None
        if messages and isinstance(messages[0], SystemMessage):
            system_prompt = _flatten_message_content_to_text(messages[0].content)

        text_only_turn = bool(kwargs.pop("_grok_text_only_turn", False))
        state.text_only_turn_active = text_only_turn

        # 6. Lazy start of proxy MCP + ACP client. Text-only turns use a
        # fresh grok session without MCP — skip touching the proxy listener so
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
        # mid-turn tool_call return without another ainvoke (e.g.
        # SendMessageCompletion).  sent_count then lags behind state.
        # On a fresh turn (no in-flight session/prompt) send only the trailing
        # HumanMessage delta so grok is not re-fed stale tool rows.
        if (
            not tool_resolved
            and not saw_builtin_resume
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
                        GROK_BUILTIN_MARKER, False
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
            # AIMessage — already in grok session. SystemMessage routing is
            # handled in step 11 via ``_collect_runtime_system_chunks`` (see
            # claude-code ``System:>`` forwarding).

        # 7c. Text-only reply turns spin a fresh grok session (step 10b) with
        # no prior CLI history — include every HumanMessage from state so the
        # model sees the original user ask AND the inter-agent delivery.
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
        #    this branch too, else we'd fire a second concurrent
        #    session/prompt (ACP violation). _consume_turn re-uses the same
        #    current_notif_queue without a new subscription.
        if (tool_resolved or saw_builtin_resume) and (
            state.pending_session_prompt_future is not None
        ):
            # Take over from the background pump (fix #2 2026-06-18): cancel
            # it and wait for it to settle so the accumulators
            # (queued_text/queued_tool_calls/queued_response) are stable
            # before _consume_turn reads them. No-op when no pump is running.
            await self._adopt_pump(state)
            return await self._consume_turn(state, agent_thread_id)

        # 10. New turn → ensure a session exists AND is live in THIS grok
        #     subprocess. Three cases:
        #       (a) no session id           → session/new
        #       (b) persisted id, not live  → session/load (restart resume)
        #       (c) id already live here    → use directly
        #
        # Only register the proxy MCP server with grok when it actually
        # exposes at least one deep-worker tool. grok 0.2.54 ABORTS a turn
        # (empty output) if handed an MCP server whose tools/list is empty
        # (live integration 2026-06-18). With no bound tools we omit the
        # server; grok's built-ins still flow through the channel-B
        # passthrough.
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

        if state.grok_session_id is None:
            # (a) Brand-new session.
            session_id = await self._acp_client.session_new(
                cwd=self._working_dir or ".",
                mcp_servers=mcp_servers,
            )
            state.grok_session_id = session_id
            state.session_live_in_process = True
            self._client_state.register_session(session_id, agent_thread_id)
            self._client_state.persist(agent_thread_id)
            if hasattr(self._acp_client, "wait_session_ready"):
                await self._acp_client.wait_session_ready(session_id)
            is_first_prompt_in_session = True
        elif not state.session_live_in_process:
            # (b) A session id loaded from disk after a restart. grok keeps
            # sessions in memory, so the NEW subprocess does not know this id
            # — prompting it directly is silently dropped (empty turn: the
            # 2026-06-18 "Agent did not produce a response" on every restarted
            # agent's first message). Re-attach it with session/load (memory
            # is preserved CLI-side). On success the prior history is intact,
            # so this is NOT a first prompt (system block already delivered).
            loaded = False
            try:
                await self._acp_client.session_load(
                    state.grok_session_id,
                    cwd=self._working_dir or ".",
                    mcp_servers=mcp_servers,
                )
                if hasattr(self._acp_client, "wait_session_ready"):
                    await self._acp_client.wait_session_ready(
                        state.grok_session_id
                    )
                # grok may still have a stale pending prompt from the previous
                # lifecycle (session/load restores it). Cancel it so the next
                # session/prompt doesn't get a "cancelled" response.
                with contextlib.suppress(Exception):
                    await self._acp_client.session_cancel(
                        state.grok_session_id
                    )
                loaded = True
            except Exception:  # noqa: BLE001 — stale/un-loadable session → fresh
                logger.warning(
                    "grok: session/load failed for %s (thread %s); starting a "
                    "fresh session and re-sending the full prompt",
                    state.grok_session_id, agent_thread_id,
                )
                self._client_state._evict_session(state.grok_session_id)
                session_id = await self._acp_client.session_new(
                    cwd=self._working_dir or ".",
                    mcp_servers=mcp_servers,
                )
                state.grok_session_id = session_id
                if hasattr(self._acp_client, "wait_session_ready"):
                    await self._acp_client.wait_session_ready(session_id)
                # The fresh session knows NOTHING. Re-send the entire visible
                # history's user text as this turn's prompt and re-deliver the
                # system block, exactly like a brand-new session. sent_count
                # was already advanced to len(messages) in step 8 (correct for
                # the post-turn state); we only rebuild the prompt text here.
                user_text_parts = [
                    m.content if isinstance(m.content, str) else str(m.content)
                    for m in state_messages
                    if isinstance(m, HumanMessage)
                ]
            state.session_live_in_process = True
            self._client_state.register_session(
                state.grok_session_id, agent_thread_id
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
            and state.grok_session_id is not None
        ):
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(state.grok_session_id)
            if self._tools_metadata and not state.mcp_detached_for_text_only:
                await self._acp_client.session_load(
                    state.grok_session_id,
                    cwd=self._working_dir or ".",
                    mcp_servers=[],
                )
                if hasattr(self._acp_client, "wait_session_ready"):
                    await self._acp_client.wait_session_ready(
                        state.grok_session_id
                    )
                with contextlib.suppress(Exception):
                    await self._acp_client.session_cancel(
                        state.grok_session_id
                    )
                state.mcp_detached_for_text_only = True
        elif (
            not text_only_turn
            and self._tools_metadata
            and state.mcp_detached_for_text_only
            and state.session_live_in_process
            and state.grok_session_id is not None
        ):
            proxy_url = await self._proxy_mcp.url_async()
            await self._acp_client.session_load(
                state.grok_session_id,
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
                await self._acp_client.wait_session_ready(state.grok_session_id)
            state.mcp_detached_for_text_only = False

        # 11. Build prompt blocks — first prompt prepends the <system> block.
        # Runtime SystemMessages (middleware cognitive hints) are forwarded as
        # ``System:>`` lines prepended to the user text on later turns, or as
        # suffix deltas when middleware appends to the first SystemMessage.
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
        if is_first_prompt_in_session and system_prompt:
            system_parts: list[str] = []
            if system_prompt:
                system_parts.append(system_prompt)
            if self._tools_metadata:
                tool_contract = _build_tool_contract(
                    [tool["name"] for tool in self._tools_metadata]
                )
                if tool_contract:
                    system_parts.append(tool_contract)
            system_text = "\n\n".join(system_parts)
            text = (
                f"<system>\n{system_text}\n</system>\n\n"
                f"{user_text}"
            )
        else:
            text = user_text
        blocks = [{"type": "text", "text": text}]

        # 12. Fire session/prompt + consume until response / tool_calls.
        state.current_notif_queue = asyncio.Queue()
        state.pending_session_prompt_future = self._acp_client.session_prompt(
            session_id=state.grok_session_id,
            prompt=blocks,
            notification_sink=state.current_notif_queue,
        )
        try:
            return await self._consume_turn(state, agent_thread_id)
        finally:
            state.text_only_turn_active = False

    async def _consume_turn(self, state: Any, agent_thread_id: str) -> AIMessage:
        """Consume session/update notifications until either pending
        tool_call(s) need LangGraph routing (return AIMessage(tool_calls),
        keeping the in-flight future for the next ainvoke) or the
        session/prompt response arrives (return AIMessage(content)).

        On re-entry from a mid-turn ainvoke this method PREPENDS the
        pump-accumulated ``state.queued_text`` and ``state.queued_tool_calls``
        — they were collected by ``_drain_pump`` while LangGraph's ToolNode
        was running the channel-B passthrough body. If the pump already
        observed the response future resolving with a terminal stopReason it
        stashed the result in ``state.queued_response`` so we short-circuit
        without waiting for ``response_fut`` (already done).

        The mid-turn pause and the natural turn end both clear/keep the
        future correctly; the finally clause guards abnormal exits so a
        crashed/cancelled ainvoke never leaves a dangling future.
        """
        final_text_parts: list[str] = list(state.queued_text)
        pending_tool_calls: list[dict] = list(state.queued_tool_calls)
        state.queued_text = []
        state.queued_tool_calls = []
        queued_response = state.queued_response
        state.queued_response = None

        notif_q: asyncio.Queue = state.current_notif_queue
        response_fut: asyncio.Future = state.pending_session_prompt_future
        cleared_normally = False
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
                        "grok text-only turn surfaced tool_calls %r — "
                        "dropping to avoid hung ToolNode",
                        [tc.get("name") for tc in pending_tool_calls],
                    )
                    with contextlib.suppress(Exception):
                        await self._acp_client.session_cancel(
                            state.grok_session_id
                        )
                    state.pending_session_prompt_future = None
                    cleared_normally = True
                    return AIMessage(content="".join(final_text_parts))
                await self._correlate_proxy_tool_calls(
                    state, pending_tool_calls
                )
                cleared_normally = True
                return AIMessage(
                    content="".join(final_text_parts),
                    tool_calls=pending_tool_calls,
                )

            while True:
                get_task = asyncio.ensure_future(notif_q.get())
                done, _pending = await asyncio.wait(
                    {get_task, response_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if response_fut in done:
                    # The turn's response arrived. CRITICAL (live integration
                    # 2026-06-18): grok streams all of its agent_message_chunk
                    # text and THEN resolves the response in the SAME event
                    # loop iteration, so by the time we get here the notif
                    # queue can still hold the chunks that make up the answer.
                    # Finalizing straight away returned an EMPTY AIMessage
                    # (the intermittent "content=''" failures). So first absorb
                    # the get_task we already launched, then drain whatever
                    # else is queued, applying every update — only THEN
                    # finalize. (A tool_call surfacing here is contradictory —
                    # a turn either ends or pauses for a tool — so we keep it
                    # simple and just include any accumulated text.)
                    if get_task in done and not get_task.cancelled():
                        with contextlib.suppress(asyncio.CancelledError):
                            self._apply_update(
                                state, agent_thread_id, get_task.result(),
                                final_text_parts, pending_tool_calls,
                            )
                    else:
                        get_task.cancel()
                    while not notif_q.empty():
                        self._apply_update(
                            state, agent_thread_id, notif_q.get_nowait(),
                            final_text_parts, pending_tool_calls,
                        )
                    response = response_fut.result()
                    state.pending_session_prompt_future = None
                    cleared_normally = True
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

                if pending_tool_calls:
                    if state.text_only_turn_active:
                        logger.warning(
                            "grok text-only turn mid-stream tool_calls %r — "
                            "dropping",
                            [tc.get("name") for tc in pending_tool_calls],
                        )
                        pending_tool_calls.clear()
                        continue
                    # Mid-turn pause: give LangGraph the tool_calls. KEEP the
                    # response future — the next ainvoke re-enters
                    # _consume_turn. Spawn the pump so the queue keeps being
                    # drained while ToolNode runs the passthrough body (fix
                    # #2 2026-06-18 — test_grok Shell hang).
                    await self._correlate_proxy_tool_calls(
                        state, pending_tool_calls
                    )
                    cleared_normally = True
                    self._start_pump(state, agent_thread_id)
                    return AIMessage(
                        content="".join(final_text_parts),
                        tool_calls=pending_tool_calls,
                    )
        finally:
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
                    await self._acp_client.session_cancel(state.grok_session_id)

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
            text_acc.append((upd.get("content") or {}).get("text", ""))
        elif kind == "agent_thought_chunk":
            return  # reasoning chunks — never surfaced
        elif kind == "tool_call":
            tc = self._map_tool_call(state, upd)
            if tc is not None:
                tool_calls_acc.append(tc)
        elif kind == "tool_call_update":
            self._resolve_builtin_update(agent_thread_id, upd)
        elif kind == "usage_update":
            # Context gauge (DESIGN_AGENT_TOKEN_STATS.md §5.2): used/size at
            # the top level of the update per ACP spec, with a content{}-dict
            # fallback. Keeps the LATEST snapshot of the current turn; emitted
            # once at the turn boundary in _finalize_response.
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
        elif kind == "plan":
            tc = _plan_to_tool_call(upd)
            if tc is not None:
                tool_calls_acc.append(tc)
        # other kinds — silently ignored

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

        ``session/prompt`` resolves its future with the RAW reply (see
        ``acp_client._dispatch``) and bypasses ``_call``'s error check, so an
        error reply reaches here as ``{"id", "error"}`` with no ``result``.
        Detect it FIRST and raise (usage/billing cap -> failover to deepseek,
        anything else -> hard error) instead of silently returning an empty
        AIMessage. Incident 2026-07-03 (grok HTTP 402 "Payment Required")."""
        from grok_provider.acp_client import (  # noqa: PLC0415
            raise_for_error_reply,
        )

        if isinstance(response, dict):
            raise_for_error_reply("session/prompt", response)
            result = response.get("result", {})
        else:
            result = {}
        await self._maybe_report_model(result)
        stop_reason = result.get("stopReason")
        if stop_reason == "cancelled":
            raise GrokTurnCancelledError("grok turn cancelled by ACP")
        if stop_reason == "refusal":
            raise GrokRefusalError("grok refused to continue")
        # Token-stats (DESIGN §2.2 live-probe, live-verified 0.2.73): the
        # session/prompt RESULT._meta carries per-turn token usage
        # (inputTokens/outputTokens/cachedReadTokens/totalTokens) — record it
        # as a plain token record (cache hit = cachedReadTokens). The
        # usage_update-derived context gauge (usage_snapshot) is the fallback
        # when no _meta usage was seen. Emit once at the turn boundary and
        # reset the accumulator. Best-effort.
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
            if meta_usage is not None:
                await self._emit_usage_event(
                    event="terminal",
                    status="ok",
                    usage=meta_usage,
                    session_id=getattr(state, "grok_session_id", None),
                )
                state.usage_snapshot = None
            else:
                snapshot = getattr(state, "usage_snapshot", None)
                if snapshot is not None:
                    await self._emit_usage_event(
                        event="terminal",
                        status="ok",
                        session_id=getattr(state, "grok_session_id", None),
                        context_used=snapshot.get("used"),
                        context_size=snapshot.get("size"),
                    )
                    state.usage_snapshot = None
        return AIMessage(content="".join(text_acc))

    # ------------------------------------------------------------------
    # Mid-turn pump (fix #2 2026-06-18)
    # ------------------------------------------------------------------
    def _start_pump(self, state: Any, agent_thread_id: str) -> None:
        """Spawn the background drain task. Idempotent — a second call while
        the previous pump is still alive is a programming bug; we cancel the
        old task to keep the queue single-consumer."""
        self._cancel_pump(state)
        state.pump_task = asyncio.create_task(
            self._drain_pump(state, agent_thread_id),
            name=f"grok-pump-{agent_thread_id}",
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
          - accumulates ``agent_message_chunk`` text into ``state.queued_text``;
          - queues any fresh ``tool_call`` events into
            ``state.queued_tool_calls`` so the next ainvoke surfaces them;
          - stashes the response future's reply in ``state.queued_response``
            if grok ends the turn while ToolNode is still running, so the
            next ainvoke finalizes without re-reading anything.

        Exits cleanly on cancellation (the next ainvoke calls ``_adopt_pump``)
        or when ``pending_session_prompt_future`` resolves. Never raises out —
        any exception during update handling is logged and the pump keeps
        running so a single malformed event can't kill the turn.
        """
        notif_q: asyncio.Queue = state.current_notif_queue
        response_fut: asyncio.Future = state.pending_session_prompt_future
        if notif_q is None or response_fut is None:
            return
        try:
            while True:
                get_task = asyncio.ensure_future(notif_q.get())
                done, _pending = await asyncio.wait(
                    {get_task, response_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if response_fut in done:
                    # Same drain-before-finalize discipline as _consume_turn:
                    # absorb the in-flight get_task plus any remaining queued
                    # updates so trailing agent_message_chunk text is not lost
                    # when grok resolves the response in the same loop tick.
                    if get_task in done and not get_task.cancelled():
                        with contextlib.suppress(Exception):
                            self._apply_update(
                                state, agent_thread_id, get_task.result(),
                                state.queued_text, state.queued_tool_calls,
                            )
                    else:
                        get_task.cancel()
                    while not notif_q.empty():
                        with contextlib.suppress(Exception):
                            self._apply_update(
                                state, agent_thread_id, notif_q.get_nowait(),
                                state.queued_text, state.queued_tool_calls,
                            )
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
                except Exception:  # noqa: BLE001 — pump must survive a single bad event
                    logger.exception("grok pump: failed to apply update")
        except asyncio.CancelledError:
            # Normal hand-off path: the next ainvoke called _adopt_pump.
            raise

    # ------------------------------------------------------------------
    # tool_call mapping
    # ------------------------------------------------------------------
    def _bound_tool_names(self) -> set[str]:
        return {t["name"] for t in self._tools_metadata}

    def _proxy_tool_from_update(self, upd: dict) -> tuple[str, dict] | None:
        """Return (name, args) when this update is a channel-A proxy tool."""
        if self._proxy_mcp is None:
            return None
        title = upd.get("title")
        raw_input = upd.get("rawInput") or {}
        if acp_types.is_proxy_mcp_dispatch(upd, self._proxy_mcp.server_name):
            return (
                str(raw_input.get("toolName") or ""),
                raw_input.get("arguments") or {},
            )
        if title in self._bound_tool_names():
            return title, raw_input
        return None

    async def _correlate_proxy_tool_calls(
        self, state: Any, tool_calls: list[dict]
    ) -> None:
        """Map ACP toolCallIds to proxy handler call_ids before LangGraph runs.

        The MCP handler parks under a synthetic id; LangGraph later resolves
        via ToolMessage.tool_call_id (the ACP id). Without this map,
        resolve_call is a no-op and grok hangs until the job-handle timeout.
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
            try:
                # Match by tool name only — grok's ACP tool_call args and the
                # MCP tools/call payload can differ (key order, coercion).
                pending = await self._proxy_mcp.wait_for_pending_calls(
                    tool_name=name,
                    arguments=None,
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "grok_provider: no pending MCP handler for %r "
                    "(toolCallId=%s) within 10s",
                    name,
                    tcid,
                )
                continue
            if pending:
                state.proxy_call_id_by_acp_id[tcid] = pending[0]["id"]

    def _map_tool_call(self, state: Any, upd: dict) -> dict | None:
        """Map a grok ``tool_call`` update to a LangChain tool_call dict.

        CASE A (proxy MCP): CallMcpTool dispatch OR grok calling a bound
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

        proxy_tool = self._proxy_tool_from_update(upd)
        if proxy_tool is not None:
            name, args = proxy_tool
            if name:
                # Narrow Grok arg-compat: rewrite Cursor-style keys / coerce
                # human-readable values into the shape our tool schemas expect.
                # CASE A only — built-in passthrough (CASE B) is verbatim.
                # See docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md.
                args = normalize_tool_args(name, args)
                return {
                    "id": tcid,
                    "name": name,
                    "args": args,
                    "type": "tool_call",
                }

        # Built-in (CASE B). Register the passthrough Future for the reader.
        if tcid not in state.builtin_passthrough_futures:
            state.builtin_passthrough_futures[tcid] = (
                asyncio.get_event_loop().create_future()
            )
        if title in acp_types.KNOWN_GROK_BUILTINS:
            surfaced_name = title
            surfaced_args = raw_input
        elif title:
            logger.warning(
                "grok_provider: unknown built-in %r — routing via GrokBuiltin",
                title,
            )
            surfaced_name = "GrokBuiltin"
            surfaced_args = {**raw_input, "_grok_original_title": title}
        else:
            surfaced_name = "GrokBuiltin"
            surfaced_args = raw_input
        return {
            "id": tcid,
            "name": surfaced_name,
            "args": surfaced_args,
            "type": "tool_call",
        }

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
    """Collect ``System:>`` lines for runtime system content not yet in grok.

    grok accepts a ``<system>`` block only on the first ``session/prompt`` of
    an ACP session. Later middleware cognitive hints (InterAgentReply,
    ProjectMemory, …) MUST ride on the same user prompt as the HumanMessage
    they refer to — mirroring ``claude_code_provider/chat_model.py`` lines
    1078–1149 (see a related internal incident analysis, not included here).

    Two paths:

    1. **new_messages scan** — any SystemMessage in the send delta *after* the
       conversation's first SystemMessage (agent default prompt) becomes a
       ``System:>`` line. Covers InterAgentReply prepend and ProjectMemory
       injections that arrive as separate SystemMessage objects.

    2. **First-system append delta** — when middleware appends to the existing
       first SystemMessage (InterAgentReply / ProjectMemory append path), only
       the suffix since ``forwarded_first_system_chars`` is forwarded.

    Returns ``(chunks, new_forwarded_first_system_chars,
    new_forwarded_extra_system_keys)``.
    """
    chunks: list[str] = []
    new_first_chars = forwarded_first_system_chars
    new_extra_keys = set(forwarded_extra_system_keys)

    # Path 1 — claude-code pre-pass on the send delta only.
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

    Grok puts result text in ``content`` (a list of content blocks) and
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
    grok_meta = dict(metadata)
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
        grok_meta["artifact_path"] = artifact_path
        if mime:
            grok_meta["mime"] = mime
        raw_type = str(metadata.get("type", "")).lower()
        grok_meta["kind"] = (
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
        "grok_meta": grok_meta,
    }


def _plan_to_tool_call(upd: dict) -> dict | None:
    """Surface grok's ``plan`` update as an ``update_plan`` tool_call so the
    plan lands in LangGraph history (codex update_plan analog)."""
    tcid = upd.get("toolCallId") or "plan"
    entries = upd.get("entries")
    return {
        "id": tcid,
        "name": "update_plan",
        "args": {"entries": entries} if entries is not None else {},
        "type": "tool_call",
    }


# Re-export the meta key so callers can attach grok_meta to ToolMessages.
__all__ = ["GrokChatModel", "GrokRefusalError", "GROK_META_KEY"]
