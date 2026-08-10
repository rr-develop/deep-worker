"""ClaudeCodeChatModel - LangChain wrapper around Claude Code SDK.

This module provides a LangChain BaseChatModel implementation that uses Claude Code SDK
(ClaudeSDKClient) as the underlying AI provider. It handles:
- Message conversion between LangChain and Claude formats
- Tool call delegation to deep-worker via ProxyMCPServer
- Session management via ClientManager (thread_id isolation)
- Async message streaming from Claude
"""

import asyncio  # type: ignore[reportUnusedImport] — used in async methods
import logging
import sys
from pathlib import Path
from typing import Any, Callable

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage as ClaudeAssistantMessage,
    ResultMessage,
    SystemMessage as ClaudeSystemMessage,
    UserMessage as ClaudeUserMessage,
)

from collections.abc import Sequence  # noqa: E402

from langchain_core.language_models.base import LanguageModelInput  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402
from pydantic import ConfigDict, PrivateAttr  # noqa: E402

from claude_code_provider.client_manager import get_global_client_manager  # noqa: E402
from claude_code_provider.custom_types import MCPResult, ToolMetadata  # noqa: E402
from claude_code_provider.proxy_mcp_server import DispatchTimeout  # noqa: E402


_logger = logging.getLogger(__name__)

# NEVER use print() in this module — see docs/POLICIES.md §1 and
# docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md.  ``stderr`` can block
# under load and BlockingIOError used to propagate up to the agent
# invoke catch-all, flipping the whole agent into AgentStatus.ERROR.
# Always use ``_logger.*`` (non-blocking handler).

class ClaudeUsageLimitError(RuntimeError):
    """Raised when the Claude CLI reports the account's usage/session limit.

    The CLI surfaces an exhausted usage cap as a single degenerate response
    whose entire text is ``You've hit your session limit · resets <time>``.
    It is NOT a real answer. Returning it as ordinary ``AIMessage`` content
    made the agent fail silently (observed live, 2026-05-21): the TaskManager kept
    "correcting" the useless reply, the agent settled to idle, and the web
    UI showed no error. Raising a typed error instead lets the agent process
    transition to ``status=ERROR`` with this human-readable message.
    """


class ClaudeApiResponseError(RuntimeError):
    """Raised when the Claude CLI returns an upstream API error AS the turn.

    Distinct from :class:`ClaudeUsageLimitError`: the account is fine, only
    this request failed (e.g. HTTP 400 on a malformed message payload). The
    error is *recoverable* — the runtime keeps the conversation alive and the
    next request may succeed once the trigger is gone.

    Registered in ``deep_worker_cli.exceptions.PROVIDER_API_ERROR_TYPES``
    (NOT in ``FATAL_EXCEPTION_TYPES``) so ``agent_process._handle_invoke``'s
    recoverable branch surfaces the real API text as a persistent
    ``type="error"`` notify and appends it to conversation history, while
    the agent stays IDLE. See docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md
    and docs/POLICIES.md §9.

    Carries structured context so diagnostics stay honest and a future
    failover policy has something to key on. ``str(exc)`` renders the real
    upstream text (never an empty string, never ordinary partial assistant
    prose — see the §3.2 precedence in the design).
    """

    def __init__(
        self,
        message: str,
        *,
        assistant_error: str | None = None,
        result_subtype: str | None = None,
        result_errors: tuple[str, ...] | list[str] = (),
        stop_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.assistant_error = assistant_error
        self.result_subtype = result_subtype
        self.result_errors: tuple[str, ...] = tuple(result_errors)
        self.stop_reason = stop_reason


# Standalone marker the Claude CLI emits when the account usage cap is hit.
# Matched at the START of a stripped response (it is the whole turn), so a
# normal answer that merely discusses session limits is not a false positive.
_SESSION_LIMIT_MARKER = "You've hit your session limit"


# ───────────────────────────────────────────────────────────────────────────
# Proxy MCP tool naming
# ───────────────────────────────────────────────────────────────────────────
# Claude Agent SDK exposes tools registered via ``create_sdk_mcp_server`` to
# the LLM using the canonical ``mcp__<server_name>__<tool_name>`` naming.
# Our proxy server is created as ``create_sdk_mcp_server(name="deepworker",
# ...)`` in :mod:`claude_code_provider.proxy_mcp_server`, so every
# ``ToolUseBlock`` Claude emits for a proxy tool has a ``name`` of the form
# ``mcp__deepworker__<tool_name>``. The internal :class:`ProxyMCPServer`
# pending-call registry, in contrast, stores the BARE tool name (because
# ``_create_tool_handler(tool_name)`` is invoked with the bare name from
# ``ToolMetadata``). We must therefore strip the prefix before:
#   1. checking whether a ``ToolUseBlock`` refers to one of our proxy
#      tools (membership against ``ToolMetadata`` names);
#   2. building the ``expected_calls`` argument for
#      :meth:`ProxyMCPServer.wait_for_matching_pending_calls`, which
#      matches against the bare names stored in ``pending_calls``.
_PROXY_MCP_TOOL_PREFIX = "mcp__deepworker__"


def _strip_proxy_mcp_prefix(tool_name: str) -> str:
    """Return the bare proxy tool name, stripping the SDK MCP prefix if any.

    Examples:
        >>> _strip_proxy_mcp_prefix("mcp__deepworker__ls")
        'ls'
        >>> _strip_proxy_mcp_prefix("ls")
        'ls'
        >>> _strip_proxy_mcp_prefix("WebSearch")
        'WebSearch'
    """
    if tool_name.startswith(_PROXY_MCP_TOOL_PREFIX):
        return tool_name[len(_PROXY_MCP_TOOL_PREFIX):]
    return tool_name


def _build_mcp_content(content: Any) -> list[dict[str, Any]]:
    """Convert LangChain ToolMessage content to MCP content blocks.

    Handles both simple string content and multimodal content (text + images).
    Converts LangChain ``image_url`` blocks to MCP ``image`` blocks.

    Args:
        content: ToolMessage.content — either a string or a list of content blocks.

    Returns:
        List of MCP-compatible content blocks.
    """
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    mcp_blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            # Convert LangChain image_url → MCP image block
            # data URL format: "data:image/png;base64,iVBOR..."
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                parts = url.split(",", 1)
                mime = parts[0].split(";")[0].split(":")[1] if ":" in parts[0] else "image/png"
                data = parts[1] if len(parts) > 1 else ""
                mcp_blocks.append({"type": "image", "data": data, "mimeType": mime})
            else:
                # Non-data URL — pass as text fallback
                mcp_blocks.append({"type": "text", "text": f"[Image: {url}]"})
        elif isinstance(block, dict) and block.get("type") == "text":
            mcp_blocks.append(block)
        else:
            mcp_blocks.append({"type": "text", "text": str(block)})

    return mcp_blocks


class ClaudeCodeChatModel(BaseChatModel):
    """LangChain chat model wrapper around Claude Code SDK.

    This model integrates Claude Code SDK with LangChain, enabling:
    1. Message-based conversation through LangChain interface
    2. Tool execution in deep-worker (not in Claude CLI)
    3. Session isolation via thread_id
    4. Full transparency for LangGraph workflows

    The model uses ClientManager to maintain separate ClaudeSDKClient instances
    per thread_id, and ProxyMCPServer to delegate tool execution to deep-worker.

    Attributes:
        _base_options: Base options for ClaudeSDKClient (private, not serialized)
        _client_manager: Manages ClaudeSDKClient instances per thread_id (private)
        _tools_metadata: Metadata for tools to expose through MCP (private)

    Example:
        >>> from claude_agent_sdk import ClaudeAgentOptions
        >>> from langchain_core.messages import HumanMessage
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def calculator(a: float, b: float) -> float:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>>
        >>> options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        >>> model = ClaudeCodeChatModel(base_options=options)
        >>> model = model.bind_tools([calculator])
        >>>
        >>> # Simple conversation
        >>> result = await model.ainvoke(
        ...     [HumanMessage(content="Hello")],
        ...     config={"configurable": {"thread_id": "session-1"}}
        ... )
        >>>
        >>> # Tool call handling
        >>> result = await model.ainvoke(
        ...     [HumanMessage(content="Calculate 5+3")],
        ...     config={"configurable": {"thread_id": "session-1"}}
        ... )
        >>> # result.tool_calls contains pending tool calls
    """

    # Pydantic configuration
    # - arbitrary_types_allowed: Allow non-Pydantic types in private attributes
    # - extra='allow': Allow extra fields (needed for model, max_tokens, etc.)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    # Private attributes (not included in schema, not serialized)
    # Using PrivateAttr ensures these don't appear in JSON Schema
    _base_options: Any = PrivateAttr(default=None)
    _client_manager: Any = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    # ⚠️ NO _tool_executors! Tools execute via LangGraph ToolNode.
    # See ProxyMCPServer module docstring for architecture explanation.
    _working_dir: str | None = PrivateAttr(default=None)
    # Token-stats usage-event callback (DESIGN_AGENT_TOKEN_STATS.md §5.2):
    # async callable installed by create_model via set_usage_event_callback.
    _usage_event_callback: Any = PrivateAttr(default=None)

    # Properties for backward compatibility with existing code
    @property
    def base_options(self) -> Any:
        """Get base options."""
        return self._base_options

    @property
    def client_manager(self) -> Any:
        """Get client manager."""
        return self._client_manager

    @property
    def tools_metadata(self) -> list[ToolMetadata]:
        """Get tools metadata."""
        return self._tools_metadata

    def set_usage_event_callback(self, callback: Any) -> None:
        """Install the async usage-event callback (§5.2).

        ``create_model`` injects the token-stats state-machine callback here
        (custom providers never receive a LangChain callback handler — the
        event callback is the single usage source, MAJOR-2).
        """
        self._usage_event_callback = callback

    async def _emit_usage_event(self, event: Any) -> None:
        """Report a terminal ``ResultMessage`` to the stats callback (§5.2).

        Best-effort: the callback is awaited; failures are logged and never
        break the provider turn (except ``asyncio.CancelledError``, which
        propagates). ``ResultMessage.usage`` is already a plain dict — copied
        defensively before handing over (SDK objects are not assumed
        JSON-serializable, MAJOR-4).
        """
        cb = getattr(self, "_usage_event_callback", None)
        if cb is None:
            return
        try:
            usage = getattr(event, "usage", None)
            if isinstance(usage, dict):
                usage = dict(usage)
            await cb(
                event="terminal",
                status="failed" if bool(getattr(event, "is_error", False)) else "ok",
                usage=usage,
                model=getattr(self, "model", None) or "",
                session_id=getattr(event, "session_id", None) or None,
                uuid=getattr(event, "uuid", None) or None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort
            _logger.exception("[token-stats] claude-code usage event callback failed")

    async def drain_after_inter_agent_send(self, thread_id: str) -> None:
        """Drain the rest of the current Claude turn after a successful
        ``send_message_to_agent`` tool result was injected through the
        proxy MCP Future.

        Called from ``InterAgentDeliveryResolveMiddleware.awrap_tool_call``
        right after ``resolve_pending_tool``. The reason: the proxy MCP
        ``send_message_to_agent`` ``ToolUseBlock`` makes us break out of
        ``receive_response`` early to surface the tool_call to LangGraph.
        Claude only emits the corresponding ``UserMessage(ToolResultBlock)``
        + closing ``AssistantMessage(TextBlock)`` + ``ResultMessage``
        AFTER the MCP Future is resolved. If we don't read them here,
        they stay buffered in the SDK queue and the NEXT ``ainvoke`` (the
        one carrying the peer's reply) consumes that buffered tail
        BEFORE Claude sees the actual reply, returns the buffered
        ``TextBlock`` (e.g. "Message sent. Awaiting its reply.") to
        LangGraph as A's assistant message, and ends the graph — so the
        peer's reply is never actually processed in that ainvoke.

        See a related internal incident analysis (not included here)
        (12-step trace, "Experiment C").

        Best-effort: any error here is logged and swallowed so a draining
        glitch can never break an inter-agent turn.
        """
        try:
            entry = self._client_manager.clients.get(thread_id)
            if entry is None:
                return
            client, _proxy = entry
            from claude_agent_sdk import ResultMessage as _RM
            count = 0
            async for ev in client.receive_response():
                count += 1
                if isinstance(ev, _RM):
                    return
                if count > 30:
                    # Safety cap; in practice tail is 3-4 events.
                    return
        except Exception:  # noqa: BLE001
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "drain_after_inter_agent_send failed; swallowed"
            )

    def resolve_pending_tool(
        self, thread_id: str, tool_call_id: str, content: MCPResult
    ) -> None:
        """Resolve a pending proxy-MCP tool Future for ``thread_id`` NOW.

        The normal path resolves a tool's Future in the send loop of the next
        ``ainvoke`` (see the ``ToolMessage`` branch). For a turn-ending tool
        like ``send_message_to_agent`` that next ``ainvoke`` never happens in
        the same graph run (SendMessageCompletionMiddleware jumps to end), so
        the Future would otherwise hang until the next external impulse and the
        Claude CLI would report the tool call as timed out — even though
        delivery already succeeded. This lets the claude-code-only
        delivery-resolve middleware resolve the Future at DELIVERY time so the
        120s handler timeout bounds delivery only, not the recipient's work.

        Best-effort and idempotent: a missing client/thread, an unknown
        ``tool_call_id``, or an already-resolved Future are all safe no-ops
        (``ProxyMCPServer.resolve_call`` guards on both presence and
        ``future.done()``). Never raises into the caller.
        """
        try:
            entry = self._client_manager.clients.get(thread_id)
            if entry is None:
                return
            _client, proxy_mcp = entry
            proxy_mcp.resolve_call(tool_call_id, content)
        except Exception:  # noqa: BLE001 — must never break a tool turn
            pass

    @staticmethod
    def _has_persisted_session(thread_id: str) -> bool:
        """True iff this thread has persisted session state on disk — i.e. it
        RAN BEFORE and this is a genuine restart.

        This is the signal that gates history-restore: it separates a real
        restart (state was persisted by a prior run's update_sent_count) from
        (a) an in-process live tool round-trip and (b) unit tests with an
        isolated empty HOME — in both of those there is no persisted state, so
        restore must NOT fire and the normal send loop runs.
        """
        try:
            from claude_code_provider.session_state import (  # noqa: PLC0415
                load_session_state,
            )

            return load_session_state(thread_id) is not None
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        """Override JSON Schema generation to exclude problematic fields.

        This method is called by Pydantic when generating JSON Schema.
        We return a minimal schema that excludes callable fields (like
        custom_get_token_ids from BaseChatModel) which would cause
        PydanticInvalidForJsonSchema errors.

        BaseChatModel has a field `custom_get_token_ids: Callable[[str], list[int]] | None`
        that cannot be serialized to JSON Schema. We exclude it by returning
        a minimal schema.

        Args:
            core_schema: Core schema from Pydantic
            handler: Schema generation handler

        Returns:
            Minimal JSON Schema for ClaudeCodeChatModel
        """
        # Get the parent schema but filter out callable fields
        # This prevents issues with BaseChatModel.custom_get_token_ids and similar
        try:
            parent_schema = handler(core_schema)
        except Exception:
            # If parent schema generation fails, use minimal schema
            parent_schema = {
                "type": "object",
                "properties": {},
            }

        # Remove properties that are callable or cause issues
        if "properties" in parent_schema and isinstance(parent_schema["properties"], dict):
            # Filter out callable fields like custom_get_token_ids
            # Keep only serializable fields
            properties: dict[str, Any] = parent_schema["properties"]  # type: ignore[assignment]
            parent_schema["properties"] = {
                k: v
                for k, v in properties.items()
                if k
                not in {
                    "custom_get_token_ids",  # Callable from BaseChatModel
                    "callbacks",  # May contain callables
                    "rate_limiter",  # May have callable fields
                }
            }

        parent_schema["title"] = cls.__name__
        parent_schema["description"] = "Claude Code chat model for LangChain"

        return parent_schema

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to exclude callable fields.

        LangGraph tries to serialize models during task execution. We need to
        exclude callable fields from BaseChatModel to avoid PydanticInvalidForJsonSchema.

        Args:
            **kwargs: Arguments to pass to parent model_dump

        Returns:
            Model data dict with callable fields excluded
        """
        # Get parent model dump but exclude callable fields
        exclude_fields: set[str] = {
            "custom_get_token_ids",  # Callable from BaseChatModel
            "callbacks",  # May contain callables
            "rate_limiter",  # May have callable fields
        }

        # Merge with any user-provided exclude
        if "exclude" in kwargs:
            user_exclude = kwargs["exclude"]
            if isinstance(user_exclude, set):
                exclude_fields = exclude_fields | user_exclude
                kwargs["exclude"] = exclude_fields
            # If user_exclude is a dict, keep it as-is (don't merge)

        else:
            kwargs["exclude"] = exclude_fields

        return super().model_dump(**kwargs)

    def __init__(
        self,
        model: str | None = None,
        base_options: Any = None,  # ClaudeAgentOptions at runtime, Any for type checking
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ClaudeCodeChatModel.

        Args:
            model: Model name (sonnet, haiku, opus, or full model name).
                  If provided and base_options is None, creates ClaudeAgentOptions.
            base_options: Base options to use for all ClaudeSDKClient instances (ClaudeAgentOptions).
                         Takes precedence over model parameter.
            max_tokens: Maximum tokens (used when creating ClaudeAgentOptions from model).
            **kwargs: Additional arguments passed to BaseChatModel
        """
        # CRITICAL: Set callable fields to None BEFORE calling super().__init__
        # BaseChatModel.custom_get_token_ids is a Callable field that causes
        # PydanticInvalidForJsonSchema errors. We must set it to None to prevent
        # Pydantic from trying to serialize it.
        kwargs["custom_get_token_ids"] = None

        super().__init__(**kwargs)

        # Create base_options from model if needed
        # Import here to avoid Pydantic schema issues
        if base_options is None and model is not None:
            from claude_agent_sdk import ClaudeAgentOptions
            base_options = ClaudeAgentOptions(model=model)

        # Initialize private attributes
        # These are excluded from Pydantic schema and serialization
        self._base_options = base_options
        # Use global singleton ClientManager so state persists across model instances
        # This is CRITICAL because LangGraph may create new model instances during
        # deserialization, but we need _sent_count to persist
        self._client_manager = get_global_client_manager()
        self._tools_metadata = []

        # NOTE: _sent_count is now stored in global ClientManager singleton
        # This ensures it persists even when LangGraph creates new model instances

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "ClaudeCodeChatModel":
        """Bind tools to this model.

        Returns a NEW ``ClaudeCodeChatModel`` whose ``_tools_metadata``
        carries one ``ToolMetadata`` entry per input tool. The original
        ``self`` is left unchanged (LangChain immutability contract).

        ════════════════════════════════════════════════════════════════
        ⚠ INPUT CONTRACT — DO NOT WEAKEN. READ THE 2026-06-12 AND
        2026-06-13 (#2) INCIDENT ANALYSES BEFORE TOUCHING THIS METHOD.
        ════════════════════════════════════════════════════════════════
        The signature follows ``BaseChatModel.bind_tools`` (Liskov-
        compliant override) but the RUNTIME contract narrows it to two
        shapes per element:
          1. ``BaseTool`` instance (already a LangChain tool).
          2. plain ``Callable`` (``def`` / ``async def`` / lambda) —
             normalised here via ``StructuredTool.from_function``.

        ``dict`` and ``type`` items declared in the base signature are
        NOT supported and raise ``TypeError`` at runtime with an
        actionable message. **Silently dropping unknown items is
        forbidden** — that "defensive" reflex is the root cause of
        the MCP-tools-lost cluster (four incidents in three weeks). A
        single dropped tool means the Claude CLI subprocess starts
        without it, the agent then truthfully reports "I have no
        ``read_file`` / ``execute`` / …", and the user spends hours
        chasing the gap. Loud failure at startup beats silent failure
        four hours into a session.

        ⚠ INVARIANT (the 2026-06-12 incident): ``bind_tools([])`` is a
        no-op that returns ``self`` unchanged. LangGraph internals and
        ``SummarizationMiddleware`` call ``bind_tools([])`` 100s of
        times per session as part of internal re-binding. Before the
        2026-06-12 fix, that empty rebind silently wiped
        ``_tools_metadata`` on the freshly-allocated instance and the
        emptiness then flowed into ``ProxyMCPServer(tools_metadata=[])``
        on the next session rotation. Keep the early-return.

        ⚠ INVARIANT (the 2026-06-13 #2 incident): plain ``Callable`` tools
        MUST be normalised, NOT skipped. ``agent_process.web_tools``
        contains ``http_request`` / ``fetch_url`` / ``read_image`` as
        bare ``def`` functions; LangGraph ``ToolNode`` wraps them in
        ``StructuredTool`` internally and the agent CAN call them — but
        if WE drop them on the way to ``ProxyMCPServer`` the CLI
        subprocess has no MCP advertisement for them and Claude refuses
        to use them after restart.

        See a related internal incident analysis (not included here)
        (MCP tools lost after restart).

        Args:
            tools: Sequence of ``BaseTool`` instances and/or plain
                ``Callable``s. Mixed input is the normal case (production
                ``web_tools`` is exactly that).
            **kwargs: Reserved for future use; ignored today.

        Returns:
            A new ``ClaudeCodeChatModel`` with ``_tools_metadata``
            populated — one entry per input tool, in input order — or
            ``self`` unchanged when ``tools`` is empty.

        Raises:
            TypeError: An element is neither a ``BaseTool`` nor a
                ``Callable``. Message names the offending item and
                hints at the ``@tool`` decorator.
        """
        import sys
        msg = f"[ClaudeCodeChatModel.bind_tools] Called with {len(tools)} tools\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        # ⚠️ Empty rebind is a no-op — see invariant above. Returning self
        # preserves the previously-bound tools across LangGraph internal
        # re-bind calls (which is the failure mode that caused the
        # MCP-tools-lost-after-restart incident). Without this guard, a freshly-allocated model with
        # empty _tools_metadata would propagate into ProxyMCPServer and
        # strip every mcp__deepworker__* tool from the CLI subprocess.
        if not tools:
            return self
        # Local import to keep module import cost low — StructuredTool
        # pulls a non-trivial pydantic graph that we don't need on plain
        # ``ainvoke``-only code paths.
        from langchain_core.tools import StructuredTool  # noqa: PLC0415

        # Normalise input to a list of BaseTool. Anything we cannot map
        # raises TypeError — see a related internal incident analysis for
        # why silent skip is forbidden.
        normalised_tools: list[BaseTool] = []
        for tool_item in tools:
            if isinstance(tool_item, BaseTool):
                normalised_tools.append(tool_item)
                continue
            if callable(tool_item):
                # Plain function (decorated with nothing, or async def).
                # ``StructuredTool.from_function`` derives a JSON schema
                # from the signature + docstring — same mechanism
                # LangGraph's ``ToolNode`` uses internally, so the agent
                # side and the MCP side now see the SAME tool surface.
                normalised_tools.append(StructuredTool.from_function(tool_item))
                continue
            # Loud failure: name the offender and tell the operator how
            # to fix it. See module docstring "INPUT CONTRACT" block.
            raise TypeError(
                f"ClaudeCodeChatModel.bind_tools: unsupported tool form "
                f"{type(tool_item).__name__}={tool_item!r}. Expected "
                f"BaseTool or Callable. Did you forget to wrap a plain "
                f"function with @tool? (Silent skip is forbidden — see "
                f"a related internal incident analysis (not included here))"
            )

        # Extract metadata from the normalised list.
        tools_metadata: list[ToolMetadata] = []
        for tool in normalised_tools:
            # Get schema — try Pydantic first, fallback to function signature
            schema = {}
            if tool.args_schema:
                try:
                    if hasattr(tool.args_schema, "model_json_schema"):
                        schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
                    elif hasattr(tool.args_schema, "schema"):
                        schema = tool.args_schema.schema()  # type: ignore[union-attr,deprecated]
                except Exception:
                    schema = {}

            # If Pydantic schema failed (empty properties or missing), build
            # from function signature. This handles StructuredTool.from_function
            # tools that use ToolRuntime (causes CallableSchema error).
            if not schema.get("properties"):
                import inspect
                func = getattr(tool, 'coroutine', None) or getattr(tool, 'func', None)
                if func:
                    try:
                        sig = inspect.signature(func)
                        props = {}
                        required = []
                        for pname, param in sig.parameters.items():
                            if pname in ('runtime', 'self', 'kwargs'):
                                continue
                            ann = param.annotation
                            if ann == int:
                                ptype = "integer"
                            elif ann == bool:
                                ptype = "boolean"
                            elif ann == float:
                                ptype = "number"
                            else:
                                ptype = "string"
                            p: dict[str, Any] = {"type": ptype, "title": pname.replace('_', ' ').title()}
                            # Extract description from Annotated metadata
                            if hasattr(ann, '__metadata__'):
                                for meta in ann.__metadata__:
                                    if isinstance(meta, str):
                                        p["description"] = meta
                                        break
                            if param.default != inspect.Parameter.empty:
                                p["default"] = param.default
                            else:
                                required.append(pname)
                            props[pname] = p
                        if props:
                            schema = {"type": "object", "properties": props, "required": required}
                    except (ValueError, TypeError):
                        pass

            metadata: ToolMetadata = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            }
            tools_metadata.append(metadata)

        # ⚠️ NO tool_executors! Tools execute via LangGraph ToolNode, not MCP handler.
        # ProxyMCPServer uses Future-based proxy — handler blocks until LangGraph
        # executes tool and calls resolve_call(). This keeps tool calls visible
        # in result.messages for dwcli broadcast, checkpointing, and HITL.

        # Create new instance with tools
        new_model = ClaudeCodeChatModel(base_options=self._base_options)
        new_model._tools_metadata = tools_metadata
        new_model._working_dir = self._working_dir
        # Token-stats (DESIGN_AGENT_TOKEN_STATS.md §5.2): LangChain's
        # bind_tools contract returns a NEW instance, so PrivateAttrs are
        # NOT carried over automatically. The real model used by LangGraph
        # (turn-node bind + _bind_tools_for_middleware) is this copy — it
        # must keep the usage-event callback or stats silently go missing.
        new_model._usage_event_callback = self._usage_event_callback

        return new_model

    # ⚠️ No executor factory methods here!
    # Tools execute via LangGraph ToolNode (which handles ToolRuntime injection,
    # InjectedToolCallId, working_dir, etc. automatically).
    # ProxyMCPServer Future-based proxy returns tool calls to LangGraph.
    # See DESIGN_restore_proxy_architecture.md for architecture details.

    async def ainvoke(
        self,
        input: LanguageModelInput,  # BaseChatModel uses LanguageModelInput type
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model asynchronously.

        This method implements proper message mirroring between LangGraph and Claude:
        1. Tracks which messages have been sent to Claude session
        2. Sends ONLY new messages (no duplicates, no resending old messages)
        3. Receives response and returns to LangGraph

        Architecture:
        - LangGraph maintains full message history in state (via checkpointer)
        - ClaudeSDKClient maintains session history in Claude CLI (via session_id)
        - ClaudeCodeChatModel tracks sent_count to determine new messages
        - New messages = messages[sent_count:] are sent to Claude

        Args:
            input: Language model input (messages, prompt, or string)
            config: Runnable config with thread_id in configurable.thread_id
            **kwargs: Additional arguments (unused)

        Returns:
            AIMessage with either:
            - tool_calls: If Claude wants to call tools
            - content: If Claude has final answer
        """
        # 1. Convert input to messages
        from langchain_core.prompt_values import PromptValue

        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)  # type: ignore[arg-type]

        # 2. Extract thread_id from config
        # First, pull config from LangGraph context variable if available.
        # When LangGraph calls model.ainvoke(messages) from within a graph node,
        # it sets var_child_runnable_config with the parent's config (including thread_id).
        from langchain_core.runnables.config import ensure_config
        config = ensure_config(config)

        # ⚠️ MANDATORY: thread_id MUST be explicitly provided in every call.
        # ClaudeCodeChatModel is STATEFUL — each thread_id maps to a separate
        # Claude SDK session with its own sent_count and message history.
        #
        # DO NOT add a fallback to "default" or any other hardcoded value!
        # A shared thread_id between main agent and subagent causes:
        #   1. sent_count corruption (subagent overwrites main agent's count)
        #   2. Message re-sending (main agent resends already-seen messages)
        #   3. Claude SDK deadlock (inconsistent session state → hang)
        #
        # Every caller MUST explicitly provide its own thread_id.
        # If thread_id is missing, the caller has a configuration bug — fail loudly.
        if "configurable" not in config or "thread_id" not in config.get("configurable", {}):
            raise ValueError(
                "ClaudeCodeChatModel requires explicit thread_id in config. "
                "Pass config={'configurable': {'thread_id': '...'}}. "
                "Each caller (main agent, subagent, TaskManager) must have its own thread_id. "
                "Implicit defaults are forbidden to prevent session state corruption."
            )
        thread_id = config["configurable"]["thread_id"]

        # Token-stats: mark the invocation start (no record is written here —
        # the record closes on the terminal ResultMessage, §5.2). Best-effort.
        _cb = getattr(self, "_usage_event_callback", None)
        if _cb is not None:
            try:
                await _cb(event="invocation_start")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                _logger.exception("[token-stats] invocation_start callback failed")

        # 3. Extract system prompt from SystemMessage (if present)
        # ⚠️ CRITICAL: SystemMessage contains cognitive hints from middleware
        # (e.g. ProjectMemoryMiddleware injects project_memory usage instructions).
        # If this is dropped, agents lose awareness of critical tools.
        # DO NOT remove this extraction — it is NOT redundant.
        system_prompt_content: str | None = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt_content = str(msg.content)
                break

        # 4. HISTORY RESTORE ON RESTART (claude-code).
        # If this is the very first turn for this thread on a fresh process
        # (the CLI client hasn't been created yet) AND there is prior history
        # to carry over, resume the agent's conversation in the new Claude CLI
        # session so it actually REMEMBERS what it did. Without this, the new
        # CLI session is empty: the send loop below skips AIMessages (`pass`)
        # and resolve_call's ToolMessages into Futures that don't exist in a
        # fresh session, so Claude would see only the raw user prompts and
        # "forget" everything. We pick the session to resume via
        # resolve_restore_session (match an existing CLI transcript by tail, or
        # synthesize one from the LangGraph history for cross-provider
        # continuation). resume does NOT re-execute tools (verified — E1).
        # See docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md. Best-effort: any
        # failure → None → fresh session (today's behavior).
        resume_session_id: str | None = None
        # 4a. Determine how many messages have already been sent before
        # restore selection. On restart with a new user tail, only the already
        # sent prefix belongs in the resumed/synthesized Claude transcript; the
        # unsent tail must still be delivered through the normal send loop.
        sent_count = self._client_manager.get_sent_count(thread_id)

        # ``has_client`` is the cross-instance "first turn after (re)start"
        # guard: it lives on the global ClientManager singleton (LangGraph may
        # build new model instances, but the manager is stable), so it is True
        # for every turn after the CLI client is created and restore runs at
        # most once.
        # Tool-call ids orphaned by a restart (populated on the restart
        # turn below; applied to the proxy once it is available).
        pending_orphan_ids: list[str] = []
        already_connected = self._client_manager.has_client(thread_id)
        if not already_connected:
            restore_source = messages
            if sent_count > 0:
                restore_source = messages[: min(sent_count, len(messages))]
            restorable = [m for m in restore_source if not isinstance(m, SystemMessage)]
            # Restore only when this is genuinely a RESTART of a prior
            # conversation, evidenced by BOTH:
            #   (a) the history carries prior turns (an AIMessage/ToolMessage —
            #       a brand-new agent's first turn is just a HumanMessage), AND
            #   (b) persisted session state exists for this thread (we ran
            #       before). (b) is what distinguishes a real restart from an
            #       in-process live tool round-trip (no persisted state yet) and
            #       from unit tests with an isolated empty HOME — in those
            #       cases we must NOT synthesize and must let the normal send
            #       loop run. See DESIGN_HISTORY_RESTORE_ON_RESTART.md.
            has_prior_turns = any(
                isinstance(m, (AIMessage, ToolMessage)) for m in restorable
            )
            _is_restart = self._has_persisted_session(thread_id)
            if has_prior_turns and _is_restart:
                from claude_code_provider.session_restore import (  # noqa: PLC0415
                    resolve_restore_session,
                )
                resume_session_id = resolve_restore_session(
                    thread_id, restorable, cwd=self._working_dir or "/"
                )

            # Restart turn: a tool call the model requested but for which no
            # ToolMessage result exists was orphaned by the restart. Detect
            # those now so we can tell the proxy to reconcile them (inform
            # the model, never wait). Without this the resumed CLI replays
            # the dangling tool_use and the turn deadlocks on an id whose
            # result never comes.
            #
            # CRITICAL: scan the FULL ``messages``, not ``restorable``. The
            # orphaned call typically sits in the UNSENT tail (it was the
            # in-flight call when the process died, after the last persisted
            # ``sent_count``), so ``restorable = messages[:sent_count]``
            # would miss it. And gate on ``_is_restart`` alone (persisted
            # session) — NOT on ``resolve_restore_session`` succeeding —
            # because the deadlock happens via the CLI replay regardless of
            # whether our LangGraph-side restore synthesis fired.
            # See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md and the
            # internal root-cause memo (2026-06-15).
            if _is_restart:
                from claude_code_provider.orphaned_tool_calls import (  # noqa: PLC0415
                    find_orphaned_tool_calls,
                    find_orphaned_tool_calls_in_transcript,
                )

                _orphan_ids: dict[str, str] = {}

                # (1) LangGraph history — catches orphans that DID get
                #     checkpointed (AIMessage tool_call with no ToolMessage).
                for o in find_orphaned_tool_calls(
                    [m for m in messages if not isinstance(m, SystemMessage)]
                ):
                    _orphan_ids.setdefault(o.id, o.name)

                # (2) THE AUTHORITATIVE SOURCE — the resumed CLI transcript.
                #     When the process is killed mid-tool-call the LangGraph
                #     checkpoint never recorded the interrupted turn, but the
                #     CLI ``.jsonl`` transcript ends with a dangling
                #     ``tool_use`` (no ``tool_result``). On resume the CLI
                #     REPLAYS that block and the turn deadlocks. We read the
                #     exact transcript being resumed and reconcile its
                #     dangling tool_use ids. Verified live 2026-06-15: the
                #     orphan lives here, not in LangGraph history.
                if resume_session_id:
                    try:
                        from claude_agent_sdk import (  # noqa: PLC0415
                            get_session_messages,
                        )

                        _tx = get_session_messages(
                            resume_session_id,
                            directory=self._working_dir or "/",
                        )
                        for o in find_orphaned_tool_calls_in_transcript(
                            _tx or []
                        ):
                            _orphan_ids.setdefault(o.id, o.name)
                    except Exception as _exc:  # noqa: BLE001
                        # Best-effort: never let transcript inspection break
                        # the turn. Worst case we fall back to history-only.
                        _logger.warning(
                            "[RESTART] transcript orphan-scan skipped for "
                            "%s: %r",
                            resume_session_id, _exc,
                        )

                if _orphan_ids:
                    pending_orphan_ids = list(_orphan_ids)
                    _logger.warning(
                        "[RESTART] %d orphaned tool call(s) detected "
                        "(interrupted by restart): %s — will reconcile, "
                        "not wait",
                        len(_orphan_ids),
                        list(_orphan_ids.items()),
                    )

        # [MCP_DIAG] Log the exact shape of arguments we pass to the
        # first-path get_or_create_client. This is the BASELINE — fresh
        # start where MCP is known to work. Compared against the
        # rotation_recreate log below, the two must look identical.
        _logger.info(
            f"[MCP_DIAG] fresh_create thread={thread_id} "
            f"resume_session={resume_session_id!r} "
            f"system_prompt_is_none={system_prompt_content is None} "
            f"system_prompt_len="
            f"{len(system_prompt_content) if system_prompt_content else 0} "
            f"tools_metadata_count={len(self._tools_metadata)}"
        )
        # [MCP_DIAG] Log the exact shape of arguments we pass to the
        # first-path get_or_create_client. This is the BASELINE — fresh
        # start where MCP is known to work. Compared against the
        # rotation_recreate log below, the two must look identical.
        _logger.info(
            f"[MCP_DIAG] fresh_create thread={thread_id} "
            f"resume_session={resume_session_id!r} "
            f"system_prompt_is_none={system_prompt_content is None} "
            f"system_prompt_len="
            f"{len(system_prompt_content) if system_prompt_content else 0} "
            f"tools_metadata_count={len(self._tools_metadata)}"
        )
        # [MCP_DIAG] Log the exact shape of arguments we pass to the
        # first-path get_or_create_client. This is the BASELINE — fresh
        # start where MCP is known to work. Compared against the
        # rotation_recreate log below, the two must look identical.
        _logger.info(
            f"[MCP_DIAG] fresh_create thread={thread_id} "
            f"resume_session={resume_session_id!r} "
            f"system_prompt_is_none={system_prompt_content is None} "
            f"system_prompt_len="
            f"{len(system_prompt_content) if system_prompt_content else 0} "
            f"tools_metadata_count={len(self._tools_metadata)}"
        )
        # Get or create client for this thread (resume if we resolved one).
        client, proxy_mcp = await self._client_manager.get_or_create_client(
            thread_id=thread_id,
            tools_metadata=self._tools_metadata,
            working_dir=self._working_dir,
            system_prompt=system_prompt_content,
            resume_session_id=resume_session_id,
        )

        # Reconcile any tool calls orphaned by the restart: the proxy will
        # return the interrupted-notice result immediately for these ids
        # instead of parking on a Future that will never resolve. No
        # timeout — structural. See DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
        if pending_orphan_ids:
            proxy_mcp.mark_orphaned_tool_calls(pending_orphan_ids)

        # DEBUG: Log what we received (both to stderr AND to file)
        import sys
        import datetime
        debug_log_path = "/tmp/claude_provider_debug.log"
        timestamp = datetime.datetime.now().isoformat()
        try:
            with open(debug_log_path, "a") as f:
                f.write(f"\n[{timestamp}] ========== START AINVOKE ==========\n")
                f.write(f"[{timestamp}] Model instance id={id(self)}\n")
                f.write(f"[{timestamp}] ClientManager id={id(self._client_manager)}\n")
                f.write(f"[{timestamp}] thread_id={thread_id}\n")
                f.write(f"[{timestamp}] sent_count={sent_count}\n")
                f.write(f"[{timestamp}] _sent_count state={dict(self._client_manager._sent_count)}\n")
                f.write(f"[{timestamp}] len(messages)={len(messages)}\n")
                for i, msg in enumerate(messages):
                    content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
                    f.write(f"[{timestamp}]   messages[{i}]: {type(msg).__name__} - {content_preview}\n")
                f.flush()
        except Exception as e:
            _logger.info(f"[DEBUG] Failed to write debug log: {e}")

        # ALSO print to stderr
        _logger.info(f"\n[{timestamp}] ========== START AINVOKE ==========")
        _logger.info(f"[{timestamp}] thread_id={thread_id}, sent_count={sent_count}, len(messages)={len(messages)}")

        _logger.info(f"\n[DEBUG ainvoke] ========== START ==========")
        _logger.info(f"[DEBUG ainvoke] Model instance id={id(self)}")
        _logger.info(f"[DEBUG ainvoke] ClientManager id={id(self._client_manager)}")
        _logger.info(f"[DEBUG ainvoke] thread_id={thread_id}")
        _logger.info(f"[DEBUG ainvoke] sent_count={sent_count}")
        _logger.info(f"[DEBUG ainvoke] _sent_count state={dict(self._client_manager._sent_count)}")
        _logger.info(f"[DEBUG ainvoke] len(messages)={len(messages)}")
        for i, msg in enumerate(messages):
            content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
            _logger.info(f"[DEBUG ainvoke]   messages[{i}]: {type(msg).__name__} - {content_preview}")

        # ⚠️ CRITICAL: Detect LangGraph summarization (message history reset).
        # When SummarizationMiddleware replaces N messages with a summary,
        # sent_count (from the old history) exceeds len(messages) (new, shorter).
        # We MUST rotate to a new Claude CLI session_id — otherwise the CLI
        # session still holds the full old history and will eventually hit
        # "Prompt is too long" (observed on a production agent after 1255 turns).
        # DO NOT remove this rotation — it is the ONLY mechanism that prevents
        # Claude CLI session overflow for long-running agents.
        if sent_count > len(messages):
            # [MCP_DIAG] Why is this rotation firing? Two very different
            # causes look identical at this point:
            #   (A) Real LangGraph SummarizationMiddleware run — first
            #       HumanMessage carries lc_source="summarization" and
            #       its content starts with "Here is a summary of the
            #       conversation to date:".
            #   (B) Something else removed N messages from state
            #       (ContextEditingMiddleware, RemoveMessage, interrupt
            #       resume path…) — Δ is small (1–3), no summary marker.
            # We've observed (B) firing per-turn in the log, which would
            # respawn the CLI subprocess every turn and could plausibly
            # break MCP. Tag every rotation with its cause so we can
            # split the failure modes in the log post-mortem.
            try:
                first_human_idx = next(
                    (i for i, m in enumerate(messages)
                     if isinstance(m, HumanMessage)),
                    -1,
                )
                rotation_cause = "unknown"
                summary_marker = False
                lc_source = None
                first_human_preview = ""
                if first_human_idx >= 0:
                    fhm = messages[first_human_idx]
                    fhm_content = (
                        fhm.content if isinstance(fhm.content, str)
                        else str(fhm.content)
                    )
                    first_human_preview = fhm_content[:120]
                    summary_marker = fhm_content.startswith(
                        "Here is a summary of the conversation to date:"
                    )
                    lc_source = (
                        fhm.additional_kwargs.get("lc_source")
                        if hasattr(fhm, "additional_kwargs")
                        else None
                    )
                if summary_marker or lc_source == "summarization":
                    rotation_cause = "summarization"
                elif (sent_count - len(messages)) <= 3:
                    rotation_cause = "small_delta_no_summary"
                else:
                    rotation_cause = "large_delta_no_summary"
                _logger.info(
                    f"[MCP_DIAG] rotation_trigger thread={thread_id} "
                    f"cause={rotation_cause} "
                    f"delta={sent_count - len(messages)} "
                    f"sent_count={sent_count} len_messages={len(messages)} "
                    f"summary_marker={summary_marker} "
                    f"lc_source={lc_source!r} "
                    f"first_human_idx={first_human_idx} "
                    f"first_human_preview={first_human_preview!r}"
                )
            except Exception as exc:  # noqa: BLE001
                # Diagnostic must never break the rotation path.
                _logger.info(
                    f"[MCP_DIAG] rotation_trigger probe_failed "
                    f"thread={thread_id} err={exc!r}"
                )

            new_session_id = await self._client_manager.rotate_session(thread_id)
            _logger.info(f"[DEBUG ainvoke] SESSION ROTATED: {thread_id} → {new_session_id} "
                f"(old sent_count={sent_count}, new messages={len(messages)})")
            sent_count = 0
            # ⚠️ rotate_session() popped + disconnected the old client from
            # self.clients[thread_id], but the local ``client`` / ``proxy_mcp``
            # variables (line 679) still reference the stale disconnected
            # instance.  Any subsequent ``client.query(...)`` or
            # ``client.receive_response()`` would raise CLIConnectionError
            # ("MCP transport used before connect() completed").  Fetch a
            # fresh client (and its associated ProxyMCPServer) here.
            #
            # resume_session_id is explicitly None — the old session was
            # rotated away; do NOT try to resume an old transcript.
            # [MCP_DIAG] Verify the system_prompt we pass on rotation is
            # the same shape as on fresh start (it should be — same code
            # path, but the symptom keeps insisting otherwise).
            _logger.info(
                f"[MCP_DIAG] rotation_recreate thread={thread_id} "
                f"new_session={new_session_id} "
                f"system_prompt_is_none={system_prompt_content is None} "
                f"system_prompt_len="
                f"{len(system_prompt_content) if system_prompt_content else 0} "
                f"tools_metadata_count={len(self._tools_metadata)}"
            )
            # [MCP_DIAG] Verify the system_prompt we pass on rotation is
            # the same shape as on fresh start (it should be — same code
            # path, but the symptom keeps insisting otherwise).
            _logger.info(
                f"[MCP_DIAG] rotation_recreate thread={thread_id} "
                f"new_session={new_session_id} "
                f"system_prompt_is_none={system_prompt_content is None} "
                f"system_prompt_len="
                f"{len(system_prompt_content) if system_prompt_content else 0} "
                f"tools_metadata_count={len(self._tools_metadata)}"
            )
            # [MCP_DIAG] Verify the system_prompt we pass on rotation is
            # the same shape as on fresh start (it should be — same code
            # path, but the symptom keeps insisting otherwise).
            _logger.info(
                f"[MCP_DIAG] rotation_recreate thread={thread_id} "
                f"new_session={new_session_id} "
                f"system_prompt_is_none={system_prompt_content is None} "
                f"system_prompt_len="
                f"{len(system_prompt_content) if system_prompt_content else 0} "
                f"tools_metadata_count={len(self._tools_metadata)}"
            )
            client, proxy_mcp = await self._client_manager.get_or_create_client(
                thread_id=thread_id,
                tools_metadata=self._tools_metadata,
                working_dir=self._working_dir,
                system_prompt=system_prompt_content,
                resume_session_id=None,
            )

        # 5. Extract ONLY new messages (not yet sent to Claude)
        new_messages = messages[sent_count:]

        _logger.info(f"[DEBUG ainvoke] len(new_messages)={len(new_messages)}")
        for i, msg in enumerate(new_messages):
            content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
            _logger.info(f"[DEBUG ainvoke]   new_messages[{i}]: {type(msg).__name__} - {content_preview}")

        # 6. Send each new message to Claude
        user_message_sent = False
        tool_resolved = False

        # ⚠️ Runtime SystemMessages from LangGraph middleware (e.g. the
        # InterAgentReplyMiddleware cognitive hint) must reach Claude in
        # the SAME prompt as the user message they refer to. The Claude
        # SDK has no public API to update ``system_prompt`` after
        # ``connect()`` — the first SystemMessage of the conversation is
        # the agent's default prompt and was passed via
        # ``ClaudeAgentOptions.system_prompt`` at client creation. Any
        # later SystemMessage is collected here and prepended (with a
        # ``"System:> "`` line prefix) to the very next HumanMessage in
        # ``new_messages``. We do NOT issue a separate ``client.query``
        # for the hint, because that would make Claude run a whole turn
        # responding to the hint first and only later see the actual
        # peer message. See a related internal incident analysis (not
        # included here).
        _pending_system_chunks: list[str] = []
        _first_system_seen_before = False
        for _prior in messages:
            if isinstance(_prior, SystemMessage) and _prior not in new_messages:
                _first_system_seen_before = True
                break
        _first_system_consumed_in_pass = _first_system_seen_before
        for _m in new_messages:
            if isinstance(_m, SystemMessage):
                if not _first_system_consumed_in_pass:
                    _first_system_consumed_in_pass = True
                else:
                    _sc = _m.content if isinstance(_m.content, str) else str(_m.content)
                    _pending_system_chunks.append(f"System:> {_sc}")

        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                # Send user message to Claude
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Prepend any runtime SystemMessage(s) that accumulated
                # earlier in this send loop — see the pre-pass above
                # for the rationale. Sent as one combined prompt so
                # Claude sees the hint and the user message together.
                if _pending_system_chunks:
                    content = "\n\n".join(_pending_system_chunks + [content])
                    _pending_system_chunks = []
                # ⚠️ Use effective_session_id (rotated after summarization),
                # NOT raw thread_id. See rotate_session() for details.
                effective_sid = self._client_manager.get_effective_session_id(thread_id)
                _logger.info(f"[DEBUG ainvoke] Sending HumanMessage to Claude:")
                _logger.info(f"[DEBUG ainvoke]   content={content[:300]!r}")
                _logger.info(f"[DEBUG ainvoke]   session_id='{effective_sid}'")
                await client.query(content, session_id=effective_sid)
                user_message_sent = True

            elif isinstance(msg, ToolMessage):
                # ⚠️ PROXY ARCHITECTURE: resolve pending MCP Future with tool result.
                # LangGraph executed the tool via ToolNode and returned ToolMessage.
                # We resolve the Future so the MCP handler returns result to Claude CLI.
                # DO NOT use client.query() here — that sends result as user message,
                # bypassing the MCP protocol and breaking the proxy.
                mcp_result: MCPResult = {
                    "content": _build_mcp_content(msg.content),
                }
                _logger.info(f"[DEBUG ainvoke] resolve_call for tool_call_id={msg.tool_call_id}")
                proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)
                tool_resolved = True

            elif isinstance(msg, SystemMessage):
                # SystemMessages are routed at the top of this method:
                #  * the FIRST SystemMessage of the conversation is the
                #    agent's default prompt and is delivered to Claude
                #    via ``ClaudeAgentOptions.system_prompt`` at client
                #    creation — no re-delivery needed here.
                #  * any LATER SystemMessage is a runtime middleware
                #    cognitive hint; it was pre-collected into
                #    ``_pending_system_chunks`` and will be prepended
                #    (as a ``"System:> ..."`` line) to the next
                #    HumanMessage in this same send loop.
                # Either way nothing to do here.
                pass

            elif isinstance(msg, AIMessage):
                # AIMessage is already in Claude's history
                # (It was generated by Claude in previous turn)
                # Just skip - no need to send back to Claude
                pass

        # 7. Update sent count BEFORE receiving response
        # This ensures we don't resend messages if receive_response fails
        # Store in ClientManager (not self) because LangGraph may create new model instances
        self._client_manager.update_sent_count(thread_id, len(messages))

        # 8. Receive response from Claude (if user message sent OR tool was resolved)
        response_text = ""
        pending_tool_calls: list[dict[str, Any]] | None = None
        proxy_tool_names = {tool["name"] for tool in self._tools_metadata}

        # ── API-error signal capture (§3.3 of
        # DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md) ──
        # Terminal-signal state for the post-loop API-error detector.
        # Declared at ainvoke top-level so EVERY exit path (including the
        # orphan-reconcile shortcut that skips the receive_response block
        # entirely, and the empty-turn path where neither ``user_message_sent``
        # nor ``tool_resolved`` is true) sees defined bindings. Last non-None
        # wins for scalar fields inside the receive loop: the terminal
        # message is authoritative.
        last_assistant_error: str | None = None
        last_usage: Any | None = None
        result_is_error: bool = False
        result_subtype: str | None = None
        result_errors: tuple[str, ...] = ()
        result_text: str | None = None
        result_stop_reason: str | None = None

        _logger.info(f"[DEBUG ainvoke] user_message_sent={user_message_sent}, tool_resolved={tool_resolved}")

        if user_message_sent or tool_resolved:
            _logger.info(f"[DEBUG ainvoke] Calling client.receive_response() [EVENT-BASED]...")

            # ═══════════════════════════════════════════════════════════════
            # PROXY ARCHITECTURE: receive_response with ToolUseBlock break
            # ═══════════════════════════════════════════════════════════════
            # Claude SDK sends AssistantMessage(ToolUseBlock) BEFORE tool
            # execution. We break on ToolUseBlock to return tool_calls to
            # LangGraph. LangGraph executes tools via ToolNode, sends
            # ToolMessage back, and we resolve the MCP Future.
            #
            # DO NOT wait for ResultMessage when ToolUseBlock is present!
            # Claude CLI waits for MCP handler to return result before
            # sending ResultMessage. If we wait → circular dependency:
            #   receive_response waits for ResultMessage
            #   → Claude waits for tool result
            #   → MCP handler waits for resolve_call
            #   → resolve_call happens after receive_response
            #   → DEADLOCK
            # ═══════════════════════════════════════════════════════════════
            from claude_agent_sdk import TextBlock, ToolUseBlock

            # ───────────────────────────────────────────────────────────
            # Why we return on the FIRST proxy ToolUseBlock, not the last
            # ───────────────────────────────────────────────────────────
            # Claude can emit several ``ToolUseBlock`` items in one
            # ``AssistantMessage`` (or stream them across several events),
            # but the Claude CLI dispatches the corresponding MCP
            # ``control_request`` messages **sequentially** — the next
            # tool's control_request is only sent after it receives the
            # previous tool's ``tool_result``. Consequently, even when N
            # blocks are visible in the message stream, only ONE handler
            # task is actually spawned at a time and only ONE matching
            # entry exists in ``proxy_mcp.pending_calls``.
            #
            # If we tried to wait for all N handlers to register, the
            # second wait would never complete: Claude is waiting for our
            # tool_result for #1 before sending the control_request for
            # #2, and we'd be waiting for #2 to register before returning
            # #1 to LangGraph. Deadlock — exactly the test_dev freeze.
            #
            # Correct semantics: on the first proxy ToolUseBlock, wait
            # event-driven (Future, NO polling) for the corresponding
            # handler to register its pending call, then surface whatever
            # is currently in pending_calls to LangGraph and exit. The
            # next ainvoke iteration (after LangGraph runs the tool and
            # we ``resolve_call`` it) will pick up the next pending call
            # the same way.
            #
            # ─────────────────────────────────────────────────────────
            # A related issue: sequential multi-tool hang fix (2026-06-06)
            # ─────────────────────────────────────────────────────────
            # When Claude emits several ``ToolUseBlock`` items in one
            # ``AssistantMessage``, the CLI dispatches MCP
            # ``control_request`` messages **sequentially** — the next
            # tool's handler only registers after we ``resolve_call``
            # the current one.  The ``AssistantMessage`` containing
            # ALL ToolUseBlocks was already consumed in a previous
            # ``ainvoke`` iteration, so a fresh ``receive_response()``
            # has nothing to yield and ``__anext__()`` hangs forever.
            #
            # Fix: when ``tool_resolved=True`` and no UserMessage was
            # sent (meaning we are in the "return tool calls" loop),
            # check ``proxy_mcp.has_pending_calls()`` before entering
            # ``receive_response()``.  If a pending call already exists
            # (or appears within a brief async wait), skip
            # ``receive_response()`` entirely — the pending call is
            # from a remaining ToolUseBlock in the already-consumed
            # ``AssistantMessage`` and will be returned by section 9.
            # ─────────────────────────────────────────────────────────
            # ─────────────────────────────────────────────────────────
            # POST-RESOLVE DISCRIMINATOR (no polling, no timeouts).
            # After resolving a ToolMessage we have TWO possibilities and
            # there is exactly one source of truth that tells us which:
            #
            #   (a) The previous ``ainvoke`` already-consumed an
            #       ``AssistantMessage`` carrying MULTIPLE
            #       ``ToolUseBlock``s. The handler for the next one of
            #       those blocks has already registered its pending call
            #       (sequential SDK dispatch). ``proxy_mcp.pending_calls``
            #       is non-empty RIGHT NOW. We must skip
            #       ``receive_response()`` — there is nothing left in the
            #       Claude stream until we resolve this next pending
            #       call. Section 9 returns the pending tool_calls.
            #
            #   (b) The just-resolved tool was the LAST one Claude wanted
            #       to call in this turn. The next stream event from
            #       ``receive_response()`` is the final text (and/or
            #       ``ResultMessage``). ``proxy_mcp.pending_calls`` is
            #       empty RIGHT NOW.
            #
            # Discriminator: ``proxy_mcp.has_pending_calls()`` — checked
            # SYNCHRONOUSLY, after ``resolve_call`` has already
            # popped the resolved id (see ``proxy_mcp_server.py``
            # docstring on ``resolve_call`` — POLICY: sync source-of-
            # truth). No ``wait_for_any_pending_call`` here: the
            # absence of a pending call is itself the answer, not a
            # "wait a bit more and see" — see the 2026-06-15 incident
            # (test_claude_code_mcp_prefix hang) and the original
            # tool→text hang. Waiting for "the next handler
            # that might register" is a polling assumption with no
            # finite ground truth: when the turn is over, no handler
            # is coming, ever.
            # ─────────────────────────────────────────────────────────
            skip_receive_response = False
            if tool_resolved and not user_message_sent:
                if proxy_mcp.has_pending_calls():
                    skip_receive_response = True
                    _logger.info(
                        f"[DEBUG ainvoke] Pending calls "
                        f"({len(proxy_mcp.get_pending_calls())}) after "
                        f"tool resolution — skipping receive_response "
                        f"(case (a): more tools in same AssistantMessage)"
                    )
                else:
                    _logger.info(
                        f"[DEBUG ainvoke] No pending calls after tool "
                        f"resolution — entering receive_response for "
                        f"final text (case (b): last tool of the turn)"
                    )

            if not skip_receive_response:
                response_iter = client.receive_response().__aiter__()
                saw_proxy_tool = False
                expected_proxy_tool_call_id: str | None = None
                expected_proxy_tool_name: str | None = None

                while True:
                    try:
                        event = await response_iter.__anext__()
                    except StopAsyncIteration:
                        _logger.info("[DEBUG ainvoke] receive_response stream ended")
                        break

                    if isinstance(event, ClaudeAssistantMessage):
                        # Capture API-error signals off the AssistantMessage
                        # (see DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.3).
                        _err = getattr(event, "error", None)
                        if _err is not None:
                            last_assistant_error = str(_err)
                        _usage = getattr(event, "usage", None)
                        if _usage is not None:
                            last_usage = _usage
                        for block in event.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text
                                _logger.info(f"[DEBUG ainvoke] Received TextBlock: '{block.text[:80]}'")
                            elif isinstance(block, ToolUseBlock):
                                _logger.info(f"[DEBUG ainvoke] Received ToolUseBlock: {block.name}")

                        tool_blocks = [
                            block for block in event.content
                            if isinstance(block, ToolUseBlock)
                        ]
                        # ── Proxy classification: PREFIX-ONLY (do NOT match
                        # on bare base name). ──
                        # Claude Code exposes our MCP tools to the model ONLY
                        # as ``mcp__deepworker__<tool>`` (the canonical MCP
                        # naming — verified, see
                        # docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md). A real proxy
                        # call therefore ALWAYS arrives prefixed. A BARE block
                        # (e.g. ``execute``/``read_file``) is NOT one of our
                        # calls even if its name collides with a base name we
                        # registered — it is a built-in tool or a model
                        # hallucination of a short name; the CLI does NOT route
                        # it through our MCP server, so NO handler / pending
                        # call ever appears for it.
                        #
                        # ⚠️ DO NOT revert to ``_strip_proxy_mcp_prefix(name)
                        # in proxy_tool_names`` (accepts bare names). That
                        # base-name match is what deadlocked a live agent
                        # 2026-06-15: a bare built-in ``execute`` was treated
                        # as our proxy ``execute``; chat_model parked on its
                        # ``tool_use_id`` which never arrives over our control
                        # channel. Identify OUR call ONLY by the prefixed name
                        # + the control-channel tool_use_id.
                        proxy_blocks = [
                            block for block in tool_blocks
                            if block.name.startswith(_PROXY_MCP_TOOL_PREFIX)
                            and _strip_proxy_mcp_prefix(block.name)
                            in proxy_tool_names
                        ]

                        if proxy_blocks:
                            # First proxy ToolUseBlock — break out and let the
                            # event-driven wait below pick up the control-channel
                            # pending call. Do NOT keep iterating; Claude will not
                            # send ResultMessage until we resolve.
                            #
                            # The visible block name/args are captured ONLY for
                            # diagnostics. They are NOT a reliable matching key:
                            # Claude streams the ToolUseBlock in the assistant
                            # message with argument JSON that can differ from the
                            # finalized args delivered to the MCP handler over the
                            # control channel (observed live:
                            # visible grep(pattern='unsubscribe_messages') vs
                            # control-channel grep(pattern='unsubscribe');
                            # visible read_file(limit=30) vs control read_file(
                            # limit=40)). The authoritative call — the one that
                            # gets executed and later resolve_call'd by its
                            # handler-assigned call_id — is whatever the proxy
                            # registers in pending_calls. So we wait for ANY
                            # handler registration, then surface get_pending_calls().
                            first_proxy_block = proxy_blocks[0]
                            expected_proxy_tool_call_id = first_proxy_block.id
                            expected_proxy_tool_name = _strip_proxy_mcp_prefix(
                                first_proxy_block.name
                            )
                            saw_proxy_tool = True
                            _logger.info(f"[DEBUG ainvoke] MCP tool detected — breaking for LangGraph "
                                f"(visible blocks in this AssistantMessage: "
                                f"{[b.name for b in proxy_blocks]})")
                            break
                        if tool_blocks:
                            # Built-in Claude tool (ToolSearch, WebSearch, etc.)
                            # — Claude handles execution internally; keep
                            # consuming events until ResultMessage.
                            _logger.info("[DEBUG ainvoke] Built-in tool — continuing receive_response")
                            continue

                    elif isinstance(event, ResultMessage):
                        _logger.info("[DEBUG ainvoke] Received ResultMessage")
                        # Persist the CLI's session uuid as a restore HINT for the
                        # next restart (authoritative validation is still the
                        # tail-match). DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.4.
                        _sid = getattr(event, "session_id", None)
                        if _sid:
                            self._client_manager.record_session_uuid(thread_id, _sid)
                        # Capture API-error signals off the ResultMessage
                        # (see DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.3).
                        result_is_error = bool(getattr(event, "is_error", False))
                        _sub = getattr(event, "subtype", None)
                        if _sub is not None:
                            result_subtype = str(_sub)
                        _errs = getattr(event, "errors", None)
                        if _errs:
                            try:
                                result_errors = tuple(str(e) for e in _errs)
                            except Exception:
                                result_errors = ()
                        _rt = getattr(event, "result", None)
                        if _rt is not None:
                            result_text = str(_rt)
                        _sr = getattr(event, "stop_reason", None)
                        if _sr is not None:
                            result_stop_reason = str(_sr)
                        _u = getattr(event, "usage", None)
                        if _u is not None and last_usage is None:
                            # Fallback: use ResultMessage usage only if no
                            # AssistantMessage carried it.
                            last_usage = _u
                        # Token-stats: terminal event → exactly one record per
                        # observed ResultMessage (§5.2).
                        await self._emit_usage_event(event)
                        break

                # ── A: event-driven wait FOR THE EXACT id ──
                # After breaking on a proxy ToolUseBlock we wait for the
                # MCP handler to register the pending call with the SAME
                # id (``ToolUseBlock.id`` == ``_meta.claudecode/
                # toolUseId`` propagated through the SDK ContextVar — see
                # POLICIES.md §7 / the 2026-06-14 incident). The wait is
                # event-driven (Future), id-keyed, and HAS NO TIMEOUT.
                #
                # ⚠️ ``ToolUseBlock.id`` here = the per-call ``toolu_*``
                # (the tool_use_id), NOT the message-block id (``msg_*``).
                # Those are different namespaces and are NEVER equal
                # (measured 0/46683 — see docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md).
                # The per-call ``toolu_*`` is the same on the stream and on
                # the MCP control channel, so id-keying is correct — BUT
                # only for OUR proxy call. KNOWN TRAP (deadlocked a live agent
                # 2026-06-15): a built-in Claude tool and our proxy tool can
                # share a base name (``execute`` vs
                # ``mcp__deepworker__execute``); if the visible block of the
                # BUILT-IN tool is misclassified as a proxy block, we wait
                # here on ITS ``toolu_*`` forever — it never arrives over
                # OUR control channel. ``expected_proxy_tool_call_id`` MUST
                # come only from a block whose name actually starts with
                # ``mcp__deepworker__``. A message block can also carry many
                # calls (up to 52) — completion = ALL of them resolved, not
                # just ``proxy_blocks[0]``.
                #
                # ⚠ DO NOT add a timeout here. The previous code wrapped
                # this in ``asyncio.wait_for(..., timeout=5.0)`` and, on
                # timeout, INJECTED THE VISIBLE BLOCK into
                # ``pending_tool_calls`` as a fallback — which gave
                # LangGraph ToolNode a tool_call without a matching MCP
                # pending and produced the "three identical copies of one
                # block" symptom (the real handler arrived later, never
                # got ``resolve_call``-ed, returned an artificial error
                # to Claude after its own 120s phantom timeout, Claude
                # retried, ToolNode executed again).
                #
                # We only reach this wait for a block whose name is
                # ``mcp__deepworker__<tool>`` AND whose stripped name is a
                # tool we registered (prefix-only classification above).
                # The CLI routes exactly those over the control channel, so
                # the matching ``tools/call`` arrives and the id-keyed
                # waiter resolves. A non-routable name (bare/built-in/
                # unknown) never reaches here, so we never park on a call
                # that won't come — this is the structural fix for the
                # 2026-06-15 deadlock (see
                # docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md).
                #
                # NOTE: the earlier claim "the SDK GUARANTEES the
                # control-channel message will arrive" was FALSE — it only
                # holds for OUR prefixed tools, which is exactly what the
                # classification now enforces. As a belt-and-braces escape
                # hatch, ``ProxyMCPServer.cancel_pending_waiter`` +
                # ``parse_no_such_tool`` let any future caller break a wait
                # if the CLI reports ``No such tool available: <name>`` for
                # an id we are holding. If the CLI is genuinely hung (no
                # control request EVER), the operator's per-turn deadline
                # surfaces it as an infrastructure error.
                if (
                    saw_proxy_tool
                    and pending_tool_calls is None
                    and not proxy_mcp.has_pending_calls()
                ):
                    if expected_proxy_tool_call_id is None:
                        # Diagnostic guard (D): a visible ToolUseBlock
                        # MUST have an id. If not, the Claude side
                        # violated its own invariant — raise so the bug
                        # is surfaced, not silently masked.
                        raise RuntimeError(
                            "Claude emitted a proxy MCP ToolUseBlock "
                            "without a tool_use_id. The Claude side "
                            "invariant is broken — escalate, do NOT "
                            "patch around it. See POLICIES.md §7."
                        )
                    # Stage A is BOUNDED by the single source of truth
                    # (variant A). If the control request never arrives, the
                    # waiter raises DispatchTimeout. There is NO pending call to
                    # execute, so we do NOT fabricate a tool_call (that would
                    # run a phantom tool via ToolNode). Instead end the turn
                    # with a NON-error in_progress notice as the assistant text:
                    # the model is informed and continues / retries. Never
                    # hangs, never errors (POLICIES.md §7.1).
                    # See DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.
                    try:
                        await proxy_mcp.wait_for_pending_call_by_id(
                            expected_proxy_tool_call_id
                        )
                    except DispatchTimeout:
                        # No pending call ever registered → no work was
                        # parked. Advertising a job_id here (as we used to)
                        # promised wait_for_job/kill_job for something that
                        # does not exist — the same "no background job"
                        # symptom the 6bd042b registry fix removed for real
                        # parks. Emit the honest not_started notice instead.
                        # See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.
                        from deepagents.middleware.bounded_tool_call import (  # noqa: PLC0415
                            build_dispatch_timeout_notice,
                            resolve_tool_wait_seconds,
                        )
                        response_text = build_dispatch_timeout_notice(
                            tool_name=expected_proxy_tool_name or "tool",
                            wait_seconds=resolve_tool_wait_seconds(),
                        )

        # Usage-cap guard: when the account hits its limit, the CLI returns
        # the marker as the WHOLE turn (no tool calls, no real work). Detect
        # it on a text-only turn and raise so the agent surfaces ERROR rather
        # than passing the useless reply downstream (observed live, 2026-05-21).
        if (
            pending_tool_calls is None
            and not proxy_mcp.has_pending_calls()
            and response_text.strip().startswith(_SESSION_LIMIT_MARKER)
        ):
            raise ClaudeUsageLimitError(response_text.strip())

        # 8.b Guard: an upstream API error must NOT become the agent's answer
        # (No-Silent-API-Errors Policy, docs/POLICIES.md §9). Consult the
        # detector iff we are NOT about to return tool_calls to LangGraph —
        # same authoritative guard as the usage-limit check above. When
        # returning tool_calls, the terminal turn is a tool-dispatch turn
        # and classification is not our concern. See
        # docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.2b, §3.3.
        returning_tool_calls = bool(
            pending_tool_calls or proxy_mcp.has_pending_calls()
        )
        if not returning_tool_calls:
            from claude_code_provider.api_error_detect import detect_api_error  # noqa: PLC0415
            detected = detect_api_error(
                response_text=response_text,
                assistant_error=last_assistant_error,
                result_is_error=result_is_error,
                result_subtype=result_subtype,
                result_errors=result_errors,
                result_text=result_text,
                stop_reason=result_stop_reason,
                usage=last_usage,
            )
            if detected is not None:
                await self._client_manager.rotate_session(
                    thread_id, invalidate_persisted=True
                )
                _logger.warning(
                    "[ainvoke] Claude API error detected on terminal turn — "
                    "raising ClaudeApiResponseError (recoverable): %s",
                    detected.message[:200],
                )
                raise ClaudeApiResponseError(
                    detected.message,
                    assistant_error=detected.assistant_error,
                    result_subtype=detected.result_subtype,
                    result_errors=detected.result_errors,
                    stop_reason=detected.stop_reason,
                )

        # 9. Check for pending tool calls from MCP proxy.
        if pending_tool_calls is not None:
            pending = pending_tool_calls
        elif proxy_mcp.has_pending_calls():
            pending = proxy_mcp.get_pending_calls()
        else:
            pending = []

        if pending:
            # Convert to LangChain tool_calls format
            tool_calls = [
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                }
                for call in pending
            ]

            msg = f"[ainvoke] Returning AIMessage with {len(tool_calls)} tool_calls: {[tc['name'] for tc in tool_calls]}\n"
            sys.stderr.write(msg)
            sys.stderr.flush()

            # IMPORTANT: Include response_text with tool_calls
            # This ensures Claude's explanation/reasoning is not lost
            return AIMessage(content=response_text, tool_calls=tool_calls)

        # 10. Return final answer
        _logger.info(f"[DEBUG ainvoke] Returning AIMessage with content: '{response_text[:80]}'")
        _logger.info(f"[DEBUG ainvoke] ========== END ==========\n")
        return AIMessage(content=response_text)

    async def cleanup_thread(self, thread_id: str) -> None:
        """Cleanup resources for a thread.

        Should be called when thread is no longer needed to free resources.
        Removes sent count tracking and disconnects the Claude client.

        Args:
            thread_id: Thread ID to cleanup
        """
        # Cleanup client and proxy (also removes sent_count tracking)
        await self._client_manager.cleanup_client(thread_id)

    def _convert_to_claude(
        self, message: BaseMessage
    ) -> ClaudeUserMessage | ClaudeAssistantMessage | ClaudeSystemMessage | None:
        """Convert LangChain message to Claude format.

        Args:
            message: LangChain message to convert

        Returns:
            Claude message or None if message should be skipped
        """
        if isinstance(message, HumanMessage):
            return ClaudeUserMessage(content=str(message.content))
        elif isinstance(message, SystemMessage):
            # SystemMessage requires subtype and data fields
            return ClaudeSystemMessage(subtype="system", data={"content": str(message.content)})
        elif isinstance(message, AIMessage):
            # Only convert AIMessage without tool_calls
            # (AIMessage with tool_calls should be skipped in ainvoke)
            if not message.tool_calls:
                from claude_agent_sdk import TextBlock

                return ClaudeAssistantMessage(
                    content=[TextBlock(text=str(message.content))],
                    model="claude-sonnet-4-5-20250929",
                )
        return None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate method (required by BaseChatModel).

        Not implemented - use ainvoke instead.

        Raises:
            NotImplementedError: This model only supports async operations
        """
        raise NotImplementedError("ClaudeCodeChatModel only supports async operations. Use ainvoke().")

    @property
    def _llm_type(self) -> str:
        """Return type of language model.

        Returns:
            String identifier for this model type
        """
        return "claude-code"
