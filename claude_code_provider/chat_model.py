"""ClaudeCodeChatModel - LangChain wrapper around Claude Code SDK.

This module provides a LangChain BaseChatModel implementation that uses Claude Code SDK
(ClaudeSDKClient) as the underlying AI provider. It handles:
- Message conversion between LangChain and Claude formats
- Tool call delegation to deep-worker via ProxyMCPServer
- Session management via ClientManager (thread_id isolation)
- Async message streaming from Claude
"""

import asyncio  # type: ignore[reportUnusedImport] — used in async methods
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


class ClaudeUsageLimitError(RuntimeError):
    """Raised when the Claude CLI reports the account's usage/session limit.

    The CLI surfaces an exhausted usage cap as a single degenerate response
    whose entire text is ``You've hit your session limit · resets <time>``.
    It is NOT a real answer. Returning it as ordinary ``AIMessage`` content
    made the agent fail silently (observed 2026-05-21): the TaskManager kept
    "correcting" the useless reply, the agent settled to idle, and the web
    UI showed no error. Raising a typed error instead lets the agent process
    transition to ``status=ERROR`` with this human-readable message.
    """


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

        Extracts metadata from LangChain tools and returns a new model instance
        with these tools configured. This follows LangChain's pattern of returning
        new instances rather than mutating existing ones.

        Args:
            tools: List of LangChain tools to bind (only BaseTool instances supported)
            **kwargs: Additional arguments (unused)

        Returns:
            New ClaudeCodeChatModel instance with tools bound
        """
        import sys
        msg = f"[ClaudeCodeChatModel.bind_tools] Called with {len(tools)} tools\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        # Extract metadata from LangChain tools
        tools_metadata: list[ToolMetadata] = []
        for tool_item in tools:
            # Only process BaseTool instances
            if not isinstance(tool_item, BaseTool):
                continue

            tool: BaseTool = tool_item

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
            if has_prior_turns and self._has_persisted_session(thread_id):
                from claude_code_provider.session_restore import (  # noqa: PLC0415
                    resolve_restore_session,
                )
                resume_session_id = resolve_restore_session(
                    thread_id, restorable, cwd=self._working_dir or "/"
                )

        # Get or create client for this thread (resume if we resolved one).
        client, proxy_mcp = await self._client_manager.get_or_create_client(
            thread_id=thread_id,
            tools_metadata=self._tools_metadata,
            working_dir=self._working_dir,
            system_prompt=system_prompt_content,
            resume_session_id=resume_session_id,
        )

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
            print(f"[DEBUG] Failed to write debug log: {e}", file=sys.stderr)

        # ALSO print to stderr
        print(f"\n[{timestamp}] ========== START AINVOKE ==========", file=sys.stderr, flush=True)
        print(f"[{timestamp}] thread_id={thread_id}, sent_count={sent_count}, len(messages)={len(messages)}", file=sys.stderr, flush=True)

        print(f"\n[DEBUG ainvoke] ========== START ==========", file=sys.stderr)
        print(f"[DEBUG ainvoke] Model instance id={id(self)}", file=sys.stderr)
        print(f"[DEBUG ainvoke] ClientManager id={id(self._client_manager)}", file=sys.stderr)
        print(f"[DEBUG ainvoke] thread_id={thread_id}", file=sys.stderr)
        print(f"[DEBUG ainvoke] sent_count={sent_count}", file=sys.stderr)
        print(f"[DEBUG ainvoke] _sent_count state={dict(self._client_manager._sent_count)}", file=sys.stderr)
        print(f"[DEBUG ainvoke] len(messages)={len(messages)}", file=sys.stderr)
        for i, msg in enumerate(messages):
            content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
            print(f"[DEBUG ainvoke]   messages[{i}]: {type(msg).__name__} - {content_preview}", file=sys.stderr)

        # ⚠️ CRITICAL: Detect LangGraph summarization (message history reset).
        # When SummarizationMiddleware replaces N messages with a summary,
        # sent_count (from the old history) exceeds len(messages) (new, shorter).
        # We MUST rotate to a new Claude CLI session_id — otherwise the CLI
        # session still holds the full old history and will eventually hit
        # "Prompt is too long" (observed after 1255 turns).
        # DO NOT remove this rotation — it is the ONLY mechanism that prevents
        # Claude CLI session overflow for long-running agents.
        if sent_count > len(messages):
            new_session_id = self._client_manager.rotate_session(thread_id)
            print(
                f"[DEBUG ainvoke] SESSION ROTATED: {thread_id} → {new_session_id} "
                f"(old sent_count={sent_count}, new messages={len(messages)})",
                file=sys.stderr,
            )
            sent_count = 0

        # 5. Extract ONLY new messages (not yet sent to Claude)
        new_messages = messages[sent_count:]

        print(f"[DEBUG ainvoke] len(new_messages)={len(new_messages)}", file=sys.stderr)
        for i, msg in enumerate(new_messages):
            content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
            print(f"[DEBUG ainvoke]   new_messages[{i}]: {type(msg).__name__} - {content_preview}", file=sys.stderr)

        # 6. Send each new message to Claude
        user_message_sent = False
        tool_resolved = False

        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                # Send user message to Claude
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # ⚠️ Use effective_session_id (rotated after summarization),
                # NOT raw thread_id. See rotate_session() for details.
                effective_sid = self._client_manager.get_effective_session_id(thread_id)
                print(f"[DEBUG ainvoke] Sending HumanMessage to Claude:", file=sys.stderr)
                print(f"[DEBUG ainvoke]   content='{content}'", file=sys.stderr)
                print(f"[DEBUG ainvoke]   session_id='{effective_sid}'", file=sys.stderr)
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
                print(f"[DEBUG ainvoke] resolve_call for tool_call_id={msg.tool_call_id}", file=sys.stderr)
                proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)
                tool_resolved = True

            elif isinstance(msg, SystemMessage):
                # ⚠️ CRITICAL: DO NOT REMOVE THIS BRANCH!
                # SystemMessage contains middleware cognitive hints (e.g. project_memory
                # instructions from ProjectMemoryMiddleware). The content is already
                # forwarded to Claude SDK via system_prompt in get_or_create_client().
                # We skip it here to avoid sending it as a user message.
                # If this branch is removed, SystemMessage falls through to the
                # default case and could be sent incorrectly.
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

        print(f"[DEBUG ainvoke] user_message_sent={user_message_sent}, tool_resolved={tool_resolved}", file=sys.stderr)

        if user_message_sent or tool_resolved:
            print(f"[DEBUG ainvoke] Calling client.receive_response() [EVENT-BASED]...", file=sys.stderr)

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
            # #1 to LangGraph. Deadlock — exactly the freeze observed in live testing.
            #
            # Correct semantics: on the first proxy ToolUseBlock, wait
            # event-driven (Future, NO polling) for the corresponding
            # handler to register its pending call, then surface whatever
            # is currently in pending_calls to LangGraph and exit. The
            # next ainvoke iteration (after LangGraph runs the tool and
            # we ``resolve_call`` it) will pick up the next pending call
            # the same way.
            response_iter = client.receive_response().__aiter__()
            saw_proxy_tool = False
            expected_proxy_tool_call_id: str | None = None
            expected_proxy_tool_name: str | None = None
            expected_proxy_args: dict[str, Any] | None = None

            while True:
                try:
                    event = await response_iter.__anext__()
                except StopAsyncIteration:
                    print("[DEBUG ainvoke] receive_response stream ended", file=sys.stderr)
                    break

                if isinstance(event, ClaudeAssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            print(f"[DEBUG ainvoke] Received TextBlock: '{block.text[:80]}'", file=sys.stderr)
                        elif isinstance(block, ToolUseBlock):
                            print(f"[DEBUG ainvoke] Received ToolUseBlock: {block.name}", file=sys.stderr)

                    tool_blocks = [
                        block for block in event.content
                        if isinstance(block, ToolUseBlock)
                    ]
                    # Claude SDK prefixes proxy MCP tool names with
                    # ``mcp__deepworker__`` (see ``_PROXY_MCP_TOOL_PREFIX``
                    # at module top). Strip the prefix before comparing
                    # against the bare names registered in ``ToolMetadata``,
                    # otherwise every proxy tool is misclassified as a
                    # built-in Claude tool and LangGraph never receives
                    # the tool_call → agent hang.
                    proxy_blocks = [
                        block for block in tool_blocks
                        if _strip_proxy_mcp_prefix(block.name) in proxy_tool_names
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
                        # control channel (live observation 2026-05-26:
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
                        expected_proxy_args = dict(first_proxy_block.input or {})
                        saw_proxy_tool = True
                        print(
                            f"[DEBUG ainvoke] MCP tool detected — breaking for LangGraph "
                            f"(visible blocks in this AssistantMessage: "
                            f"{[b.name for b in proxy_blocks]})",
                            file=sys.stderr,
                        )
                        break
                    if tool_blocks:
                        # Built-in Claude tool (ToolSearch, WebSearch, etc.)
                        # — Claude handles execution internally; keep
                        # consuming events until ResultMessage.
                        print("[DEBUG ainvoke] Built-in tool — continuing receive_response", file=sys.stderr)
                        continue

                elif isinstance(event, ResultMessage):
                    print("[DEBUG ainvoke] Received ResultMessage", file=sys.stderr)
                    # Persist the CLI's session uuid as a restore HINT for the
                    # next restart (authoritative validation is still the
                    # tail-match). DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.4.
                    _sid = getattr(event, "session_id", None)
                    if _sid:
                        self._client_manager.record_session_uuid(thread_id, _sid)
                    break

            # After breaking on a proxy ToolUseBlock, wait event-driven
            # (Future, NOT polling) for the MCP handler task spawned by
            # the SDK control channel to register its pending call. The
            # ToolUseBlock arrives via the message stream BEFORE the
            # corresponding control_request reaches the SDK control
            # channel, so a tiny registration race is normal and the wait
            # below typically resolves in milliseconds. The 30s timeout
            # only bounds this registration race, NOT actual tool
            # execution (handlers block on ``self.timeout`` once
            # registered — see ``ProxyMCPServer._create_tool_handler``).
            #
            # Recovery policy: if the wait times out, the Claude CLI
            # subprocess itself is hung (evidence: observed 2026-05-19
            # — 90+ s with no control_request delivered, Python side
            # healthy, no 429/rate-limit, no API errors). Retrying
            # in-place against the same subprocess is futile. Instead
            # kill the stuck client and raise; the next ainvoke spawns
            # a fresh Claude CLI subprocess and resumes the same
            # session_id via the rotation tracking in ClientManager.
            if (
                saw_proxy_tool
                and pending_tool_calls is None
                and not proxy_mcp.has_pending_calls()
            ):
                try:
                    # Wait for ANY handler to register a pending call. We
                    # cannot match on the visible block's args (they may
                    # diverge from the control-channel args — see capture
                    # comment above), so the authoritative pending call(s)
                    # are read from get_pending_calls() afterwards.
                    await proxy_mcp.wait_for_any_pending_call(timeout=30.0)
                except asyncio.TimeoutError as exc:
                    print(
                        f"[WARN ainvoke] wait_for_any_pending_call timed out "
                        f"after 30s for thread_id={thread_id}; visible proxy "
                        f"block tool={expected_proxy_tool_name!r} args="
                        f"{expected_proxy_args!r}, but no MCP handler registered "
                        f"a pending call; returning the visible block as a "
                        f"normal tool_call fallback.",
                        file=sys.stderr,
                    )
                    if (
                        expected_proxy_tool_call_id is not None
                        and expected_proxy_tool_name is not None
                        and expected_proxy_args is not None
                    ):
                        pending_tool_calls = [
                            {
                                "id": expected_proxy_tool_call_id,
                                "name": expected_proxy_tool_name,
                                "args": expected_proxy_args,
                            }
                        ]
                    else:
                        raise RuntimeError(
                            "Claude emitted a proxy MCP ToolUseBlock, but no "
                            "MCP handler registered a pending call and the "
                            "visible block could not be reconstructed."
                        ) from exc

        # Usage-cap guard: when the account hits its limit, the CLI returns
        # the marker as the WHOLE turn (no tool calls, no real work). Detect
        # it on a text-only turn and raise so the agent surfaces ERROR rather
        # than passing the useless reply downstream (observed 2026-05-21).
        if (
            pending_tool_calls is None
            and not proxy_mcp.has_pending_calls()
            and response_text.strip().startswith(_SESSION_LIMIT_MARKER)
        ):
            raise ClaudeUsageLimitError(response_text.strip())

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
        print(f"[DEBUG ainvoke] Returning AIMessage with content: '{response_text[:80]}'", file=sys.stderr)
        print(f"[DEBUG ainvoke] ========== END ==========\n", file=sys.stderr)
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
