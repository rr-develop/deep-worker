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

        # 4. Get or create client for this thread
        client, proxy_mcp = await self._client_manager.get_or_create_client(
            thread_id=thread_id,
            tools_metadata=self._tools_metadata,
            working_dir=self._working_dir,
            system_prompt=system_prompt_content,
        )

        # 4. Determine how many messages have already been sent
        # Get from ClientManager (not self) because LangGraph may create new model instances
        sent_count = self._client_manager.get_sent_count(thread_id)

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
        # "Prompt is too long" (observed in practice after ~1255 turns).
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
        got_tool_use = False

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

            got_tool_use = False

            async for event in client.receive_response():
                if isinstance(event, ClaudeAssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            print(f"[DEBUG ainvoke] Received TextBlock: '{block.text[:80]}'", file=sys.stderr)
                        elif isinstance(block, ToolUseBlock):
                            got_tool_use = True
                            print(f"[DEBUG ainvoke] Received ToolUseBlock: {block.name}", file=sys.stderr)

                    if got_tool_use:
                        # ToolUseBlock detected. Need to determine if this is:
                        # A) MCP tool (our proxy) — must break for LangGraph
                        # B) Built-in Claude tool — Claude handles internally
                        #
                        # Check: yield control multiple times so MCP handler task
                        # (spawned by Claude SDK via asyncio.create_task) can
                        # receive the control_request and populate pending_calls.
                        # The control_request may arrive AFTER ToolUseBlock in
                        # the message stream because they go through different
                        # channels (messages vs control protocol).
                        import asyncio
                        for _ in range(10):  # Up to 10ms
                            await asyncio.sleep(0.001)
                            if proxy_mcp.has_pending_calls():
                                break

                        if proxy_mcp.has_pending_calls():
                            # ⚠️ MCP tool — break to return tool_calls to LangGraph!
                            # Claude CLI waits for MCP handler to return result.
                            # We MUST break here to avoid circular dependency.
                            print(f"[DEBUG ainvoke] MCP tool detected — breaking for LangGraph", file=sys.stderr)
                            break
                        else:
                            # Built-in Claude tool (ToolSearch, WebSearch, etc.) —
                            # Claude CLI handles execution internally.
                            # Continue receiving — ResultMessage will come.
                            print(f"[DEBUG ainvoke] Built-in tool — continuing receive_response", file=sys.stderr)
                            got_tool_use = False  # Reset for next potential tool

                elif isinstance(event, ResultMessage):
                    print(f"[DEBUG ainvoke] Received ResultMessage", file=sys.stderr)
                    break

        # 9. Check for pending tool calls from MCP proxy.
        # After breaking on ToolUseBlock, the MCP handler task (spawned by
        # Claude SDK via asyncio.create_task) may not have populated
        # pending_calls yet. Yield control to let the task run.
        import asyncio
        if not proxy_mcp.has_pending_calls() and got_tool_use:
            await asyncio.sleep(0)  # Yield to let MCP handler task create Future
        if proxy_mcp.has_pending_calls():
            pending = proxy_mcp.get_pending_calls()
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
