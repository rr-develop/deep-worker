"""ClaudeCodeChatModel - LangChain wrapper around Claude Code SDK.

This module provides a LangChain BaseChatModel implementation that uses Claude Code SDK
(ClaudeSDKClient) as the underlying AI provider. It handles:
- Message conversion between LangChain and Claude formats
- Tool call delegation to deep-worker via ProxyMCPServer
- Session management via ClientManager (thread_id isolation)
- Async message streaming from Claude
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage as ClaudeAssistantMessage,
    ClaudeAgentOptions,
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

from claude_code_provider.client_manager import ClientManager  # noqa: E402
from claude_code_provider.custom_types import MCPResult, ToolMetadata  # noqa: E402


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
        client_manager: Manages ClaudeSDKClient instances per thread_id
        tools_metadata: Metadata for tools to expose through MCP
        base_options: Base options for ClaudeSDKClient

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

    # Type annotations for pyright (these are set in __init__ via object.__setattr__)
    if TYPE_CHECKING:
        base_options: ClaudeAgentOptions | None
        client_manager: ClientManager
        tools_metadata: list[ToolMetadata]

    def __init__(
        self,
        base_options: ClaudeAgentOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ClaudeCodeChatModel.

        Args:
            base_options: Base options to use for all ClaudeSDKClient instances.
                         If None, defaults will be used.
            **kwargs: Additional arguments passed to BaseChatModel
        """
        super().__init__(**kwargs)
        # Set attributes using object.__setattr__ to bypass Pydantic
        object.__setattr__(self, "base_options", base_options)
        object.__setattr__(self, "client_manager", ClientManager(base_options=base_options))
        object.__setattr__(self, "tools_metadata", [])

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
        # Extract metadata from LangChain tools
        tools_metadata: list[ToolMetadata] = []
        for tool_item in tools:
            # Only process BaseTool instances
            if not isinstance(tool_item, BaseTool):
                continue

            tool: BaseTool = tool_item

            # Get schema using Pydantic V2 method if available
            if tool.args_schema:
                if hasattr(tool.args_schema, "model_json_schema") and callable(
                    getattr(tool.args_schema, "model_json_schema")
                ):
                    schema = tool.args_schema.model_json_schema()  # type: ignore[attr-defined]
                elif hasattr(tool.args_schema, "schema") and callable(getattr(tool.args_schema, "schema")):
                    schema = tool.args_schema.schema()  # type: ignore[attr-defined]
                else:
                    schema = {}
            else:
                schema = {}

            metadata: ToolMetadata = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            }
            tools_metadata.append(metadata)

        # Create new instance with tools
        new_model = ClaudeCodeChatModel(base_options=self.base_options)
        object.__setattr__(new_model, "tools_metadata", tools_metadata)

        return new_model

    async def ainvoke(
        self,
        input: LanguageModelInput,  # BaseChatModel uses LanguageModelInput type
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model asynchronously.

        This is the main method for interacting with the model. It:
        1. Extracts thread_id from config
        2. Gets or creates ClaudeSDKClient for this thread
        3. Converts LangChain messages to Claude format
        4. Sends query to Claude
        5. Processes response stream
        6. Returns AIMessage (with tool_calls if pending, or final content)

        Args:
            input: Language model input (messages, prompt, or string)
            config: Runnable config with thread_id in configurable.thread_id
            **kwargs: Additional arguments (unused)

        Returns:
            AIMessage with either:
            - tool_calls: If Claude wants to call tools
            - content: If Claude has final answer
        """
        # Convert input to messages
        from langchain_core.prompt_values import PromptValue

        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)  # type: ignore[arg-type]

        # Extract thread_id from config
        thread_id = "default"
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id", "default")

        # Get or create client for this thread
        client, proxy_mcp = await self.client_manager.get_or_create_client(
            thread_id=thread_id, tools_metadata=self.tools_metadata
        )

        # Process ToolMessage if present (resolve pending calls)
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Create MCP result from ToolMessage
                mcp_result: MCPResult = {
                    "content": [{"type": "text", "text": str(msg.content)}]
                }
                proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)

        # Convert messages to Claude format (skip AIMessage with tool_calls)
        claude_messages: list[ClaudeUserMessage | ClaudeAssistantMessage | ClaudeSystemMessage] = []
        for msg in messages:
            # Skip AIMessage with tool_calls (these were already sent to Claude)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                continue
            # Skip ToolMessage (already processed above)
            if isinstance(msg, ToolMessage):
                continue

            claude_msg = self._convert_to_claude(msg)
            if claude_msg is not None:
                claude_messages.append(claude_msg)

        # Send query to Claude if we have new messages
        if claude_messages:
            # Note: query() signature expects str | AsyncIterable[dict], but accepts message objects in practice
            await client.query(claude_messages)  # type: ignore[arg-type]

        # Process response stream
        response_text = ""
        async for event in client.receive_response():
            if isinstance(event, ClaudeAssistantMessage):
                # Extract text from AssistantMessage content blocks
                from claude_agent_sdk import TextBlock

                for block in event.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
            elif isinstance(event, ResultMessage):
                # End of response
                pass

        # Check for pending tool calls
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
            return AIMessage(content="", tool_calls=tool_calls)

        # Return final answer
        return AIMessage(content=response_text)

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
