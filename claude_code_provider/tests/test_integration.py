"""Integration tests for ClaudeCodeChatModel with full workflow.

These tests verify end-to-end functionality with mock ClaudeSDKClient,
testing the complete flow from LangChain messages through tool calls to final responses.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, TextBlock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from claude_code_provider.chat_model import ClaudeCodeChatModel


# Test tools for integration tests
@langchain_tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@langchain_tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@langchain_tool("read_file")
def read_file_tool(path: str, limit: int) -> str:
    """Read a file (test stub)."""
    return f"read {path} limit={limit}"


class TestIntegrationSimpleConversation:
    """Integration tests for simple conversation flow without tools."""

    @pytest.mark.asyncio
    async def test_simple_conversation_flow(self) -> None:
        """Should handle simple conversation from start to finish."""
        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        model = ClaudeCodeChatModel(base_options=options)

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        async def mock_receive() -> Any:
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Hello! I'm doing great, thanks for asking!")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test-1",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Hello! How are you?")],
                config={"configurable": {"thread_id": "test-1"}},
            )

        # Verify result
        assert isinstance(result, AIMessage)
        content = result.content if isinstance(result.content, str) else str(result.content)
        assert "doing great" in content.lower()
        assert len(result.tool_calls) == 0

        # Verify client was used correctly
        mock_client.query.assert_called_once()
        mock_client.connect.assert_called_once()


class TestIntegrationToolCalls:
    """Integration tests for tool call workflow."""

    @pytest.mark.asyncio
    async def test_complete_tool_call_cycle(self) -> None:
        """Should handle complete tool call cycle: request → execution → response."""
        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        model = ClaudeCodeChatModel(base_options=options)
        model = model.bind_tools([add_numbers])

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        # First call - Claude requests tool use
        async def mock_receive_first() -> Any:
            return
            yield  # Make it a generator but return immediately

        # Second call - Claude provides final answer
        async def mock_receive_second() -> Any:
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="The sum of 15 and 27 is 42.")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=2,
                session_id="test-2",
            )

        # First invocation - tool call
        mock_client.receive_response = mock_receive_first

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=True)
        mock_proxy.get_pending_calls = Mock(
            return_value=[{"id": "call_1", "name": "add_numbers", "args": {"a": 15, "b": 27}}]
        )
        mock_proxy.resolve_call = Mock()

        # Create separate patches: one for ClaudeSDKClient, one for get_or_create_client
        with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
            with patch.object(
                model.client_manager,
                "get_or_create_client",
                return_value=(mock_client, mock_proxy),
            ):
                # Step 1: Model requests tool call
                result1 = await model.ainvoke(
                    [HumanMessage(content="What is 15 + 27?")],
                    config={"configurable": {"thread_id": "test-2"}},
                )

                assert isinstance(result1, AIMessage)
                assert len(result1.tool_calls) == 1
                assert result1.tool_calls[0]["name"] == "add_numbers"
                assert result1.tool_calls[0]["args"] == {"a": 15, "b": 27}

                # Step 2: Execute tool (in LangGraph, this would be done by ToolNode).
                # The real ClientManager caches the client after Step 1; keep
                # this mock integration faithful so the ToolMessage round-trip
                # is not mistaken for a restart restore.
                model.client_manager.clients["test-2"] = (mock_client, mock_proxy)
                tool_result = add_numbers.invoke(result1.tool_calls[0]["args"])
                assert tool_result == 42.0

                # Step 3: Send tool result back and get final answer
                mock_client.receive_response = mock_receive_second
                mock_proxy.has_pending_calls = Mock(return_value=False)

                result2 = await model.ainvoke(
                    [
                        HumanMessage(content="What is 15 + 27?"),
                        result1,  # AIMessage with tool_calls
                        ToolMessage(content="42.0", tool_call_id="call_1"),
                    ],
                    config={"configurable": {"thread_id": "test-2"}},
                )

                # Verify final answer
                assert isinstance(result2, AIMessage)
                assert "42" in result2.content
                assert len(result2.tool_calls) == 0

                # ⚠️ PROXY: Only HumanMessage uses query. ToolMessage uses resolve_call.
                assert mock_client.query.call_count >= 1  # HumanMessage only

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self) -> None:
        """Should handle multiple tool calls in parallel."""
        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([add_numbers, multiply_numbers])

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        async def mock_receive() -> Any:
            return
            yield

        mock_client.receive_response = mock_receive

        # Mock multiple pending calls
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=True)
        mock_proxy.get_pending_calls = Mock(
            return_value=[
                {"id": "call_1", "name": "add_numbers", "args": {"a": 5, "b": 3}},
                {"id": "call_2", "name": "multiply_numbers", "args": {"a": 4, "b": 6}},
            ]
        )

        with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
            with patch.object(
                model.client_manager,
                "get_or_create_client",
                return_value=(mock_client, mock_proxy),
            ):
                result = await model.ainvoke(
                    [HumanMessage(content="Calculate 5+3 and 4*6")],
                    config={"configurable": {"thread_id": "test-3"}},
                )

        # Should return both tool calls
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add_numbers"
        assert result.tool_calls[1]["name"] == "multiply_numbers"


class TestIntegrationParallelSessions:
    """Integration tests for parallel session handling."""

    @pytest.mark.asyncio
    async def test_parallel_sessions_isolated(self) -> None:
        """Should handle multiple sessions in parallel with proper isolation."""
        model = ClaudeCodeChatModel(base_options=None)

        # Create separate mock clients for each session
        mock_client1 = Mock()
        mock_client1.query = AsyncMock()
        mock_client1.connect = AsyncMock()
        mock_client1.disconnect = AsyncMock()

        mock_client2 = Mock()
        mock_client2.query = AsyncMock()
        mock_client2.connect = AsyncMock()
        mock_client2.disconnect = AsyncMock()

        async def mock_receive1() -> Any:
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Session 1 response")], model="claude-sonnet-4-5-20250929"
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="session-1",
            )

        async def mock_receive2() -> Any:
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Session 2 response")], model="claude-sonnet-4-5-20250929"
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="session-2",
            )

        mock_client1.receive_response = mock_receive1
        mock_client2.receive_response = mock_receive2

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        # Track which client is returned
        client_counter = {"count": 0}

        def client_factory(*args: Any, **kwargs: Any) -> Mock:
            client_counter["count"] += 1
            return mock_client1 if client_counter["count"] == 1 else mock_client2

        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient", side_effect=client_factory
        ):
            # Run both sessions in parallel
            task1 = model.ainvoke(
                [HumanMessage(content="Test 1")], config={"configurable": {"thread_id": "session-1"}}
            )
            task2 = model.ainvoke(
                [HumanMessage(content="Test 2")], config={"configurable": {"thread_id": "session-2"}}
            )

            results = await asyncio.gather(task1, task2)

        # Verify results are different and correct
        assert "Session 1" in results[0].content
        assert "Session 2" in results[1].content

        # Verify both clients were used
        mock_client1.query.assert_called_once()
        mock_client2.query.assert_called_once()


class TestIntegrationErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_claude_errors_gracefully(self) -> None:
        """Should handle errors from Claude gracefully."""
        model = ClaudeCodeChatModel(base_options=None)

        # Mock client that returns error
        mock_client = Mock()
        mock_client.query = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        async def mock_receive() -> Any:
            yield ResultMessage(
                subtype="error",
                duration_ms=100,
                duration_api_ms=50,
                is_error=True,
                num_turns=0,
                session_id="test-error",
                result="API Error",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Test")], config={"configurable": {"thread_id": "test-error"}}
            )

        # Should return AIMessage even on error (empty content is OK)
        assert isinstance(result, AIMessage)


class TestIntegrationMessageConversions:
    """Integration tests for message format conversions."""

    @pytest.mark.asyncio
    async def test_converts_all_message_types(self) -> None:
        """Should correctly convert all LangChain message types."""
        from langchain_core.messages import SystemMessage

        model = ClaudeCodeChatModel(base_options=None)

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        async def mock_receive() -> Any:
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Response")], model="claude-sonnet-4-5-20250929"
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test-convert",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch(
            "claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client
        ):
            # Send mix of message types
            result = await model.ainvoke(
                [
                    SystemMessage(content="You are a helpful assistant"),
                    HumanMessage(content="Hello"),
                ],
                config={"configurable": {"thread_id": "test-convert"}},
            )

        # Should succeed
        assert isinstance(result, AIMessage)
        assert result.content == "Response"


class TestIntegrationDivergentVisibleVsControlArgs:
    """End-to-end against a REAL ProxyMCPServer (no mocked proxy).

    Reproduces the production failure observed on a live agent (2026-05-26): the visible
    message-stream ToolUseBlock and the actual MCP control-channel call are
    the same logical call, but Claude streams slightly different argument
    JSON in the assistant message than the finalized args delivered to the
    MCP handler. With strict (tool_name, args) matching this dead-locked for
    30s and surfaced as agent `error`. The fix waits for ANY handler
    registration and then surfaces the authoritative control-channel call.
    """

    @pytest.mark.asyncio
    async def test_divergent_args_surfaces_control_channel_call(self) -> None:
        from claude_agent_sdk import ToolUseBlock

        from claude_code_provider.custom_types import ToolMetadata
        from claude_code_provider.proxy_mcp_server import ProxyMCPServer

        tools_metadata: list[ToolMetadata] = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            }
        ]
        # REAL proxy — its handler registers the authoritative pending call.
        proxy = ProxyMCPServer(tools_metadata)
        real_handler = proxy.tool_handlers[0].handler

        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([read_file_tool])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        handler_task: dict[str, Any] = {}

        async def mock_receive() -> Any:
            # The SDK control channel runs the MCP handler in a SEPARATE task,
            # concurrently with stream consumption. It invokes the handler with
            # the FINALIZED args: limit=40 (different from the visible block).
            handler_task["task"] = asyncio.create_task(
                real_handler({"path": "/tmp/x.py", "limit": 40})
            )
            # Visible block carries the streamed (earlier) args: limit=30.
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_read",
                        name="mcp__deepworker__read_file",
                        input={"path": "/tmp/x.py", "limit": 30},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, proxy),
        ):
            result = await asyncio.wait_for(
                model.ainvoke(
                    [HumanMessage(content="read the file")],
                    config={"configurable": {"thread_id": "test-divergent-e2e"}},
                ),
                timeout=5.0,  # MUST NOT hit the 30s control-channel timeout.
            )

        # ainvoke must surface the AUTHORITATIVE control-channel call (limit=40),
        # NOT the visible block's args (limit=30), and must NOT deadlock/error.
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["args"] == {"path": "/tmp/x.py", "limit": 40}

        # Resolve the real handler so its task completes cleanly.
        for call in proxy.get_pending_calls():
            proxy.resolve_call(
                call["id"], {"content": [{"type": "text", "text": "ok"}]}
            )
        await handler_task["task"]

