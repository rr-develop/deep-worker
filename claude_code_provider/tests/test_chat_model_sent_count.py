"""Tests for sent_count tracking in ClaudeCodeChatModel."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage, TextBlock, ResultMessage

from claude_code_provider.chat_model import ClaudeCodeChatModel


class TestSentCountTracking:
    """Tests for proper message mirroring via sent_count tracking."""

    @pytest.mark.asyncio
    async def test_tracks_sent_messages(self) -> None:
        """Should track sent messages and send only new ones."""
        model = ClaudeCodeChatModel()

        # Reset sent_count for this test (global singleton may have state from other tests)
        test_thread_id = "test"
        if test_thread_id in model.client_manager._sent_count:
            del model.client_manager._sent_count[test_thread_id]

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="4")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            # First call - send 1 message
            result = await model.ainvoke(
                [HumanMessage("2+2")],
                config={"configurable": {"thread_id": "test"}}
            )

        # Check sent_count - should equal len(messages) passed to ainvoke
        assert model.client_manager._sent_count["test"] == 1  # Only HumanMessage from input
        assert mock_client.query.call_count == 1
        mock_client.query.assert_called_with("2+2", session_id="test")

        # Second call - send only NEW message
        mock_client.query.reset_mock()

        async def mock_receive_2():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="6")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=2,
                session_id="test",
            )

        mock_client.receive_response = mock_receive_2

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [
                    HumanMessage("2+2"),  # Old
                    AIMessage("4"),       # Old
                    HumanMessage("3+3")   # NEW!
                ],
                config={"configurable": {"thread_id": "test"}}
            )

        # Check sent_count updated - should equal len(messages) passed to ainvoke
        assert model.client_manager._sent_count["test"] == 3  # All 3 messages from input
        assert mock_client.query.call_count == 1  # ONLY 1 new query
        mock_client.query.assert_called_with("3+3", session_id="test")  # NEW message
        assert "6" in result.content

    @pytest.mark.asyncio
    async def test_sent_count_isolation_between_threads(self) -> None:
        """Should isolate sent_count between different thread_ids."""
        model = ClaudeCodeChatModel()

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Response")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            # Thread 1
            await model.ainvoke(
                [HumanMessage("Hello from thread 1")],
                config={"configurable": {"thread_id": "thread-1"}}
            )

            # Thread 2
            await model.ainvoke(
                [HumanMessage("Hello from thread 2")],
                config={"configurable": {"thread_id": "thread-2"}}
            )

        # Check sent_count for each thread - should equal len(messages) for each
        assert model.client_manager._sent_count["thread-1"] == 1  # 1 message in first call
        assert model.client_manager._sent_count["thread-2"] == 1  # 1 message in second call
        assert mock_client.query.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_thread_removes_sent_count(self) -> None:
        """Should remove sent_count when thread is cleaned up."""
        model = ClaudeCodeChatModel()

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Response")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            # Send message
            await model.ainvoke(
                [HumanMessage("Test")],
                config={"configurable": {"thread_id": "test"}}
            )

        # Check sent_count exists
        assert "test" in model.client_manager._sent_count

        # Cleanup - need to disconnect client first before cleanup
        # Mock the disconnect to avoid actual cleanup operations
        if "test" in model.client_manager.clients:
            client, _ = model.client_manager.clients["test"]
            with patch.object(client, "disconnect", new=AsyncMock()):
                await model.cleanup_thread("test")
        else:
            await model.cleanup_thread("test")

        # Check sent_count removed
        assert "test" not in model.client_manager._sent_count

    @pytest.mark.asyncio
    async def test_handles_tool_messages_without_resending(self) -> None:
        """Should handle ToolMessage without resending old messages."""
        model = ClaudeCodeChatModel()

        # Mock client
        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            # No response when only ToolMessage is sent
            return
            yield

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)
        mock_proxy.resolve_call = Mock()

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            # First: send user message (sent_count = 0)
            # Simulate that sent_count is already 1 (HumanMessage was sent before)
            model.client_manager._sent_count["test"] = 1

            # Second: send ToolMessage (sent_count = 1)
            await model.ainvoke(
                [
                    HumanMessage("Calculate 5+3"),  # idx 0 (already sent)
                    AIMessage(content="", tool_calls=[{"id": "call_1", "name": "calc", "args": {}}]),  # idx 1 (new)
                    ToolMessage(content="8", tool_call_id="call_1"),  # idx 2 (new)
                ],
                config={"configurable": {"thread_id": "test"}}
            )

        # ⚠️ PROXY: ToolMessage resolved via resolve_call, NOT query
        assert mock_client.query.call_count == 0, "ToolMessage should NOT use query!"
        mock_proxy.resolve_call.assert_called_once()

        # Should update sent_count
        assert model.client_manager._sent_count["test"] == 3
