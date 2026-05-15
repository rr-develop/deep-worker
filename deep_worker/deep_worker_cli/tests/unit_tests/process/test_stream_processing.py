"""Unit tests for stream processing in agent_process.

CRITICAL: ALWAYS TEST LOCAL REPOSITORY CODE
============================================
These tests MUST use code from deep_worker/deepagents.
conftest.py ensures sys.path is configured correctly.

NEVER test against installed packages from site-packages!
Testing wrong code leads to false test results and production bugs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestStreamProcessing:
    """Test extraction of all message types from LangGraph stream."""

    @pytest.mark.asyncio
    async def test_stream_uses_correct_parameters(self) -> None:
        """Test that agent.astream is called with required parameters.

        TODO: This is a placeholder test for Phase 2 implementation.
        Will test that _handle_invoke calls agent.astream with:
        - stream_mode=["messages", "updates"]
        - subgraphs=True

        For now, just verify imports work.
        """
        # Verify imports
        from deep_worker_cli.process.stream_processor import process_message_chunk

        # This will be implemented in Phase 2 when we update _handle_invoke
        # to use agent.astream() with proper stream_mode and subgraphs params
        assert process_message_chunk is not None

    @pytest.mark.asyncio
    async def test_extract_text_chunk_from_ai_message(self) -> None:
        """Test extraction of text chunks from AIMessage content_blocks.

        Text blocks are BATCHED into a buffer per-callback (see
        `_add_text_to_buffer` / `_flush_text_buffer` in
        `stream_processor.py`) to reduce RPyC overhead. The test
        therefore asserts the final flushed concatenation, not a
        chunk-per-block fan-out.

        `usage_metadata` is explicitly set to None — otherwise
        Mock() makes it a nested MagicMock and `usage.get(...)`
        returns Mock objects that can't be added together.
        """
        from deep_worker_cli.process.stream_processor import (
            process_message_chunk,
            flush_all_text_buffers,
        )

        ai_message = Mock()
        ai_message.content_blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"},
        ]
        ai_message.usage_metadata = None

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        await process_message_chunk(
            message=ai_message,
            metadata={},
            notify_callback=mock_notify,
        )
        # End-of-stream: flush all pending text buffers. In production
        # this is done by the driver that owns the agent stream; in
        # tests we do it explicitly.
        await flush_all_text_buffers()

        # After the flush there is exactly one assistant_chunk
        # carrying the full concatenated text.
        assistant_chunks = [n for n in notifications if n.get("type") == "assistant_chunk"]
        assert len(assistant_chunks) == 1, notifications
        assert assistant_chunks[0]["role"] == "assistant"
        assert assistant_chunks[0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_extract_tool_call_from_content_blocks(self) -> None:
        """Test extraction of tool calls from AIMessage content_blocks."""
        # Arrange
        from deep_worker_cli.process.stream_processor import process_message_chunk

        # Create AIMessage with tool_call in content_blocks.
        # See note in test_extract_text_chunk_from_ai_message re:
        # `usage_metadata = None`.
        ai_message = Mock()
        ai_message.content_blocks = [
            {
                "type": "tool_call",
                "name": "read_file",
                "args": {"path": "config.yaml"},
                "id": "call_123",
            }
        ]
        ai_message.usage_metadata = None

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        await process_message_chunk(
            message=ai_message,
            metadata={},
            notify_callback=mock_notify,
        )

        # Assert
        assert len(notifications) == 1
        assert notifications[0] == {
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "config.yaml"},
            "id": "call_123",
        }

    @pytest.mark.asyncio
    async def test_extract_tool_result_from_tool_message(self) -> None:
        """Test extraction of tool results from ToolMessage."""
        # Arrange
        from deep_worker_cli.process.stream_processor import process_message_chunk

        # Create ToolMessage
        tool_message = ToolMessage(
            content="File content here",
            tool_call_id="call_123",
            name="read_file",
            status="success",
        )

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        await process_message_chunk(
            message=tool_message,
            metadata={},
            notify_callback=mock_notify,
        )

        # Assert
        assert len(notifications) == 1
        assert notifications[0] == {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "content": "File content here",
            "status": "success",
            "name": "read_file",
        }

    @pytest.mark.asyncio
    async def test_ignore_human_messages(self) -> None:
        """Test that HumanMessage is ignored (already notified)."""
        # Arrange
        from deep_worker_cli.process.stream_processor import process_message_chunk

        human_message = HumanMessage(content="User input")

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        await process_message_chunk(
            message=human_message,
            metadata={},
            notify_callback=mock_notify,
        )

        # Assert - no notifications for HumanMessage
        assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_tool_call_chunk_accumulation(self) -> None:
        """Test that tool_call_chunk pieces are accumulated."""
        # Arrange
        from deep_worker_cli.process.stream_processor import process_message_chunk

        # Simulate streaming tool call in chunks.
        # See note re: `usage_metadata = None` in
        # test_extract_text_chunk_from_ai_message.
        ai_message1 = Mock()
        ai_message1.content_blocks = [
            {
                "type": "tool_call_chunk",
                "name": "read_file",
                "args": '{"path": "co',
                "id": "call_123",
                "index": 0,
            }
        ]
        ai_message1.usage_metadata = None

        ai_message2 = Mock()
        ai_message2.usage_metadata = None
        ai_message2.content_blocks = [
            {
                "type": "tool_call_chunk",
                "name": None,
                "args": 'nfig.yaml"}',
                "id": "call_123",
                "index": 0,
            }
        ]

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        buffer: dict[str, Any] = {}
        await process_message_chunk(
            message=ai_message1,
            metadata={},
            notify_callback=mock_notify,
            tool_call_buffer=buffer,
        )

        await process_message_chunk(
            message=ai_message2,
            metadata={},
            notify_callback=mock_notify,
            tool_call_buffer=buffer,
        )

        # Assert - only send notification when complete
        # First chunk doesn't send (incomplete JSON)
        # Second chunk completes and sends
        assert len(notifications) == 1
        assert notifications[0]["type"] == "tool_call"
        assert notifications[0]["name"] == "read_file"
        assert notifications[0]["args"] == {"path": "config.yaml"}


class TestStreamFiltering:
    """Test filtering of messages (e.g., summarization)."""

    @pytest.mark.asyncio
    async def test_filter_summarization_messages(self) -> None:
        """Test that summarization messages are filtered out."""
        # Arrange
        from deep_worker_cli.process.stream_processor import process_message_chunk

        ai_message = Mock()
        ai_message.content_blocks = [{"type": "text", "text": "Summary text"}]

        metadata = {"lc_source": "summarization"}

        notifications: list[dict[str, Any]] = []

        async def mock_notify(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Act
        await process_message_chunk(
            message=ai_message,
            metadata=metadata,
            notify_callback=mock_notify,
        )

        # Assert - summarization messages are filtered
        assert len(notifications) == 0


class TestTypeAnnotations:
    """Verify strict type annotations on stream processor."""

    def test_process_message_chunk_has_type_hints(self) -> None:
        """Verify process_message_chunk has proper type annotations."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        import inspect

        sig = inspect.signature(process_message_chunk)

        # Check all parameters have annotations
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert (
                param.annotation != inspect.Parameter.empty
            ), f"Parameter {param_name} missing type annotation"

        # Check return type annotation
        assert (
            sig.return_annotation != inspect.Signature.empty
        ), "Missing return type annotation"
