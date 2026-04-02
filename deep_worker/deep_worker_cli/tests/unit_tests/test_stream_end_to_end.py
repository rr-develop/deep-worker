"""End-to-end tests for message streaming with tools.

CRITICAL: ALWAYS TEST LOCAL REPOSITORY CODE
============================================
These tests MUST use code from deep_worker/deepagents.
conftest.py ensures sys.path is configured correctly.

NEVER test against installed packages from site-packages!
Testing wrong code leads to false test results and production bugs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestStreamEndToEnd:
    """End-to-end tests for complete message streaming flow."""

    @pytest.mark.asyncio
    async def test_stream_processor_with_mocked_stream(self) -> None:
        """Test stream_processor with mocked LangGraph stream."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import AIMessage, ToolMessage

        # Track notifications
        notifications: list[dict[str, Any]] = []

        async def notify_callback(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Simulate streaming messages

        # 1. Text chunk
        ai_msg_1 = Mock(spec=AIMessage)
        ai_msg_1.content_blocks = [{"type": "text", "text": "Let me read "}]
        await process_message_chunk(
            message=ai_msg_1,
            metadata={},
            notify_callback=notify_callback,
        )

        # 2. Another text chunk
        ai_msg_2 = Mock(spec=AIMessage)
        ai_msg_2.content_blocks = [{"type": "text", "text": "the file."}]
        await process_message_chunk(
            message=ai_msg_2,
            metadata={},
            notify_callback=notify_callback,
        )

        # 3. Tool call
        ai_msg_3 = Mock(spec=AIMessage)
        ai_msg_3.content_blocks = [
            {
                "type": "tool_call",
                "name": "read_file",
                "args": {"path": "test.txt"},
                "id": "call_abc",
            }
        ]
        await process_message_chunk(
            message=ai_msg_3,
            metadata={},
            notify_callback=notify_callback,
        )

        # 4. Tool result
        tool_msg = ToolMessage(
            content="File content here",
            tool_call_id="call_abc",
            name="read_file",
            status="success",
        )
        await process_message_chunk(
            message=tool_msg,
            metadata={},
            notify_callback=notify_callback,
        )

        # Verify notifications
        assert len(notifications) == 4

        # Verify text chunks
        assert notifications[0] == {
            "type": "assistant_chunk",
            "role": "assistant",
            "content": "Let me read ",
        }
        assert notifications[1] == {
            "type": "assistant_chunk",
            "role": "assistant",
            "content": "the file.",
        }

        # Verify tool call
        assert notifications[2] == {
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "test.txt"},
            "id": "call_abc",
        }

        # Verify tool result
        assert notifications[3] == {
            "type": "tool_result",
            "tool_call_id": "call_abc",
            "content": "File content here",
            "status": "success",
            "name": "read_file",
        }

    @pytest.mark.asyncio
    async def test_ui_handler_with_full_message_flow(self) -> None:
        """Test UI handler processes complete message flow correctly."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app
        app = InteractiveApp(agent_name="test")

        # Mock query_one to return a mock container for assistant_chunk
        mock_container = Mock()
        mock_container.children = []

        # Mock _mount_message to track widget creation
        mounted_widgets: list[Any] = []

        async def track_mount(widget: Any) -> None:
            # Mock append_content for AssistantMessage
            if hasattr(widget, "append_content"):
                widget.append_content = AsyncMock()
            if hasattr(widget, "write_initial_content"):
                widget.write_initial_content = AsyncMock()
            mounted_widgets.append(widget)
            # Add widget to mock container for assistant_chunk tests
            mock_container.children.append(widget)

        app._mount_message = AsyncMock(side_effect=track_mount)
        app.query_one = Mock(return_value=mock_container)

        # Simulate complete message flow

        # 1. User message
        await app._handle_realtime_message({
            "type": "user",
            "role": "user",
            "content": "Read test.txt",
        })

        # 2. Assistant text chunks
        await app._handle_realtime_message({
            "type": "assistant_chunk",
            "role": "assistant",
            "content": "Let me ",
        })
        await app._handle_realtime_message({
            "type": "assistant_chunk",
            "role": "assistant",
            "content": "read that.",
        })

        # 3. Tool call
        await app._handle_realtime_message({
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "test.txt"},
            "id": "call_123",
        })

        # 4. Tool result
        await app._handle_realtime_message({
            "type": "tool_result",
            "tool_call_id": "call_123",
            "content": "File content",
            "status": "success",
            "name": "read_file",
        })

        # 5. Final assistant response
        await app._handle_realtime_message({
            "type": "assistant",
            "role": "assistant",
            "content": "Here's the content.",
        })

        # Verify widgets were created
        # UserMessage, AssistantMessage (for chunks), ToolCallMessage, AssistantMessage (final)
        assert len(mounted_widgets) >= 4

        # Verify types
        assert mounted_widgets[0].__class__.__name__ == "UserMessage"
        assert mounted_widgets[1].__class__.__name__ == "AssistantMessage"
        assert mounted_widgets[2].__class__.__name__ == "ToolCallMessage"
        assert mounted_widgets[3].__class__.__name__ == "AssistantMessage"

        # Verify ToolCallMessage was updated with result
        tool_widget = mounted_widgets[2]
        assert "call_123" not in app._current_tool_messages  # Removed after update

    @pytest.mark.asyncio
    async def test_streaming_tool_call_chunks(self) -> None:
        """Test accumulation of streaming tool_call_chunk messages."""
        from deep_worker_cli.process.stream_processor import process_message_chunk

        # Track notifications
        notifications: list[dict[str, Any]] = []

        async def notify_callback(msg: dict[str, Any]) -> None:
            notifications.append(msg)

        # Tool call buffer for accumulation
        buffer: dict[str, Any] = {}

        # Simulate streaming tool call in 3 chunks
        chunk1 = Mock()
        chunk1.content_blocks = [
            {
                "type": "tool_call_chunk",
                "name": "read_file",
                "args": '{"path": "te',
                "id": "call_abc",
                "index": 0,
            }
        ]

        chunk2 = Mock()
        chunk2.content_blocks = [
            {
                "type": "tool_call_chunk",
                "name": None,
                "args": 'st.txt',
                "id": "call_abc",
                "index": 0,
            }
        ]

        chunk3 = Mock()
        chunk3.content_blocks = [
            {
                "type": "tool_call_chunk",
                "name": None,
                "args": '"}',
                "id": "call_abc",
                "index": 0,
            }
        ]

        # Process chunks
        await process_message_chunk(
            message=chunk1,
            metadata={},
            notify_callback=notify_callback,
            tool_call_buffer=buffer,
        )
        # First chunk: incomplete JSON, no notification
        assert len(notifications) == 0

        await process_message_chunk(
            message=chunk2,
            metadata={},
            notify_callback=notify_callback,
            tool_call_buffer=buffer,
        )
        # Second chunk: still incomplete
        assert len(notifications) == 0

        await process_message_chunk(
            message=chunk3,
            metadata={},
            notify_callback=notify_callback,
            tool_call_buffer=buffer,
        )
        # Third chunk: JSON complete, notification sent
        assert len(notifications) == 1
        assert notifications[0] == {
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "test.txt"},
            "id": "call_abc",
        }
