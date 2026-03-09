"""Unit tests for _handle_realtime_message with new message types.

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


class TestRealtimeMessageHandler:
    """Test _handle_realtime_message with streaming and tool message types."""

    @pytest.mark.asyncio
    async def test_handle_assistant_chunk_creates_new_message(self) -> None:
        """Test that assistant_chunk creates new message if none exists."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app with mocked dependencies
        app = InteractiveApp(agent_name="test")

        # Mock the messages container
        mock_container = Mock()
        mock_container.children = []  # No existing messages

        # Mock query_one to return the container
        app.query_one = Mock(return_value=mock_container)

        # Mock _mount_message
        app._mount_message = AsyncMock()

        # Track created widgets to mock their append_content
        created_widgets = []

        async def track_mount(widget):
            # Mock the widget's append_content since it won't be actually mounted
            widget.append_content = AsyncMock()
            created_widgets.append(widget)

        app._mount_message.side_effect = track_mount

        # Create assistant_chunk message
        msg = {
            "type": "assistant_chunk",
            "role": "assistant",
            "content": "Hello",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify new AssistantMessage was created and mounted
        assert app._mount_message.call_count == 1
        assert len(created_widgets) == 1
        mounted_widget = created_widgets[0]
        assert mounted_widget.__class__.__name__ == "AssistantMessage"

        # Verify append_content was called on the new widget
        assert mounted_widget.append_content.call_count == 1
        assert mounted_widget.append_content.call_args[0][0] == "Hello"

    @pytest.mark.asyncio
    async def test_handle_assistant_chunk_appends_to_existing(self) -> None:
        """Test that assistant_chunk appends to existing AssistantMessage."""
        from deep_worker_cli.cli.interactive import InteractiveApp
        from deep_worker_cli.widgets.messages import AssistantMessage

        # Create app
        app = InteractiveApp(agent_name="test")

        # Create existing AssistantMessage
        existing_msg = AssistantMessage("Initial", id="test-msg")
        existing_msg.append_content = AsyncMock()

        # Mock the messages container with existing message
        mock_container = Mock()
        mock_container.children = [existing_msg]

        # Mock query_one to return the container
        app.query_one = Mock(return_value=mock_container)

        # Mock _mount_message (should NOT be called)
        app._mount_message = AsyncMock()

        # Create assistant_chunk message
        msg = {
            "type": "assistant_chunk",
            "role": "assistant",
            "content": " World",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify append_content was called on existing message
        assert existing_msg.append_content.call_count == 1
        assert existing_msg.append_content.call_args[0][0] == " World"

        # Verify no new message was mounted
        assert app._mount_message.call_count == 0

    @pytest.mark.asyncio
    async def test_handle_tool_call_message(self) -> None:
        """Test that tool_call creates ToolCallMessage and tracks it."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app
        app = InteractiveApp(agent_name="test")

        # Mock _mount_message
        app._mount_message = AsyncMock()

        # Create tool_call message
        msg = {
            "type": "tool_call",
            "name": "read_file",
            "args": {"path": "config.yaml"},
            "id": "call_123",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify ToolCallMessage was mounted
        assert app._mount_message.call_count == 1
        mounted_widget = app._mount_message.call_args[0][0]
        assert mounted_widget.__class__.__name__ == "ToolCallMessage"

        # Verify widget is tracked
        assert "call_123" in app._current_tool_messages
        assert app._current_tool_messages["call_123"] is mounted_widget

        # Verify tool call details
        assert mounted_widget._tool_name == "read_file"
        assert mounted_widget._args == {"path": "config.yaml"}
        assert mounted_widget._status == "running"

    @pytest.mark.asyncio
    async def test_handle_tool_result_success(self) -> None:
        """Test that tool_result updates corresponding ToolCallMessage."""
        from deep_worker_cli.cli.interactive import InteractiveApp
        from deep_worker_cli.widgets.messages import ToolCallMessage

        # Create app
        app = InteractiveApp(agent_name="test")

        # Create a ToolCallMessage and add it to tracking
        tool_widget = ToolCallMessage(
            tool_name="read_file",
            args={"path": "config.yaml"},
            id="test-tool"
        )
        app._current_tool_messages["call_123"] = tool_widget

        # Mock the widget's set_success method
        tool_widget.set_success = Mock()

        # Create tool_result message (success)
        msg = {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "content": "File content here",
            "status": "success",
            "name": "read_file",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify set_success was called
        assert tool_widget.set_success.call_count == 1
        assert tool_widget.set_success.call_args[0][0] == "File content here"

        # Verify widget was removed from tracking
        assert "call_123" not in app._current_tool_messages

    @pytest.mark.asyncio
    async def test_handle_tool_result_error(self) -> None:
        """Test that tool_result updates ToolCallMessage with error."""
        from deep_worker_cli.cli.interactive import InteractiveApp
        from deep_worker_cli.widgets.messages import ToolCallMessage

        # Create app
        app = InteractiveApp(agent_name="test")

        # Create a ToolCallMessage and add it to tracking
        tool_widget = ToolCallMessage(
            tool_name="read_file",
            args={"path": "config.yaml"},
            id="test-tool"
        )
        app._current_tool_messages["call_123"] = tool_widget

        # Mock the widget's set_error method
        tool_widget.set_error = Mock()

        # Create tool_result message (error)
        msg = {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "content": "File not found",
            "status": "error",
            "name": "read_file",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify set_error was called
        assert tool_widget.set_error.call_count == 1
        assert tool_widget.set_error.call_args[0][0] == "File not found"

        # Verify widget was removed from tracking
        assert "call_123" not in app._current_tool_messages

    @pytest.mark.asyncio
    async def test_handle_user_message_backwards_compat(self) -> None:
        """Test backwards compatibility with old protocol (no type field)."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app
        app = InteractiveApp(agent_name="test")

        # Mock _mount_message
        app._mount_message = AsyncMock()

        # Create old-style message (no type field, just role)
        msg = {
            "role": "user",
            "content": "Hello agent",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify UserMessage was mounted
        assert app._mount_message.call_count == 1
        mounted_widget = app._mount_message.call_args[0][0]
        assert mounted_widget.__class__.__name__ == "UserMessage"

    @pytest.mark.asyncio
    async def test_handle_assistant_message_backwards_compat(self) -> None:
        """Test backwards compatibility with old protocol (no type field)."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app
        app = InteractiveApp(agent_name="test")

        # Mock _mount_message
        app._mount_message = AsyncMock()

        # Create old-style message (no type field, just role)
        msg = {
            "role": "assistant",
            "content": "Hello user",
        }

        # Handle the message
        await app._handle_realtime_message(msg)

        # Verify AssistantMessage was mounted
        assert app._mount_message.call_count == 1
        mounted_widget = app._mount_message.call_args[0][0]
        assert mounted_widget.__class__.__name__ == "AssistantMessage"

    @pytest.mark.asyncio
    async def test_handle_orphaned_tool_result(self) -> None:
        """Test that orphaned tool_result (no matching call) is logged and ignored."""
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Create app
        app = InteractiveApp(agent_name="test")

        # Mock _mount_message (should NOT be called)
        app._mount_message = AsyncMock()

        # Create tool_result message without corresponding tool_call
        msg = {
            "type": "tool_result",
            "tool_call_id": "call_999",  # No matching tool call
            "content": "Some result",
            "status": "success",
            "name": "read_file",
        }

        # Handle the message (should log error and do nothing)
        await app._handle_realtime_message(msg)

        # Verify no message was mounted
        assert app._mount_message.call_count == 0

        # Verify tracking dict is still empty
        assert len(app._current_tool_messages) == 0
