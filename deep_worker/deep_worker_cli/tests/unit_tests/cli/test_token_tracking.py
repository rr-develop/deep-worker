"""Unit tests for token tracking in InteractiveApp.

Tests cover:
- Token extraction from AIMessage metadata
- Token display in status bar
- Token updates during streaming
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_worker_cli.cli.interactive import InteractiveApp


class TestTokenTracking:
    """Test token tracking functionality."""

    @pytest.mark.asyncio
    async def test_token_update_message_updates_status_bar(self):
        """Test token_update message updates status bar."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock status bar
            status_updates = []
            mock_status_bar = MagicMock()

            def mock_set_tokens(count):
                status_updates.append(count)

            mock_status_bar.set_tokens = mock_set_tokens
            app._status_bar = mock_status_bar

            # Simulate token_update message from agent
            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 1500,
                "input_tokens": 1000,
                "output_tokens": 500
            })

            # Should update status bar
            assert len(status_updates) == 1
            assert status_updates[0] == 1500

    @pytest.mark.asyncio
    async def test_token_count_tracked_locally(self):
        """Test token count is tracked in app state."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock status bar
            mock_status_bar = MagicMock()
            mock_status_bar.set_tokens = MagicMock()
            app._status_bar = mock_status_bar

            # Initially zero
            assert app._current_token_count == 0

            # Receive token update
            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 2500
            })

            # Should track locally
            assert app._current_token_count == 2500

    @pytest.mark.asyncio
    async def test_multiple_token_updates(self):
        """Test multiple token updates during streaming."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            status_updates = []
            mock_status_bar = MagicMock()

            def mock_set_tokens(count):
                status_updates.append(count)

            mock_status_bar.set_tokens = mock_set_tokens
            app._status_bar = mock_status_bar

            # Simulate progressive token updates
            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 500
            })

            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 1000
            })

            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 1500
            })

            # Should have all updates
            assert status_updates == [500, 1000, 1500]
            assert app._current_token_count == 1500

    @pytest.mark.asyncio
    async def test_token_update_without_status_bar(self):
        """Test token update doesn't crash if status bar not ready."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # No status bar
            app._status_bar = None

            # Should not crash
            await app._handle_realtime_message({
                "type": "token_update",
                "total_tokens": 1500
            })

            # Should still track locally
            assert app._current_token_count == 1500


class TestStreamProcessorTokenExtraction:
    """Test token extraction in stream processor."""

    @pytest.mark.asyncio
    async def test_extract_tokens_from_ai_message(self):
        """Test extracting tokens from AIMessage usage_metadata."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import AIMessage

        # Create AIMessage with usage_metadata
        msg = AIMessage(content="Hello")
        msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }

        # Track notifications
        notifications = []

        async def notify(msg_dict):
            notifications.append(msg_dict)

        # Process message
        await process_message_chunk(
            message=msg,
            metadata={},
            notify_callback=notify,
            tool_call_buffer={}
        )

        # Should send token_update notification
        token_updates = [n for n in notifications if n.get("type") == "token_update"]
        assert len(token_updates) == 1
        assert token_updates[0]["total_tokens"] == 150
        assert token_updates[0]["input_tokens"] == 100
        assert token_updates[0]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_no_token_update_if_no_metadata(self):
        """Test no token update if usage_metadata missing."""
        from deep_worker_cli.process.stream_processor import process_message_chunk
        from langchain_core.messages import AIMessage

        # AIMessage without usage_metadata
        msg = AIMessage(content="Hello")

        notifications = []

        async def notify(msg_dict):
            notifications.append(msg_dict)

        await process_message_chunk(
            message=msg,
            metadata={},
            notify_callback=notify,
            tool_call_buffer={}
        )

        # Should NOT send token_update
        token_updates = [n for n in notifications if n.get("type") == "token_update"]
        assert len(token_updates) == 0
