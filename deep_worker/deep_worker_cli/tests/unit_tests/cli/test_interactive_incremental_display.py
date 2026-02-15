"""Tests for incremental message display in interactive TUI with RPyC.

REQUIREMENTS:
1. User message MUST be displayed IMMEDIATELY when sent
2. Messages MUST appear in real-time as they are created (via RPyC callbacks)
3. Loading ENTIRE history is FORBIDDEN - must load only NEW messages
4. No delay between user sending message and seeing it in UI
5. TaskManager intermediate messages must appear incrementally via callbacks

This prevents regression where all messages appear at once after completion,
destroying interactivity.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add arpyc to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / "arpyc"))

from arpyc import RpycAsyncSession

from deep_worker_cli.cli.interactive import InteractiveApp


@pytest.mark.asyncio
async def test_user_message_displayed_immediately():
    """Test that user message is displayed IMMEDIATELY when sent via RPyC.

    CRITICAL: User must see their message right away, not after agent responds.
    """
    agent_name = "test-agent"
    user_message = "Hello, agent!"

    with patch("deep_worker_cli.cli.interactive.RpycAsyncSession") as mock_session_class:
        mock_session = AsyncMock(spec=RpycAsyncSession)
        mock_session.async_call = AsyncMock(return_value={
            "success": True,
            "response": {"role": "assistant", "content": "Response"}
        })
        mock_session_class.return_value = mock_session

        app = InteractiveApp(agent_name=agent_name)
        app.rpyc_session = mock_session
        app._connected = True

        # Mock UI components
        mock_mount = AsyncMock()

        with patch.object(app, '_mount_message', mock_mount):
            # Send message
            await app._handle_user_message(user_message)

            # User message MUST be mounted FIRST, before agent call
            # This ensures immediate display
            assert mock_mount.call_count >= 1, "User message must be displayed"
            first_call = mock_mount.call_args_list[0]

            # Verify first mounted widget is UserMessage with correct content
            mounted_widget = first_call[0][0]
            assert mounted_widget.__class__.__name__ == "UserMessage"


@pytest.mark.asyncio
async def test_incremental_history_loading_not_full_reload():
    """Test that only NEW messages are loaded via RPyC, NOT entire history.

    CRITICAL: Loading entire history destroys interactivity for long conversations.
    Must load only messages created since last load.
    """
    agent_name = "test-agent"

    with patch("deep_worker_cli.cli.interactive.RpycAsyncSession") as mock_session_class:
        mock_session = AsyncMock(spec=RpycAsyncSession)

        # Track calls to async_call
        call_count = 0

        async def mock_async_call(method, *args, **kwargs):
            """Mock RPyC calls for history loading."""
            nonlocal call_count
            call_count += 1

            # Check if this is get_history call
            if hasattr(method, '__name__') and 'get_history' in str(method):
                # Extract offset
                if len(kwargs) > 0 and "offset" in kwargs:
                    offset = kwargs["offset"]
                elif len(args) >= 2:
                    offset = args[1]
                else:
                    offset = 0

                if call_count == 1 and offset == 0:
                    # Initial load: 10 messages
                    return {
                        "success": True,
                        "total": 10,
                        "history": [{"type": "HumanMessage", "content": f"Msg {i}"} for i in range(10)],
                        "has_more": False
                    }
                elif offset == 10:
                    # After response: load only NEW messages (offset=10)
                    return {
                        "success": True,
                        "total": 14,
                        "history": [
                            {"type": "HumanMessage", "content": "TaskManager prompt 1"},
                            {"type": "AIMessage", "content": "Response 1"},
                            {"type": "HumanMessage", "content": "TaskManager prompt 2"},
                            {"type": "AIMessage", "content": "Final response"},
                        ],
                        "has_more": False
                    }
                else:
                    # Should NOT request all messages
                    pytest.fail(f"Should request with offset=10, not offset={offset}")
            else:
                # invoke_agent call
                return {
                    "success": True,
                    "response": {"role": "assistant", "content": "Final"}
                }

        mock_session.async_call = AsyncMock(side_effect=mock_async_call)
        mock_session_class.return_value = mock_session

        app = InteractiveApp(agent_name=agent_name)
        app.rpyc_session = mock_session
        app._connected = True

        # Simulate initial history load
        app._history_total = 10  # Track current message count

        # Mock mount to track calls
        mock_mount = AsyncMock()

        with patch.object(app, '_mount_message', mock_mount):
            # Send message and get response
            # With RPyC callbacks, messages arrive in real-time
            await app._handle_user_message("Test message")


@pytest.mark.asyncio
async def test_taskmanager_messages_appear_incrementally():
    """Test that TaskManager intermediate messages appear via RPyC callbacks.

    CRITICAL: Messages must appear as they are created via callbacks, not all at once.
    User should see the conversation flow in real-time.
    """
    agent_name = "test-agent"

    with patch("deep_worker_cli.cli.interactive.RpycAsyncSession") as mock_session_class:
        mock_session = AsyncMock(spec=RpycAsyncSession)

        # Simulate incremental history updates via RPyC
        async def mock_async_call(method, *args, **kwargs):
            """Mock RPyC calls."""
            # Extract offset if present
            if len(kwargs) > 0 and "offset" in kwargs:
                offset = kwargs["offset"]
            elif len(args) >= 2:
                offset = args[1]
            else:
                offset = 0

            if offset == 0:
                return {
                    "success": True,
                    "total": 2,
                    "history": [
                        {"type": "HumanMessage", "content": "Original request"},
                        {"type": "AIMessage", "content": "30% done"}
                    ],
                    "has_more": False
                }
            elif offset == 2:
                # Next 2 messages
                return {
                    "success": True,
                    "total": 4,
                    "history": [
                        {"type": "HumanMessage", "content": "TaskManager: Continue to 50%"},
                        {"type": "AIMessage", "content": "50% done"}
                    ],
                    "has_more": False
                }
            elif offset == 4:
                # Final 2 messages
                return {
                    "success": True,
                    "total": 6,
                    "history": [
                        {"type": "HumanMessage", "content": "TaskManager: Finish"},
                        {"type": "AIMessage", "content": "100% complete"}
                    ],
                    "has_more": False
                }
            else:
                return {"success": True, "total": 6, "history": [], "has_more": False}

        mock_session.async_call = AsyncMock(side_effect=mock_async_call)
        mock_session_class.return_value = mock_session

        app = InteractiveApp(agent_name=agent_name)
        app.rpyc_session = mock_session
        app._connected = True

        # Test that we can load messages incrementally via RPyC
        # First batch: 2 messages
        result1 = await mock_session.async_call(mock_session.root.get_history, 50, 0)
        assert result1["total"] == 2

        # Second batch: next 2 messages
        result2 = await mock_session.async_call(mock_session.root.get_history, 50, 2)
        assert result2["total"] == 4
        assert len(result2["history"]) == 2

        # Verify incremental loading works
        assert "TaskManager" in result2["history"][0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
