"""Basic tests for Approval Menu functionality.

Note: Full HITL middleware integration is complex and requires LangGraph middleware setup.
These tests cover basic approval menu display and interaction via RPyC messages.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_worker_cli.cli.interactive import InteractiveApp


class TestApprovalMenuBasic:
    """Test basic approval menu functionality."""

    @pytest.mark.asyncio
    async def test_approval_request_shows_menu(self):
        """Test approval_request message shows approval menu."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Simulate approval_request message
            await app._handle_realtime_message({
                "type": "approval_request",
                "approval_id": "test_123",
                "actions": [
                    {
                        "tool": "bash",
                        "args": {"command": "ls -la"}
                    }
                ]
            })

            # Should have pending approval widget
            assert app._pending_approval_widget is not None
            assert app._current_approval_id == "test_123"

    @pytest.mark.asyncio
    async def test_approval_result_hides_menu(self):
        """Test approval_result message hides menu."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock approval widget
            mock_approval = MagicMock()
            mock_approval.remove = AsyncMock()
            app._pending_approval_widget = mock_approval
            app._current_approval_id = "test_123"

            # Simulate approval_result from another client
            await app._handle_realtime_message({
                "type": "approval_result",
                "approval_id": "test_123",
                "decision": {"type": "approve"}
            })

            # Should hide menu
            assert app._pending_approval_widget is None
            assert app._current_approval_id is None

    @pytest.mark.asyncio
    async def test_escape_rejects_approval(self):
        """Test Escape key rejects approval menu."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock approval widget with action_select_reject
            mock_approval = MagicMock()
            mock_approval.action_select_reject = MagicMock()
            app._pending_approval_widget = mock_approval

            # Trigger escape
            app.action_interrupt()

            # Should call reject
            mock_approval.action_select_reject.assert_called_once()


class TestAutoApproveBasic:
    """Test basic auto-approve functionality."""

    @pytest.mark.asyncio
    async def test_toggle_auto_approve(self):
        """Test toggle_auto_approve action."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Initially False
            assert app._auto_approve is False

            # Toggle
            app.action_toggle_auto_approve()

            # Should be True
            assert app._auto_approve is True

            # Toggle again
            app.action_toggle_auto_approve()

            # Should be False
            assert app._auto_approve is False

    @pytest.mark.asyncio
    async def test_auto_approve_skips_menu(self):
        """Test auto-approve mode skips showing menu."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Enable auto-approve
            app._auto_approve = True

            # Mock RPyC session
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True
            })
            app.rpyc_session = mock_session
            app._connected = True

            # Simulate approval request
            await app._handle_realtime_message({
                "type": "approval_request",
                "approval_id": "test_auto",
                "actions": [{"tool": "bash", "args": {"command": "ls"}}]
            })

            # Should NOT show menu (auto-approved)
            assert app._pending_approval_widget is None

            # Should have sent approval via RPyC
            mock_session.async_call.assert_called()
            call_args = str(mock_session.async_call.call_args)
            assert "respond_to_approval" in call_args or "test_auto" in call_args
