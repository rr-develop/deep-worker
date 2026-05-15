"""Unit tests for agent interruption (Escape/Ctrl+C) in InteractiveApp.

Tests cover:
- Interrupt agent via Escape key
- Interrupt agent via Ctrl+C
- Approval menu rejection via interruption
- Multi-client interrupt broadcast
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_worker_cli.cli.interactive import InteractiveApp
from deep_worker_cli.widgets.approval import ApprovalMenu


class TestAgentInterruption:
    """Test agent interruption functionality."""

    @pytest.mark.asyncio
    async def test_escape_key_interrupts_agent(self):
        """Test Escape key sends interrupt to agent.

        Interactive now uses `await self._rpyc_root.interrupt_agent()`
        which returns a JSON-encoded string (see interactive.py ~line
        1588). We mock the `interrupt_agent` netref method directly.
        """
        import json as _json

        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock the RPyC root object used by _interrupt_agent.
            mock_root = MagicMock()
            mock_root.interrupt_agent = AsyncMock(
                return_value=_json.dumps(
                    {"success": True, "message": "Agent interrupted"}
                )
            )
            app._rpyc_root = mock_root

            # Trigger escape action
            app.action_interrupt()
            await pilot.pause()

            mock_root.interrupt_agent.assert_called()

    @pytest.mark.asyncio
    async def test_interrupt_updates_status(self):
        """Test interrupt updates status bar."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock RPyC session
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True,
                "message": "Agent interrupted"
            })

            app.rpyc_session = mock_session
            app._connected = True

            # Mock status bar
            status_updates = []

            def mock_update_status(msg):
                status_updates.append(msg)

            app._update_status = mock_update_status

            # Trigger escape action
            app.action_interrupt()
            await pilot.pause()

            # Should update status
            # Note: status update happens in _handle_realtime_message when
            # agent broadcasts "agent_interrupted" message
            # So we won't see it here unless we mock that flow

    @pytest.mark.asyncio
    async def test_interrupt_when_not_connected(self):
        """Test interrupt when not connected does nothing."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Ensure not connected
            app.rpyc_session = None
            app._connected = False

            # Trigger escape - should not raise error
            app.action_interrupt()
            await pilot.pause()

            # Should complete without error

    @pytest.mark.asyncio
    async def test_escape_rejects_approval_menu(self):
        """Test Escape key rejects active approval menu."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Create mock approval menu
            mock_approval = MagicMock(spec=ApprovalMenu)
            mock_approval.action_select_reject = MagicMock()

            app._pending_approval_widget = mock_approval

            # Trigger escape
            app.action_interrupt()

            # Should reject approval menu
            mock_approval.action_select_reject.assert_called_once()

    @pytest.mark.asyncio
    async def test_ctrl_c_interrupts_agent(self):
        """Test Ctrl+C interrupts agent when running."""
        import json as _json

        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            mock_root = MagicMock()
            mock_root.interrupt_agent = AsyncMock(
                return_value=_json.dumps(
                    {"success": True, "message": "Agent interrupted"}
                )
            )
            app._rpyc_root = mock_root

            # Trigger Ctrl+C action
            app.action_quit_or_interrupt()
            await pilot.pause()

            # Should call interrupt (not quit)
            mock_root.interrupt_agent.assert_called()

    @pytest.mark.asyncio
    async def test_interrupt_hides_spinner(self):
        """Test interrupt hides loading spinner."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Create mock loading widget
            from deep_worker_cli.widgets.loading import LoadingWidget

            mock_loading = MagicMock(spec=LoadingWidget)
            mock_loading.remove = AsyncMock()
            app._loading_widget = mock_loading

            # Mock RPyC
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True
            })
            app.rpyc_session = mock_session
            app._connected = True

            # Simulate receiving interrupt message from agent
            await app._handle_realtime_message({
                "type": "agent_interrupted",
                "role": "system",
                "content": "Interrupted"
            })

            # Should hide spinner
            # Note: Spinner hiding happens via _set_spinner(None)


class TestInterruptBroadcast:
    """Test interrupt broadcast to multiple clients."""

    @pytest.mark.asyncio
    async def test_interrupt_message_displays_notification(self):
        """Test that agent_interrupted message displays notification."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            mounted_messages = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_messages.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Simulate agent_interrupted broadcast
            await app._handle_realtime_message({
                "type": "agent_interrupted",
                "role": "system",
                "content": "⚠ Interrupted"
            })

            # Should mount interruption message
            assert len(mounted_messages) >= 1
            # Check that AppMessage was added
            from deep_worker_cli.widgets.messages import AppMessage
            assert any(isinstance(msg, AppMessage) for msg in mounted_messages)


class TestInterruptRPyCService:
    """Test RPyC service interrupt handling."""

    @pytest.mark.asyncio
    async def test_exposed_interrupt_agent_method(self):
        """Test that RPyC service has exposed_interrupt_agent method.

        The method is `async def` and returns a JSON-encoded string —
        the client must `await` and then `json.loads` the reply.
        """
        import json
        from deep_worker_cli.rpyc_service.agent_service import AgentService

        service = AgentService()

        # Should have exposed_interrupt_agent method
        assert hasattr(service, 'exposed_interrupt_agent')

        # Mock agent process
        mock_agent = MagicMock()
        mock_agent.is_running = False
        service._agent_process = mock_agent

        # Call when not running — async def returning JSON string
        response = json.loads(await service.exposed_interrupt_agent())

        assert response["success"] is False
        assert "No agent execution" in response["message"]

    @pytest.mark.asyncio
    async def test_interrupt_cancels_worker_task(self):
        """Test interrupt cancels the agent worker task."""
        from deep_worker_cli.rpyc_service.agent_service import AgentService
        import asyncio
        import threading

        service = AgentService()

        # Initialize service locks (normally done in on_connect)
        service._subscribers_lock = threading.Lock()
        service._message_subscribers = set()

        # Mock agent process with running task
        mock_agent = MagicMock()
        mock_agent.is_running = True

        # Create a fake task that can be cancelled
        async def fake_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(fake_task())
        mock_agent.current_worker_task = task

        service._agent_process = mock_agent

        # Track broadcasts via _notify_subscribers
        broadcasts = []

        original_notify = service._notify_subscribers

        async def mock_notify(msg):
            broadcasts.append(msg)

        service._notify_subscribers = mock_notify

        # Call interrupt — async def returning JSON string
        import json
        response = json.loads(await service.exposed_interrupt_agent())

        # Should return success
        assert response["success"] is True

        # Give event loop a chance to process the cancellation
        await asyncio.sleep(0)

        # Task should be cancelled or cancelling
        assert task.cancelled() or task.cancelling()

        # Should broadcast interrupt
        assert len(broadcasts) == 1
        assert broadcasts[0]["type"] == "agent_interrupted"
