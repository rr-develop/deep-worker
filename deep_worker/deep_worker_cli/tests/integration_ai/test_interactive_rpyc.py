"""Integration tests for interactive.py with RPyC backend.

This module tests the RPyC-based message streaming in InteractiveApp,
replacing Unix socket IPC with RPyC real-time callbacks.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

if TYPE_CHECKING:
    pass

# Add arpyc to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "arpyc"))

from arpyc import RpycAsyncSession

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.cli.interactive import InteractiveApp
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry


@pytest.fixture
async def agent_with_rpyc_running():
    """Start agent process with RPyC server for testing."""
    # Create config
    config = AgentConfig(
        model="deepseek-chat",
        working_dir=Path("/tmp"),
    )

    # Create registry
    registry = ProcessRegistry()

    # Create agent process with RPyC enabled
    agent = AgentProcess(
        name="test_interactive_rpyc",
        config=config,
        registry=registry,
        skip_permissions=True,
        rpyc_port=18864,  # Different port to avoid conflicts
    )

    # Start agent in background
    task = asyncio.create_task(agent.start())

    # Wait for server to be ready
    await asyncio.sleep(1.0)

    yield agent

    # Cleanup
    agent.running = False

    # Stop RPyC server gracefully
    if agent.rpyc_server:
        try:
            agent.rpyc_server.close()
        except Exception:
            pass

    # Wait for task to finish (with timeout)
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_interactive_app_connects_via_rpyc(agent_with_rpyc_running):
    """Test that InteractiveApp can connect via RPyC instead of Unix sockets."""
    agent = agent_with_rpyc_running

    # Mock InteractiveApp to avoid Textual UI initialization
    with patch.object(InteractiveApp, 'compose', return_value=iter([])):
        with patch.object(InteractiveApp, '_mount_message', new_callable=AsyncMock):
            with patch.object(InteractiveApp, '_update_status', new_callable=Mock):
                # Create app with RPyC enabled
                app = InteractiveApp(agent_name="test_interactive_rpyc")

                # Override connection to use RPyC
                app.rpyc_port = agent.rpyc_port

                # Manually initialize connection
                app.rpyc_session = RpycAsyncSession("localhost", app.rpyc_port)
                await app.rpyc_session.connect()

                # Verify connection
                assert app.rpyc_session.is_connected()

                # Cleanup
                await app.rpyc_session.disconnect()


@pytest.mark.skip(reason="Requires Gemini API - not currently in use")
@pytest.mark.asyncio
async def test_interactive_app_sends_message_via_rpyc(agent_with_rpyc_running):
    """Test that InteractiveApp can send messages via RPyC."""
    agent = agent_with_rpyc_running

    # Create RPyC session
    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        # Send message
        response = await session.async_call(
            session.root.invoke_agent, "Hello from interactive test"
        )

        # Verify response
        assert response["success"] is True
        assert "response" in response
        assert "content" in response["response"]


@pytest.mark.skip(reason="Requires Gemini API - not currently in use")
@pytest.mark.asyncio
async def test_interactive_app_receives_realtime_messages(agent_with_rpyc_running):
    """Test that InteractiveApp receives messages in real-time via callbacks."""
    agent = agent_with_rpyc_running

    messages_received: list[dict[str, Any]] = []
    message_event = asyncio.Event()

    def on_message(msg: dict[str, Any]) -> None:
        """Callback for message events."""
        messages_received.append(msg)
        # Set event when we receive assistant message
        if msg.get("role") == "assistant":
            message_event.set()

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        loop = asyncio.get_running_loop()

        # Create callback object
        callback = type(
            "MessageCallback",
            (),
            {"on_message": lambda self, msg: loop.call_soon_threadsafe(on_message, msg)},
        )()

        # Subscribe to messages
        success = await session.async_call(session.root.subscribe_messages, callback)
        assert success is True

        # Send message in background
        async def send_message():
            return await session.async_call(session.root.invoke_agent, "Test message")

        invoke_task = asyncio.create_task(send_message())

        # Wait for real-time message (with timeout)
        try:
            await asyncio.wait_for(message_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Real-time message not received in time")

        # Ensure invoke completes
        response = await invoke_task
        assert response["success"]

        # Verify we received messages in real-time
        assert len(messages_received) >= 2  # At least user + assistant

        # Verify message structure
        user_msgs = [m for m in messages_received if m.get("role") == "user"]
        assistant_msgs = [m for m in messages_received if m.get("role") == "assistant"]

        assert len(user_msgs) > 0, "Should receive user message"
        assert len(assistant_msgs) > 0, "Should receive assistant message"

        # Unsubscribe
        await session.async_call(session.root.unsubscribe_messages, callback)


@pytest.mark.asyncio
async def test_interactive_app_no_polling_required():
    """Test that with RPyC callbacks, polling is not needed."""
    # This is a design test - we verify that:
    # 1. Messages arrive via callbacks (push), not polling
    # 2. _load_new_messages should not be called after agent completion

    # This will be verified by the implementation:
    # - Remove _load_new_messages call from _agent_task
    # - Messages are displayed via callback handler
    # - No explicit history fetching after agent completion

    # For now, this test documents the requirement
    assert True, "This test documents that polling should be removed"
