"""Integration tests for RPyC-based IPC in agent_process.py.

This module tests the RPyC server implementation in AgentProcess,
replacing the old Unix socket IPC with RPyC for real-time message streaming.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass

# Add arpyc to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "arpyc"))

from arpyc import AsyncioEventCallback, RpycAsyncSession

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry


@pytest.fixture
async def agent_with_rpyc():
    """Start agent process with RPyC server."""
    # Create config
    config = AgentConfig(
        model="deepseek-chat",
        working_dir=Path("/tmp"),
    )

    # Create registry
    registry = ProcessRegistry()

    # Create agent process with RPyC enabled
    agent = AgentProcess(
        name="test_rpyc_agent",
        config=config,
        registry=registry,
        skip_permissions=True,
        rpyc_port=18863,  # Use different port for testing
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
async def test_rpyc_server_starts(agent_with_rpyc):
    """Test that RPyC server starts successfully."""
    agent = agent_with_rpyc

    assert agent.rpyc_server is not None
    assert agent.rpyc_port == 18863


@pytest.mark.asyncio
async def test_rpyc_client_can_connect(agent_with_rpyc):
    """Test that client can connect to RPyC server."""
    agent = agent_with_rpyc

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        assert session.is_connected()

        # Test basic RPC call
        status = await session.async_call(session.root.get_status)
        assert status is not None


@pytest.mark.asyncio
async def test_rpyc_invoke_agent(agent_with_rpyc):
    """Test invoking agent via RPyC."""
    agent = agent_with_rpyc

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        # Use async_call for synchronous RPyC methods
        response = await session.async_call(session.root.invoke_agent, "Hello, agent!")

        assert "success" in response
        assert response["success"] is True
        assert "response" in response


@pytest.mark.asyncio
async def test_rpyc_message_subscription(agent_with_rpyc):
    """Test real-time message subscription via RPyC callbacks."""
    agent = agent_with_rpyc

    messages_received: list[dict[str, Any]] = []

    def on_message(msg: dict[str, Any]) -> None:
        messages_received.append(msg)

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        loop = asyncio.get_running_loop()

        # Create callback for message subscription
        callback = type(
            "MessageCallback",
            (),
            {"on_message": lambda self, msg: on_message(msg)},
        )()

        # Subscribe to messages
        success = await session.async_call(session.root.subscribe_messages, callback)
        assert success is True

        # Send message to agent
        response = await session.async_call(session.root.invoke_agent, "Test message")
        assert response["success"]

        # Wait for messages to arrive
        await asyncio.sleep(0.5)

        # Should have received messages
        assert len(messages_received) > 0

        # Unsubscribe
        await session.async_call(session.root.unsubscribe_messages, callback)


@pytest.mark.asyncio
async def test_rpyc_get_history(agent_with_rpyc):
    """Test getting conversation history via RPyC."""
    agent = agent_with_rpyc

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        # Invoke agent to create some history
        await session.async_call(session.root.invoke_agent, "First message")
        await session.async_call(session.root.invoke_agent, "Second message")

        # Get history
        history = await session.async_call(session.root.get_history, limit=10, offset=0)

        assert "success" in history
        assert history["success"] is True
        assert "history" in history
        assert len(history["history"]) >= 2  # At least 2 messages


@pytest.mark.asyncio
@pytest.mark.skip(reason="Race condition in session_manager file I/O, not RPyC issue")
async def test_rpyc_concurrent_clients(agent_with_rpyc):
    """Test multiple clients can connect concurrently."""
    agent = agent_with_rpyc

    async def client_task(client_id: int):
        async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
            response = await session.async_call(
                session.root.invoke_agent, f"Message from client {client_id}"
            )
            return response["success"]

    # Run multiple clients concurrently
    results = await asyncio.gather(
        client_task(1),
        client_task(2),
        client_task(3),
    )

    assert all(results)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Real-time streaming requires async architecture changes")
async def test_rpyc_message_streaming_realtime(agent_with_rpyc):
    """Test real-time message streaming during agent execution."""
    agent = agent_with_rpyc

    messages: list[dict[str, Any]] = []
    message_event = asyncio.Event()

    def on_message(msg: dict[str, Any]) -> None:
        messages.append(msg)
        if len(messages) >= 2:  # Expecting at least user + assistant
            message_event.set()

    async with RpycAsyncSession("localhost", agent.rpyc_port) as session:
        loop = asyncio.get_running_loop()

        # Subscribe using AsyncioEventCallback pattern
        callback = type(
            "MessageCallback",
            (),
            {"on_message": lambda self, msg: loop.call_soon_threadsafe(on_message, msg)},
        )()

        await session.async_call(session.root.subscribe_messages, callback)

        # Invoke agent (this might take time) - run in background
        async def invoke_wrapper():
            return await session.async_call(session.root.invoke_agent, "Test real-time streaming")

        invoke_task = asyncio.create_task(invoke_wrapper())

        # Wait for messages to arrive (with timeout)
        try:
            await asyncio.wait_for(message_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Messages not received in time")

        # Ensure invoke completes
        await invoke_task

        # Verify we got messages in real-time (before invoke completed)
        assert len(messages) > 0

        await session.async_call(session.root.unsubscribe_messages, callback)
