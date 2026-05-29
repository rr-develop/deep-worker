"""Tests for AgentService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    """Create a mock ProcessRegistry."""
    registry = MockProcessRegistry()
    # Add some test agents
    registry.add_mock_agent("test-agent-1", 12345, 18861, "/tmp/test1")
    registry.add_mock_agent("test-agent-2", 12346, 18862, "/tmp/test2")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    """Create AgentService with mock registry."""
    return AgentService(registry=mock_registry)


class TestAgentServiceInitialization:
    """Test AgentService initialization."""

    async def test_initialize_discovers_agents(
        self, agent_service: AgentService, mock_registry: MockProcessRegistry
    ) -> None:
        """Test that initialize discovers agents from registry."""
        await agent_service.initialize()

        agents = await agent_service.list_agents()
        assert len(agents) == 2
        assert agents[0].name == "test-agent-1"
        assert agents[1].name == "test-agent-2"

    async def test_initialize_empty_registry(self) -> None:
        """Test initialization with empty registry."""
        empty_registry = MockProcessRegistry()
        service = AgentService(registry=empty_registry)
        await service.initialize()

        agents = await service.list_agents()
        assert len(agents) == 0


class TestAgentServiceListAgents:
    """Test listing agents."""

    async def test_list_agents_returns_all_agents(
        self, agent_service: AgentService
    ) -> None:
        """Test that list_agents returns all discovered agents."""
        await agent_service.initialize()
        agents = await agent_service.list_agents()

        assert len(agents) == 2
        assert all(hasattr(agent, "name") for agent in agents)
        assert all(hasattr(agent, "status") for agent in agents)

    async def test_list_agents_before_initialize(
        self, agent_service: AgentService
    ) -> None:
        """Test that list_agents works before initialize (returns empty)."""
        agents = await agent_service.list_agents()
        assert len(agents) == 0


class TestAgentServiceGetAgent:
    """Test getting individual agent info."""

    async def test_get_agent_existing(
        self, agent_service: AgentService
    ) -> None:
        """Test getting info for existing agent."""
        await agent_service.initialize()
        agent = await agent_service.get_agent("test-agent-1")

        assert agent is not None
        assert agent.name == "test-agent-1"
        assert agent.pid == 12345
        assert agent.rpyc_port == 18861

    async def test_get_agent_nonexistent(
        self, agent_service: AgentService
    ) -> None:
        """Test getting info for non-existent agent."""
        await agent_service.initialize()
        agent = await agent_service.get_agent("nonexistent")

        assert agent is None


class TestAgentServiceConnection:
    """Test agent connection management."""

    @pytest.mark.skip(reason="Requires real RPyC connection - integration test")
    async def test_connect_to_agent(
        self, agent_service: AgentService
    ) -> None:
        """Test connecting to an agent via RPyC."""
        # This would require a real agent process with RPyC server
        # Will be tested in integration tests
        pass

    async def test_is_connected_no_connection(
        self, agent_service: AgentService
    ) -> None:
        """Test is_connected returns False when not connected."""
        await agent_service.initialize()
        connected = await agent_service.is_connected("test-agent-1")
        assert connected is False

    async def test_is_connected_returns_false_when_conn_closed(
        self, agent_service: AgentService
    ) -> None:
        """Regression for a related internal incident analysis (not included here).

        ``is_connected`` previously checked only the local
        ``_connections`` dict. After the peer process died and rpyc
        marked the Connection as ``_closed=True``, ``is_connected``
        still returned True, so every ``fire_and_forget_async(root.…)``
        call kept registering AsyncResult chains in
        ``Connection._request_callbacks`` that no reply would ever
        clear. With ~115k chains accumulated the live process hit
        9.4 GB RSS.

        The contract ``is_connected`` must enforce: if rpyc has
        already flipped ``conn._closed`` (channel torn down,
        ``aclose`` finished, peer died and was noticed), report
        not-connected.
        """
        await agent_service.initialize()

        dead_conn = MagicMock()
        dead_conn._closed = True
        agent_service._connections["test-agent-1"] = dead_conn

        connected = await agent_service.is_connected("test-agent-1")

        assert connected is False, (
            "is_connected must not lie about a torn-down rpyc Connection"
        )

    async def test_is_connected_evicts_closed_connection(
        self, agent_service: AgentService
    ) -> None:
        """A torn-down connection must be removed from
        ``_connections`` so the next caller does not pull the same
        dead handle out and call ``root.method()`` on it.

        Without this, every subsequent caller would see ``False`` from
        ``is_connected`` and trigger a reconnect — but if the eviction
        is missing, the dead handle keeps coming back, every old
        in-flight ``fire_and_forget_async`` keeps firing on the dead
        ``root``, and the AsyncResult leak resumes.
        """
        await agent_service.initialize()

        dead_conn = MagicMock()
        dead_conn._closed = True
        agent_service._connections["test-agent-1"] = dead_conn

        await agent_service.is_connected("test-agent-1")

        assert "test-agent-1" not in agent_service._connections, (
            "is_connected must evict torn-down connections so "
            "subsequent callers do not reuse the dead handle"
        )

    async def test_is_connected_true_for_live_connection(
        self, agent_service: AgentService
    ) -> None:
        """Happy path: ``conn._closed = False`` reports connected.

        This guards against an over-zealous fix that calls
        ``conn._closed`` something falsy and then evicts every
        connection on every probe.
        """
        await agent_service.initialize()

        live_conn = MagicMock()
        live_conn._closed = False
        agent_service._connections["test-agent-1"] = live_conn

        connected = await agent_service.is_connected("test-agent-1")

        assert connected is True
        assert agent_service._connections["test-agent-1"] is live_conn

    async def test_is_connected_handles_conn_without_closed_attr(
        self, agent_service: AgentService
    ) -> None:
        """A test-double or stub Connection may not expose ``_closed``.

        In that case the existing local-dict semantics must apply
        (entry present → connected). This keeps existing integration
        tests using mock connections green.
        """
        await agent_service.initialize()

        plain_conn = object()  # no _closed attribute at all
        agent_service._connections["test-agent-1"] = plain_conn

        connected = await agent_service.is_connected("test-agent-1")

        assert connected is True


class TestAgentServiceShutdown:
    """Test service shutdown."""

    async def test_shutdown_cleans_up(
        self, agent_service: AgentService
    ) -> None:
        """Test that shutdown properly cleans up resources."""
        await agent_service.initialize()
        await agent_service.shutdown()

        # After shutdown, service should be in clean state
        agents = await agent_service.list_agents()
        assert len(agents) == 0


class TestAgentServiceMessageCallbacks:
    """Test message callback subscription."""

    async def test_register_message_callback(
        self, agent_service: AgentService
    ) -> None:
        """Test registering a message callback."""
        callback_called = False
        received_message = None

        async def test_callback(message: dict) -> None:
            nonlocal callback_called, received_message
            callback_called = True
            received_message = message

        agent_service.register_message_callback(test_callback)

        # Simulate receiving a message
        await agent_service._broadcast_message({
            "type": "agent_message",
            "agent_name": "test-agent-1",
            "content": "Hello",
        })

        assert callback_called is True
        assert received_message is not None
        assert received_message["agent_name"] == "test-agent-1"

    async def test_unregister_message_callback(
        self, agent_service: AgentService
    ) -> None:
        """Test unregistering a message callback."""
        callback_called = False

        async def test_callback(message: dict) -> None:
            nonlocal callback_called
            callback_called = True

        agent_service.register_message_callback(test_callback)
        agent_service.unregister_message_callback(test_callback)

        # Simulate receiving a message
        await agent_service._broadcast_message({
            "type": "agent_message",
            "agent_name": "test-agent-1",
            "content": "Hello",
        })

        # Callback should not be called after unregistering
        assert callback_called is False


class TestAgentServiceStatusConversion:
    """Test status conversion between different formats."""

    async def test_convert_agent_status_idle(
        self, agent_service: AgentService
    ) -> None:
        """Test converting idle status."""
        # This tests the internal status conversion logic
        status = agent_service._convert_status("idle")
        assert status == AgentStatus.IDLE

    async def test_convert_agent_status_unknown(
        self, agent_service: AgentService
    ) -> None:
        """Test converting unknown status defaults to idle."""
        status = agent_service._convert_status("unknown_status")
        # Unknown statuses should default to IDLE or be handled gracefully
        assert status in [AgentStatus.IDLE, AgentStatus.ERROR]


class TestAgentServiceSendMessageFireAndForget:
    """Test send_message with fire-and-forget pattern."""

    async def test_send_message_returns_immediately(
        self, agent_service: AgentService
    ) -> None:
        """Test that send_message returns immediately without blocking."""
        import asyncio
        import time

        await agent_service.initialize()

        # Mock agent connection
        mock_root = AsyncMock()
        # Simulate slow agent response (10 seconds)
        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(10)
            return {"success": True, "response": "done"}

        mock_root.invoke_agent = slow_invoke
        mock_conn = MagicMock()

        agent_service._roots["test-agent-1"] = mock_root
        agent_service._connections["test-agent-1"] = mock_conn

        # Call should return immediately (not wait 10s)
        start = time.time()
        result = await agent_service.send_message("test-agent-1", "hello")
        elapsed = time.time() - start

        # Should be instant (< 1 second)
        assert elapsed < 1.0, f"send_message took {elapsed}s, expected < 1s"
        assert result["success"] is True
        assert "WebSocket" in result["message"]

    async def test_send_message_broadcasts_success_callback(
        self, agent_service: AgentService
    ) -> None:
        """Test that send_message success callback broadcasts result."""
        import asyncio

        await agent_service.initialize()

        # Track broadcast messages
        broadcast_messages = []
        async def capture_broadcast(msg):
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        # Mock agent connection
        mock_root = AsyncMock()
        mock_root.invoke_agent = AsyncMock(
            return_value={"success": True, "response": "agent response"}
        )
        mock_conn = MagicMock()

        agent_service._roots["test-agent-1"] = mock_root
        agent_service._connections["test-agent-1"] = mock_conn

        # Send message
        result = await agent_service.send_message("test-agent-1", "hello")
        assert result["success"] is True

        # Wait for fire-and-forget callback to execute
        await asyncio.sleep(0.5)

        # Should have received broadcast with agent_response
        assert len(broadcast_messages) > 0
        response_msg = next((m for m in broadcast_messages if m.get("type") == "agent_response"), None)
        assert response_msg is not None
        assert response_msg["agent_name"] == "test-agent-1"
        assert response_msg["success"] is True

    async def test_send_message_broadcasts_error_callback(
        self, agent_service: AgentService
    ) -> None:
        """Test that send_message error callback broadcasts errors."""
        import asyncio

        await agent_service.initialize()

        # Track broadcast messages
        broadcast_messages = []
        async def capture_broadcast(msg):
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        # Mock agent connection that raises error
        mock_root = AsyncMock()
        mock_root.invoke_agent = AsyncMock(
            side_effect=Exception("Agent error")
        )
        mock_conn = MagicMock()

        agent_service._roots["test-agent-1"] = mock_root
        agent_service._connections["test-agent-1"] = mock_conn

        # Send message
        result = await agent_service.send_message("test-agent-1", "hello")
        assert result["success"] is True  # Still returns success (fire-and-forget)

        # Wait for fire-and-forget error callback
        await asyncio.sleep(0.5)

        # Should have received broadcast with agent_error
        assert len(broadcast_messages) > 0
        error_msg = next((m for m in broadcast_messages if m.get("type") == "agent_error"), None)
        assert error_msg is not None
        assert error_msg["agent_name"] == "test-agent-1"
        assert error_msg["success"] is False
        assert "Agent error" in error_msg["error"]

    async def test_send_message_interrupted_result_is_not_an_error_or_response(
        self, agent_service: AgentService
    ) -> None:
        """A user-initiated interrupt comes back as a clean
        ``{"success": False, "interrupted": True}`` result (NOT a remote
        exception — see exposed_invoke_agent). The web must NOT broadcast
        it as ``agent_error`` (no spurious popup / Last Error) and NOT as
        ``agent_response`` (no false "processed successfully" toast). The
        agent already broadcast ``agent_interrupted`` separately.
        Regression for the 2026-05-20 interrupt-as-error bug.
        """
        import asyncio

        await agent_service.initialize()

        broadcast_messages = []
        async def capture_broadcast(msg):
            broadcast_messages.append(msg)
        agent_service.register_message_callback(capture_broadcast)

        mock_root = AsyncMock()
        mock_root.invoke_agent = AsyncMock(
            return_value={
                "success": False,
                "interrupted": True,
                "message": "Agent invocation interrupted by user",
            }
        )
        mock_conn = MagicMock()
        agent_service._roots["test-agent-1"] = mock_root
        agent_service._connections["test-agent-1"] = mock_conn

        await agent_service.send_message("test-agent-1", "hello")
        await asyncio.sleep(0.5)

        # No agent_error broadcast (the user's complaint).
        assert not any(
            m.get("type") == "agent_error" for m in broadcast_messages
        ), f"interrupt must not broadcast agent_error; got {broadcast_messages}"
        # No agent_response broadcast (no false success).
        assert not any(
            m.get("type") == "agent_response" for m in broadcast_messages
        ), f"interrupt must not broadcast agent_response; got {broadcast_messages}"

    async def test_send_message_has_no_timeout(
        self, agent_service: AgentService
    ) -> None:
        """Test that send_message does NOT have a timeout.

        Agent invocation can take minutes (complex tasks, TaskManager loops,
        interrupt approval waiting). A timeout here only causes a false error
        in the UI while the agent keeps working. Connection errors are handled
        by RPyC (EOFError/ConnectionError).
        """
        await agent_service.initialize()

        mock_root = AsyncMock()
        mock_conn = MagicMock()
        agent_service._roots["test-agent-1"] = mock_root
        agent_service._connections["test-agent-1"] = mock_conn

        # Patch fire_and_forget_async to capture the timeout argument
        from unittest.mock import patch
        with patch("rpyc.utils.helpers.fire_and_forget_async") as mock_faf:
            await agent_service.send_message("test-agent-1", "hello")

            assert mock_faf.called
            kwargs = mock_faf.call_args.kwargs
            assert kwargs.get("timeout") is None, \
                "invoke_agent must NOT have a timeout — see comment in send_message()"

    async def test_send_message_not_connected_auto_connects(
        self, agent_service: AgentService
    ) -> None:
        """Test that send_message auto-connects if not connected."""
        await agent_service.initialize()

        # Mock connect_to_agent
        agent_service.connect_to_agent = AsyncMock(return_value=False)

        # Send message to unconnected agent
        result = await agent_service.send_message("test-agent-1", "hello")

        # Should fail to connect and return error
        assert result["success"] is False
        assert "Failed to connect" in result["error"]
