"""Integration tests for AgentService with real agents."""

import asyncio
from typing import Any

import pytest

from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from tests.integration.agent_fixture import agent_fixture


@pytest.mark.integration
class TestAgentServiceRealConnection:
    """Test AgentService with real agent connections."""

    def test_connect_to_real_agent(self) -> None:
        """Test connecting to a real agent via RPyC."""
        with agent_fixture() as agent_info:
            # Create AgentService with real ProcessRegistry
            registry = ProcessRegistry()
            service = AgentService(registry=registry)

            # Initialize should discover the agent
            asyncio.run(service.initialize())

            # Get the agent
            agent = asyncio.run(service.get_agent(agent_info.name))
            assert agent is not None
            assert agent.name == agent_info.name
            assert agent.pid == agent_info.pid
            assert agent.rpyc_port == agent_info.rpyc_port

            # Test connection
            connected = asyncio.run(service.is_connected(agent_info.name))
            # Initially not connected
            assert connected is False

            # Connect to the agent
            success = asyncio.run(service.connect_to_agent(agent_info.name))
            assert success is True

            # Now should be connected
            connected = asyncio.run(service.is_connected(agent_info.name))
            assert connected is True

            # Get root proxy
            root = service.get_root(agent_info.name)
            assert root is not None

            # Cleanup
            asyncio.run(service.shutdown())

    def test_send_message_to_real_agent(self) -> None:
        """Test getting history from a real agent (doesn't require AI)."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Connect to the agent
            success = asyncio.run(service.connect_to_agent(agent_info.name))
            assert success is True

            # Get connection
            root = service.get_root(agent_info.name)
            assert root is not None

            # Get history (doesn't require AI, just returns conversation history)
            response = asyncio.run(service.get_agent_history(agent_info.name))
            # Should get a response with messages list
            assert isinstance(response, dict)
            assert "messages" in response or "success" in response

            asyncio.run(service.shutdown())

    def test_get_agent_status_from_real_agent(self) -> None:
        """Test getting status from a real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Get the agent status
            agent = asyncio.run(service.get_agent(agent_info.name))
            assert agent is not None

            # Agent should have valid status
            assert agent.status is not None
            # Status should be one of the valid AgentStatus values
            from app.schemas.agent_schemas import AgentStatus
            assert agent.status in [
                AgentStatus.IDLE,
                AgentStatus.RUNNING,
                AgentStatus.WAITING_INPUT,
                AgentStatus.WAITING_APPROVAL,
                AgentStatus.ERROR,
            ]

            asyncio.run(service.shutdown())

    def test_message_callback_with_real_agent(self) -> None:
        """Test message callbacks with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Track callback invocations
            callback_messages: list[dict[str, Any]] = []

            async def test_callback(message: dict[str, Any]) -> None:
                callback_messages.append(message)

            # Register callback
            service.register_message_callback(test_callback)

            # Simulate broadcasting a message
            test_message = {
                "type": "agent_message",
                "agent_name": agent_info.name,
                "content": "Test message",
            }
            asyncio.run(service._broadcast_message(test_message))

            # Give async callback time to execute
            asyncio.run(asyncio.sleep(0.1))

            # Verify callback was called
            assert len(callback_messages) == 1
            assert callback_messages[0]["agent_name"] == agent_info.name

            asyncio.run(service.shutdown())

    def test_multiple_agents_discovery(self) -> None:
        """Test that AgentService discovers multiple agents."""
        from tests.integration.agent_fixture import multiple_agents

        with multiple_agents(count=2) as agents:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Should discover both agents
            discovered_agents = asyncio.run(service.list_agents())
            assert len(discovered_agents) >= 2  # >= because other agents might exist

            # Find our test agents
            our_agent_names = {agent.name for agent in agents}
            discovered_names = {agent.name for agent in discovered_agents}

            for agent_name in our_agent_names:
                assert agent_name in discovered_names

            asyncio.run(service.shutdown())

    def test_reconnect_after_disconnect(self) -> None:
        """Test reconnecting to agent after connection is lost."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Connect first time
            success = asyncio.run(service.connect_to_agent(agent_info.name))
            assert success is True

            root1 = service.get_root(agent_info.name)
            assert root1 is not None

            # Disconnect
            asyncio.run(service.disconnect_from_agent(agent_info.name))

            # Should be disconnected
            connected = asyncio.run(service.is_connected(agent_info.name))
            assert connected is False

            # Try to reconnect
            success = asyncio.run(service.connect_to_agent(agent_info.name))
            assert success is True

            root2 = service.get_root(agent_info.name)
            assert root2 is not None

            # Should be connected again
            connected = asyncio.run(service.is_connected(agent_info.name))
            assert connected is True

            asyncio.run(service.shutdown())
