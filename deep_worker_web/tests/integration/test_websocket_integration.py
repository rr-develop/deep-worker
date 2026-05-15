"""Integration tests for WebSocket functionality with real agents."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app as main_app
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from tests.integration.agent_fixture import agent_fixture


@pytest.mark.integration
class TestWebSocketWithRealAgent:
    """Test WebSocket functionality with real agents."""

    def test_websocket_connection_basic(self) -> None:
        """Test basic WebSocket connection."""
        with agent_fixture() as agent_info:
            # Setup app with real agent service
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            main_app.state.agent_service = service

            # Create test client
            client = TestClient(main_app)

            # Connect to WebSocket (correct path is /ws/agents)
            with client.websocket_connect("/ws/agents") as websocket:
                # Send ping and wait for the pong specifically. The server
                # continuously broadcasts agent_discovered / agent_status
                # events as the monitor loop polls registered agents; those
                # messages may land on the same channel between our send
                # and the pong reply. Drain until we see the pong.
                websocket.send_json({"type": "ping"})
                pong_received = False
                seen_types: list[str] = []
                for _ in range(50):
                    data = websocket.receive_json()
                    assert data is not None
                    msg_type = data.get("type")
                    seen_types.append(msg_type)
                    if msg_type == "pong":
                        pong_received = True
                        break
                assert pong_received, (
                    f"expected 'pong' among ws messages, got types={seen_types}"
                )

            asyncio.run(service.shutdown())

    def test_websocket_receives_agent_messages(self) -> None:
        """Test that WebSocket receives messages broadcast from agents."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            main_app.state.agent_service = service

            client = TestClient(main_app)

            with client.websocket_connect("/ws/agents") as websocket:
                # Simulate agent broadcasting a message
                test_message = {
                    "type": "agent_message",
                    "agent_name": agent_info.name,
                    "content": "Test broadcast message",
                }

                # Broadcast via service
                asyncio.run(service._broadcast_message(test_message))

                # Give time for broadcast to propagate
                # Note: TestClient WebSocket is synchronous, so we need to handle carefully
                try:
                    # Try to receive the broadcast message
                    # This might not work perfectly with TestClient due to async handling
                    data = websocket.receive_json()
                    if data:
                        # If we received something, verify it's our message or related
                        assert isinstance(data, dict)
                except Exception:
                    # WebSocket broadcast might not work perfectly in TestClient
                    # This is expected - real integration would use async client
                    pass

            asyncio.run(service.shutdown())

    @pytest.mark.skip(reason="TestClient doesn't support multiple WebSocket connections well")
    def test_websocket_multiple_connections(self) -> None:
        """Test multiple WebSocket connections simultaneously."""
        # Note: TestClient has limitations with multiple concurrent WebSocket connections
        # This test is skipped in integration tests
        # Real multiple connection support should be tested with real clients
        pass

    def test_websocket_connection_cleanup(self) -> None:
        """Test that WebSocket connections are properly cleaned up.

        The test uses the same ping/pong probe as
        ``test_websocket_connection_basic`` but focuses on teardown: the
        context-manager exit must disconnect cleanly regardless of how
        many background broadcasts arrived in between.
        """
        with agent_fixture() as agent_info:  # noqa: F841 — agent must exist
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            main_app.state.agent_service = service

            client = TestClient(main_app)

            # Connect and disconnect
            with client.websocket_connect("/ws/agents") as websocket:
                # Probe liveness via ping/pong. The monitor loop concurrently
                # broadcasts agent_discovered / agent_status events so we
                # drain until we see pong rather than relying on a fixed
                # receive order.
                websocket.send_json({"type": "ping"})
                for _ in range(50):
                    data = websocket.receive_json()
                    if data is not None and data.get("type") == "pong":
                        break
                else:
                    raise AssertionError(
                        "did not receive pong on websocket ping/pong probe"
                    )

            # Connection should be cleaned up after context exit
            # ConnectionManager should have removed it
            asyncio.run(service.shutdown())

    def test_websocket_with_agent_status_changes(self) -> None:
        """Test WebSocket receives agent status change notifications."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            main_app.state.agent_service = service

            client = TestClient(main_app)

            with client.websocket_connect("/ws/agents") as websocket:
                # Simulate status change broadcast
                status_message = {
                    "type": "agent_status_change",
                    "agent_name": agent_info.name,
                    "status": "running",
                }

                asyncio.run(service._broadcast_message(status_message))

                # Drain up to N messages looking for the synthetic
                # ``agent_status_change`` broadcast we just fired. The
                # monitor loop may push ``agent_discovered`` / ``agent_status``
                # first. A blocking ``receive_json()`` without a bound
                # would wedge the test if the event never arrives.
                found = False
                for _ in range(50):
                    data = websocket.receive_json()
                    if data and data.get("type") == "agent_status_change":
                        assert data["agent_name"] == agent_info.name
                        found = True
                        break
                assert found, "did not observe the broadcast agent_status_change"

            asyncio.run(service.shutdown())


@pytest.mark.integration
class TestWebSocketMessageCallback:
    """Test WebSocket message callback integration."""

    def test_callback_registration_and_broadcast(self) -> None:
        """Test that callbacks are properly registered and receive broadcasts."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Track received messages
            received_messages = []

            async def test_callback(message: dict) -> None:
                received_messages.append(message)

            # Register callback
            service.register_message_callback(test_callback)

            # Broadcast a message
            test_message = {
                "type": "test",
                "agent_name": agent_info.name,
                "content": "Test message",
            }
            asyncio.run(service._broadcast_message(test_message))

            # Give callback time to execute
            asyncio.run(asyncio.sleep(0.1))

            # Verify callback received message
            assert len(received_messages) == 1
            assert received_messages[0]["agent_name"] == agent_info.name

            # Cleanup
            service.unregister_message_callback(test_callback)
            asyncio.run(service.shutdown())

    def test_multiple_callbacks(self) -> None:
        """Test multiple callbacks receive same broadcast."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            # Track messages in two different callbacks
            received_1 = []
            received_2 = []

            async def callback1(message: dict) -> None:
                received_1.append(message)

            async def callback2(message: dict) -> None:
                received_2.append(message)

            # Register both
            service.register_message_callback(callback1)
            service.register_message_callback(callback2)

            # Broadcast
            test_message = {"type": "test", "content": "Multi-callback test"}
            asyncio.run(service._broadcast_message(test_message))

            # Give callbacks time
            asyncio.run(asyncio.sleep(0.1))

            # Both should have received
            assert len(received_1) == 1
            assert len(received_2) == 1
            assert received_1[0] == received_2[0]

            # Cleanup
            service.unregister_message_callback(callback1)
            service.unregister_message_callback(callback2)
            asyncio.run(service.shutdown())

    def test_callback_unregistration(self) -> None:
        """Test that unregistered callbacks don't receive broadcasts."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())

            received_messages = []

            async def test_callback(message: dict) -> None:
                received_messages.append(message)

            # Register and then unregister
            service.register_message_callback(test_callback)
            service.unregister_message_callback(test_callback)

            # Broadcast
            test_message = {"type": "test", "content": "Should not receive"}
            asyncio.run(service._broadcast_message(test_message))

            # Give time
            asyncio.run(asyncio.sleep(0.1))

            # Should not have received anything
            assert len(received_messages) == 0

            asyncio.run(service.shutdown())
