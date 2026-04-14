"""Tests for WebSocket router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.websocket import ConnectionManager, router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def connection_manager() -> ConnectionManager:
    """Create a ConnectionManager for testing."""
    return ConnectionManager()


class TestConnectionManager:
    """Test ConnectionManager functionality."""

    async def test_connect_adds_connection(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that connect adds a connection to active list."""
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()

        await connection_manager.connect(mock_websocket)

        assert len(connection_manager.active_connections) == 1
        assert mock_websocket in connection_manager.active_connections
        mock_websocket.accept.assert_called_once()

    def test_disconnect_removes_connection(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that disconnect removes a connection."""
        mock_websocket = MagicMock()
        connection_manager.active_connections.append(mock_websocket)

        connection_manager.disconnect(mock_websocket)

        assert len(connection_manager.active_connections) == 0
        assert mock_websocket not in connection_manager.active_connections

    def test_disconnect_nonexistent_connection(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that disconnecting non-existent connection doesn't error."""
        mock_websocket = MagicMock()
        
        # Should not raise exception
        connection_manager.disconnect(mock_websocket)
        
        assert len(connection_manager.active_connections) == 0

    async def test_broadcast_to_all_connections(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that broadcast sends to all connections."""
        from fastapi.websockets import WebSocketState
        
        # Create mock connections
        mock_ws1 = MagicMock()
        mock_ws1.client_state = WebSocketState.CONNECTED
        mock_ws1.send_json = AsyncMock()

        mock_ws2 = MagicMock()
        mock_ws2.client_state = WebSocketState.CONNECTED
        mock_ws2.send_json = AsyncMock()

        connection_manager.active_connections = [mock_ws1, mock_ws2]

        message = {"type": "test", "content": "hello"}
        await connection_manager.broadcast(message)

        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    async def test_broadcast_removes_failed_connections(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that broadcast removes connections that fail to send."""
        from fastapi.websockets import WebSocketState
        
        # Create one good and one bad connection
        mock_ws_good = MagicMock()
        mock_ws_good.client_state = WebSocketState.CONNECTED
        mock_ws_good.send_json = AsyncMock()

        mock_ws_bad = MagicMock()
        mock_ws_bad.client_state = WebSocketState.DISCONNECTED  # Already disconnected

        connection_manager.active_connections = [mock_ws_good, mock_ws_bad]

        await connection_manager.broadcast({"type": "test"})

        # Good connection should still be active
        assert mock_ws_good in connection_manager.active_connections
        # Bad connection should be removed
        assert mock_ws_bad not in connection_manager.active_connections

    async def test_broadcast_empty_connections(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test that broadcast with no connections doesn't error."""
        # Should not raise exception
        await connection_manager.broadcast({"type": "test"})


class TestWebSocketEndpoint:
    """Test WebSocket endpoint."""

    @pytest.mark.skip(reason="WebSocket testing requires more complex setup")
    def test_websocket_connection(self) -> None:
        """Test WebSocket connection and message flow.
        
        This would require TestClient with WebSocket support and proper
        async handling. Will be covered in integration tests.
        """
        pass

    @pytest.mark.skip(reason="WebSocket testing requires more complex setup")
    def test_websocket_ping_pong(self) -> None:
        """Test ping/pong keep-alive.
        
        Will be covered in integration tests.
        """
        pass

    @pytest.mark.skip(reason="WebSocket testing requires more complex setup")
    def test_websocket_broadcast_agent_events(self) -> None:
        """Test that agent events are broadcast to WebSocket clients.
        
        Will be covered in integration tests.
        """
        pass
