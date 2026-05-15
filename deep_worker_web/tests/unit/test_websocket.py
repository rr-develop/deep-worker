"""Tests for WebSocket router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketState

from app.routers.websocket import (
    ConnectionManager,
    _make_message_callback,
    router,
)
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


class TestMessageCallbackClosesDeadConnection:
    """Regression for the CLOSE-WAIT leak seen in an earlier incident.

    The per-connection ``message_callback`` registered with
    ``AgentService`` is the only place where a broken peer is
    discovered when there is no inbound traffic. If it merely logs the
    exception and returns, the main receive loop keeps blocking on
    ``websocket.receive_json()`` and the OS-level socket sits in
    CLOSE-WAIT until the process exits. Eighteen of these accumulated
    on the live process and drove a non-stop log storm.

    Contract: when ``send_json`` raises, the callback MUST trigger a
    server-side ``close`` so that the main loop wakes up via
    ``WebSocketDisconnect`` and tears the connection down through the
    normal ``finally`` cleanup.
    """

    async def test_callback_closes_websocket_on_send_failure(self) -> None:
        """A failed send schedules ``websocket.close()`` exactly once."""
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.send_json = AsyncMock(side_effect=RuntimeError("peer is gone"))
        ws.close = AsyncMock()

        callback = _make_message_callback(ws)

        await callback({"type": "agent_status_change"})

        ws.close.assert_awaited_once()

    async def test_callback_does_not_close_when_send_succeeds(self) -> None:
        """Healthy callback path must not touch ``close``."""
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()

        callback = _make_message_callback(ws)

        await callback({"type": "agent_status_change"})

        ws.send_json.assert_awaited_once()
        ws.close.assert_not_called()

    async def test_callback_close_is_idempotent_across_repeated_failures(
        self,
    ) -> None:
        """Once the peer is dead the callback must not keep calling close.

        ``_broadcast_message`` fans out concurrently; multiple in-flight
        callbacks for the same dead websocket can land here at almost
        the same instant. We must close at most once and tolerate the
        race — if Starlette has already moved the socket to
        ``DISCONNECTED`` we must skip ``close`` entirely.
        """
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.send_json = AsyncMock(side_effect=RuntimeError("peer is gone"))
        ws.close = AsyncMock()

        callback = _make_message_callback(ws)

        await callback({"type": "agent_status_change"})
        ws.client_state = WebSocketState.DISCONNECTED
        await callback({"type": "agent_status_change"})

        assert ws.close.await_count == 1

    async def test_callback_swallows_close_failure(self) -> None:
        """``close`` itself can raise on a half-torn-down socket.

        We must not let that exception propagate out of the callback —
        ``asyncio.gather(..., return_exceptions=True)`` would log it
        but the WS would never get cleaned up if anything downstream
        re-raises.
        """
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.send_json = AsyncMock(side_effect=RuntimeError("peer is gone"))
        ws.close = AsyncMock(side_effect=RuntimeError("already closed"))

        callback = _make_message_callback(ws)

        await callback({"type": "agent_status_change"})  # must not raise
