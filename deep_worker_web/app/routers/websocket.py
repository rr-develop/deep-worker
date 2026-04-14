"""WebSocket router for real-time agent communication.

This module provides WebSocket endpoints for:
- Real-time message streaming from agents
- Agent status updates
- Agent discovery/termination events

ARCHITECTURE:
- Single WebSocket connection per client at /ws/agents
- Broadcasts all agent events to all connected clients
- Client-side filtering by selected agent
- Ping/pong for connection keep-alive
"""

import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections and broadcasting.
    
    This follows the standard WebSocket connection manager pattern,
    maintaining a list of active connections and broadcasting messages
    to all connected clients.
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to accept.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to remove.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message dict to broadcast (will be JSON serialized).
        """
        import json

        # Add uptime data to every broadcast message (BEFORE checking connections)
        from app.utils.uptime import get_uptime_data
        message["uptime"] = get_uptime_data()

        if not self.active_connections:
            return

        # CRITICAL: Ensure message is JSON serializable before broadcasting
        # This handles RPyC proxies and other non-serializable objects
        try:
            json_str = json.dumps(message, default=str)
            serializable_message = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            # Fallback: convert all problematic fields to strings
            serializable_message = {}
            for key, value in message.items():
                try:
                    json.dumps({key: value})
                    serializable_message[key] = value
                except (TypeError, ValueError):
                    serializable_message[key] = str(value)
                    logger.warning(f"Converted field '{key}' to string for WebSocket")

        logger.debug(f"Broadcasting message to {len(self.active_connections)} clients")

        # Send to all connections, handling failures gracefully
        disconnected = []
        for connection in self.active_connections:
            try:
                # Check if connection is still open
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(serializable_message)
                else:
                    disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Remove failed connections
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
# NOTE: This is thread-safe because FastAPI runs in a single event loop
connection_manager = ConnectionManager()


@router.websocket("/ws/agents")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent events.
    
    Protocol:
    - Client → Server: {"type": "ping"}
    - Server → Client: {"type": "pong"}
    - Server → Client: {"type": "agent_message", "agent_name": "...", "message": {...}}
    - Server → Client: {"type": "agent_status", "agent_name": "...", "status": "..."}
    - Server → Client: {"type": "agent_discovered", "agent": {...}}
    - Server → Client: {"type": "agent_terminated", "agent_name": "..."}
    
    Args:
        websocket: WebSocket connection.
    """
    await connection_manager.connect(websocket)

    # Get AgentService from app state (declare outside try for finally access)
    agent_service: AgentService = websocket.app.state.agent_service

    # Register callback for agent messages
    async def message_callback(message: dict[str, Any]) -> None:
        """Callback to broadcast agent messages to this WebSocket."""
        await connection_manager.broadcast(message)

    agent_service.register_message_callback(message_callback)

    try:

        # Send initial agent list
        agents = await agent_service.list_agents()
        for agent in agents:
            # Add uptime data to initial agent discovery messages
            from app.utils.uptime import get_uptime_data
            message = {
                "type": "agent_discovered",
                "agent": agent.model_dump(mode="json"),
                "uptime": get_uptime_data(),
            }
            await websocket.send_json(message)

        # Keep connection alive and handle ping/pong
        while True:
            try:
                # Wait for messages from client (ping, etc.)
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    # Add uptime data to ping/pong for keep-alive monitoring
                    from app.utils.uptime import get_uptime_data
                    await websocket.send_json({"type": "pong", "uptime": get_uptime_data()})
                else:
                    logger.warning(f"Unknown message type from client: {data.get('type')}")

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        # Cleanup
        connection_manager.disconnect(websocket)
        
        # Unregister callback if we registered it
        try:
            if 'message_callback' in locals():
                agent_service.unregister_message_callback(message_callback)
        except Exception as e:
            logger.error(f"Error unregistering callback: {e}")


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager.
    
    This is used by other components that need to broadcast messages.
    
    Returns:
        The global ConnectionManager instance.
    """
    return connection_manager
