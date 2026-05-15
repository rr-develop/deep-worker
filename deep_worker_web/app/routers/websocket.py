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

import json
import logging
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


def _client_repr(ws: Any) -> str:
    """Best-effort "host:port" identity for a WebSocket connection.

    Used in log lines so operators can tell *which* client caused a
    broadcast failure. Never raises.
    """
    try:
        client = getattr(ws, "client", None)
        if client is None:
            return "client=<unknown>"
        host = getattr(client, "host", None) or (
            client[0] if isinstance(client, tuple) and client else "?"
        )
        port = getattr(client, "port", None) or (
            client[1] if isinstance(client, tuple) and len(client) > 1 else "?"
        )
        return f"client={host}:{port}"
    except Exception:  # noqa: BLE001
        return "client=<repr-failed>"


def _coerce_to_serializable(message: dict[str, Any]) -> dict[str, Any]:
    """Produce a JSON-safe version of ``message``.

    Three stages, in order of preference:

    1. ``json.dumps(message, default=str)`` — fast path for clean
       messages (handles datetimes, enums, pathlib).
    2. Per-field fallback: any key whose value can't be coerced is
       replaced by its ``str(value)``. The per-field warning is
       logged.
    3. If even the per-field fallback produces something that can't
       round-trip through ``json.dumps``, return a synthetic
       ``broadcast_failed`` envelope so the UI knows the payload was
       unrecoverable. The original_type field preserves the class of
       message that was lost, and the error field carries the
       reason.

    Stage 3 exists because ``str(x)`` can itself raise — e.g. an
    object whose ``__str__`` blows up. Silently letting the broadcast
    vanish is worse than telling the UI "something was lost".
    """
    # Stage 1: fast path.
    try:
        return json.loads(json.dumps(message, default=str))
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "broadcast: payload not directly serialisable (%s); "
            "falling back to per-field stringification",
            exc,
        )

    # Stage 2: per-field fallback.
    safe: dict[str, Any] = {}
    for key, value in message.items():
        try:
            json.dumps({key: value})
            safe[key] = value
        except (TypeError, ValueError):
            try:
                safe[key] = str(value)
                logger.warning(
                    "broadcast: coerced field %r to str (original type=%s)",
                    key,
                    type(value).__name__,
                )
            except Exception as inner:  # noqa: BLE001
                # str() itself raised — the value is truly hostile.
                # Emit a structured "broadcast_failed" envelope so
                # the UI can surface the loss instead of silently
                # dropping the message.
                logger.error(
                    "broadcast: field %r is fully unserialisable "
                    "(even str() failed: %s); emitting broadcast_failed",
                    key,
                    inner,
                    exc_info=True,
                )
                return {
                    "type": "broadcast_failed",
                    "original_type": message.get("type"),
                    "reason": "unserialisable_field",
                    "field": key,
                    "error": f"{type(inner).__name__}: {inner}",
                }
        except Exception as other:  # noqa: BLE001
            logger.error(
                "broadcast: unexpected error coercing field %r (%s)",
                key,
                other,
                exc_info=True,
            )
            return {
                "type": "broadcast_failed",
                "original_type": message.get("type"),
                "reason": "coerce_exception",
                "field": key,
                "error": f"{type(other).__name__}: {other}",
            }

    # Stage 3 sanity: if even the per-field dict can't round-trip,
    # fall back to the broadcast_failed envelope as the final
    # guarantee that ``send_json`` won't raise.
    try:
        json.dumps(safe)
        return safe
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "broadcast: per-field fallback still not serialisable (%s); "
            "emitting broadcast_failed",
            exc,
            exc_info=True,
        )
        return {
            "type": "broadcast_failed",
            "original_type": message.get("type"),
            "reason": "roundtrip_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


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

        Failures never propagate: a broken client is pruned, a
        non-serialisable payload is replaced by a synthetic
        ``broadcast_failed`` envelope so the UI sees "something was
        lost" rather than a silent drop. Every failure is logged at
        ERROR with the exception trace so operators can diagnose.

        Args:
            message: Message dict to broadcast (will be JSON serialized).
        """
        # Add uptime data to every broadcast message (BEFORE checking
        # connections). Doing so requires an already-well-formed
        # message dict; ``get_uptime_data`` is pure in-process so it
        # cannot fail — wrap it defensively anyway.
        from app.utils.uptime import get_uptime_data
        try:
            message["uptime"] = get_uptime_data()
        except Exception:  # noqa: BLE001
            logger.exception("get_uptime_data failed; broadcasting without uptime")

        if not self.active_connections:
            return

        serializable_message = _coerce_to_serializable(message)

        logger.debug(
            "Broadcasting message to %d clients", len(self.active_connections)
        )

        # Send to all connections, handling failures gracefully.
        # We identify each client by a short repr so log lines are
        # useful in production ("client=127.0.0.1:12345" rather than
        # "Error broadcasting to connection").
        disconnected = []
        for connection in self.active_connections:
            client_id = _client_repr(connection)
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(serializable_message)
                else:
                    logger.info(
                        "pruning already-disconnected client %s", client_id
                    )
                    disconnected.append(connection)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "broadcast send failed to %s (%s): %s",
                    client_id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
# NOTE: This is thread-safe because FastAPI runs in a single event loop
connection_manager = ConnectionManager()


def _make_message_callback(
    websocket: WebSocket,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Build the per-connection broadcast sink used by ``AgentService``.

    The callback sends each broadcast frame to *this* websocket only
    (broadcasting to all from inside the callback would be N² —
    every callback would re-fan-out to every websocket).

    Lifecycle on send failure (regression-guarded by
    ``test_callback_closes_websocket_on_send_failure``):

    * If ``send_json`` raises, the peer is gone — but the main
      receive loop is parked in ``await websocket.receive_json()``
      and won't notice until traffic resumes from the client side,
      which it never will. The OS-level socket sits in CLOSE-WAIT
      until the process exits.
    * To wake the receive loop we explicitly call
      ``websocket.close()``. Starlette routes that into a
      ``WebSocketDisconnect`` on the receive side, the main loop
      breaks, and the ``finally`` block tears down the
      ``ConnectionManager`` entry and the ``AgentService`` callback
      registration.
    * The close is idempotent: we skip when ``client_state`` is
      already ``DISCONNECTED`` (concurrent broadcasts can hit this
      path many times in the same tick) and we swallow any exception
      from ``close`` itself, since by that point the socket may
      already be half-torn-down by another path.
    """

    async def message_callback(message: dict[str, Any]) -> None:
        try:
            await websocket.send_json(message)
        except Exception as exc:  # noqa: BLE001
            # Logged at DEBUG: a closing tab is the common case and
            # ERROR-level here would reproduce the earlier debug-log-storm
            # incident. The exc_info
            # preserves the trace for anyone raising the level.
            logger.debug(
                "per-connection send failed on %s: %s; closing peer",
                _client_repr(websocket),
                exc,
                exc_info=True,
            )
            # Wake the parked receive loop so the connection is
            # actually torn down instead of lingering in CLOSE-WAIT.
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close()
                except Exception as close_exc:  # noqa: BLE001
                    logger.debug(
                        "close after send failure raised on %s: %s",
                        _client_repr(websocket),
                        close_exc,
                    )

    return message_callback


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

    # Per-connection sink. See ``_make_message_callback`` for the
    # close-on-failure lifecycle that guards against the CLOSE-WAIT
    # leak seen in an earlier incident. ⚠️ Each callback
    # sends to ITS OWN websocket only — NOT broadcast to all (N²
    # duplication otherwise).
    message_callback = _make_message_callback(websocket)
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
            except json.JSONDecodeError as exc:
                # A malformed client frame is a client-side bug — do
                # NOT kill the connection over it. Log at WARNING
                # (not ERROR: this is routinely a human pasting raw
                # text into the console), send the client a
                # structured error, and keep looping.
                logger.warning(
                    "invalid_json frame from %s: %s",
                    _client_repr(websocket),
                    exc,
                )
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "invalid_json",
                            "detail": str(exc),
                        }
                    )
                except Exception:  # noqa: BLE001
                    # If we can't even send back the error, the
                    # connection really is gone — fall through to
                    # disconnect on the next receive.
                    pass
                continue
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}", exc_info=True)
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
