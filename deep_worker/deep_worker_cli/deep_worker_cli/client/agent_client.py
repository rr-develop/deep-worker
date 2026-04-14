"""Agent client SDK for connecting to deep_worker agents via RPyC.

This client ONLY supports RPyC connections. Unix sockets are NOT supported.
All communication is done via native rpyc with AsyncioServer.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deep_worker_cli.exceptions import AgentNotFoundError, IPCConnectionError
from deep_worker_cli.process.registry import ProcessRegistry

# ============================================================================
# CRITICAL: RPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP!
# ============================================================================
# ALWAYS import rpyc from the local rpyc_async source checkout.
# This ensures we use rpyc_async with AsyncioServer support.
# ============================================================================
_rpyc_async_path = Path("rpyc_async")
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc
from rpyc.core.async_connect import async_connect  # type: ignore[import-untyped]
from rpyc.utils.classic import obtain

if TYPE_CHECKING:
    from deep_worker_cli.basesession.protocol_types import SessionInfo


class AgentClient:
    """
    Client SDK for connecting to deep_worker agent processes via RPyC.

    ONLY RPyC connections are supported. Unix sockets are NOT supported.

    Usage:
        client = AgentClient("project-alpha")
        await client.connect()  # Connects via RPyC

        response = await client.invoke_agent("Hello")

        await client.disconnect()
    """

    def __init__(
        self,
        agent_name: str,
        registry: ProcessRegistry | None = None,
    ):
        """Initialize the agent client.

        Args:
            agent_name: Name of the agent to connect to
            registry: Optional ProcessRegistry instance. If None, uses default.
        """
        self.agent_name = agent_name
        self.registry = registry or ProcessRegistry()
        self.client_id = str(uuid.uuid4())
        self.connected = False

        # RPyC is the ONLY connection method
        self._conn: rpyc.Connection | None = None
        self._root: Any = None  # RPyC root reference (obtained before enable_asyncio_serving)

    async def connect(self) -> None:
        """Connect to the agent process via RPyC.

        RPyC is the ONLY supported connection method.

        Raises:
            AgentNotFoundError: If agent is not found in registry
            IPCConnectionError: If connection fails or RPyC port not configured
        """
        # Lookup agent in registry
        agent_info = self.registry.lookup(self.agent_name)
        if not agent_info:
            raise AgentNotFoundError(
                f"Agent '{self.agent_name}' not found. "
                f"Start it with: deep-worker {self.agent_name}"
            )

        # RPyC port is REQUIRED
        if not agent_info.rpyc_port:
            raise IPCConnectionError(
                f"RPyC port not configured for agent '{self.agent_name}'. "
                f"Agent must be started with RPyC support."
            )

        # Connect via RPyC (ONLY method)
        try:
            # Get event loop
            loop = asyncio.get_running_loop()

            # FULLY ASYNC connection - NO blocking, NO threads!
            # Uses asyncio.sock_connect() under the hood for non-blocking socket I/O
            self._conn = await async_connect(
                "127.0.0.1",
                agent_info.rpyc_port,
                config={"allow_public_attrs": True},
                timeout=10.0,
                loop=loop
            )

            # NOTE: async_connect() already enables asyncio serving internally,
            # so we don't need to call enable_asyncio_serving() again!

            # CRITICAL: DO NOT access self._conn.root here!
            # Accessing .root triggers RPC handshake which BLOCKS in async context.
            # Instead, use lazy access via _ensure_root() (accessed on first use).
            # self._root stays None until first RPC call.

            self.connected = True

        except asyncio.TimeoutError as e:
            raise IPCConnectionError(f"Connection timeout after 10s") from e
        except Exception as e:
            raise IPCConnectionError(f"Failed to connect via RPyC: {e}") from e

    def _ensure_root(self) -> None:
        """Lazy initialization of _root.

        CRITICAL: This MUST be called before any RPC method that uses self._root.
        First call will access self._conn.root which triggers RPC handshake (may block once).
        Subsequent calls use cached self._root (fast).
        """
        if self._root is None and self._conn is not None:
            self._root = self._conn.root

    async def disconnect(self) -> None:
        """Disconnect from the agent (closes RPyC session)."""
        if not self.connected:
            return

        # Close RPyC connection
        if self._conn:
            try:
                self._conn.disable_asyncio_serving()
                self._conn.close()
            except Exception:
                pass  # Ignore errors during disconnect
            self._conn = None
            self._root = None

        self.connected = False

    async def _send_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send a command via RPyC.

        OPTIMIZATION: Uses low-level HANDLE_ASYNC_CALLATTR to eliminate extra
        RPyC messages. Instead of:
          1. SEND getattr(root, 'method') -> RECV method_netref -> SEND call() -> RECV result
        We do:
          1. SEND callattr(root, 'method', args) -> RECV result

        This reduces RPyC protocol messages from 6 to 2 per call (see docs/RPYC_SERIALIZATION_DESIGN.md).

        Args:
            command: Command dictionary to send

        Returns:
            Response dictionary

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        if not self.connected or not self._conn:
            raise IPCConnectionError("Not connected to agent")

        # Lazy root initialization (first call triggers RPC handshake)
        self._ensure_root()

        try:
            command_type = command.get("command")

            # Import RPyC consts for low-level protocol handlers
            from rpyc.core import consts

            # Map commands to RPyC service methods using low-level HANDLE_ASYNC_CALLATTR
            # This bypasses Python's attribute access (which creates intermediate netrefs)
            # and directly calls the remote method in a single RPC roundtrip.

            if command_type == "status":
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                # NOTE: Must use "exposed_" prefix for actual method name
                # Format: (handler, root_netref, method_name, args_tuple, kwargs_list)
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,              # root object
                    "exposed_get_status",    # method name (with exposed_ prefix!)
                    (),                      # args tuple (empty for get_status)
                    ()                       # kwargs list (empty)
                )
                # Wait for result (still JSON string from server)
                response_json_str = await async_result
                # Deserialize JSON string to local dict (no RPC overhead for field access)
                response = json.loads(response_json_str)
                return response

            elif command_type == "invoke":
                message = command.get("message", "")
                stream = command.get("stream", False)
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_invoke_agent",  # method name (with exposed_ prefix!)
                    (message, stream),       # args tuple
                    ()                       # kwargs list (empty)
                )
                response_json_str = await async_result
                # Deserialize JSON string to local dict
                response = json.loads(response_json_str)
                return response

            elif command_type == "get_history":
                limit = command.get("limit", 50)
                offset = command.get("offset", 0)
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_get_history",   # method name (with exposed_ prefix!)
                    (limit, offset),         # args tuple
                    ()                       # kwargs list (empty)
                )
                response_json_str = await async_result
                # Deserialize JSON string to local dict
                response = json.loads(response_json_str)
                return response

            elif command_type in ("attach", "detach"):
                # RPyC doesn't need explicit attach/detach
                return {"success": True}

            else:
                raise IPCConnectionError(f"Unknown command type: {command_type}")

        except Exception as e:
            raise IPCConnectionError(f"RPyC communication error: {e}") from e

    async def get_status(self) -> "SessionInfo":
        """Get agent status as SessionInfo.

        This method queries the agent's status and returns a SessionInfo object
        compatible with the BaseSession protocol.

        OPTIMIZATION: Agent now returns minimal status response (status + error only)
        to reduce serialization overhead. We reconstruct minimal SessionInfo locally.

        Returns:
            SessionInfo with current agent state

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({"command": "status"})

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to get status: {response.get('error', 'Unknown error')}"
            )

        # Import here to avoid circular dependency
        from deep_worker_cli.basesession.protocol_types import SessionInfo
        from datetime import datetime

        # OPTIMIZATION: Handle both old format (session_info) and new format (minimal)
        session_data = response.get("session_info")

        if session_data:
            # Old format - full SessionInfo (backward compatibility)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[CLIENT] Using old session_info format")
            result = SessionInfo.model_validate(session_data)
            logger.info(f"[CLIENT] model_validate() completed, returning SessionInfo")
            return result
        else:
            # New format - minimal status response
            # Reconstruct minimal SessionInfo with only status field populated
            status_str = response.get("status", "unknown")
            error = response.get("error")

            return SessionInfo(
                name=self.agent_name,
                session_type="langchain",
                status=status_str,
                working_dir="/tmp",  # Placeholder (not used)
                created_at=datetime.now(),  # Placeholder (not used)
                last_activity=None,
                tasks_completed=0,
                error_message=error,
                data=None,
                metadata=None,
            )

    async def invoke_agent(
        self, message: str, stream: bool = False
    ) -> dict[str, Any]:
        """Send a message to the agent and get response.

        Args:
            message: User message to send to the agent
            stream: Whether to stream the response (default: False)

        Returns:
            Response dictionary with format:
                {
                    "success": bool,
                    "response": {
                        "role": "assistant",
                        "content": str
                    }
                }

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({
            "command": "invoke",
            "message": message,
            "stream": stream,
        })

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to invoke agent: {response.get('error', 'Unknown error')}"
            )

        return response

    async def get_history(
        self, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Get conversation history from the agent.

        Args:
            limit: Maximum number of messages to retrieve (default: 50)
            offset: Number of recent messages to skip for pagination (default: 0)

        Returns:
            Response dictionary with format:
                {
                    "success": bool,
                    "history": [list of message dicts],
                    "total": total message count,
                    "has_more": bool indicating if more messages exist
                }

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({
            "command": "get_history",
            "limit": limit,
            "offset": offset,
        })

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to get history: {response.get('error', 'Unknown error')}"
            )

        return response
