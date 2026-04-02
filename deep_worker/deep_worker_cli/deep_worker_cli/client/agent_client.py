"""Agent client SDK for connecting to deep_worker agents via RPyC.

This client ONLY supports RPyC connections. Unix sockets are NOT supported.
All communication is done via native rpyc with AsyncioServer.
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deep_worker_cli.exceptions import AgentNotFoundError, IPCConnectionError
from deep_worker_cli.process.registry import ProcessRegistry

import rpyc

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
            # SECURITY: Connect only to localhost (127.0.0.1)
            self._conn = rpyc.connect(
                "127.0.0.1",
                agent_info.rpyc_port,
                config={"allow_public_attrs": True}
            )

            # CRITICAL: Enable asyncio serving BEFORE accessing root!
            # With AsyncioServer, we must enable asyncio serving first,
            # otherwise conn.root will hang/timeout.
            # See: rpyc_async tests/test_asyncio_server_bidirectional.py:180-183
            loop = asyncio.get_running_loop()
            self._conn.enable_asyncio_serving(loop=loop)

            # Now safe to get root (after asyncio serving is enabled)
            self._root = self._conn.root

            self.connected = True

        except Exception as e:
            raise IPCConnectionError(f"Failed to connect via RPyC: {e}") from e

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

        Args:
            command: Command dictionary to send

        Returns:
            Response dictionary

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        if not self.connected or not self._root:
            raise IPCConnectionError("Not connected to agent")

        try:
            command_type = command.get("command")

            # Map commands to RPyC service methods
            if command_type == "status":
                import logging
                logger = logging.getLogger(__name__)
                logger.info("[CLIENT] About to call await root.get_status()...")
                response = await self._root.get_status()
                logger.info(f"[CLIENT] await root.get_status() returned: {response}")
                # Server already returns {"success": True, "session_info": {...}}, no need to wrap again
                return response

            elif command_type == "invoke":
                message = command.get("message", "")
                stream = command.get("stream", False)
                response = await self._root.invoke_agent(message, stream)
                return response

            elif command_type == "get_history":
                limit = command.get("limit", 50)
                offset = command.get("offset", 0)
                response = await self._root.get_history(limit, offset)
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

        # Parse session_info from response
        session_data = response.get("session_info")
        if not session_data:
            raise IPCConnectionError("No session_info in status response")

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[CLIENT] About to call SessionInfo.model_validate()...")
        result = SessionInfo.model_validate(session_data)
        logger.info(f"[CLIENT] model_validate() completed, returning SessionInfo")
        return result

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
