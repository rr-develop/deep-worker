"""Agent client SDK for connecting to deep_worker agents."""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from deep_worker_cli.exceptions import AgentNotFoundError, IPCConnectionError
from deep_worker_cli.ipc.protocol import IPCProtocol
from deep_worker_cli.process.registry import ProcessRegistry

if TYPE_CHECKING:
    from deep_worker_cli.basesession.protocol_types import SessionInfo


class AgentClient:
    """
    Client SDK for connecting to deep_worker agent processes.

    Usage:
        client = AgentClient("project-alpha")
        await client.connect()

        response = await client.send_command({"command": "status"})

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
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.client_id = str(uuid.uuid4())
        self.connected = False

    async def connect(self) -> None:
        """Connect to the agent process.

        Raises:
            AgentNotFoundError: If agent is not found in registry
            IPCConnectionError: If connection fails
        """
        # Lookup agent in registry
        agent_info = self.registry.lookup(self.agent_name)
        if not agent_info:
            raise AgentNotFoundError(
                f"Agent '{self.agent_name}' not found. "
                f"Start it with: deep-worker {self.agent_name}"
            )

        # Connect to Unix socket
        try:
            self.reader, self.writer = await asyncio.open_unix_connection(
                str(agent_info.socket_path)
            )
        except (OSError, ConnectionError) as e:
            raise IPCConnectionError(f"Failed to connect to agent: {e}") from e

        # Send attach command
        response = await self._send_command({
            "command": "attach",
            "client_id": self.client_id,
        })

        if not response.get("success"):
            raise IPCConnectionError(f"Failed to attach: {response.get('error')}")

        self.connected = True

    async def disconnect(self) -> None:
        """Disconnect from the agent."""
        if not self.connected:
            return

        try:
            # Send detach command
            await self._send_command({"command": "detach", "client_id": self.client_id})
        except IPCConnectionError:
            pass  # Ignore errors during disconnect

        # Close connection
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

        self.connected = False
        self.reader = None
        self.writer = None

    async def _send_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send a command and receive response.

        Args:
            command: Command dictionary to send

        Returns:
            Response dictionary

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        if not self.writer or not self.reader:
            raise IPCConnectionError("Not connected to agent")

        try:
            # Send command
            data = IPCProtocol.encode(command)
            self.writer.write(data)
            await self.writer.drain()

            # Receive response
            response_data = await self.reader.readline()
            if not response_data:
                raise IPCConnectionError("Connection closed by agent")

            return IPCProtocol.decode(response_data)

        except (OSError, ConnectionError) as e:
            raise IPCConnectionError(f"Communication error: {e}") from e

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

        return SessionInfo.model_validate(session_data)
