"""IPC Server for deep_worker agents."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from deep_worker_cli.exceptions import IPCConnectionError, IPCError
from deep_worker_cli.ipc.protocol import IPCProtocol

# Type alias for message handler
MessageHandler = Callable[[dict[str, Any], "IPCConnection"], Awaitable[dict[str, Any]]]


class IPCConnection:
    """Represents a connection to an IPC client."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        client_id: str,
    ):
        """Initialize the IPC connection.

        Args:
            reader: Stream reader for receiving data
            writer: Stream writer for sending data
            client_id: Unique identifier for this client
        """
        self.reader = reader
        self.writer = writer
        self.client_id = client_id
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message to the client.

        Args:
            message: Dictionary to send

        Raises:
            IPCConnectionError: If connection is closed or send fails
        """
        if self._closed:
            raise IPCConnectionError("Connection is closed")

        try:
            data = IPCProtocol.encode(message)
            self.writer.write(data)
            await self.writer.drain()
        except (OSError, ConnectionError) as e:
            raise IPCConnectionError(f"Failed to send message: {e}") from e

    async def receive(self) -> dict[str, Any]:
        """Receive a message from the client.

        Returns:
            Decoded message dictionary

        Raises:
            IPCConnectionError: If connection is closed or receive fails
        """
        if self._closed:
            raise IPCConnectionError("Connection is closed")

        try:
            # Read until newline (message delimiter)
            data = await self.reader.readline()
            if not data:
                raise IPCConnectionError("Connection closed by client")

            return IPCProtocol.decode(data)
        except (OSError, ConnectionError) as e:
            raise IPCConnectionError(f"Failed to receive message: {e}") from e

    async def close(self) -> None:
        """Close the connection."""
        if not self._closed:
            self._closed = True
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:  # noqa: S110
                pass  # Ignore errors during close


class IPCServer:
    """IPC Server using Unix Domain Sockets."""

    def __init__(
        self,
        socket_path: Path,
        handler: MessageHandler,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix socket
            handler: Async function to handle incoming messages
        """
        self.socket_path = socket_path
        self.handler = handler
        self._server: asyncio.Server | None = None
        self._connections: list[IPCConnection] = []
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start the IPC server.

        Raises:
            IPCError: If server fails to start
        """
        # Remove existing socket if present
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Ensure parent directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Start Unix socket server
            self._server = await asyncio.start_unix_server(
                self._handle_client, path=str(self.socket_path)
            )
        except OSError as e:
            raise IPCError(f"Failed to start IPC server: {e}") from e

    async def stop(self) -> None:
        """Stop the IPC server."""
        if self._server is None:
            return

        # Close server
        self._server.close()
        await self._server.wait_closed()

        # Close all connections
        for connection in self._connections:
            await connection.close()
        self._connections.clear()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Remove socket
        if self.socket_path.exists():
            self.socket_path.unlink()

        self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection.

        Args:
            reader: Stream reader for this client
            writer: Stream writer for this client
        """
        # Generate client ID from address
        peername = writer.get_extra_info("peername")
        client_id = f"client-{id(writer)}"

        connection = IPCConnection(reader, writer, client_id)
        self._connections.append(connection)

        try:
            while not connection._closed:  # noqa: SLF001
                # Receive message
                message = await connection.receive()

                # Handle message
                response = await self.handler(message, connection)

                # Send response
                await connection.send(response)

        except IPCConnectionError:
            # Connection closed or error
            pass
        except Exception as e:
            # Send error response
            try:
                await connection.send({"success": False, "error": str(e)})
            except IPCConnectionError:
                pass
        finally:
            # Cleanup
            await connection.close()
            if connection in self._connections:
                self._connections.remove(connection)
