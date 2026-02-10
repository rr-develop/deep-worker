"""Tests for IPC server."""

import asyncio
import json
import socket
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.exceptions import IPCConnectionError
from deep_worker_cli.ipc.server import IPCConnection, IPCServer


@pytest.fixture
def temp_socket_path():
    """Create a temporary socket path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.sock"


@pytest.mark.asyncio
class TestIPCConnection:
    """Tests for IPCConnection."""

    async def test_send_message(self):
        """Test sending a message through connection."""
        # Create a pair of connected sockets
        server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            # Create connection from server socket
            reader, writer = await asyncio.open_unix_connection(sock=server_sock)
            connection = IPCConnection(reader, writer, "test-client")

            # Send message
            message = {"command": "status", "data": {}}
            await connection.send(message)

            # Read from client side
            data = client_sock.recv(1024)
            received = json.loads(data.decode("utf-8").strip())

            assert received == message
        finally:
            server_sock.close()
            client_sock.close()

    async def test_receive_message(self):
        """Test receiving a message through connection."""
        # Create a pair of connected sockets
        server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            # Create connection from server socket
            reader, writer = await asyncio.open_unix_connection(sock=server_sock)
            connection = IPCConnection(reader, writer, "test-client")

            # Send message from client side
            message = {"command": "invoke", "input": "test"}
            client_sock.sendall(json.dumps(message).encode("utf-8") + b"\n")

            # Receive message
            received = await connection.receive()

            assert received == message
        finally:
            server_sock.close()
            client_sock.close()

    async def test_close_connection(self):
        """Test closing a connection."""
        server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            reader, writer = await asyncio.open_unix_connection(sock=server_sock)
            connection = IPCConnection(reader, writer, "test-client")

            await connection.close()

            # Connection should be marked as closed
            assert connection._closed  # noqa: SLF001

            # Subsequent sends should raise error
            with pytest.raises(IPCConnectionError, match="closed"):
                await connection.send({"test": "message"})

        finally:
            server_sock.close()
            client_sock.close()


@pytest.mark.asyncio
class TestIPCServer:
    """Tests for IPCServer."""

    async def test_server_start_stop(self, temp_socket_path):
        """Test starting and stopping server."""

        async def handler(message, connection):
            return {"success": True}

        server = IPCServer(socket_path=temp_socket_path, handler=handler)

        # Start server
        await server.start()

        # Socket should exist
        assert temp_socket_path.exists()

        # Stop server
        await server.stop()

        # Socket should be removed
        assert not temp_socket_path.exists()

    async def test_server_accept_connection(self, temp_socket_path):
        """Test server accepting a connection."""
        connections = []

        async def handler(message, connection):
            connections.append(connection)
            return {"success": True, "received": message}

        server = IPCServer(socket_path=temp_socket_path, handler=handler)

        try:
            # Start server
            await server.start()

            # Give server time to start
            await asyncio.sleep(0.1)

            # Connect client
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            # Send message
            message = {"command": "status"}
            writer.write(json.dumps(message).encode("utf-8") + b"\n")
            await writer.drain()

            # Receive response
            data = await reader.readline()
            response = json.loads(data.decode("utf-8"))

            assert response["success"] is True
            assert response["received"] == message

            # Cleanup
            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()

    async def test_server_multiple_connections(self, temp_socket_path):
        """Test server handling multiple concurrent connections."""
        connection_count = 0

        async def handler(message, connection):
            nonlocal connection_count
            connection_count += 1
            return {"success": True, "count": connection_count}

        server = IPCServer(socket_path=temp_socket_path, handler=handler)

        try:
            await server.start()
            await asyncio.sleep(0.1)

            # Create multiple concurrent connections
            async def send_message(msg):
                reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))
                writer.write(json.dumps(msg).encode("utf-8") + b"\n")
                await writer.drain()
                data = await reader.readline()
                writer.close()
                await writer.wait_closed()
                return json.loads(data.decode("utf-8"))

            # Send 5 concurrent messages
            messages = [{"command": f"test-{i}"} for i in range(5)]
            results = await asyncio.gather(*[send_message(msg) for msg in messages])

            # All should succeed
            assert all(r["success"] for r in results)
            assert connection_count == 5

        finally:
            await server.stop()
