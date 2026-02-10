"""Tests for AgentClient SDK."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.exceptions import AgentNotFoundError
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        registry_file = tmpdir_path / "registry.json"
        lock_file = tmpdir_path / "registry.lock"
        socket_path = tmpdir_path / "test.sock"
        yield {
            "tmpdir": tmpdir_path,
            "registry_file": registry_file,
            "lock_file": lock_file,
            "socket_path": socket_path,
        }


@pytest.fixture
def mock_registry(temp_dirs):
    """Create a mock registry."""
    return ProcessRegistry(
        registry_file=temp_dirs["registry_file"], lock_file=temp_dirs["lock_file"]
    )


@pytest.mark.asyncio
class TestAgentClient:
    """Tests for AgentClient."""

    async def test_connect_to_nonexistent_agent(self, mock_registry, temp_dirs):
        """Test connecting to a non-existent agent raises error."""
        client = AgentClient("nonexistent-agent", registry=mock_registry)

        with pytest.raises(AgentNotFoundError, match="not found"):
            await client.connect()

    async def test_connect_to_registered_agent(self, mock_registry, temp_dirs):
        """Test connecting to a registered agent."""
        import os

        # Register an agent
        agent_info = AgentProcessInfo(
            name="test-agent",
            pid=os.getpid(),
            socket_path=temp_dirs["socket_path"],
            pid_file=temp_dirs["tmpdir"] / "test.pid",
            working_dir="/tmp/test",
            started_at=datetime.now(),
        )
        mock_registry.register(agent_info)

        # Create a mock server
        async def mock_handler(message, connection):
            if message.get("command") == "attach":
                return {"success": True, "agent_name": "test-agent"}
            return {"success": False}

        from deep_worker_cli.ipc.server import IPCServer

        server = IPCServer(socket_path=temp_dirs["socket_path"], handler=mock_handler)
        await server.start()

        try:
            # Connect client
            client = AgentClient("test-agent", registry=mock_registry)
            await client.connect()

            assert client.connected
            assert client.agent_name == "test-agent"

            # Disconnect
            await client.disconnect()
            assert not client.connected

        finally:
            await server.stop()

    async def test_send_command(self, mock_registry, temp_dirs):
        """Test sending a command to agent."""
        import os

        # Register agent
        agent_info = AgentProcessInfo(
            name="test-agent",
            pid=os.getpid(),
            socket_path=temp_dirs["socket_path"],
            pid_file=temp_dirs["tmpdir"] / "test.pid",
            working_dir="/tmp/test",
            started_at=datetime.now(),
        )
        mock_registry.register(agent_info)

        # Create mock server
        async def mock_handler(message, connection):
            if message.get("command") == "attach":
                return {"success": True}
            elif message.get("command") == "status":
                return {"success": True, "status": "idle"}
            return {"success": False}

        from deep_worker_cli.ipc.server import IPCServer

        server = IPCServer(socket_path=temp_dirs["socket_path"], handler=mock_handler)
        await server.start()

        try:
            client = AgentClient("test-agent", registry=mock_registry)
            await client.connect()

            # Send command
            response = await client._send_command({"command": "status"})  # noqa: SLF001

            assert response["success"] is True
            assert response["status"] == "idle"

            await client.disconnect()

        finally:
            await server.stop()
