"""Test that AgentClient connects via RPyC.

This test ensures that AgentClient uses RPyC for all connections.
Unix sockets are NOT supported - only RPyC.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


class TestAgentClientRPyC:
    """Test AgentClient with RPyC connection mode."""

    @pytest.mark.asyncio
    async def test_connect_via_rpyc(self):
        """Test that AgentClient connects via RPyC (only supported method)."""
        # Create mock agent info with RPyC port
        from pathlib import Path
        from datetime import datetime

        mock_agent_info = AgentProcessInfo(
            name="test_agent",
            pid=12345,
            pid_file=Path("/tmp/test.pid"),
            working_dir="/tmp",
            started_at=datetime.fromisoformat("2026-02-15T10:00:00"),
            rpyc_port=18861,  # RPyC is the ONLY way to connect
            last_activity=datetime.fromisoformat("2026-02-15T10:00:00"),
        )

        # Mock registry to return agent info
        with patch("deep_worker_cli.client.agent_client.ProcessRegistry") as mock_registry_cls:
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_cls.return_value = mock_registry

            # Mock RPyC session
            with patch("deep_worker_cli.client.agent_client.RpycAsyncSession") as mock_rpyc_cls:
                mock_session = AsyncMock()
                mock_session.connect = AsyncMock()
                mock_session.async_call = AsyncMock(return_value={"success": True})
                mock_rpyc_cls.return_value = mock_session

                # Create client and connect
                client = AgentClient("test_agent")
                await client.connect()

                # Verify RPyC was used
                assert mock_rpyc_cls.called, "RpycAsyncSession should be created"
                assert mock_session.connect.called, "RPyC connect should be called"
                assert client.connected, "Client should be connected"

    @pytest.mark.asyncio
    async def test_send_prompt_via_rpyc(self):
        """Test that send_prompt works via RPyC connection."""
        from pathlib import Path
        from datetime import datetime

        mock_agent_info = AgentProcessInfo(
            name="test_agent",
            pid=12345,
            pid_file=Path("/tmp/test.pid"),
            working_dir="/tmp",
            started_at=datetime.fromisoformat("2026-02-15T10:00:00"),
            rpyc_port=18861,
            last_activity=datetime.fromisoformat("2026-02-15T10:00:00"),
        )

        with patch("deep_worker_cli.client.agent_client.ProcessRegistry") as mock_registry_cls:
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_cls.return_value = mock_registry

            with patch("deep_worker_cli.client.agent_client.RpycAsyncSession") as mock_rpyc_cls:
                mock_session = AsyncMock()
                mock_session.connect = AsyncMock()

                # Mock invoke response
                mock_session.async_call = AsyncMock(return_value={
                    "success": True,
                    "response": {
                        "role": "assistant",
                        "content": "The answer is 6"
                    }
                })
                mock_rpyc_cls.return_value = mock_session

                # Create client and send prompt
                client = AgentClient("test_agent")
                await client.connect()
                response = await client.invoke_agent("3+3", stream=False)

                # Verify response
                assert response["success"], "Response should be successful"
                assert response["response"]["content"] == "The answer is 6"
                assert mock_session.async_call.called, "async_call should be invoked"

    @pytest.mark.asyncio
    async def test_fails_without_rpyc_port(self):
        """Test that AgentClient fails if rpyc_port is not available.

        RPyC is the ONLY supported connection method.
        """
        from pathlib import Path
        from datetime import datetime

        # Create agent info with valid rpyc_port, then set to None
        # to simulate corrupted registry data
        mock_agent_info = AgentProcessInfo(
            name="test_agent",
            pid=12345,
            pid_file=Path("/tmp/test.pid"),
            working_dir="/tmp",
            started_at=datetime.fromisoformat("2026-02-15T10:00:00"),
            rpyc_port=18861,  # Valid initially
            last_activity=datetime.fromisoformat("2026-02-15T10:00:00"),
        )
        # Simulate corrupted/missing rpyc_port
        mock_agent_info.rpyc_port = None  # type: ignore[assignment]

        with patch("deep_worker_cli.client.agent_client.ProcessRegistry") as mock_registry_cls:
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_cls.return_value = mock_registry

            from deep_worker_cli.exceptions import IPCConnectionError

            # Create client and try to connect
            client = AgentClient("test_agent")

            # Should raise error because RPyC port is required
            with pytest.raises(IPCConnectionError, match="RPyC port not configured"):
                await client.connect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
