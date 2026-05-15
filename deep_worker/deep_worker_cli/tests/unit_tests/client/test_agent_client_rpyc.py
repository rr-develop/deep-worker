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

        # Mock registry to return agent info.
        # NOTE: AgentClient no longer uses a dedicated `RpycAsyncSession`
        # wrapper — it calls `rpyc.core.async_connect.async_connect`
        # directly (see client/agent_client.py ~line 101). We mock
        # that entry point instead.
        with patch("deep_worker_cli.client.agent_client.ProcessRegistry") as mock_registry_cls:
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_cls.return_value = mock_registry

            with patch(
                "deep_worker_cli.client.agent_client.async_connect",
                new_callable=AsyncMock,
            ) as mock_async_connect:
                mock_conn = MagicMock()
                mock_async_connect.return_value = mock_conn

                # Create client and connect
                client = AgentClient("test_agent")
                await client.connect()

                # async_connect must have been awaited exactly once
                # with this agent's port.
                mock_async_connect.assert_awaited_once()
                ((host, port), _kw) = (
                    mock_async_connect.await_args.args,
                    mock_async_connect.await_args.kwargs,
                )
                assert host == "127.0.0.1"
                assert port == 18861
                assert client.connected is True
                assert client._conn is mock_conn

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

        # invoke_agent now routes through `_send_command` which uses
        # low-level `self._conn.async_request(HANDLE_ASYNC_CALLATTR,
        # self._root, "exposed_invoke_agent", (message, stream), ())`
        # and awaits the result (a JSON STRING from the server).
        # See client/agent_client.py ~line 204-218.
        import json as _json

        with patch("deep_worker_cli.client.agent_client.ProcessRegistry") as mock_registry_cls:
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_cls.return_value = mock_registry

            with patch(
                "deep_worker_cli.client.agent_client.async_connect",
                new_callable=AsyncMock,
            ) as mock_async_connect:
                # Fake AsyncResult returned by conn.async_request — it's
                # awaitable and resolves to a JSON-encoded reply.
                async def fake_async_request(*args, **kwargs):
                    return _json.dumps({
                        "success": True,
                        "response": {
                            "role": "assistant",
                            "content": "The answer is 6",
                        },
                    })

                mock_conn = MagicMock()
                mock_conn.async_request = MagicMock(side_effect=fake_async_request)
                # `.root` is accessed lazily by `_ensure_root`; any
                # MagicMock works for our purposes.
                mock_conn.root = MagicMock()
                mock_async_connect.return_value = mock_conn

                client = AgentClient("test_agent")
                await client.connect()
                response = await client.invoke_agent("3+3", stream=False)

                assert response["success"], "Response should be successful"
                assert response["response"]["content"] == "The answer is 6"
                # The RPC must have been dispatched exactly once.
                assert mock_conn.async_request.call_count == 1
                # And aimed at `exposed_invoke_agent`.
                handler_args = mock_conn.async_request.call_args.args
                assert "exposed_invoke_agent" in handler_args

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
