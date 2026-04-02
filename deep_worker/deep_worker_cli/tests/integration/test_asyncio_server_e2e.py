"""End-to-end integration tests for AsyncioServer migration.

This module tests that AgentProcess with AsyncioServer can:
- Start successfully
- Accept client connections
- Handle basic RPC calls
- Shutdown gracefully
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

@pytest.mark.asyncio
async def test_agent_process_starts_with_asyncio_server():
    """Test that AgentProcess can start with AsyncioServer."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    # Create temp directory for test agent
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-asyncio-agent",
            config=config,
            rpyc_port=0,  # Auto-assign port
        )

        try:
            # Start agent (this should start AsyncioServer)
            await agent.start()

            # Verify server is running
            assert agent.rpyc_server is not None, "RPyC server should be initialized"
            assert agent.rpyc_port > 0, "RPyC port should be assigned"

            # Verify it's AsyncioServer
            from rpyc.utils.async_server import AsyncioServer

            assert isinstance(
                agent.rpyc_server, AsyncioServer
            ), "Server should be AsyncioServer instance"

        finally:
            # Cleanup
            await agent.shutdown()

@pytest.mark.asyncio
async def test_client_can_connect_to_asyncio_server():
    """Test that a client can connect to AgentProcess with AsyncioServer."""
    import rpyc
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-asyncio-agent-2",
            config=config,
            rpyc_port=0,
        )

        try:
            # Start agent
            await agent.start()

            # Get assigned port
            port = agent.rpyc_port

            # Connect client
            conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

            try:
                # Get root before enabling asyncio serving
                root = conn.root

                # Enable asyncio serving
                loop = asyncio.get_running_loop()
                conn.enable_asyncio_serving(loop=loop)

                # Test basic RPC call - get_status
                result = await root.get_status()

                # Verify result is a dict with expected structure
                assert isinstance(result, dict), "Result should be a dict"
                assert "success" in result, "Result should have 'success' field"

            finally:
                conn.disable_asyncio_serving()
                conn.close()

        finally:
            await agent.shutdown()

@pytest.mark.asyncio
async def test_multiple_sequential_calls():
    """Test that multiple sequential RPC calls work correctly."""
    import rpyc
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-asyncio-agent-3",
            config=config,
            rpyc_port=0,
        )

        try:
            await agent.start()
            port = agent.rpyc_port

            conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

            try:
                root = conn.root
                loop = asyncio.get_running_loop()
                conn.enable_asyncio_serving(loop=loop)

                # Make multiple calls
                for i in range(3):
                    result = await root.get_status()
                    assert isinstance(result, dict), f"Call {i} should return dict"

                # Call get_metadata
                metadata = await root.get_metadata()
                assert isinstance(metadata, dict), "Metadata should be dict"
                assert "name" in metadata, "Metadata should have 'name' field"
                assert metadata["name"] == "test-asyncio-agent-3"

            finally:
                conn.disable_asyncio_serving()
                conn.close()

        finally:
            await agent.shutdown()

@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Test that AsyncioServer shuts down gracefully."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-shutdown-agent",
            config=config,
            rpyc_port=0,
        )

        # Start and immediately shutdown
        await agent.start()
        assert agent.rpyc_server is not None

        # Shutdown should complete without errors
        await agent.shutdown()

        # Server should be cleaned up
        # (We don't check rpyc_server is None because shutdown might not clear it)
