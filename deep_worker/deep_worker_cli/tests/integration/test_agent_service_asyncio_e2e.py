"""End-to-end integration tests for AgentService with AsyncioServer.

This module tests that AgentService works correctly with AsyncioServer:
- Server starts and accepts connections
- Async RPC calls work correctly
- Multiple sequential calls work
- Server shuts down gracefully
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Add the local rpyc_async source checkout to path BEFORE importing rpyc
_rpyc_async_path = Path("rpyc_async")
if str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc


@pytest.mark.asyncio
async def test_agent_service_with_asyncio_server():
    """Test that AgentService works with AsyncioServer."""
    from rpyc.utils.async_server import AsyncioServer
    from deep_worker_cli.rpyc_service.agent_service import AgentService
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from pathlib import Path
    import tempfile

    # Create a bound service class
    class BoundAgentService(AgentService):
        def __init__(self):
            super().__init__()
            # Create minimal mock agent_process
            self._mock_agent = type('MockAgent', (), {
                'name': 'test-agent',
                'config': type('Config', (), {'working_dir': '/tmp'})(),
                'session_manager': None,
            })()
            self.set_agent_process(self._mock_agent)

    # Start AsyncioServer with AgentService
    server = AsyncioServer(BoundAgentService, port=0)
    await server.start()

    try:
        # Get assigned port
        port = server.listener.getsockname()[1]
        assert port > 0, "Server should get assigned port"

        # Connect client
        conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

        try:
            # Get root before enabling asyncio serving
            root = conn.root

            # Enable asyncio serving
            loop = asyncio.get_running_loop()
            conn.enable_asyncio_serving(loop=loop)

            # Test exposed_get_status
            result = await root.get_status()
            assert isinstance(result, dict), "Result should be dict"
            assert "success" in result, "Result should have success field"

            # Test exposed_get_metadata
            metadata = await root.get_metadata()
            assert isinstance(metadata, dict), "Metadata should be dict"
            assert "name" in metadata, "Metadata should have name field"

            # Test multiple sequential calls
            for i in range(3):
                result = await root.get_status()
                assert isinstance(result, dict), f"Call {i} should return dict"

        finally:
            conn.disable_asyncio_serving()
            conn.close()

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_agent_service_overridden_base_methods():
    """Test that base RPyC Service methods are properly async."""
    from rpyc.utils.async_server import AsyncioServer
    from deep_worker_cli.rpyc_service.agent_service import AgentService
    import asyncio

    class BoundAgentService(AgentService):
        def __init__(self):
            super().__init__()
            self._mock_agent = type('MockAgent', (), {
                'name': 'test-agent-2',
                'config': type('Config', (), {'working_dir': '/tmp'})(),
                'session_manager': None,
            })()
            self.set_agent_process(self._mock_agent)

    server = AsyncioServer(BoundAgentService, port=0)
    await server.start()

    try:
        port = server.listener.getsockname()[1]
        conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

        try:
            root = conn.root
            loop = asyncio.get_running_loop()
            conn.enable_asyncio_serving(loop=loop)

            # Test exposed_get_service_name (should be async)
            service_name = await root.get_service_name()
            assert service_name == "AGENT_SERVICE"

            # Test exposed_get_service_aliases (should be async)
            aliases = await root.get_service_aliases()
            assert isinstance(aliases, list)

        finally:
            conn.disable_asyncio_serving()
            conn.close()

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_server_handles_multiple_clients_sequentially():
    """Test that AsyncioServer can handle multiple clients connecting sequentially."""
    from rpyc.utils.async_server import AsyncioServer
    from deep_worker_cli.rpyc_service.agent_service import AgentService
    import asyncio

    class BoundAgentService(AgentService):
        def __init__(self):
            super().__init__()
            self._mock_agent = type('MockAgent', (), {
                'name': 'test-multi-client',
                'config': type('Config', (), {'working_dir': '/tmp'})(),
                'session_manager': None,
            })()
            self.set_agent_process(self._mock_agent)

    server = AsyncioServer(BoundAgentService, port=0)
    await server.start()

    try:
        port = server.listener.getsockname()[1]

        # Connect first client
        conn1 = rpyc.connect("localhost", port, config={"allow_all_attrs": True})
        try:
            root1 = conn1.root
            loop = asyncio.get_running_loop()
            conn1.enable_asyncio_serving(loop=loop)

            result1 = await root1.get_metadata()
            assert result1["name"] == "test-multi-client"

        finally:
            conn1.disable_asyncio_serving()
            conn1.close()

        # Connect second client (after first disconnects)
        conn2 = rpyc.connect("localhost", port, config={"allow_all_attrs": True})
        try:
            root2 = conn2.root
            conn2.enable_asyncio_serving(loop=loop)

            result2 = await root2.get_metadata()
            assert result2["name"] == "test-multi-client"

        finally:
            conn2.disable_asyncio_serving()
            conn2.close()

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_asyncio_server_graceful_shutdown():
    """Test that AsyncioServer shuts down gracefully."""
    from rpyc.utils.async_server import AsyncioServer
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    class BoundAgentService(AgentService):
        def __init__(self):
            super().__init__()
            self._mock_agent = type('MockAgent', (), {
                'name': 'test-shutdown',
                'config': type('Config', (), {'working_dir': '/tmp'})(),
                'session_manager': None,
            })()
            self.set_agent_process(self._mock_agent)

    server = AsyncioServer(BoundAgentService, port=0)
    await server.start()

    # Verify server started
    assert server.listener is not None

    # Shutdown should complete without errors
    await server.close()
