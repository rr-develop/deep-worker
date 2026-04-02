"""Test rpyc_async integration.

This module verifies that the installed rpyc distribution is the rpyc_async
fork (published on PyPI as ``rpyc-async``), which provides AsyncioServer,
enable_asyncio_serving and the async callbacks used throughout deep_worker.
"""

from __future__ import annotations

import pytest


def test_rpyc_distribution_is_rpyc_async():
    """The installed rpyc must come from the rpyc-async distribution."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("rpyc-async")
    except PackageNotFoundError:
        pytest.fail(
            "rpyc-async distribution not installed. deep_worker requires the "
            "AsyncioServer-capable rpyc fork; add rpyc-async>=1.0.0 to your "
            "environment."
        )

def test_asyncio_server_available():
    """Test that AsyncioServer is available."""
    import rpyc

    assert hasattr(rpyc.utils, "async_server"), (
        "AsyncioServer module not found - wrong rpyc version?"
    )

    from rpyc.utils.async_server import AsyncioServer

    assert AsyncioServer is not None

def test_protocol_version():
    """Test that protocol version is >= 5.1."""
    from rpyc.core import consts

    assert consts.PROTOCOL_VERSION >= (5, 1), (
        f"Wrong protocol version: {consts.PROTOCOL_VERSION}"
    )

def test_enable_asyncio_serving_method_exists():
    """Test that Connection has enable_asyncio_serving method."""
    from rpyc.core.protocol import Connection

    assert hasattr(Connection, "enable_asyncio_serving"), (
        "Connection.enable_asyncio_serving method not found"
    )
    assert hasattr(Connection, "disable_asyncio_serving"), (
        "Connection.disable_asyncio_serving method not found"
    )

def test_async_handlers_available():
    """Test that async handlers module is available."""
    from rpyc.core import async_handlers

    assert async_handlers is not None
    assert hasattr(async_handlers, "_handle_async_call"), (
        "_handle_async_call not found in async_handlers"
    )

def test_asyncresult_awaitable():
    """Test that AsyncResult is awaitable."""
    from rpyc.core.async_ import AsyncResult

    # Check that AsyncResult has __await__ method
    assert hasattr(AsyncResult, "__await__"), (
        "AsyncResult is not awaitable - missing __await__ method"
    )

@pytest.mark.asyncio
async def test_basic_asyncio_server_can_start():
    """Test that AsyncioServer can be instantiated and started."""
    import rpyc
    from rpyc.utils.async_server import AsyncioServer

    class TestService(rpyc.Service):
        async def exposed_hello(self):
            return "Hello from AsyncioServer!"

    # Create server with port=0 (auto-assign)
    server = AsyncioServer(TestService, port=0)

    # Start server
    await server.start()

    # Get assigned port
    port = server.listener.getsockname()[1]
    assert port > 0

    # Close server
    await server.close()

@pytest.mark.asyncio
async def test_native_rpyc_connection_with_asyncio():
    """Test native rpyc connection with enable_asyncio_serving."""
    import asyncio

    import rpyc
    from rpyc.utils.async_server import AsyncioServer

    class TestService(rpyc.Service):
        async def exposed_test_method(self):
            await asyncio.sleep(0.01)
            return "test_response"

    # Start server
    server = AsyncioServer(TestService, port=0)
    await server.start()
    port = server.listener.getsockname()[1]

    try:
        # Connect client
        conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

        try:
            # Get root BEFORE enabling asyncio serving (sync call)
            root = conn.root

            # Now enable asyncio serving (CRITICAL!)
            loop = asyncio.get_running_loop()
            conn.enable_asyncio_serving(loop=loop)

            # Make async call
            result = await root.test_method()
            assert result == "test_response"
        finally:
            conn.disable_asyncio_serving()
            conn.close()
    finally:
        await server.close()

@pytest.mark.asyncio
async def test_bidirectional_async_callbacks():
    """Test that bidirectional async callbacks work (no deadlock)."""
    import asyncio

    import rpyc
    from rpyc.utils.async_server import AsyncioServer

    class ServerService(rpyc.Service):
        async def exposed_call_with_callback(self, callback, value):
            # Server calls client async callback
            result = await callback(value * 2)
            return result + 100

    # Start server
    server = AsyncioServer(ServerService, port=0)
    await server.start()
    port = server.listener.getsockname()[1]

    try:
        # Connect client
        conn = rpyc.connect("localhost", port, config={"allow_all_attrs": True})

        try:
            # Get root BEFORE enabling asyncio serving
            root = conn.root

            # Enable asyncio serving
            loop = asyncio.get_running_loop()
            conn.enable_asyncio_serving(loop=loop)

            # Define client async callback
            async def client_callback(value):
                await asyncio.sleep(0.01)
                return value + 10

            # Server calls client callback - should NOT deadlock!
            result = await root.call_with_callback(client_callback, 5)
            # 5 * 2 = 10, 10 + 10 = 20, 20 + 100 = 120
            assert result == 120
        finally:
            conn.disable_asyncio_serving()
            conn.close()
    finally:
        await server.close()
