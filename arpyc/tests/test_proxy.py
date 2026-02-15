"""Tests for arpyc.proxy module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arpyc.proxy import AsyncMethodProxy, AsyncProxy
from arpyc.session import RpycAsyncSession

if TYPE_CHECKING:
    pass


async def test_async_proxy_creation(rpyc_server):
    """Test AsyncProxy initialization."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        # Create proxy from root
        proxy = AsyncProxy(session.root, session)

        assert proxy._netref is session.root
        assert proxy._session is session


async def test_async_proxy_getattr(rpyc_server):
    """Test AsyncProxy __getattr__ returns AsyncMethodProxy."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = AsyncProxy(session.root, session)

        # Get method proxy
        echo_method = proxy.echo
        assert isinstance(echo_method, AsyncMethodProxy)

        add_method = proxy.add
        assert isinstance(add_method, AsyncMethodProxy)


async def test_async_proxy_method_call(rpyc_server):
    """Test calling methods through AsyncProxy."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = AsyncProxy(session.root, session)

        # Call method asynchronously
        result = await proxy.echo("test message")
        assert result == "echo: test message"

        result2 = await proxy.add(10, 20)
        assert result2 == 30


async def test_async_proxy_method_sync_call(rpyc_server):
    """Test synchronous call through AsyncMethodProxy.sync()."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = AsyncProxy(session.root, session)

        # Call method synchronously
        result = proxy.echo.sync("sync test")
        assert result == "echo: sync test"

        result2 = proxy.add.sync(5, 7)
        assert result2 == 12


async def test_async_proxy_dir(rpyc_server):
    """Test AsyncProxy __dir__ lists available methods."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = AsyncProxy(session.root, session)

        methods = dir(proxy)

        # Should include remote methods (RPyC exposes as "exposed_*")
        assert "exposed_echo" in methods or "echo" in methods
        assert "exposed_add" in methods or "add" in methods
        assert "exposed_get_call_count" in methods or "get_call_count" in methods


async def test_async_proxy_multiple_calls(rpyc_server):
    """Test multiple calls through same proxy."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = AsyncProxy(session.root, session)

        # Reset call count
        session.root.reset_call_count()

        # Make multiple calls
        await proxy.echo("msg1")
        await proxy.echo("msg2")
        await proxy.add(1, 2)

        count = await proxy.get_call_count()
        assert count == 3


async def test_async_method_proxy_creation():
    """Test AsyncMethodProxy initialization."""
    from unittest.mock import Mock

    netref = Mock()
    session = Mock(spec=RpycAsyncSession)
    method_name = "test_method"

    method_proxy = AsyncMethodProxy(netref, method_name, session)

    assert method_proxy._netref is netref
    assert method_proxy._method_name == method_name
    assert method_proxy._session is session


async def test_async_method_proxy_call(rpyc_server):
    """Test AsyncMethodProxy __call__ for async execution."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        # Create method proxy manually
        method_proxy = AsyncMethodProxy(session.root, "add", session)

        # Call asynchronously
        result = await method_proxy(15, 25)
        assert result == 40


async def test_async_method_proxy_sync(rpyc_server):
    """Test AsyncMethodProxy.sync() for synchronous execution."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        method_proxy = AsyncMethodProxy(session.root, "add", session)

        # Call synchronously
        result = method_proxy.sync(8, 12)
        assert result == 20


async def test_session_proxy_method(rpyc_server):
    """Test RpycAsyncSession.proxy() creates AsyncProxy."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        # Use session.proxy() method
        proxy = session.proxy(session.root)

        assert isinstance(proxy, AsyncProxy)

        result = await proxy.echo("test")
        assert result == "echo: test"


async def test_proxy_with_nested_objects(rpyc_server):
    """Test AsyncProxy with nested remote objects."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        # RPyC service should expose objects
        root_proxy = session.proxy(session.root)

        # Call method that might return another object
        # (TestService doesn't have this, but testing the pattern)
        result = await root_proxy.echo("nested")
        assert result == "echo: nested"


async def test_proxy_error_propagation(rpyc_server):
    """Test that errors from remote calls propagate correctly."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = session.proxy(session.root)

        # Call method that raises error
        with pytest.raises(ValueError, match="Test error"):
            await proxy.raise_error()


async def test_proxy_method_with_kwargs(rpyc_server):
    """Test proxy method calls with keyword arguments."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = session.proxy(session.root)

        # Call with keyword arguments
        result = await proxy.add(a=100, b=200)
        assert result == 300


async def test_proxy_chaining(rpyc_server):
    """Test that proxy can be used multiple times."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        proxy = session.proxy(session.root)

        # Multiple chained calls
        r1 = await proxy.echo("first")
        r2 = await proxy.echo("second")
        r3 = await proxy.add(1, 2)

        assert r1 == "echo: first"
        assert r2 == "echo: second"
        assert r3 == 3
