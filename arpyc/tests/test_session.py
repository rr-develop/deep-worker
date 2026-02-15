"""Tests for arpyc.session module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arpyc.exceptions import ARPyCConnectionError, NotConnectedError
from arpyc.session import RpycAsyncSession

if TYPE_CHECKING:
    pass


async def test_session_creation():
    """Test RpycAsyncSession initialization."""
    session = RpycAsyncSession("localhost", 18862)

    assert session._host == "localhost"
    assert session._port == 18862
    assert session._conn is None
    assert session._bg is None
    assert not session.is_connected()


async def test_session_custom_config():
    """Test RpycAsyncSession with custom config."""
    config = {"allow_public_attrs": True, "sync_request_timeout": 30}
    session = RpycAsyncSession(
        "localhost",
        18862,
        config=config,
        auto_reconnect=False,
        connect_timeout=15.0,
    )

    assert session._config == config
    assert not session._auto_reconnect
    assert session._connect_timeout == 15.0


async def test_session_connect_success(rpyc_server):
    """Test successful connection to RPyC server."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()

    assert session.is_connected()
    assert session._conn is not None
    assert session._bg is not None

    await session.disconnect()


async def test_session_connect_failure():
    """Test connection failure handling."""
    session = RpycAsyncSession("localhost", 99999)  # Invalid port

    with pytest.raises(ARPyCConnectionError):
        await session.connect()

    assert not session.is_connected()


async def test_session_disconnect(rpyc_server):
    """Test disconnection from server."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()
    assert session.is_connected()

    await session.disconnect()
    assert not session.is_connected()
    assert session._conn is None
    assert session._bg is None


async def test_session_disconnect_when_not_connected():
    """Test that disconnect is safe when not connected."""
    session = RpycAsyncSession("localhost", 18862)

    # Should not raise
    await session.disconnect()

    assert not session.is_connected()


async def test_session_context_manager(rpyc_server):
    """Test RpycAsyncSession as async context manager."""
    host, port = rpyc_server

    async with RpycAsyncSession(host, port) as session:
        assert session.is_connected()
        # Use session
        result = session.root.echo("test")
        assert result == "echo: test"

    # Should be disconnected after context
    assert not session.is_connected()


async def test_session_root_property(rpyc_server):
    """Test access to root service via property."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()

    # Access root service
    root = session.root
    assert root is not None

    # Call remote method
    result = root.echo("hello")
    assert result == "echo: hello"

    await session.disconnect()


async def test_session_root_when_not_connected():
    """Test that root property raises when not connected."""
    session = RpycAsyncSession("localhost", 18862)

    with pytest.raises(NotConnectedError):
        _ = session.root


async def test_session_conn_property(rpyc_server):
    """Test access to underlying connection."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()

    conn = session.conn
    assert conn is not None
    # Should be RPyC connection
    assert hasattr(conn, "root")
    assert hasattr(conn, "close")

    await session.disconnect()


async def test_session_conn_when_not_connected():
    """Test that conn property raises when not connected."""
    session = RpycAsyncSession("localhost", 18862)

    with pytest.raises(NotConnectedError):
        _ = session.conn


async def test_session_async_call(rpyc_server):
    """Test async_call method."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()

    # Call remote method asynchronously
    result = await session.async_call(session.root.add, 10, 20)
    assert result == 30

    await session.disconnect()


async def test_session_async_call_not_connected():
    """Test async_call raises when not connected."""
    session = RpycAsyncSession("localhost", 18862)

    with pytest.raises(NotConnectedError):
        await session.async_call(Mock(), 1, 2)


async def test_session_multiple_async_calls(rpyc_server):
    """Test multiple concurrent async calls."""
    host, port = rpyc_server
    session = RpycAsyncSession(host, port)

    await session.connect()

    # Make multiple concurrent calls
    results = await asyncio.gather(
        session.async_call(session.root.add, 1, 2),
        session.async_call(session.root.add, 3, 4),
        session.async_call(session.root.add, 5, 6),
    )

    assert results == [3, 7, 11]

    await session.disconnect()


async def test_session_is_connected():
    """Test is_connected method."""
    session = RpycAsyncSession("localhost", 18862)

    assert not session.is_connected()

    session._conn = Mock()
    assert session.is_connected()

    session._conn = None
    assert not session.is_connected()
