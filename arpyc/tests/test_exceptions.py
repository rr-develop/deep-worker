"""Tests for arpyc.exceptions module."""

import pytest

from arpyc.exceptions import (
    ARPyCConnectionError,
    ARPyCError,
    ARPyCTimeoutError,
    NotConnectedError,
    ReconnectFailedError,
)


def test_arpyc_error_is_base_exception():
    """Test that ARPyCError is the base exception."""
    error = ARPyCError("test message")
    assert isinstance(error, Exception)
    assert str(error) == "test message"


def test_connection_error_inherits_from_base():
    """Test that ARPyCConnectionError inherits from ARPyCError."""
    error = ARPyCConnectionError("connection failed")
    assert isinstance(error, ARPyCError)
    assert isinstance(error, Exception)
    assert str(error) == "connection failed"


def test_timeout_error_inherits_from_base():
    """Test that ARPyCTimeoutError inherits from ARPyCError."""
    error = ARPyCTimeoutError("timeout")
    assert isinstance(error, ARPyCError)
    assert isinstance(error, Exception)
    assert str(error) == "timeout"


def test_not_connected_error_inherits_from_connection_error():
    """Test that NotConnectedError inherits from ARPyCConnectionError."""
    error = NotConnectedError("not connected")
    assert isinstance(error, ARPyCConnectionError)
    assert isinstance(error, ARPyCError)
    assert str(error) == "not connected"


def test_reconnect_failed_error_inherits_from_connection_error():
    """Test that ReconnectFailedError inherits from ARPyCConnectionError."""
    error = ReconnectFailedError("reconnect failed")
    assert isinstance(error, ARPyCConnectionError)
    assert isinstance(error, ARPyCError)
    assert str(error) == "reconnect failed"


def test_exception_hierarchy():
    """Test the complete exception hierarchy."""
    # Create instances
    base = ARPyCError("base")
    conn = ARPyCConnectionError("conn")
    timeout = ARPyCTimeoutError("timeout")
    not_conn = NotConnectedError("not_conn")
    reconnect = ReconnectFailedError("reconnect")

    # Test isinstance relationships
    assert isinstance(conn, ARPyCError)
    assert isinstance(timeout, ARPyCError)
    assert isinstance(not_conn, ARPyCConnectionError)
    assert isinstance(reconnect, ARPyCConnectionError)

    # Test NOT relationships
    assert not isinstance(base, ARPyCConnectionError)
    assert not isinstance(timeout, ARPyCConnectionError)
    assert not isinstance(conn, ARPyCTimeoutError)


def test_exceptions_can_be_raised_and_caught():
    """Test that exceptions can be raised and caught properly."""
    # Test raising and catching specific exception
    with pytest.raises(NotConnectedError) as exc_info:
        raise NotConnectedError("test not connected")
    assert str(exc_info.value) == "test not connected"

    # Test catching as parent type
    with pytest.raises(ARPyCConnectionError):
        raise NotConnectedError("caught as parent")

    # Test catching as base type
    with pytest.raises(ARPyCError):
        raise ReconnectFailedError("caught as base")


def test_exceptions_without_message():
    """Test that exceptions work without messages."""
    error = ARPyCError()
    assert isinstance(error, Exception)

    error2 = NotConnectedError()
    assert isinstance(error2, ARPyCConnectionError)
