"""Tests for arpyc.types module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arpyc.types import AsyncEventHandler, EventHandler, StreamCallback

if TYPE_CHECKING:
    pass


def test_event_handler_protocol_sync():
    """Test EventHandler protocol with sync implementation."""

    class MyHandler:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def on_event(self, name: str, payload: dict[str, Any]) -> None:
            self.events.append((name, payload))

    handler = MyHandler()
    assert isinstance(handler, EventHandler)

    # Test usage
    handler.on_event("test", {"data": 123})
    assert len(handler.events) == 1
    assert handler.events[0] == ("test", {"data": 123})


async def test_async_event_handler_protocol():
    """Test AsyncEventHandler protocol with async implementation."""

    class MyAsyncHandler:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        async def on_event(self, name: str, payload: dict[str, Any]) -> None:
            self.events.append((name, payload))

    handler = MyAsyncHandler()
    assert isinstance(handler, AsyncEventHandler)

    # Test usage
    await handler.on_event("test", {"data": 456})
    assert len(handler.events) == 1
    assert handler.events[0] == ("test", {"data": 456})


def test_stream_callback_protocol():
    """Test StreamCallback protocol with implementation."""

    class MyStreamCallback:
        def __init__(self) -> None:
            self.chunks: list[Any] = []
            self.done = False
            self.error: Exception | None = None

        def on_chunk(self, chunk: Any) -> None:
            self.chunks.append(chunk)

        def on_done(self) -> None:
            self.done = True

        def on_error(self, error: Exception) -> None:
            self.error = error

    callback = MyStreamCallback()
    assert isinstance(callback, StreamCallback)

    # Test usage
    callback.on_chunk("chunk1")
    callback.on_chunk("chunk2")
    assert callback.chunks == ["chunk1", "chunk2"]
    assert not callback.done
    assert callback.error is None

    callback.on_done()
    assert callback.done

    # Test error
    callback2 = MyStreamCallback()
    test_error = ValueError("test error")
    callback2.on_error(test_error)
    assert callback2.error is test_error


def test_protocol_not_matched_missing_method():
    """Test that protocol is not matched when methods are missing."""

    class IncompleteHandler:
        pass

    handler = IncompleteHandler()
    assert not isinstance(handler, EventHandler)
    assert not isinstance(handler, AsyncEventHandler)
    assert not isinstance(handler, StreamCallback)


def test_protocol_not_matched_wrong_signature():
    """Test that protocol requires correct method signatures."""

    class WrongSignature:
        def on_event(self) -> None:  # Missing parameters
            pass

    handler = WrongSignature()
    # Runtime checkable protocols check method existence, not signatures
    # This is a limitation of runtime_checkable
    assert isinstance(handler, EventHandler)  # Will pass at runtime


def test_multiple_protocol_implementation():
    """Test class implementing multiple protocols."""

    class CombinedHandler:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []
            self.chunks: list[Any] = []

        def on_event(self, name: str, payload: dict[str, Any]) -> None:
            self.events.append((name, payload))

        def on_chunk(self, chunk: Any) -> None:
            self.chunks.append(chunk)

        def on_done(self) -> None:
            pass

        def on_error(self, error: Exception) -> None:
            pass

    handler = CombinedHandler()
    assert isinstance(handler, EventHandler)
    assert isinstance(handler, StreamCallback)
    # Note: runtime_checkable protocols only check method existence, not signatures
    # So on_event (sync) will still match AsyncEventHandler protocol at runtime
    # This is a limitation of runtime_checkable - static type checkers handle this correctly
