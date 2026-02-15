"""Tests for arpyc.helpers module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import rpyc

from arpyc.helpers import (
    AsyncioCallbackAdapter,
    AsyncioEventCallback,
    await_rpyc,
)

if TYPE_CHECKING:
    pass


async def test_await_rpyc_success():
    """Test await_rpyc with successful AsyncResult."""
    loop = asyncio.get_running_loop()

    # Create mock AsyncResult
    ar = Mock(spec=rpyc.core.async_.AsyncResult)
    ar.ready = False
    ar.value = "test_result"

    # Setup callback mechanism
    callbacks = []

    def add_callback(cb):
        callbacks.append(cb)

    ar.add_callback = add_callback

    # Create awaitable
    fut = await_rpyc(ar, loop=loop)

    # Simulate async result becoming ready
    async def simulate_completion():
        await asyncio.sleep(0.01)
        ar.ready = True
        for cb in callbacks:
            cb(ar)

    asyncio.create_task(simulate_completion())

    # Await result
    result = await fut
    assert result == "test_result"


async def test_await_rpyc_with_exception():
    """Test await_rpyc when remote call raises exception."""
    loop = asyncio.get_running_loop()

    # Create mock AsyncResult that raises
    ar = Mock(spec=rpyc.core.async_.AsyncResult)
    ar.ready = False

    callbacks = []

    def add_callback(cb):
        callbacks.append(cb)

    ar.add_callback = add_callback

    # Make value property raise exception
    test_exception = ValueError("remote error")

    def raise_error():
        raise test_exception

    type(ar).value = property(lambda self: raise_error())

    # Create awaitable
    fut = await_rpyc(ar, loop=loop)

    # Simulate completion
    async def simulate_error():
        await asyncio.sleep(0.01)
        ar.ready = True
        for cb in callbacks:
            cb(ar)

    asyncio.create_task(simulate_error())

    # Should raise the exception
    with pytest.raises(ValueError, match="remote error"):
        await fut


async def test_await_rpyc_with_timeout():
    """Test await_rpyc with timeout."""
    loop = asyncio.get_running_loop()

    # Create mock AsyncResult that never completes
    ar = Mock(spec=rpyc.core.async_.AsyncResult)
    ar.ready = False
    ar.add_callback = Mock()

    # Create awaitable with short timeout
    fut = await_rpyc(ar, loop=loop, timeout=0.1)

    # Should timeout
    with pytest.raises(asyncio.TimeoutError):
        await fut


async def test_await_rpyc_uses_current_loop_by_default():
    """Test that await_rpyc uses current loop if not specified."""
    # Don't pass loop parameter
    ar = Mock(spec=rpyc.core.async_.AsyncResult)
    ar.value = "result"

    callbacks = []
    ar.add_callback = lambda cb: callbacks.append(cb)

    fut = await_rpyc(ar)  # No loop parameter

    # Trigger callback
    async def trigger():
        await asyncio.sleep(0.01)
        for cb in callbacks:
            cb(ar)

    asyncio.create_task(trigger())

    result = await fut
    assert result == "result"


async def test_asyncio_callback_adapter_dispatch_sync():
    """Test AsyncioCallbackAdapter dispatches sync handlers."""
    loop = asyncio.get_running_loop()
    adapter = AsyncioCallbackAdapter(loop)

    results = []

    def sync_handler(x: int, y: int) -> int:
        result = x + y
        results.append(result)
        return result

    # Dispatch sync handler
    adapter._dispatch_sync(sync_handler, 5, 10)

    # Give event loop time to process
    await asyncio.sleep(0.01)

    assert results == [15]


async def test_asyncio_callback_adapter_dispatch_async():
    """Test AsyncioCallbackAdapter dispatches async handlers."""
    loop = asyncio.get_running_loop()
    adapter = AsyncioCallbackAdapter(loop)

    results = []

    async def async_handler(x: int) -> None:
        await asyncio.sleep(0.01)
        results.append(x * 2)

    # Dispatch async handler
    adapter._dispatch_async(async_handler(21))

    # Give event loop time to process
    await asyncio.sleep(0.02)

    assert results == [42]


async def test_asyncio_event_callback_sync_handler():
    """Test AsyncioEventCallback with sync handler."""
    loop = asyncio.get_running_loop()

    events = []

    def handler(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    callback = AsyncioEventCallback(loop, handler)

    # Call from "RPyC thread" (simulated)
    callback.on_event("test_event", {"data": 123})

    # Give event loop time to process
    await asyncio.sleep(0.01)

    assert len(events) == 1
    assert events[0] == ("test_event", {"data": 123})


async def test_asyncio_event_callback_async_handler():
    """Test AsyncioEventCallback with async handler."""
    loop = asyncio.get_running_loop()

    events = []

    async def handler(name: str, payload: dict[str, Any]) -> None:
        await asyncio.sleep(0.01)
        events.append((name, payload))

    callback = AsyncioEventCallback(loop, handler)

    # Call from "RPyC thread"
    callback.on_event("async_event", {"value": 456})

    # Give event loop time to process
    await asyncio.sleep(0.02)

    assert len(events) == 1
    assert events[0] == ("async_event", {"value": 456})


async def test_asyncio_event_callback_multiple_events():
    """Test AsyncioEventCallback handles multiple events."""
    loop = asyncio.get_running_loop()

    events = []

    def handler(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    callback = AsyncioEventCallback(loop, handler)

    # Send multiple events
    callback.on_event("event1", {"id": 1})
    callback.on_event("event2", {"id": 2})
    callback.on_event("event3", {"id": 3})

    # Give event loop time to process
    await asyncio.sleep(0.01)

    assert len(events) == 3
    assert events[0] == ("event1", {"id": 1})
    assert events[1] == ("event2", {"id": 2})
    assert events[2] == ("event3", {"id": 3})


async def test_asyncio_event_callback_from_thread():
    """Test AsyncioEventCallback works when called from real thread."""
    import threading

    loop = asyncio.get_running_loop()

    events = []

    def handler(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    callback = AsyncioEventCallback(loop, handler)

    # Call from different thread
    def thread_func():
        callback.on_event("thread_event", {"thread_id": 42})

    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()

    # Give event loop time to process
    await asyncio.sleep(0.01)

    assert len(events) == 1
    assert events[0] == ("thread_event", {"thread_id": 42})


async def test_asyncio_event_callback_handler_exception():
    """Test AsyncioEventCallback handles exceptions in handler gracefully."""
    loop = asyncio.get_running_loop()

    def failing_handler(name: str, payload: dict[str, Any]) -> None:
        raise ValueError("handler failed")

    callback = AsyncioEventCallback(loop, failing_handler)

    # Should not raise, but log error
    callback.on_event("error_event", {})

    # Give event loop time to process
    await asyncio.sleep(0.01)

    # Test passes if no exception propagated
