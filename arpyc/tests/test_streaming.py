"""Tests for arpyc.streaming module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from arpyc.streaming import StreamingCallback
from arpyc.types import StreamCallback as StreamCallbackProtocol

if TYPE_CHECKING:
    pass


async def test_streaming_callback_creation():
    """Test StreamingCallback initialization."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    assert callback._loop is loop
    assert isinstance(callback._queue, asyncio.Queue)
    assert not callback._done


async def test_streaming_callback_matches_protocol():
    """Test that StreamingCallback implements StreamCallback protocol."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    assert isinstance(callback, StreamCallbackProtocol)


async def test_streaming_callback_on_chunk():
    """Test on_chunk method adds chunks to queue."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    # Add chunks
    callback.on_chunk("chunk1")
    callback.on_chunk("chunk2")
    callback.on_chunk("chunk3")

    # Give event loop time to process
    await asyncio.sleep(0.01)

    # Verify chunks are in queue
    assert not callback._queue.empty()
    assert await callback._queue.get() == "chunk1"
    assert await callback._queue.get() == "chunk2"
    assert await callback._queue.get() == "chunk3"


async def test_streaming_callback_on_done():
    """Test on_done method signals end of stream."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    callback.on_chunk("chunk1")
    callback.on_done()

    assert callback._done
    # None should be in queue to signal completion
    assert await callback._queue.get() == "chunk1"
    assert await callback._queue.get() is None


async def test_streaming_callback_on_error():
    """Test on_error method signals stream error."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    callback.on_chunk("chunk1")
    test_error = ValueError("test error")
    callback.on_error(test_error)

    assert callback._done
    # Error should be in queue
    assert await callback._queue.get() == "chunk1"
    error_in_queue = await callback._queue.get()
    assert isinstance(error_in_queue, ValueError)
    assert str(error_in_queue) == "test error"


async def test_streaming_callback_async_iteration():
    """Test async iteration over stream chunks."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[int](loop)

    # Add chunks and signal done
    for i in range(5):
        callback.on_chunk(i)
    callback.on_done()

    # Collect chunks via async iteration
    chunks = []
    async for chunk in callback:
        chunks.append(chunk)

    assert chunks == [0, 1, 2, 3, 4]


async def test_streaming_callback_async_iteration_with_error():
    """Test async iteration stops with error."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    callback.on_chunk("chunk1")
    callback.on_chunk("chunk2")
    callback.on_error(RuntimeError("stream error"))

    chunks = []
    with pytest.raises(RuntimeError, match="stream error"):
        async for chunk in callback:
            chunks.append(chunk)

    # Should have collected chunks before error
    assert chunks == ["chunk1", "chunk2"]


async def test_streaming_callback_ignore_chunks_after_done():
    """Test that chunks after on_done are ignored."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    callback.on_chunk("chunk1")
    callback.on_done()
    callback.on_chunk("chunk2")  # Should be ignored

    chunks = []
    async for chunk in callback:
        chunks.append(chunk)

    assert chunks == ["chunk1"]  # chunk2 not included


async def test_streaming_callback_ignore_chunks_after_error():
    """Test that chunks after on_error are ignored."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    callback.on_chunk("chunk1")
    callback.on_error(ValueError("error"))
    callback.on_chunk("chunk2")  # Should be ignored

    chunks = []
    with pytest.raises(ValueError):
        async for chunk in callback:
            chunks.append(chunk)

    assert chunks == ["chunk1"]


async def test_streaming_callback_from_different_thread():
    """Test that on_chunk works when called from different thread."""
    import threading

    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    def add_chunks():
        for i in range(3):
            callback.on_chunk(f"chunk{i}")
        callback.on_done()

    # Call from different thread
    thread = threading.Thread(target=add_chunks)
    thread.start()
    thread.join()

    # Collect chunks
    chunks = []
    async for chunk in callback:
        chunks.append(chunk)

    assert chunks == ["chunk0", "chunk1", "chunk2"]


async def test_streaming_callback_multiple_errors():
    """Test that only first error is sent."""
    loop = asyncio.get_running_loop()
    callback = StreamingCallback[str](loop)

    error1 = ValueError("error1")
    error2 = RuntimeError("error2")

    callback.on_error(error1)
    callback.on_error(error2)  # Should be ignored

    with pytest.raises(ValueError, match="error1"):
        async for _ in callback:
            pass


async def test_streaming_callback_typed():
    """Test that StreamingCallback is generic and typed."""
    loop = asyncio.get_running_loop()

    # String callback
    str_callback = StreamingCallback[str](loop)
    str_callback.on_chunk("test")
    str_callback.on_done()

    result: list[str] = []
    async for chunk in str_callback:
        result.append(chunk)

    # Int callback
    int_callback = StreamingCallback[int](loop)
    int_callback.on_chunk(42)
    int_callback.on_done()

    int_result: list[int] = []
    async for chunk in int_callback:
        int_result.append(chunk)

    assert result == ["test"]
    assert int_result == [42]
