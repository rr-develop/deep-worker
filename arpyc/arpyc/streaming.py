"""Streaming support for real-time data."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamingCallback(Generic[T]):
    """
    Callback adapter for streaming data from RPyC to asyncio.

    This class provides a thread-safe way to stream data from RPyC threads
    into asyncio code. It implements the StreamCallback protocol and provides
    an async iterator interface for consuming streamed data.

    Example:
        >>> callback = StreamingCallback[str](asyncio.get_running_loop())
        >>> # From RPyC thread:
        >>> callback.on_chunk("data1")
        >>> callback.on_chunk("data2")
        >>> callback.on_done()
        >>> # From asyncio:
        >>> async for chunk in callback:
        ...     print(chunk)
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Initialize streaming callback.

        Args:
            loop: Asyncio event loop for thread-safe queue operations
        """
        self._queue: asyncio.Queue[T | None | Exception] = asyncio.Queue()
        self._loop = loop
        self._done = False

    def on_chunk(self, chunk: T) -> None:
        """
        Called from RPyC thread with new chunk.

        Args:
            chunk: Data chunk to stream

        Note:
            This method is thread-safe and can be called from RPyC threads.
            Chunks received after on_done() or on_error() are ignored.
        """
        if not self._done:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
            logger.debug("Streamed chunk: %s", chunk)

    def on_done(self) -> None:
        """
        Signal end of stream.

        Note:
            This method is thread-safe and can be called from RPyC threads.
            After calling this, subsequent on_chunk() calls are ignored.
        """
        if not self._done:
            self._done = True
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            logger.debug("Stream completed")

    def on_error(self, error: Exception) -> None:
        """
        Signal stream error.

        Args:
            error: Exception that occurred during streaming

        Note:
            This method is thread-safe and can be called from RPyC threads.
            After calling this, subsequent on_chunk() calls are ignored.
        """
        if not self._done:
            self._done = True
            self._loop.call_soon_threadsafe(self._queue.put_nowait, error)
            logger.error("Stream error: %s", error)

    async def __aiter__(self) -> AsyncIterator[T]:
        """
        Async iterator over stream chunks.

        Yields:
            Stream chunks until stream is done or error occurs

        Raises:
            Exception: If stream error occurred (via on_error)

        Example:
            >>> async for chunk in callback:
            ...     process(chunk)
        """
        while True:
            item = await self._queue.get()
            if item is None:
                # Stream completed
                break
            if isinstance(item, Exception):
                # Stream error
                raise item
            yield item
