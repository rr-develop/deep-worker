"""Low-level helpers for RPyC async integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

import rpyc  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


def await_rpyc(
    ar: Any,  # rpyc.core.async_.AsyncResult
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    timeout: float | None = None,
) -> Awaitable[Any]:
    """
    Convert RPyC AsyncResult to awaitable Future.

    This function bridges RPyC's callback-based async model with asyncio's
    await syntax. When the RPyC AsyncResult completes, the Future is resolved.

    Args:
        ar: RPyC AsyncResult from async_ call
        loop: Event loop (defaults to current running loop)
        timeout: Optional timeout in seconds

    Returns:
        Awaitable that resolves to remote call result

    Raises:
        asyncio.TimeoutError: If timeout exceeded
        Exception: If remote call failed (exception from server)

    Example:
        >>> from rpyc.utils.classic import async_
        >>> remote_func = async_(conn.root.some_function)
        >>> result = await await_rpyc(remote_func(args))
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    fut: asyncio.Future[Any] = loop.create_future()

    def _done_callback(_ar: Any) -> None:
        """Callback executed when AsyncResult completes."""
        try:
            # Access .value may raise if remote call failed
            result = _ar.value
        except Exception as e:
            # Remote call raised exception
            loop.call_soon_threadsafe(fut.set_exception, e)
            logger.debug("RPyC call failed: %s", e)
        else:
            # Remote call succeeded
            loop.call_soon_threadsafe(fut.set_result, result)
            logger.debug("RPyC call succeeded: %s", result)

    # Register callback with AsyncResult
    ar.add_callback(_done_callback)

    # Apply timeout if specified
    if timeout is not None:
        return asyncio.wait_for(fut, timeout=timeout)

    return fut


class AsyncioCallbackAdapter:
    """
    Base class for callbacks passed to RPyC server.

    Automatically forwards calls from RPyC thread to asyncio loop using
    thread-safe primitives.

    This class provides helper methods for dispatching both sync and async
    handlers safely across thread boundaries.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Initialize callback adapter.

        Args:
            loop: Asyncio event loop for thread-safe dispatch
        """
        self._loop = loop

    def _dispatch_sync(
        self, handler: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Dispatch sync handler to asyncio loop thread-safely.

        Args:
            handler: Synchronous callable to execute
            *args: Positional arguments for handler
            **kwargs: Keyword arguments for handler

        Note:
            The handler is executed in the asyncio loop thread.
        """

        def _wrapper() -> None:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.exception("Error in sync handler: %s", e)

        self._loop.call_soon_threadsafe(_wrapper)

    def _dispatch_async(self, coro: Awaitable[Any]) -> None:
        """
        Dispatch coroutine to asyncio loop thread-safely.

        Args:
            coro: Coroutine to execute in the event loop

        Note:
            The coroutine is scheduled as a Task in the asyncio loop.
        """

        def _wrapper() -> None:
            try:
                asyncio.ensure_future(coro, loop=self._loop)
            except Exception as e:
                logger.exception("Error dispatching async handler: %s", e)

        self._loop.call_soon_threadsafe(_wrapper)


class AsyncioEventCallback(AsyncioCallbackAdapter):
    """
    Event callback adapter for publish-subscribe patterns.

    This class wraps user-provided event handlers (sync or async) and ensures
    they are executed safely in the asyncio event loop, even when called from
    RPyC threads.

    Example:
        >>> def on_event(name: str, payload: dict):
        ...     print(f"Event: {name}")
        >>> callback = AsyncioEventCallback(loop, on_event)
        >>> conn.root.subscribe(callback)
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        handler: Callable[[str, dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        """
        Initialize event callback.

        Args:
            loop: Asyncio event loop
            handler: Event handler (sync or async function)
                Signature: (name: str, payload: dict) -> None or Awaitable[None]
        """
        super().__init__(loop)
        self._handler = handler

    def on_event(self, name: str, payload: dict[str, Any]) -> None:
        """
        Called from RPyC thread, dispatched to asyncio.

        This method can be called from any thread (typically RPyC serving thread).
        It safely dispatches the event to the user's handler in the asyncio loop.

        Args:
            name: Event name
            payload: Event data
        """
        logger.debug("Event received: %s with payload: %s", name, payload)

        # Check if handler is async or sync
        if inspect.iscoroutinefunction(self._handler):
            # Async handler - dispatch as coroutine
            coro = self._handler(name, payload)
            self._dispatch_async(coro)
        else:
            # Sync handler - dispatch directly
            self._dispatch_sync(self._handler, name, payload)
