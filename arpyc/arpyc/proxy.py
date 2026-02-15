"""Transparent async proxy for RPyC remote objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from arpyc.session import RpycAsyncSession

logger = logging.getLogger(__name__)


class AsyncProxy:
    """
    Transparent async wrapper for RPyC netref objects.

    This class allows calling remote methods with natural async syntax:
    >>> await proxy.method(args)

    Instead of the more verbose:
    >>> await session.async_call(netref.method, args)

    Example:
        >>> counter = session.proxy(session.root.create_counter())
        >>> value = await counter.inc(5)
        >>> current = counter.get.sync()  # Sync call if needed
    """

    def __init__(self, netref: Any, session: RpycAsyncSession) -> None:
        """
        Initialize async proxy.

        Args:
            netref: RPyC netref (remote object reference)
            session: RpycAsyncSession for async calls
        """
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, "_netref", netref)
        object.__setattr__(self, "_session", session)

    def __getattr__(self, name: str) -> AsyncMethodProxy:
        """
        Get async method proxy.

        Args:
            name: Method name

        Returns:
            AsyncMethodProxy for the requested method
        """
        logger.debug("Getting method proxy for: %s", name)
        # Use object.__getattribute__ to access _session safely
        session = object.__getattribute__(self, "_session")
        netref = object.__getattribute__(self, "_netref")
        return AsyncMethodProxy(netref, name, session)

    def __dir__(self) -> list[str]:
        """
        List available methods (for IDE autocomplete).

        Returns:
            List of method names
        """
        try:
            # Get methods from remote object
            return dir(self._netref)
        except Exception as e:
            logger.warning("Failed to get dir() for netref: %s", e)
            return []

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent setting attributes (read-only proxy)."""
        raise AttributeError(
            f"Cannot set attribute '{name}' on AsyncProxy (read-only)"
        )


class AsyncMethodProxy:
    """
    Proxy for individual remote method.

    Provides both async and sync call modes:
    - Async: await method(args)
    - Sync: method.sync(args)
    """

    def __init__(
        self,
        netref: Any,
        method_name: str,
        session: RpycAsyncSession,
    ) -> None:
        """
        Initialize method proxy.

        Args:
            netref: RPyC netref object
            method_name: Name of the method to proxy
            session: RpycAsyncSession for async calls
        """
        self._netref = netref
        self._method_name = method_name
        self._session = session

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """
        Execute remote method asynchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Awaitable that resolves to method result

        Example:
            >>> result = await proxy.method(arg1, arg2)
        """
        logger.debug("Async call: %s(*%s, **%s)", self._method_name, args, kwargs)

        # Get the actual method from netref
        method = getattr(self._netref, self._method_name)

        # Use session.async_call to execute
        return self._session.async_call(method, *args, **kwargs)

    def sync(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute remote method synchronously (blocking).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method result (blocks until complete)

        Example:
            >>> result = proxy.method.sync(arg1, arg2)

        Warning:
            This blocks the current thread. Use sparingly in async code.
        """
        logger.debug("Sync call: %s(*%s, **%s)", self._method_name, args, kwargs)

        # Get the actual method from netref and call directly
        method = getattr(self._netref, self._method_name)
        return method(*args, **kwargs)
