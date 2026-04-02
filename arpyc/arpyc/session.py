"""Main session class for RPyC async connections using BgServingThread.

⚠️ LEGACY APPROACH: For new code, consider RpycAsyncSessionNative instead!

This module uses BgServingThread (separate thread with polling delays).
For better performance (156x faster!), use RpycAsyncSessionNative from session_native.py.

Use THIS approach (RpycAsyncSession) ONLY if:
- You need Windows support (asyncio.add_reader has limitations on Windows)
- You have existing code that depends on BgServingThread behavior

For NEW code on Unix-like systems (Linux, macOS):
    from arpyc import RpycAsyncSessionNative  # ← Use this! 156x faster

Performance comparison:
    RpycAsyncSession:       ~301ms average latency (this file)
    RpycAsyncSessionNative: ~1.93ms average latency (session_native.py)
    Speedup:                156x faster!

See the internal asyncio-native performance analysis (not included here) for details.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

import rpyc  # type: ignore[import-untyped]
from rpyc.utils.helpers import BgServingThread  # type: ignore[import-untyped]

from arpyc.exceptions import ARPyCConnectionError, NotConnectedError
from arpyc.helpers import await_rpyc

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


class RpycAsyncSession:
    """
    Asyncio-friendly RPyC session using BgServingThread.

    ⚠️ LEGACY APPROACH: Consider RpycAsyncSessionNative for new code!

    This class uses BgServingThread which has polling delays (sleep_interval=0.1).
    For 156x better performance, use RpycAsyncSessionNative instead.

    This class provides:
    - Async connection management
    - Background serving thread for reverse calls (callbacks)
    - Async call support via await_rpyc
    - Context manager protocol
    - Optional auto-reconnect and heartbeat

    ⚠️ PERFORMANCE ISSUE:
        BgServingThread polls with sleep(0.1) delays = ~100ms overhead per cycle
        This causes ~301ms average latency for recursive calls.

    ✅ BETTER ALTERNATIVE: RpycAsyncSessionNative
        - Event-driven (no polling delays)
        - 156x faster (~1.93ms latency)
        - Callbacks in event loop (can use asyncio!)

    Use THIS class (RpycAsyncSession) ONLY if:
    - Windows platform (add_reader has limitations)
    - Legacy code compatibility

    Example:
        >>> # OLD approach (this class):
        >>> async with RpycAsyncSession("localhost", 18861) as session:
        ...     result = await session.async_call(session.root.some_method, arg)
        ...     print(result)

        >>> # RECOMMENDED for new code:
        >>> from arpyc import RpycAsyncSessionNative
        >>> async with RpycAsyncSessionNative("localhost", 18861) as session:
        ...     result = await session.async_call(session.root.some_method, arg)
        ...     print(result)  # 156x faster!

    See Also:
        - RpycAsyncSessionNative: Recommended asyncio-native approach
        - Internal asyncio-native performance analysis (not included here)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18861,
        *,
        config: dict[str, Any] | None = None,
        auto_reconnect: bool = True,
        heartbeat_interval: float = 5.0,
        connect_timeout: float = 10.0,
    ) -> None:
        """
        Initialize RPyC async session.

        Args:
            host: Server hostname
            port: Server port
            config: RPyC connection config (e.g., {"allow_public_attrs": True})
            auto_reconnect: Enable automatic reconnection (not implemented yet)
            heartbeat_interval: Heartbeat check interval in seconds (not implemented yet)
            connect_timeout: Connection timeout in seconds
        """
        self._host = host
        self._port = port
        self._config = config or {"allow_public_attrs": True}
        self._auto_reconnect = auto_reconnect
        self._heartbeat_interval = heartbeat_interval
        self._connect_timeout = connect_timeout

        # Connection state
        self._conn: rpyc.Connection | None = None
        self._bg: BgServingThread | None = None

    async def connect(self) -> None:
        """
        Establish connection and start background thread.

        Raises:
            ARPyCConnectionError: If connection fails
        """
        if self._conn is not None:
            logger.warning("Already connected to %s:%s", self._host, self._port)
            return

        try:
            logger.info("Connecting to RPyC server %s:%s", self._host, self._port)

            # Connect with timeout
            loop = asyncio.get_running_loop()

            def _connect() -> Any:
                return rpyc.connect(
                    self._host,
                    self._port,
                    config=self._config,
                )

            conn_future = loop.run_in_executor(None, _connect)

            self._conn = await asyncio.wait_for(
                conn_future,
                timeout=self._connect_timeout,
            )

            # Start background serving thread for reverse calls (callbacks)
            self._bg = BgServingThread(self._conn)

            logger.info("Connected to %s:%s", self._host, self._port)

        except asyncio.TimeoutError as e:
            raise ARPyCConnectionError(
                f"Connection timeout after {self._connect_timeout}s"
            ) from e
        except Exception as e:
            raise ARPyCConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """Gracefully close connection."""
        if self._conn is None:
            logger.debug("Not connected, nothing to disconnect")
            return

        logger.info("Disconnecting from %s:%s", self._host, self._port)

        try:
            # Stop background serving thread
            if self._bg is not None:
                self._bg.stop()
                self._bg = None

            # Close connection
            if self._conn is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._conn.close)
                self._conn = None

            logger.info("Disconnected successfully")

        except Exception as e:
            logger.exception("Error during disconnect: %s", e)
            # Force cleanup even on error
            self._conn = None
            self._bg = None

    async def __aenter__(self) -> RpycAsyncSession:
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Context manager exit."""
        await self.disconnect()

    @property
    def conn(self) -> Any:  # rpyc.Connection
        """
        Get underlying RPyC connection.

        Returns:
            RPyC connection object

        Raises:
            NotConnectedError: If not connected
        """
        if self._conn is None:
            raise NotConnectedError("Not connected to server")
        return self._conn

    @property
    def root(self) -> Any:
        """
        Get root service proxy.

        Returns:
            Root service proxy for calling remote methods

        Raises:
            NotConnectedError: If not connected

        Example:
            >>> session.root.some_method(args)
        """
        return self.conn.root

    def async_call(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[Any]:
        """
        Execute remote method asynchronously.

        Args:
            fn: Remote method to call (from session.root.*)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Awaitable that resolves to method result

        Raises:
            NotConnectedError: If not connected
            Exception: If remote call fails

        Example:
            >>> result = await session.async_call(session.root.add, 1, 2)
        """
        if self._conn is None:
            raise NotConnectedError("Not connected to server")

        # Wrap function with rpyc.async_
        async_fn = rpyc.async_(fn)

        # Call and get AsyncResult
        ar = async_fn(*args, **kwargs)

        # Convert to awaitable
        return await_rpyc(ar)

    def proxy(self, netref: Any) -> Any:  # AsyncProxy
        """
        Wrap netref in AsyncProxy for transparent await.

        Args:
            netref: RPyC netref object (remote object reference)

        Returns:
            AsyncProxy wrapper for transparent async calls

        Example:
            >>> counter = session.proxy(session.root.create_counter())
            >>> await counter.inc(5)
        """
        from arpyc.proxy import AsyncProxy

        return AsyncProxy(netref, self)

    def is_connected(self) -> bool:
        """
        Check if connection is alive.

        Returns:
            True if connected, False otherwise
        """
        return self._conn is not None
