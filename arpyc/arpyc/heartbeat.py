"""Heartbeat and auto-reconnect functionality."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from arpyc.session import RpycAsyncSession

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Manages connection health checks and auto-reconnect.

    This class monitors an RpycAsyncSession by periodically pinging the server.
    If the connection is lost, it can automatically attempt to reconnect.

    Example:
        >>> heartbeat = HeartbeatManager(session, interval=5.0)
        >>> await heartbeat.start()
        >>> # ... heartbeat runs in background ...
        >>> await heartbeat.stop()
    """

    def __init__(
        self,
        session: RpycAsyncSession,
        interval: float = 5.0,
        timeout: float = 3.0,
        max_retries: int = 3,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
        on_reconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize heartbeat manager.

        Args:
            session: RpycAsyncSession to monitor
            interval: Heartbeat check interval in seconds
            timeout: Ping timeout in seconds
            max_retries: Maximum reconnection attempts (0 = infinite)
            on_disconnect: Callback when connection lost
            on_reconnect: Callback when reconnected
        """
        self._session = session
        self._interval = interval
        self._timeout = timeout
        self._max_retries = max_retries
        self._on_disconnect = on_disconnect
        self._on_reconnect = on_reconnect

        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """
        Start heartbeat monitoring.

        Creates a background task that periodically checks connection health.
        """
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Stop heartbeat monitoring."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Heartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        retry_count = 0

        while self._running:
            try:
                await asyncio.sleep(self._interval)

                if not self._running:
                    break

                # Check if still connected
                if not self._session.is_connected():
                    logger.warning("Session not connected")
                    await self._handle_disconnect()
                    retry_count += 1

                    if self._max_retries > 0 and retry_count >= self._max_retries:
                        logger.error(
                            "Max retries (%d) reached, stopping heartbeat",
                            self._max_retries,
                        )
                        break

                    # Try to reconnect
                    if await self._try_reconnect():
                        retry_count = 0
                        await self._handle_reconnect()
                    else:
                        # Exponential backoff
                        backoff = min(30, 2**retry_count)
                        logger.info("Reconnect failed, waiting %ds", backoff)
                        await asyncio.sleep(backoff)

                else:
                    # Ping server to check health
                    try:
                        await asyncio.wait_for(
                            self._ping(),
                            timeout=self._timeout,
                        )
                        retry_count = 0  # Reset on successful ping
                        logger.debug("Heartbeat ping successful")

                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("Heartbeat ping failed: %s", e)
                        await self._handle_disconnect()
                        retry_count += 1

            except asyncio.CancelledError:
                logger.debug("Heartbeat cancelled")
                break
            except Exception as e:
                logger.exception("Error in heartbeat loop: %s", e)
                await asyncio.sleep(self._interval)

    async def _ping(self) -> None:
        """Ping the server to check connection health."""
        if not self._session.is_connected():
            raise ConnectionError("Not connected")

        try:
            # RPyC connections have a ping() method
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._session.conn.ping)
        except Exception as e:
            raise ConnectionError(f"Ping failed: {e}") from e

    async def _try_reconnect(self) -> bool:
        """
        Try to reconnect to server.

        Returns:
            True if reconnected successfully, False otherwise
        """
        try:
            logger.info("Attempting to reconnect...")
            await self._session.connect()
            logger.info("Reconnected successfully")
            return True
        except Exception as e:
            logger.error("Reconnect failed: %s", e)
            return False

    async def _handle_disconnect(self) -> None:
        """Handle disconnect event."""
        if self._on_disconnect:
            try:
                await self._on_disconnect()
            except Exception as e:
                logger.exception("Error in disconnect callback: %s", e)

    async def _handle_reconnect(self) -> None:
        """Handle reconnect event."""
        if self._on_reconnect:
            try:
                await self._on_reconnect()
            except Exception as e:
                logger.exception("Error in reconnect callback: %s", e)
