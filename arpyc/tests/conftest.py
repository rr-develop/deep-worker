"""Pytest fixtures and configuration for arpyc tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
import rpyc
from rpyc.utils.server import ThreadedServer

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for tests."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
async def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


class TestService(rpyc.Service):
    """Test RPyC service for integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.subscribers: set[Any] = set()

    def exposed_echo(self, msg: str) -> str:
        """Echo message back."""
        self.call_count += 1
        return f"echo: {msg}"

    def exposed_add(self, a: int, b: int) -> int:
        """Add two numbers."""
        self.call_count += 1
        return a + b

    def exposed_raise_error(self) -> None:
        """Raise an error for testing."""
        raise ValueError("Test error")

    def exposed_get_call_count(self) -> int:
        """Get number of calls."""
        return self.call_count

    def exposed_reset_call_count(self) -> None:
        """Reset call counter."""
        self.call_count = 0

    def exposed_subscribe(self, callback: Any) -> bool:
        """Subscribe to events."""
        self.subscribers.add(callback)
        return True

    def exposed_unsubscribe(self, callback: Any) -> bool:
        """Unsubscribe from events."""
        self.subscribers.discard(callback)
        return True

    def exposed_emit_event(self, name: str, payload: dict[str, Any]) -> None:
        """Emit event to all subscribers."""
        for sub in self.subscribers:
            try:
                sub.on_event(name, payload)
            except Exception:
                pass


@pytest.fixture(scope="module")
def rpyc_server() -> Generator[tuple[str, int], None, None]:
    """Start RPyC test server."""
    import threading

    host = "127.0.0.1"
    port = 18862  # Different port to avoid conflicts

    server = ThreadedServer(
        TestService,
        hostname=host,
        port=port,
        protocol_config={"allow_public_attrs": True},
    )

    # Start server in background thread
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    # Give server time to start
    import time

    time.sleep(0.5)

    yield (host, port)

    # Cleanup
    server.close()
