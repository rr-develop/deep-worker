# ARPyC - Async RPyC Adapter

## 📋 Overview

**ARPyC** (Asyncio RPyC) is a thin adaptation layer on top of RPyC that provides:
- ✅ Native `asyncio` support (await for remote calls)
- ✅ Safe integration of callbacks from RPyC threads into the asyncio event loop
- ✅ Transparent async proxy objects
- ✅ Heartbeat and auto-reconnect
- ✅ Streaming support for real-time data
- ✅ Type hints and Protocol-based typing

---

## 🏗️ Package architecture

```
arpyc/
├── README.md                    # Documentation and usage examples
├── pyproject.toml               # Package metadata (for a future pip install)
├── arpyc/
│   ├── __init__.py              # Public API exports
│   ├── session.py               # RpycAsyncSession - the main class
│   ├── helpers.py               # await_rpyc, callback adapters
│   ├── proxy.py                 # AsyncProxy for transparent await
│   ├── exceptions.py            # Custom exceptions
│   ├── types.py                 # Type hints, Protocols
│   ├── heartbeat.py             # Heartbeat and reconnect logic
│   └── streaming.py             # Streaming helpers for real-time data
├── examples/
│   ├── basic/
│   │   ├── server.py            # Simple server example
│   │   └── client.py            # Simple client example
│   ├── streaming/
│   │   ├── server.py            # Server with streaming
│   │   └── client.py            # Client with streaming
│   └── callbacks/
│       ├── server.py            # Server with an event bus
│       └── client.py            # Client with subscriptions
└── tests/
    ├── __init__.py
    ├── conftest.py              # Pytest fixtures
    ├── test_session.py          # RpycAsyncSession tests
    ├── test_helpers.py          # await_rpyc tests
    ├── test_proxy.py            # AsyncProxy tests
    └── test_integration.py      # End-to-end tests
```

---

## 🔧 Key components

### 1. `arpyc.helpers` - Base utilities

```python
"""Low-level helpers for RPyC async integration."""

from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, TypeVar
import rpyc

T = TypeVar("T")

def await_rpyc(
    ar: rpyc.core.async_.AsyncResult,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    timeout: float | None = None
) -> Awaitable[T]:
    """
    Convert RPyC AsyncResult to awaitable Future.

    Args:
        ar: RPyC AsyncResult from async_ call
        loop: Event loop (defaults to current)
        timeout: Optional timeout in seconds

    Returns:
        Awaitable that resolves to remote call result

    Raises:
        asyncio.TimeoutError: If timeout exceeded
        RPyCException: If remote call failed
    """
    ...

class AsyncioCallbackAdapter:
    """
    Base class for callbacks passed to RPyC server.
    Automatically forwards calls from RPyC thread to asyncio loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        ...

    def _dispatch(self, coro: Awaitable[Any]) -> None:
        """Dispatch coroutine to asyncio loop thread-safely."""
        ...

class AsyncioEventCallback(AsyncioCallbackAdapter):
    """Event callback adapter for publish-subscribe patterns."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        handler: Callable[[str, dict], Awaitable[None] | None]
    ) -> None:
        ...

    def on_event(self, name: str, payload: dict) -> None:
        """Called from RPyC thread, dispatched to asyncio."""
        ...
```

**Highlights:**
- Thread-safe via `call_soon_threadsafe`
- Support for both sync and async handlers
- Timeout support for await_rpyc
- Graceful error handling

---

### 2. `arpyc.proxy` - Transparent async proxies

```python
"""Transparent async proxy for RPyC remote objects."""

from __future__ import annotations
from typing import Any, Awaitable
import rpyc

class AsyncProxy:
    """
    Transparent async wrapper for RPyC netref objects.
    Allows: await proxy.method(...) instead of await session.async_call(...)
    """

    def __init__(
        self,
        netref: Any,  # RPyC netref
        session: RpycAsyncSession
    ) -> None:
        ...

    def __getattr__(self, name: str) -> AsyncMethodProxy:
        """Get async method proxy."""
        ...

    def __dir__(self) -> list[str]:
        """List available methods (for IDE autocomplete)."""
        ...

class AsyncMethodProxy:
    """Proxy for individual remote method."""

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """Execute remote method asynchronously."""
        ...

    def sync(self, *args: Any, **kwargs: Any) -> Any:
        """Execute remote method synchronously (blocking)."""
        ...
```

**Usage:**
```python
# Instead of:
counter = conn.root.create_counter()
result = await session.async_call(counter.inc, 5)

# You can write:
counter = session.proxy(conn.root.create_counter())
result = await counter.inc(5)

# Or sync if needed:
result = counter.inc.sync(5)
```

---

### 3. `arpyc.session` - The main class

```python
"""Main session class for RPyC async connections."""

from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, TypeVar
import rpyc
from rpyc.utils.helpers import BgServingThread

T = TypeVar("T")

class RpycAsyncSession:
    """
    Asyncio-friendly RPyC session with:
    - Background serving thread for reverse calls
    - Async call support
    - Auto-reconnect
    - Heartbeat
    - Context manager protocol
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18861,
        *,
        config: dict[str, Any] | None = None,
        auto_reconnect: bool = True,
        heartbeat_interval: float = 5.0,
        connect_timeout: float = 10.0
    ) -> None:
        ...

    async def connect(self) -> None:
        """Establish connection and start background thread."""
        ...

    async def disconnect(self) -> None:
        """Gracefully close connection."""
        ...

    async def __aenter__(self) -> RpycAsyncSession:
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        """Context manager exit."""
        await self.disconnect()

    @property
    def conn(self) -> rpyc.Connection:
        """Get underlying RPyC connection."""
        ...

    @property
    def root(self) -> Any:
        """Get root service proxy."""
        ...

    def async_call(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any
    ) -> Awaitable[T]:
        """Execute remote method asynchronously."""
        ...

    def proxy(self, netref: Any) -> AsyncProxy:
        """Wrap netref in AsyncProxy for transparent await."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is alive."""
        ...
```

**Usage:**
```python
async with RpycAsyncSession("localhost", 18861) as session:
    # Auto-connect and auto-disconnect
    counter = session.proxy(session.root.create_counter())
    await counter.inc(10)
```

---

### 4. `arpyc.heartbeat` - Heartbeat and reconnect

```python
"""Heartbeat and auto-reconnect functionality."""

from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable

class HeartbeatManager:
    """
    Manages connection health checks and auto-reconnect.
    """

    def __init__(
        self,
        session: RpycAsyncSession,
        interval: float = 5.0,
        timeout: float = 3.0,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
        on_reconnect: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        ...

    async def start(self) -> None:
        """Start heartbeat monitoring."""
        ...

    async def stop(self) -> None:
        """Stop heartbeat monitoring."""
        ...

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        ...
```

**Highlights:**
- Periodic `conn.ping()` to check connectivity
- Automatic reconnect when the connection drops
- Callbacks for handling disconnect/reconnect events
- Exponential backoff for retries

---

### 5. `arpyc.streaming` - Real-time streaming

```python
"""Streaming support for real-time data."""

from __future__ import annotations
import asyncio
from typing import AsyncIterator, Callable, TypeVar

T = TypeVar("T")

class StreamingCallback:
    """
    Callback adapter for streaming data from RPyC to asyncio.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._loop = loop

    def on_chunk(self, chunk: T) -> None:
        """Called from RPyC thread with new chunk."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

    def on_done(self) -> None:
        """Signal end of stream."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    async def __aiter__(self) -> AsyncIterator[T]:
        """Async iterator over stream chunks."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
```

**Usage:**
```python
# Server streaming
class Service(rpyc.Service):
    def exposed_stream_data(self, callback):
        for i in range(100):
            callback.on_chunk(f"chunk-{i}")
            time.sleep(0.1)
        callback.on_done()

# Client consuming
callback = StreamingCallback(asyncio.get_running_loop())
session.root.stream_data(callback)

async for chunk in callback:
    print(f"Received: {chunk}")
```

---

### 6. `arpyc.types` - Type hints and Protocols

```python
"""Type hints and Protocol definitions."""

from __future__ import annotations
from typing import Protocol, Any, Awaitable, runtime_checkable

@runtime_checkable
class EventHandler(Protocol):
    """Protocol for event handlers."""

    def on_event(self, name: str, payload: dict) -> None:
        """Handle incoming event."""
        ...

@runtime_checkable
class AsyncEventHandler(Protocol):
    """Protocol for async event handlers."""

    async def on_event(self, name: str, payload: dict) -> None:
        """Handle incoming event asynchronously."""
        ...

@runtime_checkable
class StreamCallback(Protocol):
    """Protocol for streaming callbacks."""

    def on_chunk(self, chunk: Any) -> None:
        """Handle stream chunk."""
        ...

    def on_done(self) -> None:
        """Handle stream completion."""
        ...
```

---

### 7. `arpyc.exceptions` - Exceptions

```python
"""Custom exceptions for arpyc."""

class ARPyCError(Exception):
    """Base exception for arpyc errors."""
    pass

class ConnectionError(ARPyCError):
    """Connection-related errors."""
    pass

class TimeoutError(ARPyCError):
    """Timeout errors."""
    pass

class NotConnectedError(ConnectionError):
    """Raised when operation requires active connection."""
    pass

class ReconnectFailedError(ConnectionError):
    """Raised when auto-reconnect fails."""
    pass
```

---

## 📝 Public API (`arpyc/__init__.py`)

```python
"""ARPyC - Async RPyC Adapter for asyncio."""

__version__ = "0.1.0"

from arpyc.session import RpycAsyncSession
from arpyc.helpers import (
    await_rpyc,
    AsyncioCallbackAdapter,
    AsyncioEventCallback,
)
from arpyc.proxy import AsyncProxy, AsyncMethodProxy
from arpyc.heartbeat import HeartbeatManager
from arpyc.streaming import StreamingCallback
from arpyc.exceptions import (
    ARPyCError,
    ConnectionError,
    TimeoutError,
    NotConnectedError,
    ReconnectFailedError,
)
from arpyc.types import (
    EventHandler,
    AsyncEventHandler,
    StreamCallback,
)

__all__ = [
    # Main
    "RpycAsyncSession",
    # Helpers
    "await_rpyc",
    "AsyncioCallbackAdapter",
    "AsyncioEventCallback",
    # Proxy
    "AsyncProxy",
    "AsyncMethodProxy",
    # Heartbeat
    "HeartbeatManager",
    # Streaming
    "StreamingCallback",
    # Exceptions
    "ARPyCError",
    "ConnectionError",
    "TimeoutError",
    "NotConnectedError",
    "ReconnectFailedError",
    # Types
    "EventHandler",
    "AsyncEventHandler",
    "StreamCallback",
]
```

---

## 🎯 Usage examples

### Basic Example

```python
import asyncio
from arpyc import RpycAsyncSession, AsyncioEventCallback

async def main():
    async with RpycAsyncSession("localhost", 18861) as session:
        # Event subscription
        def on_event(name: str, payload: dict):
            print(f"Event: {name} -> {payload}")

        callback = AsyncioEventCallback(
            asyncio.get_running_loop(),
            on_event
        )
        session.root.subscribe(callback)

        # Remote object with transparent async
        counter = session.proxy(session.root.create_counter())

        # Await remote methods
        result = await counter.inc(5)
        print(f"Counter: {result}")

        # Or sync if needed
        current = counter.get.sync()
        print(f"Current: {current}")

asyncio.run(main())
```

### Streaming Example

```python
from arpyc import RpycAsyncSession, StreamingCallback

async def main():
    async with RpycAsyncSession("localhost", 18861) as session:
        callback = StreamingCallback(asyncio.get_running_loop())

        # Start remote streaming
        session.root.stream_messages(callback)

        # Consume stream
        async for message in callback:
            print(f"Received: {message}")

asyncio.run(main())
```

---

## 🔍 Integration with deep-worker

For deep-worker, usage will look like this:

```python
# In agent_process.py - server
class AgentService(rpyc.Service):
    def exposed_subscribe_messages(self, callback):
        self._message_subscribers.add(callback)

    def _on_new_message(self, msg: dict):
        for cb in self._message_subscribers:
            cb.on_message(msg)

# In interactive.py - client
class InteractiveApp(App):
    async def on_mount(self):
        self.session = RpycAsyncSession(...)
        await self.session.connect()

        callback = AsyncioEventCallback(
            asyncio.get_running_loop(),
            self._on_agent_message
        )
        self.session.root.subscribe_messages(callback)

    async def _on_agent_message(self, msg: dict):
        # Real-time message display!
        await self._mount_message(AssistantMessage(msg['content']))
```

---

## 📦 Dependencies

```toml
[project]
name = "arpyc"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "rpyc>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "mypy>=1.0",
]
```

---

## ✅ Design advantages

1. **Minimal dependencies**: RPyC only
2. **Type-safe**: full typing with Protocols
3. **Thread-safe**: safe integration of RPyC threads ↔ asyncio
4. **Transparency**: `await proxy.method()` instead of boilerplate
5. **Production-ready**: heartbeat, reconnect, error handling
6. **Extensible**: easy to add new adapters
7. **Testable**: full test coverage

---

## 🚀 Next steps

1. ✅ **Design approval**
2. Implement the core functionality (`helpers.py`, `session.py`)
3. Implement the proxy layer (`proxy.py`)
4. Implement heartbeat and streaming
5. Write the examples
6. Write the tests
7. Integration with deep-worker

---

## ❓ Open questions

1. **RPyC version**: use the latest 6.x or pin a specific one?
2. **Logging**: use the standard `logging` module or something else?
3. **Metrics**: are built-in metrics (latency, errors) needed?
4. **Batching**: is support for batch remote calls needed?
5. **Compression**: is compression needed for large payloads?

---

**Ready for implementation once you approve!** 🎉
