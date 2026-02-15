"""ARPyC - Async RPyC Adapter for asyncio."""

__version__ = "0.1.0"

from arpyc.exceptions import (
    ARPyCConnectionError,
    ARPyCError,
    ARPyCTimeoutError,
    NotConnectedError,
    ReconnectFailedError,
)
from arpyc.helpers import (
    AsyncioCallbackAdapter,
    AsyncioEventCallback,
    await_rpyc,
)
from arpyc.heartbeat import HeartbeatManager
from arpyc.proxy import AsyncMethodProxy, AsyncProxy
from arpyc.session import RpycAsyncSession
from arpyc.streaming import StreamingCallback
from arpyc.types import AsyncEventHandler, EventHandler, StreamCallback

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
    "ARPyCConnectionError",
    "ARPyCTimeoutError",
    "NotConnectedError",
    "ReconnectFailedError",
    # Types
    "EventHandler",
    "AsyncEventHandler",
    "StreamCallback",
]
