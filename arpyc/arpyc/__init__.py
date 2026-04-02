"""ARPyC - Async RPyC Adapter for asyncio.

This library provides two approaches for RPyC integration with asyncio:

1. RpycAsyncSession (BgServingThread approach):
   - Uses separate background thread for serving RPC requests
   - Callbacks execute in BgServingThread
   - Has polling delays (sleep_interval=0.1)
   - Compatible with all platforms

2. RpycAsyncSessionNative (asyncio-native approach):
   - Uses loop.add_reader() for event-driven RPC serving
   - Callbacks execute in event loop (main thread)
   - No polling delays - event-driven
   - ~156x FASTER than BgServingThread
   - Requires Unix-like systems (Windows has limitations)

Choose RpycAsyncSessionNative for best performance and lower latency.
Choose RpycAsyncSession for compatibility or if using Windows.
"""

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
from arpyc.session_native import RpycAsyncSessionNative
from arpyc.streaming import StreamingCallback
from arpyc.types import AsyncEventHandler, EventHandler, StreamCallback

__all__ = [
    # Main - Two approaches
    "RpycAsyncSession",  # BgServingThread approach (compatible)
    "RpycAsyncSessionNative",  # Asyncio-native approach (faster, ~156x)
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
