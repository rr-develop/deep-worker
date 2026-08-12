"""Test rpyc_async integration.

``rpyc-async`` is a regular PyPI dependency (>=1.0.0). These tests verify
that the installed distribution is importable and carries the AsyncioServer
features this project relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_rpyc_is_importable():
    """``rpyc_async`` resolves to a real, importable package.

    Deliberately does not assert *where* it comes from: a source checkout on
    a developer machine and a site-packages install from PyPI are both valid,
    and pinning either one makes the test a statement about one machine's
    layout rather than about the dependency.
    """
    import rpyc_async as rpyc

    assert rpyc.__file__ is not None
    assert Path(rpyc.__file__).is_file()


def test_asyncio_server_available():
    """Test that AsyncioServer is available."""
    import rpyc_async as rpyc

    assert hasattr(rpyc.utils, "async_server"), (
        "AsyncioServer module not found - wrong rpyc version?"
    )

    from rpyc_async.utils.async_server import AsyncioServer

    assert AsyncioServer is not None


def test_protocol_version():
    """Test that protocol version is >= 5.1."""
    from rpyc_async.core import consts

    assert consts.PROTOCOL_VERSION >= (5, 1), (
        f"Wrong protocol version: {consts.PROTOCOL_VERSION}"
    )


def test_enable_asyncio_serving_method_exists():
    """Test that Connection has enable_asyncio_serving method."""
    from rpyc_async.core.protocol import Connection

    assert hasattr(Connection, "enable_asyncio_serving"), (
        "Connection.enable_asyncio_serving method not found"
    )
    assert hasattr(Connection, "disable_asyncio_serving"), (
        "Connection.disable_asyncio_serving method not found"
    )


def test_async_handlers_available():
    """Test that async handlers module is available."""
    from rpyc_async.core import async_handlers

    assert async_handlers is not None
    assert hasattr(async_handlers, "_handle_async_call"), (
        "_handle_async_call not found in async_handlers"
    )


def test_asyncresult_awaitable():
    """Test that AsyncResult is awaitable."""
    from rpyc_async.core.async_ import AsyncResult

    # Check that AsyncResult has __await__ method
    assert hasattr(AsyncResult, "__await__"), (
        "AsyncResult is not awaitable - missing __await__ method"
    )


@pytest.mark.asyncio
async def test_basic_asyncio_server_can_start():
    """Test that AsyncioServer can be instantiated and started."""
    import rpyc_async as rpyc
    from rpyc_async.core.async_connect import async_connect
    from rpyc_async.utils.async_server import AsyncioServer

    class TestService(rpyc.Service):
        async def exposed_hello(self):
            return "Hello from AsyncioServer!"

    # Create server with port=0 (auto-assign)
    server = AsyncioServer(TestService, port=0)

    # Start server
    await server.start()

    # Get assigned port
    port = server.listener.getsockname()[1]
    assert port > 0

    # Close server
    await server.close()


import asyncio as _asyncio_top  # noqa: E402
import contextlib  # noqa: E402
import multiprocessing  # noqa: E402
import socket as _socket_top  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from typing import Any, Callable  # noqa: E402

import rpyc_async as rpyc  # noqa: E402


def _get_free_port() -> int:
    s = _socket_top.socket(_socket_top.AF_INET, _socket_top.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _mp_server_entrypoint(
    service_factory: Callable[[], Any],
    port: int,
    ready_queue: Any,
) -> None:
    """Entry point for a child-process AsyncioServer."""
    import rpyc_async as _rpyc_child  # noqa: F401
    from rpyc_async.utils.async_server import AsyncioServer as _AsyncioServer

    async def _main() -> None:
        server = _AsyncioServer(
            service_factory(), hostname="127.0.0.1", port=port
        )
        await server.start()
        ready_queue.put("ready")
        try:
            await _asyncio_top.Event().wait()
        except _asyncio_top.CancelledError:
            pass
        finally:
            await server.close()

    try:
        _asyncio_top.run(_main())
    except KeyboardInterrupt:
        pass


@contextlib.contextmanager
def _mp_asyncio_server(
    service_factory: Callable[[], Any],
    *,
    ready_timeout: float = 5.0,
) -> Iterator[int]:
    """Start `AsyncioServer(service_factory())` in a child process.

    Local copy of `tests/support.py::mp_asyncio_server` from the rpyc-async
    project. The same-process server+client topology is forbidden by that
    project's own policy.
    """
    port = _get_free_port()
    ready_queue: Any = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_mp_server_entrypoint,
        args=(service_factory, port, ready_queue),
        daemon=True,
    )
    proc.start()
    try:
        try:
            signal = ready_queue.get(timeout=ready_timeout)
        except Exception as exc:
            raise RuntimeError(
                f"AsyncioServer child failed to report ready within "
                f"{ready_timeout}s: {exc!r}"
            ) from exc
        if signal != "ready":
            raise RuntimeError(f"unexpected signal: {signal!r}")
        yield port
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)


class _AsyncEchoService(rpyc.Service):
    """Tiny async service for the round-trip test. Must live at module
    scope so `multiprocessing` can pickle it."""

    async def exposed_test_method(self):
        import asyncio as _asyncio
        await _asyncio.sleep(0.01)
        return "test_response"


class _CallbackService(rpyc.Service):
    """Service that invokes a client-provided async callback. Must live
    at module scope for pickling by `multiprocessing`."""

    async def exposed_call_with_callback(self, callback, value):
        # Server calls client async callback
        result = await callback(value * 2)
        return result + 100


@pytest.mark.asyncio
async def test_native_rpyc_connection_with_asyncio():
    """Test native rpyc connection round-trips an `async def exposed_*`.

    Runs the server in a child process per
    `docs/DESIGN_NO_SAME_PROCESS_TESTS.md` — AsyncioServer + client in
    one process is forbidden (would deadlock on the shared event
    loop).
    """
    from rpyc_async.core.async_connect import async_connect
    with _mp_asyncio_server(_AsyncEchoService) as port:
        conn = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            result = await conn.root.test_method()
            assert result == "test_response"
        finally:
            await conn.aclose()


@pytest.mark.asyncio
async def test_bidirectional_async_callbacks():
    """Server calls a client-provided async callback — must not deadlock.

    Runs the server in a child process; the bidirectional-recursion
    fix (commit fe14779 in rpyc_async) guarantees no self-deadlock
    under this pattern.
    """
    import asyncio
    from rpyc_async.core.async_connect import async_connect

    with _mp_asyncio_server(_CallbackService) as port:
        conn = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            async def client_callback(value):
                await asyncio.sleep(0.01)
                return value + 10

            # 5 * 2 = 10, 10 + 10 = 20, 20 + 100 = 120
            result = await conn.root.call_with_callback(client_callback, 5)
            assert result == 120
        finally:
            await conn.aclose()
