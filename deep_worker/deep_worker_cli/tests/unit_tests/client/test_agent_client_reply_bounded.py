"""Regression: an in-flight RPyC reply must FAIL when the transport dies,
never park forever.

Incident (repo_graph ingest, 2026-09-23): the agent successfully called its
MCP tool (``submit_links``, 6 links, ``batch_complete=true``) and finished its
turn in ~1.5s — ``token_stats.jsonl`` recorded it at 02:55:05Z. The caller's
``converse()`` did not return for 69 MINUTES. The stack showed a coroutine
parked in ``ep_poll`` awaiting the ``AsyncResult`` from
``AgentClient._send_command``; the channel had died with
``ClosedResourceError``.

Root cause (rpyc_async ``protocol.py:714``): ``Connection._cleanup`` does
``self._request_callbacks.clear()``. Pending ``AsyncResult`` objects are
DISCARDED, not failed — nothing calls ``set_result`` / ``set_exception`` /
``cancel`` on their futures. A coroutine already parked inside
``AsyncResult.__await__`` is therefore never woken. The 2026-04-27
cancel-aware cleanup in ``__await__`` cannot save it: that fires when the
future reaches a done state, and here it never does.

Why the watchdog did not catch it: the 600s watchdog measures SILENCE, and the
successful tool-call had just re-armed the progress window via
``note_progress``. "Progress happened, but no answer will ever come" is
invisible to a silence timer — hence the 69-minute park.

Fix: ``AgentClient._await_reply`` races the reply against rpyc_async's
EVENT-DRIVEN close notification (``add_close_callback``, fired from
``_cleanup`` → ``_fire_close_notifications``). No polling of ``conn.closed``
— the project forbids status polling.

These tests drive a REAL ``AsyncResult`` over a real ``socketpair``-backed
``Connection`` so they exercise the genuine discard path, not a mock of it.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from deep_worker_cli.client.agent_client import AgentClient, IPCConnectionError


def _live_connection():
    """A real rpyc_async Connection over a socketpair, plus the peer socket.

    Closing the peer socket and the connection reproduces the production
    transport death exactly (``_cleanup`` clears ``_request_callbacks``).
    """
    from rpyc_async.core.service import VoidService
    from rpyc_async.core.stream import SocketStream
    from rpyc_async.utils.factory import connect_stream

    ours, peer = socket.socketpair()
    conn = connect_stream(
        SocketStream(ours), VoidService, config={"allow_public_attrs": True}
    )
    conn.enable_asyncio_serving()
    return conn, peer


def _pending_async_result(conn):
    """An AsyncResult for a request the peer will never answer."""
    from rpyc_async.core import consts

    return conn.async_request(
        consts.HANDLE_ASYNC_CALLATTR, conn._local_root, "exposed_never", (), ()
    )


@pytest.mark.asyncio
async def test_transport_death_fails_an_in_flight_reply() -> None:
    """The bug: awaiter parked BEFORE the transport dies is never woken.

    This is the production shape — ``converse()`` was already awaiting when
    ``ClosedResourceError`` killed the channel. Without the fix this test
    hangs until the ``wait_for`` bound below trips.
    """
    conn, peer = _live_connection()
    client = AgentClient("probe")
    client._conn = conn
    client.connected = True

    async_result = _pending_async_result(conn)
    awaiter = asyncio.ensure_future(client._await_reply(async_result))

    # Let the awaiter actually park on the future before killing the channel.
    await asyncio.sleep(0.1)
    assert not awaiter.done(), "precondition: the reply is still in flight"

    peer.close()
    conn.close()

    with pytest.raises(IPCConnectionError) as excinfo:
        # A generous bound: the point is that it resolves at all. Pre-fix this
        # raises TimeoutError instead, which is the defect.
        await asyncio.wait_for(awaiter, timeout=5.0)

    assert "closed while awaiting" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_healthy_reply_is_returned_unchanged() -> None:
    """NEGATIVE CONTROL: the same helper must NOT invent a failure.

    Without this, a helper that simply raised on every call would pass the
    test above while breaking every RPC in production. ``AsyncResult.__call__
    (is_exc, obj)`` is the real completion entry point — the same one the
    protocol uses when a genuine reply lands.
    """
    conn, peer = _live_connection()
    client = AgentClient("probe")
    client._conn = conn
    client.connected = True

    async_result = _pending_async_result(conn)
    awaiter = asyncio.ensure_future(client._await_reply(async_result))
    await asyncio.sleep(0.1)

    async_result(False, "the-reply")  # peer answers normally

    assert await asyncio.wait_for(awaiter, timeout=5.0) == "the-reply"

    peer.close()
    conn.close()


@pytest.mark.asyncio
async def test_close_before_the_await_also_fails_fast() -> None:
    """Already-closed connection: ``add_close_callback`` fires immediately.

    rpyc_async invokes the callback synchronously at registration time when
    the connection is already closed, so this path must not hang either.
    """
    conn, peer = _live_connection()
    client = AgentClient("probe")
    client._conn = conn
    client.connected = True

    async_result = _pending_async_result(conn)
    peer.close()
    conn.close()

    with pytest.raises(IPCConnectionError):
        await asyncio.wait_for(client._await_reply(async_result), timeout=5.0)
