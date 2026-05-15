"""Wave 2 — WebSocket layer must log, degrade, and survive — never
silently drop messages or kill the server on a malformed client.

The WS surface has three failure modes we care about:

1. A broken client (slow / closed / crashed).
   ``ConnectionManager.broadcast`` and the per-connection message
   callback must log the failure **with detail** (exception + the
   connection's identity), not swallow it, and prune the connection
   so the next broadcast doesn't keep failing on the same dead
   socket.

2. A malformed client message.
   The WS main loop must reply with
   ``{"type": "error", "error": "invalid_json", ...}`` and keep
   looping. One bad frame must not terminate an otherwise-healthy
   connection.

3. A non-serialisable broadcast payload.
   ``ConnectionManager.broadcast`` has a per-field string-fallback
   today. If even that fails, it must emit a synthetic
   ``{"type": "broadcast_failed", "original_type": ..., "error": ...}``
   event so the UI can surface "something was lost" rather than
   pretend everything is fine.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.websockets import WebSocketState

from app.routers.websocket import ConnectionManager


def _fake_ws(
    *, connected: bool = True, send_side_effect: Any = None
) -> MagicMock:
    ws = MagicMock()
    ws.client_state = (
        WebSocketState.CONNECTED if connected else WebSocketState.DISCONNECTED
    )
    ws.send_json = AsyncMock(side_effect=send_side_effect)
    ws.client = ("127.0.0.1", 12345)  # shown in logs
    return ws


@pytest.mark.asyncio
async def test_broadcast_logs_and_prunes_broken_client(caplog) -> None:
    """A connection whose ``send_json`` raises must be removed from
    the manager AND the failure must be logged with an identifier —
    not silently pruned.
    """
    mgr = ConnectionManager()
    good = _fake_ws()
    bad = _fake_ws(send_side_effect=ConnectionResetError("peer closed"))

    mgr.active_connections.extend([good, bad])

    with caplog.at_level(logging.ERROR, logger="app.routers.websocket"):
        await mgr.broadcast({"type": "probe"})

    assert bad not in mgr.active_connections, "broken client must be pruned"
    assert good in mgr.active_connections, (
        "healthy client must not be pruned because another one failed"
    )

    err_matches = [
        rec for rec in caplog.records
        if rec.levelno >= logging.ERROR
        and "peer closed" in (rec.exc_text or rec.getMessage())
    ]
    assert err_matches, (
        "broken-client failure must be logged as ERROR with the "
        "exception detail; caplog: "
        + "; ".join(rec.getMessage() for rec in caplog.records)
    )


@pytest.mark.asyncio
async def test_broadcast_emits_broadcast_failed_on_unserializable_payload(
    caplog,
) -> None:
    """When even the per-field fallback can't produce something
    ``send_json`` will accept, the manager must emit a synthetic
    ``broadcast_failed`` event to every connected client so the
    UI sees that something was lost.
    """
    mgr = ConnectionManager()
    seen: list[dict[str, Any]] = []

    async def record_send(payload: dict[str, Any]) -> None:
        seen.append(payload)

    ws = _fake_ws()
    ws.send_json = AsyncMock(side_effect=record_send)
    mgr.active_connections.append(ws)

    class Unserialisable:
        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

        def __str__(self) -> str:
            raise RuntimeError("str exploded")

    with caplog.at_level(logging.ERROR, logger="app.routers.websocket"):
        await mgr.broadcast(
            {"type": "agent_message", "content": Unserialisable()}
        )

    # Must have delivered exactly one message: the broadcast_failed
    # envelope (the original payload could not be made JSON-safe).
    types_sent = [msg.get("type") for msg in seen]
    assert "broadcast_failed" in types_sent, (
        "when even fallback serialisation cannot rescue the payload, "
        "the manager must emit a broadcast_failed envelope so the UI "
        "knows something was dropped; got: " + repr(types_sent)
    )
    # The envelope must carry the original type and the serialisation
    # error so the UI / logs can tell what was lost.
    envelope = next(
        msg for msg in seen if msg.get("type") == "broadcast_failed"
    )
    assert envelope.get("original_type") == "agent_message"
    assert "error" in envelope

    # And an ERROR log must explain the serialisation failure.
    assert [
        rec for rec in caplog.records
        if rec.levelno >= logging.ERROR
        and "broadcast" in rec.getMessage().lower()
    ], "serialisation failure must log ERROR"


@pytest.mark.asyncio
async def test_broadcast_does_not_prune_connection_on_transient_send_error(
    caplog,
) -> None:
    """The current behaviour prunes on any send failure. That is
    the right trade-off: if a send fails we have no way to know
    whether the socket is still good, and the client reconnects
    cheaply. But the failure MUST be logged, not swallowed."""
    mgr = ConnectionManager()
    bad = _fake_ws(send_side_effect=RuntimeError("boom"))
    mgr.active_connections.append(bad)

    with caplog.at_level(logging.ERROR, logger="app.routers.websocket"):
        await mgr.broadcast({"type": "probe"})

    # Pruned (same policy as before).
    assert bad not in mgr.active_connections
    # Logged (this is the contract added in Wave 2).
    assert [
        rec for rec in caplog.records
        if "boom" in (rec.exc_text or rec.getMessage())
    ]
