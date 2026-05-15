"""Wave 2 — the WebSocket main loop must not be killed by a single
malformed client frame.

Before this wave, any non-JSON frame from the client raised
``json.JSONDecodeError`` inside ``websocket.receive_json()`` — caught
by the bare ``except Exception`` which ``break``-ed out of the loop
and terminated the connection. A buggy or chatty client could make
the site look "dead" on every retry.

Contract
--------
When ``receive_json`` raises ``json.JSONDecodeError``:

* the loop MUST NOT break;
* the server sends ``{"type": "error", "error": "invalid_json", ...}``
  back to the same client so the UI can surface it;
* a WARNING is logged with the offending connection's identifier.

Other exceptions (unknown, protocol errors) still break the loop —
that policy is unchanged; we are strictly hardening the most common
client-side mistake (sending a plain-text ping).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.websockets import WebSocketState

import app.routers.websocket as ws_mod


@pytest.mark.asyncio
async def test_websocket_loop_survives_invalid_json(caplog) -> None:
    """Drive ``websocket_endpoint`` directly with a fake WebSocket
    whose ``receive_json`` first raises JSONDecodeError (malformed
    frame) and then raises ``WebSocketDisconnect`` to let the loop
    exit cleanly.

    We verify:
      (1) the loop did not ``break`` on the JSONDecodeError — it
          called ``receive_json`` a second time;
      (2) the server sent an ``invalid_json`` error frame to the
          client between the two receives;
      (3) the malformed-frame event was logged at WARNING level.
    """
    # Fake WebSocket mimicking just what websocket_endpoint needs.
    ws = MagicMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.client = ("127.0.0.1", 12345)
    ws.accept = AsyncMock(return_value=None)
    ws.send_json = AsyncMock(return_value=None)

    # First receive_json: malformed JSON. Second: disconnect.
    from fastapi import WebSocketDisconnect

    receive_calls = {"n": 0}

    async def receive_json_side_effect() -> Any:
        receive_calls["n"] += 1
        if receive_calls["n"] == 1:
            raise json.JSONDecodeError("Expecting value", "junk", 0)
        raise WebSocketDisconnect()

    ws.receive_json = AsyncMock(side_effect=receive_json_side_effect)

    # Minimal app state — ``websocket_endpoint`` reads
    # ``websocket.app.state.agent_service``.
    fake_service = MagicMock()
    fake_service.list_agents = AsyncMock(return_value=[])
    fake_service.register_message_callback = MagicMock()
    fake_service.unregister_message_callback = MagicMock()

    fake_app = MagicMock()
    fake_app.state = MagicMock(agent_service=fake_service)
    ws.app = fake_app

    with caplog.at_level(logging.WARNING, logger="app.routers.websocket"):
        await ws_mod.websocket_endpoint(ws)

    assert receive_calls["n"] >= 2, (
        "loop must not break on JSONDecodeError — it should call "
        "receive_json again, got only "
        + str(receive_calls["n"])
        + " calls"
    )

    # Check send_json was called with an invalid_json error envelope
    # at some point before the disconnect.
    error_frames = [
        call.args[0] for call in ws.send_json.call_args_list
        if isinstance(call.args[0], dict)
        and call.args[0].get("type") == "error"
        and call.args[0].get("error") == "invalid_json"
    ]
    assert error_frames, (
        "server must send {type: error, error: invalid_json, ...} "
        "to the client; sent frames: "
        + repr([c.args[0] for c in ws.send_json.call_args_list])
    )

    # Logged at WARNING (not ERROR — it's a client mistake, not a
    # server bug).
    assert [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "invalid_json" in rec.getMessage().lower()
    ], (
        "malformed client frame must log at WARNING with 'invalid_json' "
        "in the message; got: "
        + "; ".join(rec.getMessage() for rec in caplog.records)
    )
