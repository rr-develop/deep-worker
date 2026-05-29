"""`_notify_subscribers` must ship a serialised JSON string.

Historic behaviour was to pass the dict by reference across RPyC,
which on the subscriber side ended up being a netref. Traversing
that netref (`msg.items()`) from the subscriber's async loop
triggered RPyC's own ``sync_request() from asyncio loop`` guard
and leaked remote references. The fix: serialise once on the
agent side and ship a plain string — subscribers just
``json.loads``.

Note: with the 2026-05-20 switch to ``fire_and_forget_async``, the
dispatched ``on_message`` call runs as a separate Task, so the
caller's ``await`` does not by itself drive the subscriber.
We use an ``asyncio.Event`` to wait for delivery deterministically.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


@pytest.mark.asyncio
async def test_notify_subscribers_sends_json_string() -> None:
    received: list[object] = []
    received_event = asyncio.Event()

    async def on_message(payload: object) -> None:
        received.append(payload)
        received_event.set()

    service = AgentService()
    agent = MagicMock()
    subscriber = MagicMock()
    subscriber.on_message = on_message
    agent._message_subscribers = {subscriber}
    service._agent_process = agent

    await service._notify_subscribers({"type": "user", "content": "hi"})
    await asyncio.wait_for(received_event.wait(), timeout=1.0)

    assert len(received) == 1
    (arg,) = received
    assert isinstance(arg, str), f"expected str, got {type(arg).__name__}"
    assert json.loads(arg) == {"type": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_notify_subscribers_handles_unserialisable_payload() -> None:
    """If someone accidentally passes a non-JSON-safe dict, we must
    still deliver a payload rather than drop the broadcast silently."""
    received: list[object] = []
    received_event = asyncio.Event()

    async def on_message(payload: object) -> None:
        received.append(payload)
        received_event.set()

    service = AgentService()
    agent = MagicMock()
    subscriber = MagicMock()
    subscriber.on_message = on_message
    agent._message_subscribers = {subscriber}
    service._agent_process = agent

    class NotSerialisable:
        def __repr__(self) -> str:  # noqa: D401
            return "<NotSerialisable>"

    await service._notify_subscribers({"type": "tool_result", "payload": NotSerialisable()})
    await asyncio.wait_for(received_event.wait(), timeout=1.0)

    assert len(received) == 1
    (arg,) = received
    assert isinstance(arg, str)
    # Must be valid JSON even if we fell back to str()-ing the inner value
    parsed = json.loads(arg)
    assert parsed.get("type") in ("tool_result", "raw")
