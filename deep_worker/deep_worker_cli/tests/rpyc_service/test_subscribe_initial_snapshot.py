"""``exposed_subscribe_messages`` must push a status snapshot to every
fresh subscriber.

Without this, a client reconnecting after a status transition would
never learn the current state (there is no polling in the system any
more). The snapshot is a regular ``status_change`` event delivered on
the subscribe callback the moment it is registered — not a new wire
type, just the first message of the stream.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


def _build_service_with_agent(
    *,
    status_value: str = "running",
    tm_active: bool | None = True,
    tm_disabled: bool | None = False,
    tm_present: bool = True,
) -> tuple[AgentService, MagicMock]:
    service = AgentService()
    agent = MagicMock()
    agent._message_subscribers = set()

    # Build the snapshot envelope exactly as AgentProcess.
    # _current_status_snapshot() would. We stub the method on the
    # MagicMock so it returns a real dict rather than another mock —
    # otherwise ``json.dumps`` would choke and hide real bugs.
    snapshot: dict[str, Any] = {
        "type": "status_change",
        "status": status_value,
        "error_message": None,
    }
    if tm_present:
        snapshot["task_manager_active"] = bool(tm_active)
        snapshot["task_manager_disabled"] = bool(tm_disabled)
    agent._current_status_snapshot = MagicMock(return_value=snapshot)

    service._agent_process = agent
    return service, agent


@pytest.mark.asyncio
async def test_subscribe_sends_initial_snapshot() -> None:
    service, agent = _build_service_with_agent(status_value="running")
    callback = MagicMock()
    callback.on_message = AsyncMock()

    ok = await service.exposed_subscribe_messages(callback)
    assert ok is True

    # Snapshot delivery is scheduled as a task (see
    # ``exposed_subscribe_messages`` doc) — yield so it runs.
    import asyncio as _asyncio
    # Snapshot is delivered via ``loop.call_later(0, ...)`` so the
    # subscribe RPC can flush first. Give the event loop a real tick
    # (non-zero sleep) so that deferred task actually runs.
    await _asyncio.sleep(0.02)

    # Callback must have been invoked exactly once after yield —
    # that is the initial snapshot.
    assert callback.on_message.await_count == 1
    (arg,) = callback.on_message.await_args.args
    assert isinstance(arg, str), "payload must be a JSON string"
    payload = json.loads(arg)
    assert payload["type"] == "status_change"
    assert payload["status"] == "running"
    # Subscriber has been added.
    assert callback in agent._message_subscribers


@pytest.mark.asyncio
async def test_snapshot_includes_task_manager_fields_when_configured() -> None:
    import asyncio as _asyncio

    service, _agent = _build_service_with_agent(
        tm_present=True, tm_active=True, tm_disabled=False
    )
    cb = MagicMock()
    cb.on_message = AsyncMock()

    await service.exposed_subscribe_messages(cb)
    await _asyncio.sleep(0.02)

    (arg,) = cb.on_message.await_args.args
    payload = json.loads(arg)
    assert payload["task_manager_active"] is True
    assert payload["task_manager_disabled"] is False


@pytest.mark.asyncio
async def test_snapshot_omits_task_manager_fields_when_absent() -> None:
    """No TM configured → TM keys absent entirely. Never invent data."""
    import asyncio as _asyncio

    service, _agent = _build_service_with_agent(tm_present=False)
    cb = MagicMock()
    cb.on_message = AsyncMock()

    await service.exposed_subscribe_messages(cb)
    await _asyncio.sleep(0.02)

    (arg,) = cb.on_message.await_args.args
    payload = json.loads(arg)
    assert "task_manager_active" not in payload
    assert "task_manager_disabled" not in payload


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_block_subscription() -> None:
    """Even if the initial snapshot push fails (netref EOF etc.), the
    subscriber must still end up in the set so subsequent events can
    reach it."""
    service, agent = _build_service_with_agent()
    cb = MagicMock()
    cb.on_message = AsyncMock(side_effect=RuntimeError("flaky client"))

    ok = await service.exposed_subscribe_messages(cb)
    assert ok is True
    assert cb in agent._message_subscribers
