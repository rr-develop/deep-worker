"""Regression: the status-POLLING path must forward the agent's error
message, not just the bare status.

Two paths feed the UI's agent status:
  1. subscribe-push (``_dispatch_subscriber_payload``) — already
     forwards ``error_message`` (see test_status_change_handling.py).
  2. polling (``_update_agent_status`` → ``root.get_status()``) — this
     one used to broadcast only ``status`` and drop the agent's
     ``error`` field, so a UI that learned the agent was in error via
     a poll never got the human-readable cause to show in the
     persistent error panel (incident observed 2026-05-20).

This test drives the real ``_update_agent_status`` with a fake root
whose ``get_status()`` returns the agent's error payload, and asserts
the broadcast carries ``error_message``.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService


def _make_service(name: str = "agent_a") -> AgentService:
    svc = AgentService(registry=object())
    svc._agents[name] = AgentInfo(
        name=name,
        pid=4242,
        status=AgentStatus.IDLE,
        provider="codex",
        model="unknown",
        rpyc_port=12345,
        working_dir="/path/to/other",
        started_at=datetime.now(),
    )
    return svc


class _FakeRoot:
    """Root proxy stub: ``get_status()`` returns a JSON string the same
    shape the real agent's ``exposed_get_status`` produces."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def get_status(self) -> str:
        return json.dumps(self._payload)


async def _run_and_capture(
    svc: AgentService, agent_name: str, root: Any
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        captured.append(msg)

    svc._broadcast_message = capture  # type: ignore[method-assign]
    await svc._update_agent_status(agent_name, root)
    # _update_agent_status is fire-and-forget; give the dispatched task
    # room to run its success callback.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if captured:
            break
    return captured


@pytest.mark.asyncio
async def test_poll_forwards_error_message_when_status_error() -> None:
    svc = _make_service("agent_a")
    err = (
        "Agent invocation failed: Codex stream error: "
        '{"status":400,"error":{"message":"The \'unknown\' model is '
        'not supported when using Codex with a ChatGPT account."}}'
    )
    root = _FakeRoot({"success": True, "status": "error", "error": err})

    events = await _run_and_capture(svc, "agent_a", root)
    status_events = [e for e in events if e.get("type") == "agent_status"]
    assert status_events, f"expected agent_status broadcast; got {events}"
    e = status_events[0]
    assert e.get("status") == "error"
    assert e.get("error_message") == err, (
        "the polling path must forward the agent's error so the UI's "
        "persistent error panel can render the full cause"
    )


@pytest.mark.asyncio
async def test_poll_omits_error_message_when_healthy() -> None:
    svc = _make_service("agent_a")
    root = _FakeRoot({"success": True, "status": "idle", "error": None})

    events = await _run_and_capture(svc, "agent_a", root)
    status_events = [e for e in events if e.get("type") == "agent_status"]
    assert status_events
    assert "error_message" not in status_events[0], (
        "a healthy poll must not fabricate an error_message (would leave "
        "a stale error panel up after recovery)"
    )
