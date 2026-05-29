"""Web-side contract for the ``channel_quarantined`` agent-push event.

When a client's RPyC channel to an agent is flooded and put into inbound
quarantine, the agent broadcasts ``{"type": "channel_quarantined", ...}``.
This is a WARNING ONLY — the agent keeps working and its status is
unchanged. The web layer must:

1. Forward it as a dedicated top-level ``channel_quarantined`` WS event
   (level=warning), carrying connid/peer/content for the banner.
2. NOT translate it into an ``agent_status`` change.
3. NOT alter the in-memory ``AgentInfo`` status mirror.

Incident observed 2026-05-22: the quarantine used to (via cancelled invoke) push the
agent to INTERRUPTED; now it must be a non-blocking warning only.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService


def _make_service_with_agent(name: str = "alpha") -> AgentService:
    svc = AgentService(registry=object())
    svc._agents[name] = AgentInfo(
        name=name,
        pid=4242,
        status=AgentStatus.RUNNING,  # actively working
        provider="claude-code",
        model="opus",
        rpyc_port=12345,
        working_dir="/tmp/test",
        started_at=datetime.now(),
    )
    return svc


async def _deliver(svc: AgentService, agent_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        captured.append(msg)

    svc._broadcast_message = capture  # type: ignore[method-assign]
    await svc._dispatch_subscriber_payload(agent_name, json.dumps(payload))
    return captured


_QUAR_PAYLOAD = {
    "type": "channel_quarantined",
    "role": "system",
    "level": "warning",
    "connid": "conn7",
    "peer": "('127.0.0.1', 50816)",
    "inbound_inflight": 10000,
    "threshold": 10000,
    "content": "A client RPyC channel was flooded and put into quarantine.",
}


@pytest.mark.asyncio
async def test_quarantine_forwarded_as_warning_event() -> None:
    svc = _make_service_with_agent("alpha")
    events = await _deliver(svc, "alpha", _QUAR_PAYLOAD)

    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "channel_quarantined", (
        "must forward as a dedicated channel_quarantined event, not "
        "agent_status / agent_message"
    )
    assert ev["level"] == "warning"
    assert ev["agent_name"] == "alpha"
    assert ev["connid"] == "conn7"
    assert ev["peer"] == "('127.0.0.1', 50816)"


@pytest.mark.asyncio
async def test_quarantine_does_not_change_agent_status() -> None:
    svc = _make_service_with_agent("alpha")
    assert svc._agents["alpha"].status == AgentStatus.RUNNING

    events = await _deliver(svc, "alpha", _QUAR_PAYLOAD)

    # The agent's mirrored status must be untouched — quarantine is a warning.
    assert svc._agents["alpha"].status == AgentStatus.RUNNING, (
        "channel_quarantined must NOT change the agent's status (incident "
        "observed 2026-05-22)"
    )
    # And no agent_status event was emitted.
    assert all(e["type"] != "agent_status" for e in events)


@pytest.mark.asyncio
async def test_quarantine_not_wrapped_as_agent_message() -> None:
    """It must take the dedicated branch, not fall through to the generic
    agent_message wrapper (which the frontend would route as a chat msg)."""
    svc = _make_service_with_agent("alpha")
    events = await _deliver(svc, "alpha", _QUAR_PAYLOAD)
    assert all(e["type"] != "agent_message" for e in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
