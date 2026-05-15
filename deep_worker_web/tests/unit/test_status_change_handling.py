"""Web-side contract for the new agent-push event ``status_change``.

When the agent sends ``{"type": "status_change", "status": ...,
"task_manager_active": ..., "task_manager_disabled": ...}``
(serialised to JSON as per the subscribe-channel contract),
the web layer must:

1. Convert it into the existing frontend-facing
   ``{"type": "agent_status", ...}`` broadcast so no frontend
   change is needed on the plumbing side.
2. Update the in-memory ``AgentInfo`` mirror so a subsequent
   ``GET /api/agents`` is immediately in sync.
3. Refrain from inventing TM fields when the payload didn't carry
   them (honest-unknown preserved).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService


def _make_service_with_agent(name: str = "alpha") -> AgentService:
    svc = AgentService(registry=object())  # registry unused in these tests
    svc._agents[name] = AgentInfo(
        name=name,
        pid=4242,
        status=AgentStatus.IDLE,
        provider="claude-code",
        model="opus",
        rpyc_port=12345,
        working_dir="/tmp/test",
        started_at=datetime.now(),
    )
    return svc


async def _deliver(svc: AgentService, agent_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Deliver a subscribe-channel payload and return what was broadcast.

    Uses the same code path a real subscribe callback would take.
    """
    captured: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        captured.append(msg)

    svc._broadcast_message = capture  # type: ignore[method-assign]

    # Call the underlying dispatcher — public API for status_change lives
    # on ``AgentService`` so it can be reused outside the RPyC service
    # wrapper. This is the seam tests hit.
    await svc._dispatch_subscriber_payload(agent_name, json.dumps(payload))
    return captured


@pytest.mark.asyncio
async def test_status_change_translates_to_agent_status() -> None:
    svc = _make_service_with_agent("alpha")

    events = await _deliver(
        svc,
        "alpha",
        {
            "type": "status_change",
            "status": "running",
            "task_manager_active": True,
            "task_manager_disabled": False,
        },
    )

    assert any(
        e.get("type") == "agent_status"
        and e.get("agent_name") == "alpha"
        and e.get("status") == "running"
        and e.get("task_manager_active") is True
        and e.get("task_manager_disabled") is False
        for e in events
    ), f"expected agent_status broadcast; got {events}"


@pytest.mark.asyncio
async def test_status_change_updates_in_memory_agent_info() -> None:
    svc = _make_service_with_agent("alpha")
    assert svc._agents["alpha"].status == AgentStatus.IDLE

    await _deliver(
        svc,
        "alpha",
        {
            "type": "status_change",
            "status": "running",
            "task_manager_active": True,
            "task_manager_disabled": False,
        },
    )

    # AgentInfo mirror must reflect the new state so that
    # GET /api/agents returns the truth immediately.
    info = svc._agents["alpha"]
    # Web enum: "running" maps to RUNNING.
    assert info.status == AgentStatus.RUNNING, info.status
    # TM flags exposed on the schema.
    assert getattr(info, "task_manager_active", None) is True
    assert getattr(info, "task_manager_disabled", None) is False
    assert getattr(info, "has_task_manager", None) is True


@pytest.mark.asyncio
async def test_status_change_without_tm_preserves_unknown() -> None:
    """Payload w/o TM fields must NOT flip has_task_manager to False.
    ``None`` (unknown) stays ``None``."""
    svc = _make_service_with_agent("alpha")

    await _deliver(
        svc,
        "alpha",
        {"type": "status_change", "status": "running"},  # no TM
    )

    info = svc._agents["alpha"]
    # Not fabricated.
    assert getattr(info, "task_manager_active", None) is None
    assert getattr(info, "task_manager_disabled", None) is None
    assert getattr(info, "has_task_manager", None) is None


@pytest.mark.asyncio
async def test_unknown_event_type_is_passed_through_as_agent_message() -> None:
    """Existing behaviour: random message types still wrap to
    ``agent_message``. We only specialise ``status_change``."""
    svc = _make_service_with_agent("alpha")

    events = await _deliver(
        svc, "alpha", {"type": "assistant", "content": "hi"}
    )
    # Must be wrapped, not dropped.
    assert any(e.get("type") == "agent_message" for e in events)
