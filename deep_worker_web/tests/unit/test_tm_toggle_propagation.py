"""Regression: disabling TaskManager from the UI must STICK.

User report (2026-05-20): uncheck the TM box for an agent, switch to
another agent, switch back — the box is checked again. Disabling TM
must be permanent until the user re-enables it.

ROOT CAUSE (web side): ``AgentService.set_task_manager_enabled`` fired
the RPyC toggle to the agent (which DID flip
``task_manager.disabled``), but it never updated its own
``self._agents[name].task_manager_disabled`` cache and never
broadcast an ``agent_status`` event carrying the new value. So the
new state lived ONLY inside the agent process; the web cache and
every connected browser kept the stale ``disabled=False`` until some
later ``_update_agent_status`` poll happened to refresh it (which is
why old agents "didn't self-restore" reliably).

Contract under test:
  * ``set_task_manager_enabled`` updates the in-memory ``AgentInfo``
    mirror IMMEDIATELY (so ``GET /api/agents`` and a re-select read
    the truth), and
  * broadcasts an ``agent_status`` event with ``task_manager_disabled``
    so every connected client updates without waiting for a poll.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService


def _make_service(name: str = "tm_agent") -> AgentService:
    svc = AgentService(registry=object())
    svc._agents[name] = AgentInfo(
        name=name,
        pid=4242,
        status=AgentStatus.IDLE,
        provider="codex",
        model="gpt-5.5",
        rpyc_port=12345,
        working_dir="/tmp/test",
        started_at=datetime.now(),
        has_task_manager=True,
        task_manager_active=False,
        task_manager_disabled=False,
    )
    return svc


class _FakeRoot:
    """Root stub whose ``set_task_manager_enabled`` records the call and
    returns the agent's JSON result, mirroring the real
    ``exposed_set_task_manager_enabled``."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def set_task_manager_enabled(self, enabled: bool) -> str:
        import json
        self.calls.append(enabled)
        return json.dumps({"success": True, "enabled": enabled})


async def _toggle_and_capture(
    svc: AgentService, name: str, enabled: bool
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        captured.append(msg)

    svc._broadcast_message = capture  # type: ignore[method-assign]
    # Pretend we're connected with a fake root.
    root = _FakeRoot()
    svc._roots[name] = root  # type: ignore[index]

    async def _is_connected(_n: str) -> bool:
        return True

    svc.is_connected = _is_connected  # type: ignore[method-assign]

    await svc.set_task_manager_enabled(name, enabled)
    # The RPyC call is fire-and-forget; give the dispatched task room.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if captured:
            break
    return captured


@pytest.mark.asyncio
async def test_disable_updates_cache_immediately() -> None:
    """Disabling TM sets ``task_manager_disabled=True`` in the AgentInfo
    mirror right away — NOT after the next poll."""
    svc = _make_service("tm_agent")
    assert svc._agents["tm_agent"].task_manager_disabled is False

    await _toggle_and_capture(svc, "tm_agent", enabled=False)

    assert svc._agents["tm_agent"].task_manager_disabled is True, (
        "set_task_manager_enabled(False) must immediately mark the agent "
        "disabled in the cache so a re-select / GET /api/agents reads the "
        "truth (the UI uncheck must STICK)"
    )


@pytest.mark.asyncio
async def test_disable_broadcasts_agent_status_with_disabled_true() -> None:
    """Disabling TM broadcasts agent_status carrying
    ``task_manager_disabled=True`` so every connected client updates
    live without waiting for a poll."""
    svc = _make_service("tm_agent")

    events = await _toggle_and_capture(svc, "tm_agent", enabled=False)

    status_events = [
        e for e in events
        if e.get("type") == "agent_status" and e.get("agent_name") == "tm_agent"
    ]
    assert status_events, f"expected agent_status broadcast; got {events}"
    assert status_events[0].get("task_manager_disabled") is True


@pytest.mark.asyncio
async def test_enable_updates_cache_and_broadcasts_disabled_false() -> None:
    """Re-enabling flips it back: cache disabled=False + broadcast."""
    svc = _make_service("tm_agent")
    svc._agents["tm_agent"].task_manager_disabled = True  # start disabled

    events = await _toggle_and_capture(svc, "tm_agent", enabled=True)

    assert svc._agents["tm_agent"].task_manager_disabled is False
    status_events = [
        e for e in events
        if e.get("type") == "agent_status" and e.get("agent_name") == "tm_agent"
    ]
    assert status_events
    assert status_events[0].get("task_manager_disabled") is False
