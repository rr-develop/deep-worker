"""Regression: when an already-known agent restarts (registry entry
overwritten with a new ``started_at``), ``_reconcile_from_registry`` MUST:

  1. Update ``self._agents[name].started_at`` to the new value.
  2. Broadcast a ``agent_started_at_changed`` WS event carrying the new
     ISO-formatted ``started_at``.

This drives the web-UI's per-agent uptime display without any polling:
the ``started_at`` is pushed on change; the browser only re-renders
``now - started_at`` locally on a 1-second tick.

Design doc: ``docs/DESIGN_AGENT_UPTIME_UI.md``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    registry = MockProcessRegistry()
    registry.add_mock_agent(
        "vc_uptime", pid=1000, rpyc_port=18871, working_dir="/tmp/vc"
    )
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    return AgentService(registry=mock_registry)


async def test_reconcile_emits_started_at_changed_on_restart(
    agent_service: AgentService, mock_registry: MockProcessRegistry
) -> None:
    await agent_service.initialize()

    # Prime the in-memory _agents map by running the first reconcile
    # (the initial agent_discovered).
    with patch.object(agent_service, "_start_background_connect"):
        await agent_service._reconcile_from_registry()

    original_started_at = agent_service._agents["vc_uptime"].started_at
    assert original_started_at is not None

    # Simulate a restart: same name, new started_at (registry entry is
    # overwritten in-place by ProcessRegistry.register on restart).
    new_started_at = datetime.now() + timedelta(seconds=5)
    mock_registry._agents["vc_uptime"]["started_at"] = new_started_at.isoformat()

    broadcasts: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        broadcasts.append(msg)

    with (
        patch.object(agent_service, "_broadcast_message", side_effect=capture),
        patch.object(agent_service, "_start_background_connect"),
    ):
        await agent_service._reconcile_from_registry()

    # 1) in-memory _agents map MUST have the new started_at.
    assert agent_service._agents["vc_uptime"].started_at == new_started_at, (
        f"reconcile must update _agents[name].started_at after restart. "
        f"Got {agent_service._agents['vc_uptime'].started_at!r}, "
        f"expected {new_started_at!r}."
    )

    # 2) MUST have emitted an agent_started_at_changed broadcast with the
    #    new ISO-formatted started_at.
    changed = [b for b in broadcasts if b.get("type") == "agent_started_at_changed"]
    assert len(changed) == 1, (
        f"Expected exactly one agent_started_at_changed broadcast, "
        f"got {len(changed)}. All broadcasts: "
        f"{[b.get('type') for b in broadcasts]}"
    )
    assert changed[0]["agent_name"] == "vc_uptime"
    assert changed[0]["started_at"] == new_started_at.isoformat()


async def test_reconcile_does_not_emit_when_started_at_unchanged(
    agent_service: AgentService, mock_registry: MockProcessRegistry
) -> None:
    """No broadcast when nothing changed — avoids WS spam on every inotify
    tick when the registry file was rewritten for unrelated reasons."""
    await agent_service.initialize()

    with patch.object(agent_service, "_start_background_connect"):
        await agent_service._reconcile_from_registry()

    broadcasts: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        broadcasts.append(msg)

    # No changes to the registry — reconcile again.
    with (
        patch.object(agent_service, "_broadcast_message", side_effect=capture),
        patch.object(agent_service, "_start_background_connect"),
    ):
        await agent_service._reconcile_from_registry()

    changed = [b for b in broadcasts if b.get("type") == "agent_started_at_changed"]
    assert changed == [], (
        f"No started_at change → no broadcast expected. Got: {changed}"
    )
