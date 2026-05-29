"""TDD for Bug 1: the registry reconcile must notice an EXISTING agent's
restart and trigger reconnect (DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md).

The event-driven watcher already calls ``_reconcile_from_registry`` on every
registry change. For a *new* agent it seeds ``_agents`` and connects. But for an
ALREADY-KNOWN agent that restarted (same name, new pid/port), the reconcile
must refresh the in-memory identity to the new pid/port so the identity-aware
``is_connected`` sees the change, evicts the stale connection, and the reconcile
reconnects to the new port. This is what keeps "visible ⇒ subscribed" true
after a restart without any user action.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    registry = MockProcessRegistry()
    registry.add_mock_agent("agent_a", pid=1000, rpyc_port=18861, working_dir="/tmp/vc")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    return AgentService(registry=mock_registry)


def _live_conn() -> MagicMock:
    conn = MagicMock()
    conn._closed = False
    return conn


async def test_reconcile_refreshes_identity_and_reconnects_on_restart(
    agent_service: AgentService, mock_registry: MockProcessRegistry
) -> None:
    await agent_service.initialize()

    # Established subscription bound to the ORIGINAL identity.
    agent_service._connections["agent_a"] = _live_conn()
    agent_service._roots["agent_a"] = MagicMock()
    agent_service._callback_services["agent_a"] = MagicMock()
    agent_service._record_conn_identity("agent_a", pid=1000, rpyc_port=18861)

    # Agent restarts: registry now reports a new pid + port.
    mock_registry.add_mock_agent(
        "agent_a", pid=2000, rpyc_port=18862, working_dir="/tmp/vc"
    )

    reconnected: list[str] = []

    def fake_supervise(coro: object, name: str = "") -> None:
        # Record the reconnect intent; close the coroutine to avoid a
        # "never awaited" warning.
        if name.startswith("reconnect_to_agent") or name.startswith("connect_to_agent"):
            reconnected.append(name)
        getattr(coro, "close", lambda: None)()

    with patch("app.services.agent_service.supervise", side_effect=fake_supervise):
        await agent_service._reconcile_from_registry()

    # In-memory identity refreshed to the NEW port...
    assert agent_service._registry_identity("agent_a") == (2000, 18862), (
        "reconcile must refresh _agents[name] with the restarted agent's "
        "current pid/port so the restart is detectable"
    )
    # ...the stale connection evicted (is_connected saw the mismatch)...
    assert "agent_a" not in agent_service._connections
    # ...and a reconnect was scheduled.
    assert reconnected, "reconcile must schedule a reconnect after a restart"


async def test_reconcile_no_churn_when_identity_unchanged(
    agent_service: AgentService, mock_registry: MockProcessRegistry
) -> None:
    """Steady state (no restart): a live, identity-matching connection must
    NOT be evicted or reconnected — no duplicate subscriptions."""
    await agent_service.initialize()

    conn = _live_conn()
    agent_service._connections["agent_a"] = conn
    agent_service._roots["agent_a"] = MagicMock()
    agent_service._callback_services["agent_a"] = MagicMock()
    agent_service._record_conn_identity("agent_a", pid=1000, rpyc_port=18861)

    reconnected: list[str] = []

    def fake_supervise(coro: object, name: str = "") -> None:
        reconnected.append(name)
        getattr(coro, "close", lambda: None)()

    with patch("app.services.agent_service.supervise", side_effect=fake_supervise):
        await agent_service._reconcile_from_registry()

    assert agent_service._connections.get("agent_a") is conn
    assert reconnected == [], "no reconnect/churn when nothing changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
