"""TDD for Bug 1: web must re-subscribe when an agent restarts.

DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md. Invariant: if an agent is visible in
the UI it must be subscribed to its CURRENT process. After an ungraceful
restart the agent reappears in the registry with a new ``(pid, rpyc_port)`` but
the old RPyC connection's ``_closed`` flag may still be ``False`` — so the old
``is_connected`` fast-path lied "connected" and ``connect_to_agent`` never
reconnected to the new port. These tests pin the identity-aware behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
    """A connection rpyc still considers open (_closed is False)."""
    conn = MagicMock()
    conn._closed = False
    return conn


class TestIsConnectedIdentity:
    """is_connected must treat a process-identity change as 'not connected'."""

    async def test_restart_with_same_closed_flag_is_not_connected(
        self, agent_service: AgentService, mock_registry: MockProcessRegistry
    ) -> None:
        """Agent restarts (new pid+port) while the old conn still has
        _closed=False. is_connected MUST report False and evict the stale
        connection, so the next connect_to_agent reaches the new port."""
        await agent_service.initialize()

        # Simulate an established subscription against the ORIGINAL identity.
        conn = _live_conn()
        agent_service._connections["agent_a"] = conn
        agent_service._roots["agent_a"] = MagicMock()
        agent_service._callback_services["agent_a"] = MagicMock()
        agent_service._record_conn_identity("agent_a", pid=1000, rpyc_port=18861)

        # The agent restarts: registry now has a new pid + port.
        mock_registry.add_mock_agent(
            "agent_a", pid=2000, rpyc_port=18862, working_dir="/tmp/vc"
        )
        await agent_service._discover_agents()

        connected = await agent_service.is_connected("agent_a")

        assert connected is False, (
            "after a restart (changed pid/port) the old connection must NOT "
            "count as connected even if its _closed flag is still False"
        )
        assert "agent_a" not in agent_service._connections, (
            "the stale connection to the dead process must be evicted"
        )

    async def test_same_identity_stays_connected(
        self, agent_service: AgentService
    ) -> None:
        """No restart: identity matches and conn open → still connected."""
        await agent_service.initialize()

        conn = _live_conn()
        agent_service._connections["agent_a"] = conn
        agent_service._record_conn_identity("agent_a", pid=1000, rpyc_port=18861)

        assert await agent_service.is_connected("agent_a") is True
        assert agent_service._connections["agent_a"] is conn

    async def test_no_recorded_identity_falls_back_to_closed_flag(
        self, agent_service: AgentService
    ) -> None:
        """Backward-compat: a connection with no recorded identity keeps the
        old _closed-based semantics (integration tests rely on this)."""
        await agent_service.initialize()

        conn = _live_conn()
        agent_service._connections["agent_a"] = conn
        # No _record_conn_identity call.

        assert await agent_service.is_connected("agent_a") is True


class TestReconnectOnRestart:
    """connect_to_agent after a restart must open a NEW connection + resubscribe."""

    async def test_connect_reconnects_to_new_port_after_restart(
        self, agent_service: AgentService, mock_registry: MockProcessRegistry
    ) -> None:
        await agent_service.initialize()

        # Pre-existing (stale) connection from before the restart.
        stale = _live_conn()
        agent_service._connections["agent_a"] = stale
        agent_service._roots["agent_a"] = MagicMock()
        agent_service._callback_services["agent_a"] = MagicMock()
        agent_service._record_conn_identity("agent_a", pid=1000, rpyc_port=18861)

        # Restart → new pid/port.
        mock_registry.add_mock_agent(
            "agent_a", pid=2000, rpyc_port=18862, working_dir="/tmp/vc"
        )
        await agent_service._discover_agents()

        # New connection the reconnect should produce.
        new_root = AsyncMock()
        new_conn = _live_conn()
        new_conn.root = new_root

        connected_ports: list[int] = []

        async def fake_async_connect(host: str, port: int, **kw: object) -> MagicMock:
            connected_ports.append(port)
            return new_conn

        with patch(
            "app.services.agent_service.async_connect",
            side_effect=fake_async_connect,
        ), patch.object(
            agent_service, "_subscribe_to_agent", new=AsyncMock()
        ) as sub, patch.object(
            agent_service, "_fetch_agent_metadata", new=AsyncMock()
        ), patch.object(
            agent_service, "_update_agent_status", new=AsyncMock()
        ):
            ok = await agent_service.connect_to_agent("agent_a")

        assert ok is True
        # Reconnected to the NEW port, and resubscribed.
        assert connected_ports == [18862], (
            "connect_to_agent must dial the new rpyc_port after a restart, "
            f"got {connected_ports}"
        )
        sub.assert_awaited_once()
        assert agent_service._connections["agent_a"] is new_conn


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
