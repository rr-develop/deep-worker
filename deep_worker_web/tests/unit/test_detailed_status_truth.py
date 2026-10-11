"""`detailed_status_response` must report what the agent actually said.

Regression for the 2026-09-26 report: `list_agents` showed an agent mid-turn
while `get_agent_status` said `status=idle`, `is_busy=false`,
`message_count=0` and a `last_activity` twelve days stale.

Root cause, in one sentence: `get_agent_detailed_status` read a
`session_info` sub-dict that the agent stopped sending when
`exposed_get_status` was slimmed to a minimal payload, so every field taken
from it silently fell back to its default — and `last_activity` was not read
at all, it was hardcoded to `started_at`.

These tests drive the real callback with the real (flat) response shape, so
they fail if anyone reintroduces the `session_info` read or the
`started_at` substitution.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry

#: What the agent's ``exposed_get_status`` really returns: flat keys, no
#: ``session_info`` anywhere. Mirrored from
#: ``deep_worker_cli/rpyc_service/agent_service.py::exposed_get_status``.
AGENT_LAST_ACTIVITY = "2026-09-26T13:15:46.738659"


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    return AgentService(registry=mock_registry)


async def _detailed_status(
    agent_service: AgentService, agent_name: str, agent_response: dict
) -> dict:
    """Run the real on_success callback and return the broadcast payload."""
    mock_root = MagicMock()
    agent_service._connections[agent_name] = MagicMock()
    agent_service._roots[agent_name] = mock_root

    broadcasts: list[dict] = []

    async def capture(message: dict) -> None:
        broadcasts.append(message)

    with (
        patch("rpyc_async.utils.helpers.fire_and_forget_call") as mock_faf,
        patch.object(agent_service, "_broadcast_message", capture),
    ):
        await agent_service.get_agent_detailed_status(agent_name)

        assert mock_faf.called, "detailed status never reached the agent"
        on_success = mock_faf.call_args.kwargs["success_callback"]
        await on_success(json.dumps(agent_response))

    assert broadcasts, "no detailed_status_response was broadcast"
    message = broadcasts[-1]
    assert message["type"] == "detailed_status_response", message
    assert message["success"] is True, message
    return message["status"]


class TestDetailedStatusReflectsTheAgent:
    async def test_working_agent_is_not_reported_idle(
        self, agent_service: AgentService
    ) -> None:
        """The exact reported symptom: mid-turn agent shown as idle.

        `working` maps to RUNNING, the same value `list_agents` derives from
        the same field. One status, one meaning.
        """
        await agent_service.initialize()

        status = await _detailed_status(
            agent_service,
            "test-agent",
            {
                "success": True,
                "status": "working",
                "error": None,
                "last_activity": AGENT_LAST_ACTIVITY,
                "message_count": 7,
                "tasks_completed": 2,
            },
        )

        assert status["status"] == AgentStatus.RUNNING.value, (
            "a working agent was reported as something else — the caller "
            "cannot tell busy from idle"
        )

    async def test_last_activity_is_the_agents_value_not_started_at(
        self, agent_service: AgentService
    ) -> None:
        """`last_activity` must be the turn stamp, never the process start."""
        await agent_service.initialize()

        agent = await agent_service.get_agent("test-agent")
        started_at = agent.started_at.isoformat()

        status = await _detailed_status(
            agent_service,
            "test-agent",
            {
                "success": True,
                "status": "working",
                "error": None,
                "last_activity": AGENT_LAST_ACTIVITY,
                "message_count": 7,
                "tasks_completed": 2,
            },
        )

        assert status["last_activity"] == AGENT_LAST_ACTIVITY
        assert status["last_activity"] != started_at, (
            "last_activity fell back to started_at — this is the original bug"
        )

    async def test_message_count_comes_from_the_agent(
        self, agent_service: AgentService
    ) -> None:
        """A non-zero count must survive the trip; 0 meant 'never read'."""
        await agent_service.initialize()

        status = await _detailed_status(
            agent_service,
            "test-agent",
            {
                "success": True,
                "status": "idle",
                "error": None,
                "last_activity": AGENT_LAST_ACTIVITY,
                "message_count": 7,
                "tasks_completed": 2,
            },
        )

        assert status["message_count"] == 7

    async def test_idle_agent_still_reports_idle(
        self, agent_service: AgentService
    ) -> None:
        """The fix must not invert the bug: idle has to stay idle."""
        await agent_service.initialize()

        status = await _detailed_status(
            agent_service,
            "test-agent",
            {
                "success": True,
                "status": "idle",
                "error": None,
                "last_activity": AGENT_LAST_ACTIVITY,
                "message_count": 7,
                "tasks_completed": 2,
            },
        )

        assert status["status"] == AgentStatus.IDLE.value
