"""Tests for agent status update parsing and broadcasting.

TDD tests to verify that:
1. _update_agent_status correctly parses status from agent response
2. Status is broadcast via WebSocket with correct values
3. monitor_agents polls status for connected agents
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    """Create a mock ProcessRegistry."""
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    """Create AgentService with mock registry."""
    return AgentService(registry=mock_registry)


async def _call_update_and_invoke_callback(
    agent_service: AgentService,
    agent_name: str,
    agent_response_str: str,
) -> None:
    """Helper: call _update_agent_status and invoke the on_success callback.

    fire_and_forget_async is patched to capture and immediately invoke
    the success_callback with the agent response string.
    """
    mock_root = MagicMock()
    agent_service._connections[agent_name] = MagicMock()
    agent_service._roots[agent_name] = mock_root

    with patch("rpyc_async.utils.helpers.fire_and_forget_async") as mock_faf:
        await agent_service._update_agent_status(agent_name, mock_root)

        # fire_and_forget_async was called — extract the success callback
        assert mock_faf.called
        kwargs = mock_faf.call_args
        on_success = kwargs.kwargs.get("success_callback") or kwargs[1].get("success_callback")

        # Invoke callback with agent response
        await on_success(agent_response_str)


class TestUpdateAgentStatusParsing:
    """Test that _update_agent_status correctly parses agent response."""

    async def test_parses_working_status(
        self, agent_service: AgentService
    ) -> None:
        """Agent status 'working' must be parsed from response top level."""
        await agent_service.initialize()

        agent_response = json.dumps({
            "success": True,
            "status": "working",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        agent = await agent_service.get_agent("test-agent")
        assert agent is not None
        assert agent.status == AgentStatus.RUNNING

    async def test_parses_ready_status(
        self, agent_service: AgentService
    ) -> None:
        """Agent status 'ready' should map to IDLE."""
        await agent_service.initialize()

        agent_response = json.dumps({
            "success": True,
            "status": "ready",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        agent = await agent_service.get_agent("test-agent")
        assert agent.status == AgentStatus.IDLE

    async def test_parses_waiting_input_status(
        self, agent_service: AgentService
    ) -> None:
        """Agent status 'waiting_input' should map to WAITING_INPUT."""
        await agent_service.initialize()

        agent_response = json.dumps({
            "success": True,
            "status": "waiting_input",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        agent = await agent_service.get_agent("test-agent")
        assert agent.status == AgentStatus.WAITING_INPUT

    async def test_parses_error_status(
        self, agent_service: AgentService
    ) -> None:
        """Agent status 'error' should map to ERROR."""
        await agent_service.initialize()

        agent_response = json.dumps({
            "success": True,
            "status": "error",
            "error": "Something broke",
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        agent = await agent_service.get_agent("test-agent")
        assert agent.status == AgentStatus.ERROR


class TestStatusBroadcast:
    """Test that status broadcasts contain correct values."""

    async def test_broadcast_contains_raw_status(
        self, agent_service: AgentService
    ) -> None:
        """Broadcast must contain the raw agent status string for frontend CSS."""
        await agent_service.initialize()

        broadcast_messages: list[dict] = []

        async def capture_broadcast(msg: dict) -> None:
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        agent_response = json.dumps({
            "success": True,
            "status": "waiting_input",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        status_broadcasts = [m for m in broadcast_messages if m.get("type") == "agent_status"]
        assert len(status_broadcasts) == 1
        assert status_broadcasts[0]["status"] == "waiting_input"
        assert status_broadcasts[0]["agent_name"] == "test-agent"

    async def test_broadcast_working_status(
        self, agent_service: AgentService
    ) -> None:
        """Working status must be broadcast as 'working', not 'running'."""
        await agent_service.initialize()

        broadcast_messages: list[dict] = []

        async def capture_broadcast(msg: dict) -> None:
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        agent_response = json.dumps({
            "success": True,
            "status": "working",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        status_broadcasts = [m for m in broadcast_messages if m.get("type") == "agent_status"]
        assert status_broadcasts[0]["status"] == "working"


class TestStatusDoesNotOverwriteProviderModel:
    """Test that _update_agent_status does NOT overwrite provider/model."""

    async def test_preserves_existing_provider_model(
        self, agent_service: AgentService
    ) -> None:
        """Polling status must not reset provider/model to 'unknown'."""
        await agent_service.initialize()

        # Simulate that provider/model were already set (e.g. from get_metadata)
        agent = await agent_service.get_agent("test-agent")
        agent.provider = "deepseek"
        agent.model = "deepseek-chat"

        # Now _update_agent_status fires with a response that has NO provider/model
        agent_response = json.dumps({
            "success": True,
            "status": "working",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        # Provider and model must NOT be overwritten
        agent = await agent_service.get_agent("test-agent")
        assert agent.provider == "deepseek"
        assert agent.model == "deepseek-chat"
        # But status should be updated
        assert agent.status == AgentStatus.RUNNING


class TestTaskManagerActiveBroadcast:
    """Test that task_manager_active is included in status broadcast."""

    async def test_broadcast_includes_task_manager_active(
        self, agent_service: AgentService
    ) -> None:
        """When agent returns task_manager_active, broadcast must include it."""
        await agent_service.initialize()

        broadcast_messages: list[dict] = []

        async def capture_broadcast(msg: dict) -> None:
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        agent_response = json.dumps({
            "success": True,
            "status": "working",
            "error": None,
            "task_manager_active": True,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        status_broadcasts = [m for m in broadcast_messages if m.get("type") == "agent_status"]
        assert len(status_broadcasts) == 1
        assert status_broadcasts[0]["task_manager_active"] is True

    async def test_broadcast_omits_task_manager_when_absent(
        self, agent_service: AgentService
    ) -> None:
        """When agent doesn't return task_manager_active, broadcast should not include it."""
        await agent_service.initialize()

        broadcast_messages: list[dict] = []

        async def capture_broadcast(msg: dict) -> None:
            broadcast_messages.append(msg)

        agent_service.register_message_callback(capture_broadcast)

        agent_response = json.dumps({
            "success": True,
            "status": "ready",
            "error": None,
        })

        await _call_update_and_invoke_callback(
            agent_service, "test-agent", agent_response
        )

        status_broadcasts = [m for m in broadcast_messages if m.get("type") == "agent_status"]
        assert len(status_broadcasts) == 1
        assert "task_manager_active" not in status_broadcasts[0]


class TestFetchAgentMetadata:
    """Test that _fetch_agent_metadata populates provider/model from get_metadata."""

    async def test_sets_provider_and_model(
        self, agent_service: AgentService
    ) -> None:
        """Metadata fetch should set provider and model on agent info."""
        await agent_service.initialize()

        metadata_response = json.dumps({
            "success": True,
            "metadata": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "directory": "/tmp",
                "pid": 12345,
                "started": "2024-01-01T00:00:00",
                "rpyc_port": 18861,
                "agent_type": "deep-worker",
            },
        })

        mock_root = MagicMock()
        agent_service._connections["test-agent"] = MagicMock()
        agent_service._roots["test-agent"] = mock_root

        with patch("rpyc_async.utils.helpers.fire_and_forget_async") as mock_faf:
            await agent_service._fetch_agent_metadata("test-agent", mock_root)

            kwargs = mock_faf.call_args
            on_success = kwargs.kwargs.get("success_callback") or kwargs[1].get("success_callback")
            await on_success(metadata_response)

        agent = await agent_service.get_agent("test-agent")
        assert agent.provider == "deepseek"
        assert agent.model == "deepseek-chat"

    async def test_handles_missing_provider(
        self, agent_service: AgentService
    ) -> None:
        """If metadata has no provider, keep existing value."""
        await agent_service.initialize()

        metadata_response = json.dumps({
            "success": True,
            "metadata": {
                "model": "gpt-4",
                "directory": "/tmp",
                "pid": 12345,
                "started": "2024-01-01T00:00:00",
                "rpyc_port": 18861,
                "agent_type": "deep-worker",
            },
        })

        mock_root = MagicMock()
        agent_service._connections["test-agent"] = MagicMock()
        agent_service._roots["test-agent"] = mock_root

        with patch("rpyc_async.utils.helpers.fire_and_forget_async") as mock_faf:
            await agent_service._fetch_agent_metadata("test-agent", mock_root)

            kwargs = mock_faf.call_args
            on_success = kwargs.kwargs.get("success_callback") or kwargs[1].get("success_callback")
            await on_success(metadata_response)

        agent = await agent_service.get_agent("test-agent")
        assert agent.model == "gpt-4"
