"""Tests for task_manager_active in exposed_get_status().

TDD: Verifies that:
1. When TaskManager is configured and running, status includes task_manager_active=True
2. When TaskManager is configured but idle, status includes task_manager_active=False
3. When TaskManager is NOT configured, status does NOT include task_manager_active
"""

import json

import pytest
from unittest.mock import AsyncMock, Mock

from deep_worker_cli.rpyc_service.agent_service import AgentService


class TestGetStatusTaskManagerActive:
    """Tests for task_manager_active field in exposed_get_status()."""

    @pytest.mark.asyncio
    async def test_task_manager_running(self):
        """When TaskManager is active, status includes task_manager_active=True."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "RUNNING"
        mock_agent.session_manager.error_message = None

        # TaskManager configured and active (evaluating a task)
        mock_agent.task_manager = Mock()
        mock_agent.task_manager.state = "evaluating"
        mock_agent.task_manager.disabled = False

        service._agent_process = mock_agent

        result_str = await service.exposed_get_status()
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["task_manager_active"] is True

    @pytest.mark.asyncio
    async def test_task_manager_idle(self):
        """When TaskManager is configured but not running, task_manager_active=False."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "IDLE"
        mock_agent.session_manager.error_message = None

        mock_agent.task_manager = Mock()
        mock_agent.task_manager.state = "inactive"
        mock_agent.task_manager.disabled = False

        service._agent_process = mock_agent

        result_str = await service.exposed_get_status()
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["task_manager_active"] is False

    @pytest.mark.asyncio
    async def test_no_task_manager(self):
        """When TaskManager is not configured, field should be absent."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "IDLE"
        mock_agent.session_manager.error_message = None

        mock_agent.task_manager = None

        service._agent_process = mock_agent

        result_str = await service.exposed_get_status()
        result = json.loads(result_str)

        assert result["success"] is True
        assert "task_manager_active" not in result
