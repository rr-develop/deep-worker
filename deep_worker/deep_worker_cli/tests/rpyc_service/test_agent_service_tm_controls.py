"""Tests for TaskManager control RPyC methods."""

import json
import pytest
from unittest.mock import AsyncMock, Mock

from deep_worker_cli.rpyc_service.agent_service import AgentService


class TestExposedSetTaskManagerEnabled:

    @pytest.mark.asyncio
    async def test_disable(self):
        service = AgentService()
        mock_agent = Mock()
        mock_tm = Mock()
        mock_tm.disabled = False
        mock_agent.task_manager = mock_tm
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_set_task_manager_enabled(False))
        assert result["success"] is True
        assert result["enabled"] is False
        mock_tm.disable.assert_called_once()

    @pytest.mark.asyncio
    async def test_enable(self):
        service = AgentService()
        mock_agent = Mock()
        mock_tm = Mock()
        mock_tm.disabled = True
        mock_agent.task_manager = mock_tm
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_set_task_manager_enabled(True))
        assert result["success"] is True
        assert result["enabled"] is True
        mock_tm.enable.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_task_manager(self):
        service = AgentService()
        mock_agent = Mock()
        mock_agent.task_manager = None
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_set_task_manager_enabled(False))
        assert result["success"] is False


class TestExposedResetTaskManager:

    @pytest.mark.asyncio
    async def test_reset_success(self):
        service = AgentService()
        mock_agent = Mock()
        mock_tm = Mock()
        mock_tm.reset = AsyncMock()
        mock_agent.task_manager = mock_tm
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_reset_task_manager())
        assert result["success"] is True
        mock_tm.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_no_task_manager(self):
        service = AgentService()
        mock_agent = Mock()
        mock_agent.task_manager = None
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_reset_task_manager())
        assert result["success"] is False


class TestGetStatusIncludesDisabled:

    @pytest.mark.asyncio
    async def test_includes_disabled_field(self):
        service = AgentService()
        mock_agent = Mock()
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "IDLE"
        mock_agent.session_manager.error_message = None
        mock_agent.task_manager = Mock()
        mock_agent.task_manager.state = "inactive"
        mock_agent.task_manager.disabled = True
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_get_status())
        assert result["task_manager_disabled"] is True

    @pytest.mark.asyncio
    async def test_no_disabled_field_without_tm(self):
        service = AgentService()
        mock_agent = Mock()
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "IDLE"
        mock_agent.session_manager.error_message = None
        mock_agent.task_manager = None
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_get_status())
        assert "task_manager_disabled" not in result
