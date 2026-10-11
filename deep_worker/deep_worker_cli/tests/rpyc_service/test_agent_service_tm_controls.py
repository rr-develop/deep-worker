"""Tests for TaskManager control RPyC methods."""

import json
from datetime import datetime, timezone

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

    @pytest.mark.asyncio
    async def test_toggle_reaches_the_real_session_metadata(self, tmp_path):
        """The toggle must land on disk, against a REAL SessionManager.

        The sibling tests hand the service a bare ``Mock()`` agent, so the
        persistence call is absorbed by an auto-created attribute and proves
        nothing about the write. This one uses the production SessionManager so
        the assertion is about the file the next process will read.
        """
        from deep_worker_cli.session.session_manager import SessionManager

        data_dir = tmp_path / "agent"
        data_dir.mkdir()
        session_manager = SessionManager(
            data_dir=data_dir, launch_config={"task_manager": True}
        )

        service = AgentService()
        mock_agent = Mock()
        mock_tm = Mock()
        mock_tm.disabled = False
        mock_agent.task_manager = mock_tm
        mock_agent.session_manager = session_manager
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_set_task_manager_enabled(False))
        assert result["success"] is True
        assert result["persisted"] is True, (
            "the handler reported the switch as not durable"
        )

        on_disk = json.loads((data_dir / "session_metadata.json").read_text())
        assert on_disk["task_manager_disabled"] is True

        # And the enable direction clears it again, on disk.
        json.loads(await service.exposed_set_task_manager_enabled(True))
        on_disk = json.loads((data_dir / "session_metadata.json").read_text())
        assert on_disk["task_manager_disabled"] is False

    @pytest.mark.asyncio
    async def test_persist_failure_is_reported_not_swallowed(self):
        """A failed write must flip ``persisted`` to False, loudly.

        The in-memory toggle still applies -- the operator's intent is honoured
        for this process -- but they must be able to tell it will not survive a
        restart, rather than being told "success" unconditionally.
        """
        service = AgentService()
        mock_agent = Mock()
        mock_tm = Mock()
        mock_tm.disabled = False
        mock_agent.task_manager = mock_tm
        mock_agent.name = "probe"
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.set_task_manager_disabled = AsyncMock(
            side_effect=OSError("disk full")
        )
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_set_task_manager_enabled(False))
        assert result["success"] is True
        assert result["persisted"] is False
        mock_tm.disable.assert_called_once()


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
        # Real values for the fields exposed_get_status JSON-encodes: a bare
        # Mock for last_activity/message_count/tasks_completed is not
        # serializable, so json.dumps would raise and the method would return
        # success:False instead of the field under test.
        mock_agent.session_manager = Mock()
        mock_agent.session_manager.status = Mock()
        mock_agent.session_manager.status.value = "IDLE"
        mock_agent.session_manager.error_message = None
        mock_agent.session_manager.last_activity = datetime(
            2026, 10, 7, 12, 0, tzinfo=timezone.utc
        )
        mock_agent.session_manager.message_count = 7
        mock_agent.session_manager.tasks_completed = 3
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
