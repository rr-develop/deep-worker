"""Tests for TaskManager disable/enable functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from pathlib import Path

from deep_worker_cli.task_manager.task_manager import TaskManager


def _make_task_manager():
    """Create a TaskManager with mocked dependencies."""
    mock_config = Mock()
    mock_config.working_dir = Path("/tmp")
    mock_session_mgr = Mock()

    tm = TaskManager(
        agent_name="test-agent",
        config=mock_config,
        session_manager=mock_session_mgr,
    )
    tm._running = True
    tm.manager_agent = Mock()  # Pretend initialized
    return tm


class TestDisableEnable:
    """Tests for disable/enable toggle."""

    def test_disabled_flag_default_false(self):
        tm = _make_task_manager()
        assert tm.disabled is False

    def test_disable(self):
        tm = _make_task_manager()
        tm.disable()
        assert tm.disabled is True

    def test_enable(self):
        tm = _make_task_manager()
        tm.disable()
        tm.enable()
        assert tm.disabled is False

    @pytest.mark.asyncio
    async def test_init_task_skips_when_disabled(self):
        tm = _make_task_manager()
        tm.disable()
        await tm.init_task("do something")
        assert tm.state == "inactive"  # Not changed to 'waiting'

    @pytest.mark.asyncio
    async def test_on_subordinate_response_returns_none_when_disabled(self):
        tm = _make_task_manager()
        tm.state = "waiting"
        tm.initial_request = "do something"
        tm.disable()

        result = await tm.on_subordinate_response("response text", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_enable_restores_normal_behavior(self):
        tm = _make_task_manager()
        tm.disable()
        await tm.init_task("do something")
        assert tm.state == "inactive"

        tm.enable()
        await tm.init_task("do something")
        assert tm.state == "waiting"
