"""Tests for session adapter - converting AgentProcess to SessionInfo."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.basesession.protocol_types import SessionStatus
from deep_worker_cli.basesession.session_adapter import (
    get_session_info,
    map_agent_status,
)
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class TestMapAgentStatus:
    """Tests for mapping AgentStatus to SessionStatus."""

    def test_idle_to_ready(self):
        """Test mapping IDLE to READY."""
        assert map_agent_status(AgentStatus.IDLE) == SessionStatus.READY

    def test_running_to_working(self):
        """Test mapping RUNNING to WORKING."""
        assert map_agent_status(AgentStatus.RUNNING) == SessionStatus.WORKING

    def test_waiting_input_to_waiting_input(self):
        """Test mapping WAITING_INPUT to WAITING_INPUT."""
        assert (
            map_agent_status(AgentStatus.WAITING_INPUT)
            == SessionStatus.WAITING_INPUT
        )

    def test_error_to_error(self):
        """Test mapping ERROR to ERROR."""
        assert map_agent_status(AgentStatus.ERROR) == SessionStatus.ERROR

    def test_shutdown_to_unknown(self):
        """Test mapping SHUTDOWN to UNKNOWN."""
        assert map_agent_status(AgentStatus.SHUTDOWN) == SessionStatus.UNKNOWN


class TestGetSessionInfo:
    """Tests for converting AgentProcess to SessionInfo."""

    @pytest.fixture
    def mock_agent_process(self, tmp_path: Path):
        """Create a mock agent process for testing."""
        config = AgentConfig(
            model="claude-sonnet-4-5-20250929",
            working_dir=tmp_path,
        )

        # Create agent process
        agent = AgentProcess(
            name="test-agent",
            config=config,
            pid_file=tmp_path / "test.pid",
            data_dir=tmp_path / "data",
        )

        # Initialize session manager
        agent.session_manager = SessionManager(data_dir=tmp_path / "data")

        # Set some test values
        agent.session_manager.status = AgentStatus.IDLE
        agent.session_manager.tasks_completed = 5
        agent.session_manager.message_count = 42

        # Mock started_at (will be added to AgentProcess)
        agent.started_at = datetime(2025, 1, 1, 12, 0, 0)

        return agent

    def test_get_session_info_basic(self, mock_agent_process):
        """Test getting basic session info."""
        session_info = get_session_info(mock_agent_process)

        assert session_info.name == "test-agent"
        assert session_info.session_type == "langchain"
        assert session_info.status == SessionStatus.READY
        assert str(mock_agent_process.config.working_dir) in session_info.working_dir
        assert session_info.created_at == datetime(2025, 1, 1, 12, 0, 0)
        assert session_info.tasks_completed == 5

    def test_get_session_info_with_data(self, mock_agent_process):
        """Test session info includes data field."""
        session_info = get_session_info(mock_agent_process)

        assert session_info.data is not None
        assert session_info.data.message_count == 42
        assert session_info.data.process_id == os.getpid()

    def test_get_session_info_with_metadata(self, mock_agent_process):
        """Test session info includes metadata."""
        session_info = get_session_info(mock_agent_process)

        assert session_info.metadata is not None
        assert session_info.metadata.environment is not None
        assert session_info.metadata.properties["agent_type"] == "deep_worker"
        assert (
            session_info.metadata.properties["model"]
            == "claude-sonnet-4-5-20250929"
        )

    def test_get_session_info_with_error(self, mock_agent_process):
        """Test session info with error status."""
        mock_agent_process.session_manager.status = AgentStatus.ERROR
        mock_agent_process.session_manager.error_message = "Test error message"

        session_info = get_session_info(mock_agent_process)

        assert session_info.status == SessionStatus.ERROR
        assert session_info.error_message == "Test error message"

    def test_get_session_info_running_status(self, mock_agent_process):
        """Test session info with RUNNING status."""
        mock_agent_process.session_manager.status = AgentStatus.RUNNING

        session_info = get_session_info(mock_agent_process)

        assert session_info.status == SessionStatus.WORKING

    def test_get_session_info_last_activity(self, mock_agent_process):
        """Test session info includes last_activity."""
        session_info = get_session_info(mock_agent_process)

        assert session_info.last_activity is not None
        assert isinstance(session_info.last_activity, datetime)
