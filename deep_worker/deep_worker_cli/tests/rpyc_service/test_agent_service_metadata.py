"""Tests for AgentService metadata and shutdown methods.

TDD tests for new RPyC methods needed for DwSession migration.
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock

from deep_worker_cli.rpyc_service.agent_service import AgentService


class TestAgentServiceMetadata:
    """Tests for exposed_get_metadata() method."""

    def test_get_metadata_success(self):
        """Test successful metadata retrieval."""
        # Setup
        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861

        mock_config = Mock()
        mock_config.model = "deepseek-chat"
        mock_config.working_dir = "/test/working/dir"
        mock_agent.config = mock_config

        service._agent_process = mock_agent

        # Execute
        result = service.exposed_get_metadata()

        # Verify
        assert result["success"] is True
        assert "metadata" in result

        metadata = result["metadata"]
        assert metadata["directory"] == "/test/working/dir"
        # pid will be the current process PID, so just check that it is present
        assert "pid" in metadata
        assert isinstance(metadata["pid"], int)
        assert metadata["started"] == "2024-01-15T10:30:00"
        assert metadata["rpyc_port"] == 18861
        assert metadata["agent_type"] == "deep-worker"
        assert metadata["model"] == "deepseek-chat"

    def test_get_metadata_no_agent(self):
        """Test metadata when agent not initialized."""
        service = AgentService()
        service._agent_process = None

        result = service.exposed_get_metadata()

        assert result["success"] is False
        assert "error" in result
        assert "not initialized" in result["error"].lower()

    def test_get_metadata_no_config(self):
        """Test metadata when agent has no config."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861
        mock_agent.config = None

        service._agent_process = mock_agent

        result = service.exposed_get_metadata()

        assert result["success"] is True
        assert result["metadata"]["model"] is None
        assert result["metadata"]["directory"] is None
        assert result["metadata"]["rpyc_port"] == 18861


class TestAgentServiceShutdown:
    """Tests for exposed_shutdown() method."""

    def test_shutdown_success(self):
        """Test successful shutdown initiation."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.running = True
        mock_rpyc_server = Mock()
        mock_rpyc_server.close = Mock()
        mock_agent.rpyc_server = mock_rpyc_server

        service._agent_process = mock_agent

        result = service.exposed_shutdown()

        assert result["success"] is True
        assert "message" in result
        assert "shutdown initiated" in result["message"].lower()

        # Verify agent.running was set to False
        assert mock_agent.running is False

    def test_shutdown_no_agent(self):
        """Test shutdown when agent not initialized."""
        service = AgentService()
        service._agent_process = None

        result = service.exposed_shutdown()

        assert result["success"] is False
        assert "error" in result
        assert "not initialized" in result["error"].lower()

    def test_shutdown_handles_exception(self):
        """Test shutdown handles exceptions gracefully."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.running = True

        # Make setting running raise exception
        def raise_error():
            raise RuntimeError("Test error")

        type(mock_agent).running = property(lambda self: True, lambda self, val: raise_error())
        service._agent_process = mock_agent

        result = service.exposed_shutdown()

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result
