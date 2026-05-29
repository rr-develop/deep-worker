"""Tests for AgentService metadata and shutdown methods.

TDD tests for new RPyC methods needed for DwSession migration.

The exposed_* methods in AgentService return JSON-serialized strings
(not dicts) — see the `IMPORTANT: Returns JSON-serialized string ...`
note in `agent_service.py`. Every assertion must first `json.loads`
the return value.
"""

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock

from deep_worker_cli.rpyc_service.agent_service import AgentService


class TestAgentServiceMetadata:
    """Tests for exposed_get_metadata() method."""

    @pytest.mark.asyncio
    async def test_get_metadata_success(self):
        """Test successful metadata retrieval."""
        # Setup
        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861

        mock_config = Mock()
        mock_config.model = "deepseek-chat"
        # `exposed_get_metadata` reads these too and must serialize
        # the result as JSON — so they need to be JSON-serializable
        # primitives, not Mock autospec stand-ins.
        mock_config.resolved_model = "deepseek-chat"
        mock_config.provider = "deepseek"
        mock_config.working_dir = "/test/working/dir"
        mock_agent.config = mock_config

        service._agent_process = mock_agent

        # Execute. `exposed_get_metadata` returns a JSON string
        # to avoid RPyC netref overhead (see its docstring).
        result = json.loads(await service.exposed_get_metadata())

        # Verify
        assert result["success"] is True
        assert "metadata" in result

        metadata = result["metadata"]
        assert metadata["directory"] == "/test/working/dir"
        # pid is the current process PID, so we just check that it is present
        assert "pid" in metadata
        assert isinstance(metadata["pid"], int)
        assert metadata["started"] == "2024-01-15T10:30:00"
        assert metadata["rpyc_port"] == 18861
        assert metadata["agent_type"] == "deep-worker"
        assert metadata["model"] == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_get_metadata_no_agent(self):
        """Test metadata when agent not initialized."""
        service = AgentService()
        service._agent_process = None

        result = json.loads(await service.exposed_get_metadata())

        assert result["success"] is False
        assert "error" in result
        assert "not initialized" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_metadata_no_config(self):
        """Test metadata when agent has no config."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861
        mock_agent.config = None

        service._agent_process = mock_agent

        result = json.loads(await service.exposed_get_metadata())

        assert result["success"] is True
        assert result["metadata"]["model"] is None
        assert result["metadata"]["directory"] is None
        assert result["metadata"]["rpyc_port"] == 18861


class TestAgentServiceMetadataProvider:
    """Tests for provider field in exposed_get_metadata()."""

    @pytest.mark.asyncio
    async def test_metadata_includes_provider(self):
        """Test that metadata includes provider from config."""
        import json

        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861

        mock_config = Mock()
        mock_config.model = "deepseek-chat"
        mock_config.provider = "deepseek"
        mock_config.resolved_model = "deepseek-chat"
        mock_config.working_dir = "/test/dir"
        mock_agent.config = mock_config

        service._agent_process = mock_agent

        result_str = await service.exposed_get_metadata()
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["metadata"]["provider"] == "deepseek"
        assert result["metadata"]["model"] == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_metadata_provider_none_when_no_config(self):
        """Test that provider is None when config is absent."""
        import json

        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861
        mock_agent.config = None

        service._agent_process = mock_agent

        result_str = await service.exposed_get_metadata()
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["metadata"]["provider"] is None

    @pytest.mark.asyncio
    async def test_metadata_returns_resolved_model_when_none(self):
        """When config.model is None, metadata should return the provider default."""
        import json

        service = AgentService()
        mock_agent = Mock()
        mock_agent.started_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_agent.rpyc_port = 18861

        mock_config = Mock()
        mock_config.model = None
        mock_config.provider = "deepseek"
        mock_config.working_dir = "/test/dir"
        # resolved_model property should return "deepseek-chat"
        mock_config.resolved_model = "deepseek-chat"
        mock_config.dynamic_display_model = None  # prevent Mock auto-stub
        mock_config.dynamic_display_model = None  # prevent Mock auto-stub
        mock_agent.config = mock_config

        service._agent_process = mock_agent

        result_str = await service.exposed_get_metadata()
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["metadata"]["model"] == "deepseek-chat"


class TestAgentServiceShutdown:
    """Tests for exposed_shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown initiation."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.running = True
        mock_rpyc_server = Mock()
        mock_rpyc_server.close = AsyncMock()
        mock_agent.rpyc_server = mock_rpyc_server

        service._agent_process = mock_agent

        result = json.loads(await service.exposed_shutdown())

        assert result["success"] is True
        assert "message" in result
        assert "shutdown initiated" in result["message"].lower()

        # Verify agent.running was set to False
        assert mock_agent.running is False

    @pytest.mark.asyncio
    async def test_shutdown_no_agent(self):
        """Test shutdown when agent not initialized."""
        service = AgentService()
        service._agent_process = None

        result = json.loads(await service.exposed_shutdown())

        assert result["success"] is False
        assert "error" in result
        assert "not initialized" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_shutdown_handles_exception(self):
        """Test shutdown handles exceptions gracefully."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.running = True

        # Make setting running raise exception
        def raise_error():
            raise RuntimeError("Test error")

        type(mock_agent).running = property(lambda self: True, lambda self, val: raise_error())
        service._agent_process = mock_agent

        result = json.loads(await service.exposed_shutdown())

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_shutdown_with_grace_and_force(self, monkeypatch):
        """exposed_shutdown now accepts grace_seconds and force (fire-and-forget
        lifecycle contract). force=True must schedule an immediate _exit; the
        default graceful path must not call the blocking time.sleep."""
        service = AgentService()
        mock_agent = Mock()
        mock_agent.running = True
        mock_rpyc_server = Mock()
        mock_rpyc_server.close = AsyncMock()
        mock_agent.rpyc_server = mock_rpyc_server
        service._agent_process = mock_agent

        # Capture os._exit so we don't actually exit the pytest process.
        called: dict[str, int] = {}

        def fake_exit(code: int) -> None:
            called["code"] = code

        import os as _os
        monkeypatch.setattr(_os, "_exit", fake_exit)

        # graceful path — should schedule (loop.call_later), not exit now
        result = json.loads(
            await service.exposed_shutdown(grace_seconds=0.05, force=False)
        )
        assert result["success"] is True
        assert result["force"] is False
        assert result["grace_seconds"] == 0.05
        # force=False must NOT have called _exit yet (only scheduled)
        assert "code" not in called
        # Wait slightly past grace to let loop.call_later fire.
        await asyncio.sleep(0.12)
        assert called.get("code") == 0

        # Reset and test force=True
        called.clear()
        mock_agent.running = True
        result = json.loads(await service.exposed_shutdown(force=True))
        assert result["success"] is True
        assert result["force"] is True
        await asyncio.sleep(0.2)
        assert called.get("code") == 0
