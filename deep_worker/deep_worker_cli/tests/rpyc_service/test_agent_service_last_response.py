"""Tests for AgentService.exposed_get_last_response() RPyC method.

TDD tests for the new RPyC method that returns the last assistant response.
"""

import json

import pytest
from unittest.mock import AsyncMock, Mock

from deep_worker_cli.rpyc_service.agent_service import AgentService


class TestExposedGetLastResponse:
    """Tests for exposed_get_last_response() method."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Test successful last response retrieval."""
        service = AgentService()
        mock_agent_process = AsyncMock()
        mock_agent_process._handle_get_last_response = AsyncMock(return_value={
            "success": True,
            "last_response": {
                "type": "AIMessage",
                "id": "a1",
                "content": "7",
            },
        })
        service._agent_process = mock_agent_process

        result_str = await service.exposed_get_last_response()

        # Must return JSON string
        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["last_response"]["content"] == "7"
        assert result["last_response"]["type"] == "AIMessage"

    @pytest.mark.asyncio
    async def test_no_agent_process(self):
        """Test when agent process is not initialized."""
        service = AgentService()
        service._agent_process = None

        result_str = await service.exposed_get_last_response()

        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_ai_messages(self):
        """Test when no AIMessages in history."""
        service = AgentService()
        mock_agent_process = AsyncMock()
        mock_agent_process._handle_get_last_response = AsyncMock(return_value={
            "success": True,
            "last_response": None,
        })
        service._agent_process = mock_agent_process

        result_str = await service.exposed_get_last_response()

        result = json.loads(result_str)
        assert result["success"] is True
        assert result["last_response"] is None

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are caught and returned as error."""
        service = AgentService()
        mock_agent_process = AsyncMock()
        mock_agent_process._handle_get_last_response = AsyncMock(
            side_effect=RuntimeError("Connection lost")
        )
        service._agent_process = mock_agent_process

        result_str = await service.exposed_get_last_response()

        result = json.loads(result_str)
        assert result["success"] is False
        assert "Connection lost" in result["error"]
