"""Tests for DeepWorkerDetector - cco daemon integration."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.basesession.detector import DeepWorkerDetector
from deep_worker_cli.basesession.protocol_types import SessionStatus
from deep_worker_cli.types import AgentProcessInfo


class TestDeepWorkerDetector:
    """Tests for DeepWorkerDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return DeepWorkerDetector()

    @pytest.mark.asyncio
    async def test_detect_sessions_no_agents(self, detector):
        """Test detecting sessions when no agents are running."""
        with patch.object(
            detector.registry,
            "list_all",
            return_value=[],
        ):
            sessions = await detector.detect_sessions()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_detect_sessions_single_agent(self, detector, tmp_path):
        """Test detecting sessions with a single agent."""
        # Mock agent info
        agent_info = AgentProcessInfo(
            name="test-agent",
            pid=12345,
            socket_path=Path("/tmp/test.sock"),
            pid_file=Path("/tmp/test.pid"),
            working_dir="/tmp/work",
            started_at="2025-01-01T12:00:00",
        )

        # Mock registry to return agent
        with patch.object(
            detector.registry,
            "list_all",
            return_value=[agent_info],
        ):
            # Mock AgentClient
            mock_client = AsyncMock()
            mock_session_info = MagicMock()
            mock_session_info.name = "test-agent"
            mock_session_info.status = SessionStatus.READY
            mock_client.get_status.return_value = mock_session_info

            with patch(
                "deep_worker_cli.basesession.detector.AgentClient",
                return_value=mock_client,
            ):
                sessions = await detector.detect_sessions()

        assert len(sessions) == 1
        assert sessions[0].name == "test-agent"

    @pytest.mark.asyncio
    async def test_detect_sessions_multiple_agents(self, detector):
        """Test detecting sessions with multiple agents."""
        # Mock multiple agents
        agents = [
            AgentProcessInfo(
                name=f"agent-{i}",
                pid=12345 + i,
                socket_path=Path(f"/tmp/agent-{i}.sock"),
                pid_file=Path(f"/tmp/agent-{i}.pid"),
                working_dir="/tmp/work",
                started_at="2025-01-01T12:00:00",
            )
            for i in range(3)
        ]

        with patch.object(
            detector.registry,
            "list_all",
            return_value=agents,
        ):
            # Mock AgentClient for each agent
            mock_client = AsyncMock()
            mock_sessions = [
                MagicMock(name=f"agent-{i}", status=SessionStatus.READY)
                for i in range(3)
            ]

            async def get_status_side_effect():
                return mock_sessions.pop(0)

            mock_client.get_status = get_status_side_effect

            with patch(
                "deep_worker_cli.basesession.detector.AgentClient",
                return_value=mock_client,
            ):
                sessions = await detector.detect_sessions()

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_detect_sessions_with_failed_query(self, detector):
        """Test detecting sessions when one agent fails to query."""
        # Mock agents
        agents = [
            AgentProcessInfo(
                name="agent-1",
                pid=12345,
                socket_path=Path("/tmp/agent-1.sock"),
                pid_file=Path("/tmp/agent-1.pid"),
                working_dir="/tmp/work",
                started_at="2025-01-01T12:00:00",
            ),
            AgentProcessInfo(
                name="agent-2",
                pid=12346,
                socket_path=Path("/tmp/agent-2.sock"),
                pid_file=Path("/tmp/agent-2.pid"),
                working_dir="/tmp/work",
                started_at="2025-01-01T12:00:00",
            ),
        ]

        with patch.object(
            detector.registry,
            "list_all",
            return_value=agents,
        ):
            # Mock AgentClient - first succeeds, second fails
            call_count = {"count": 0}

            class MockClient:
                def __init__(self, agent_name, registry=None):
                    self.agent_name = agent_name
                    self.registry = registry

                async def connect(self):
                    if call_count["count"] == 1:
                        raise ConnectionError("Connection failed")
                    call_count["count"] += 1

                async def get_status(self):
                    mock = MagicMock()
                    mock.name = self.agent_name
                    mock.status = SessionStatus.READY
                    return mock

                async def disconnect(self):
                    pass

            with patch(
                "deep_worker_cli.basesession.detector.AgentClient",
                MockClient,
            ):
                sessions = await detector.detect_sessions()

        # Should return only the successful agent
        assert len(sessions) == 1
        assert sessions[0].name == "agent-1"

    @pytest.mark.asyncio
    async def test_detect_sessions_handles_exception(self, detector, caplog):
        """Test that detector handles exceptions gracefully."""
        agent_info = AgentProcessInfo(
            name="test-agent",
            pid=12345,
            socket_path=Path("/tmp/test.sock"),
            pid_file=Path("/tmp/test.pid"),
            working_dir="/tmp/work",
            started_at="2025-01-01T12:00:00",
        )

        with patch.object(
            detector.registry,
            "list_all",
            return_value=[agent_info],
        ):
            # Mock AgentClient that raises exception
            with patch(
                "deep_worker_cli.basesession.detector.AgentClient",
                side_effect=Exception("Test error"),
            ):
                sessions = await detector.detect_sessions()

        # Should return empty list and log warning
        assert sessions == []
