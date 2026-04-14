"""Tests for agent stop/restart API endpoints."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.agents import router
from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_agent_service() -> AgentService:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    service = AgentService(registry=registry)
    return service


@pytest.fixture
async def client(app: FastAPI, mock_agent_service: AgentService) -> AsyncClient:
    await mock_agent_service.initialize()
    app.state.agent_service = mock_agent_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await mock_agent_service.shutdown()


def _write_metadata(name: str, status: str = "idle", launch_config: dict | None = None):
    """Write session_metadata.json for a test agent."""
    data_dir = Path("~/.deep_worker").expanduser() / name
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": status,
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": datetime.now().isoformat(),
    }
    if launch_config is not None:
        metadata["launch_config"] = launch_config
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata))


# ===========================================================================
# POST /api/agents/{name}/stop
# ===========================================================================

class TestStopAgent:
    """Tests for POST /api/agents/{name}/stop."""

    async def test_stop_nonexistent_agent_returns_404(self, client: AsyncClient):
        response = await client.post("/api/agents/nonexistent/stop")
        assert response.status_code == 404

    async def test_stop_idle_agent_returns_200(self, client: AsyncClient):
        with patch("app.routers.agents.stop_agent_process") as mock_stop:
            mock_stop.return_value = {"success": True, "message": "Agent 'test-agent' stopped"}
            response = await client.post("/api/agents/test-agent/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    async def test_stop_actively_running_returns_409(self, client: AsyncClient):
        with patch("app.routers.agents.stop_agent_process") as mock_stop:
            mock_stop.side_effect = RuntimeError("Agent is actively running")
            response = await client.post("/api/agents/test-agent/stop")
        assert response.status_code == 409


# ===========================================================================
# POST /api/agents/{name}/restart
# ===========================================================================

class TestRestartAgent:
    """Tests for POST /api/agents/{name}/restart."""

    async def test_restart_no_launch_config_returns_404(self, client: AsyncClient):
        with patch("app.routers.agents.load_launch_config", return_value=None):
            response = await client.post("/api/agents/test-agent/restart")
        assert response.status_code == 404

    async def test_restart_with_saved_config_returns_200(self, client: AsyncClient):
        saved = {
            "provider": "claude-code", "model": "opus",
            "working_dir": "/tmp/test_project", "skip_permissions": True,
            "task_manager": False, "memory_only": False,
            "use_sqlite": False, "clean": False,
        }
        with patch("app.routers.agents.load_launch_config", return_value=saved), \
             patch("app.routers.agents.stop_agent_process", return_value={"success": True}), \
             patch("app.routers.agents.spawn_agent_process", return_value={"success": True, "pid": 99999}):
            response = await client.post("/api/agents/test-agent/restart")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    async def test_restart_actively_running_returns_409(self, client: AsyncClient):
        saved = {"provider": "claude-code", "model": "opus", "working_dir": "/tmp/test_project"}
        with patch("app.routers.agents.load_launch_config", return_value=saved), \
             patch("app.routers.agents.stop_agent_process") as mock_stop:
            mock_stop.side_effect = RuntimeError("Agent is actively running")
            response = await client.post("/api/agents/test-agent/restart")
        assert response.status_code == 409

    async def test_restart_with_model_override(self, client: AsyncClient):
        saved = {
            "provider": "claude-code", "model": "opus",
            "working_dir": "/tmp/test_project", "skip_permissions": True,
            "task_manager": False, "memory_only": False,
            "use_sqlite": False, "clean": False,
        }
        with patch("app.routers.agents.load_launch_config", return_value=saved), \
             patch("app.routers.agents.stop_agent_process", return_value={"success": True}), \
             patch("app.routers.agents.spawn_agent_process") as mock_spawn:
            mock_spawn.return_value = {"success": True, "pid": 99999}
            response = await client.post(
                "/api/agents/test-agent/restart",
                json={"model": "sonnet"},
            )
        assert response.status_code == 200
        # Verify the override was passed
        call_args = mock_spawn.call_args
        assert call_args[1].get("model") == "sonnet" or call_args[0][1].get("model") == "sonnet"


# ===========================================================================
# POST /api/agents/batch
# ===========================================================================

class TestBatchOperations:
    """Tests for POST /api/agents/batch."""

    async def test_batch_stop_returns_results(self, client: AsyncClient):
        with patch("app.routers.agents.stop_agent_process") as mock_stop:
            mock_stop.return_value = {"success": True, "message": "Stopped"}
            response = await client.post(
                "/api/agents/batch",
                json={"action": "stop", "agent_names": ["test-agent"]},
            )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "test-agent" in data["results"]

    async def test_batch_invalid_action_returns_400(self, client: AsyncClient):
        response = await client.post(
            "/api/agents/batch",
            json={"action": "invalid", "agent_names": ["test-agent"]},
        )
        assert response.status_code == 400

    async def test_batch_empty_list_returns_400(self, client: AsyncClient):
        response = await client.post(
            "/api/agents/batch",
            json={"action": "stop", "agent_names": []},
        )
        assert response.status_code == 400


# ===========================================================================
# spawn_agent_process — PYTHONPATH and error handling
# ===========================================================================

class TestSpawnAgentProcess:
    """Tests for spawn_agent_process internals."""

    async def test_spawn_sets_pythonpath(self):
        """subprocess must have deep_worker_cli and deepagents on PYTHONPATH."""
        from unittest.mock import call
        config = {
            "provider": "claude-code", "model": "opus",
            "working_dir": "/tmp/test_project", "skip_permissions": True,
            "task_manager": False, "memory_only": False,
            "use_sqlite": False,
        }
        with patch("app.routers.agents.subprocess.Popen") as mock_popen, \
             patch("app.routers.agents.asyncio.sleep"):
            mock_popen.return_value = MagicMock(pid=99999)
            # Mock registry to simulate agent appearing
            with patch("app.routers.agents._wait_for_agent_in_registry", return_value=True):
                from app.routers.agents import spawn_agent_process
                await spawn_agent_process("test-agent", config)

            # Verify env was passed with correct PYTHONPATH
            call_kwargs = mock_popen.call_args[1]
            assert "env" in call_kwargs
            pythonpath = call_kwargs["env"].get("PYTHONPATH", "")
            assert "deep_worker_cli" in pythonpath
            assert "deepagents" in pythonpath
            assert "rpyc_async" in pythonpath

    async def test_spawn_returns_failure_if_agent_not_in_registry(self):
        """If agent doesn't appear in registry, return success=False."""
        config = {
            "provider": "claude-code", "model": "opus",
            "working_dir": "/tmp/test_project",
        }
        with patch("app.routers.agents.subprocess.Popen") as mock_popen, \
             patch("app.routers.agents._wait_for_agent_in_registry", return_value=False):
            mock_popen.return_value = MagicMock(pid=99999)
            from app.routers.agents import spawn_agent_process
            result = await spawn_agent_process("test-agent", config)

        assert result["success"] is False
        assert "failed to start" in result.get("error", "").lower() or "did not appear" in result.get("error", "").lower()
