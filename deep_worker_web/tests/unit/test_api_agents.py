"""Tests for agents API router."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.routers.agents import router
from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_agent_service() -> AgentService:
    """Create mock AgentService."""
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    service = AgentService(registry=registry)
    return service


@pytest.fixture
async def client(app: FastAPI, mock_agent_service: AgentService) -> AsyncClient:
    """Create async test client."""
    from httpx import ASGITransport
    
    # Initialize the service
    await mock_agent_service.initialize()
    
    # Set the service as app state (simulating what main.py does)
    app.state.agent_service = mock_agent_service
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    
    # Cleanup
    await mock_agent_service.shutdown()


class TestListAgents:
    """Test GET /api/agents endpoint."""

    async def test_list_agents_success(self, client: AsyncClient) -> None:
        """Test listing all agents."""
        response = await client.get("/api/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test-agent"
        assert data[0]["pid"] == 12345

    async def test_list_agents_empty(self, app: FastAPI) -> None:
        """Test listing agents when none exist."""
        from httpx import ASGITransport
        
        # Create service with empty registry
        empty_service = AgentService(registry=MockProcessRegistry())
        await empty_service.initialize()
        app.state.agent_service = empty_service
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/agents")
        
        await empty_service.shutdown()
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


class TestGetAgent:
    """Test GET /api/agents/{agent_name} endpoint."""

    async def test_get_agent_success(self, client: AsyncClient) -> None:
        """Test getting a specific agent."""
        response = await client.get("/api/agents/test-agent")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-agent"
        assert data["pid"] == 12345
        assert data["rpyc_port"] == 18861

    async def test_get_agent_not_found(self, client: AsyncClient) -> None:
        """Test getting non-existent agent."""
        response = await client.get("/api/agents/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()


class TestSendMessage:
    """Test POST /api/agents/{agent_name}/message endpoint."""

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_send_message_success(self, client: AsyncClient) -> None:
        """Test sending message to agent."""
        # This requires real RPyC connection
        pass

    async def test_send_message_invalid_json(self, client: AsyncClient) -> None:
        """Test sending invalid JSON."""
        response = await client.post(
            "/api/agents/test-agent/message",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        
        # Should return 422 for invalid JSON
        assert response.status_code == 422

    async def test_send_message_agent_not_found(self, client: AsyncClient) -> None:
        """Test sending message to non-existent agent."""
        response = await client.post(
            "/api/agents/nonexistent/message",
            json={"message": "Hello"},
        )
        
        assert response.status_code == 404


class TestGetHistory:
    """Test GET /api/agents/{agent_name}/history endpoint."""

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_get_history_success(self, client: AsyncClient) -> None:
        """Test getting conversation history."""
        # This requires real RPyC connection
        pass

    async def test_get_history_agent_not_found(self, client: AsyncClient) -> None:
        """Test getting history for non-existent agent."""
        response = await client.get("/api/agents/nonexistent/history")
        
        assert response.status_code == 404


class TestStopTaskManager:
    """Test POST /api/agents/{agent_name}/stop_task_manager endpoint."""

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_stop_task_manager_success(self, client: AsyncClient) -> None:
        """Test stopping TaskManager."""
        # This requires real RPyC connection
        pass

    async def test_stop_task_manager_agent_not_found(
        self, client: AsyncClient
    ) -> None:
        """Test stopping TaskManager for non-existent agent."""
        response = await client.post(
            "/api/agents/nonexistent/stop_task_manager",
            json={"force": False},
        )
        
        assert response.status_code == 404


class TestInterruptAgent:
    """Test POST /api/agents/{agent_name}/interrupt endpoint."""

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_interrupt_agent_success(self, client: AsyncClient) -> None:
        """Test interrupting a running agent."""
        # This requires real RPyC connection
        pass

    async def test_interrupt_agent_not_found(self, client: AsyncClient) -> None:
        """Test interrupting non-existent agent returns 404."""
        response = await client.post("/api/agents/nonexistent/interrupt")

        assert response.status_code == 404

    async def test_interrupt_agent_endpoint_exists(self, client: AsyncClient) -> None:
        """Endpoint must exist (POST method) for the existing agent.

        Even if RPyC is unavailable, the endpoint should respond
        (with a fire-and-forget contract — returns 200 with success=True)
        rather than 404 / 405.
        """
        response = await client.post("/api/agents/test-agent/interrupt")

        # Must NOT be 404 (not found) nor 405 (method not allowed)
        assert response.status_code != 404, (
            "Endpoint /interrupt must exist for known agent"
        )
        assert response.status_code != 405, (
            "Endpoint /interrupt must accept POST method"
        )


class TestDialogTextEndpoint:
    """Test GET /api/agents/{agent_name}/dialog-text endpoint contract.

    Behavioural tests live in ``test_dialog_formatting.py``;
    these cover only the HTTP envelope (status codes, content-type,
    verbosity validation) using a stub of the formatter.
    """

    async def test_endpoint_404_for_unknown_agent(
        self, client: AsyncClient
    ) -> None:
        """Unknown agent → 404."""
        response = await client.get("/api/agents/nonexistent/dialog-text")
        assert response.status_code == 404

    async def test_endpoint_rejects_unknown_verbosity(
        self, client: AsyncClient
    ) -> None:
        """Unknown verbosity → 422."""
        response = await client.get(
            "/api/agents/test-agent/dialog-text?verbosity=zzz"
        )
        assert response.status_code == 422

    async def test_endpoint_default_is_final(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default (no verbosity query) must be 'final'."""
        captured: dict[str, Any] = {}

        async def fake_dialog(
            agent_name: str,
            verbosity: str = "final",
        ) -> str:
            captured["agent_name"] = agent_name
            captured["verbosity"] = verbosity
            return f"--- User ---\nx ({verbosity})"

        monkeypatch.setattr(
            mock_agent_service,
            "is_connected",
            AsyncMock(return_value=True),
        )
        monkeypatch.setattr(
            mock_agent_service,
            "get_agent_dialog_formatted",
            fake_dialog,
        )

        response = await client.get("/api/agents/test-agent/dialog-text")
        assert response.status_code == 200
        assert captured["verbosity"] == "final"

    async def test_endpoint_accepts_text_verbosity(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_dialog(agent_name: str, verbosity: str = "final") -> str:
            captured["verbosity"] = verbosity
            return ""

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", fake_dialog)

        response = await client.get(
            "/api/agents/test-agent/dialog-text?verbosity=text"
        )
        assert response.status_code == 200
        assert captured["verbosity"] == "text"

    async def test_endpoint_accepts_full_verbosity(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_dialog(agent_name: str, verbosity: str = "final") -> str:
            captured["verbosity"] = verbosity
            return ""

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", fake_dialog)

        response = await client.get(
            "/api/agents/test-agent/dialog-text?verbosity=full"
        )
        assert response.status_code == 200
        assert captured["verbosity"] == "full"

    async def test_endpoint_returns_plain_text_content_type(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_dialog(agent_name: str, verbosity: str = "final") -> str:
            return "anything"

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", fake_dialog)

        response = await client.get("/api/agents/test-agent/dialog-text")
        ct = response.headers["content-type"].lower()
        assert "text/plain" in ct
        assert "charset=utf-8" in ct


class TestGetStatus:
    """Test GET /api/agents/{agent_name}/status endpoint."""

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_get_status_success(self, client: AsyncClient) -> None:
        """Test getting detailed agent status."""
        # This requires real RPyC connection
        pass

    async def test_get_status_agent_not_found(self, client: AsyncClient) -> None:
        """Test getting status for non-existent agent."""
        response = await client.get("/api/agents/nonexistent/status")
        
        assert response.status_code == 404
