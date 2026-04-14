"""Tests for last-response functionality.

TDD tests for:
1. AgentService.get_agent_last_response() - fire-and-forget service method
2. GET /api/agents/{name}/last-response - REST endpoint
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.agents import router
from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


# ============================================================================
# Service layer tests
# ============================================================================


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    """Create a mock ProcessRegistry."""
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    """Create AgentService with mock registry."""
    return AgentService(registry=mock_registry)


class TestGetAgentLastResponse:
    """Tests for AgentService.get_agent_last_response()."""

    async def test_returns_request_id(self, agent_service: AgentService) -> None:
        """Test that method returns immediately with request_id."""
        await agent_service.initialize()

        # Mock connection
        agent_service._connections["test-agent"] = AsyncMock()
        agent_service._roots["test-agent"] = AsyncMock()

        result = await agent_service.get_agent_last_response("test-agent")

        assert result["success"] is True
        assert "request_id" in result

    async def test_returns_error_for_unknown_agent(
        self, agent_service: AgentService
    ) -> None:
        """Test error when agent not found."""
        await agent_service.initialize()

        result = await agent_service.get_agent_last_response("nonexistent")

        assert result["success"] is False
        assert "error" in result

    async def test_uses_provided_request_id(
        self, agent_service: AgentService
    ) -> None:
        """Test that provided request_id is used."""
        await agent_service.initialize()

        agent_service._connections["test-agent"] = AsyncMock()
        agent_service._roots["test-agent"] = AsyncMock()

        result = await agent_service.get_agent_last_response(
            "test-agent", request_id="my-req-123"
        )

        assert result["request_id"] == "my-req-123"


# ============================================================================
# Endpoint tests
# ============================================================================


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
    return AgentService(registry=registry)


@pytest.fixture
async def client(app: FastAPI, mock_agent_service: AgentService) -> AsyncClient:
    """Create async test client."""
    await mock_agent_service.initialize()
    app.state.agent_service = mock_agent_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await mock_agent_service.shutdown()


class TestGetLastResponseEndpoint:
    """Test GET /api/agents/{agent_name}/last-response endpoint."""

    async def test_agent_not_found(self, client: AsyncClient) -> None:
        """Test 404 for non-existent agent."""
        response = await client.get("/api/agents/nonexistent/last-response")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.skip(reason="Requires RPyC connection - integration test")
    async def test_success(self, client: AsyncClient) -> None:
        """Test successful last response retrieval."""
        pass
