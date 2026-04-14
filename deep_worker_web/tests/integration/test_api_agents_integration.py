"""Integration tests for agents API with real agents."""

import asyncio

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.routers.agents import router
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from tests.integration.agent_fixture import agent_fixture


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.integration
class TestAgentsAPIWithRealAgent:
    """Test agents API endpoints with real agent."""

    def test_list_agents_with_real_agent(self, app: FastAPI) -> None:
        """Test GET /api/agents with real agent."""
        with agent_fixture() as agent_info:
            # Create service with real registry
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            # Make API request
            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/api/agents")

                    assert response.status_code == 200
                    data = response.json()
                    assert isinstance(data, list)

                    # Find our test agent
                    agent_names = [agent["name"] for agent in data]
                    assert agent_info.name in agent_names

                    # Verify agent data
                    test_agent = next(a for a in data if a["name"] == agent_info.name)
                    assert test_agent["pid"] == agent_info.pid
                    assert test_agent["rpyc_port"] == agent_info.rpyc_port

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_get_specific_agent(self, app: FastAPI) -> None:
        """Test GET /api/agents/{agent_name} with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(f"/api/agents/{agent_info.name}")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["name"] == agent_info.name
                    assert data["pid"] == agent_info.pid
                    assert data["rpyc_port"] == agent_info.rpyc_port

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_send_message_to_real_agent(self, app: FastAPI) -> None:
        """Test POST /api/agents/{agent_name}/message with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Send a message to the agent
                    response = await client.post(
                        f"/api/agents/{agent_info.name}/message",
                        json={"message": "Hello, agent!"},
                    )

                    # Should succeed (200 or 202)
                    assert response.status_code in [200, 202]

                    # Response should have expected structure
                    data = response.json()
                    assert "status" in data or "message" in data

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_get_history_from_real_agent(self, app: FastAPI) -> None:
        """Test GET /api/agents/{agent_name}/history with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Get conversation history
                    response = await client.get(
                        f"/api/agents/{agent_info.name}/history"
                    )

                    assert response.status_code == 200
                    data = response.json()

                    # Should return a list of messages (even if empty)
                    assert isinstance(data, list)

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_stop_task_manager(self, app: FastAPI) -> None:
        """Test POST /api/agents/{agent_name}/stop_task_manager with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Try to stop TaskManager
                    response = await client.post(
                        f"/api/agents/{agent_info.name}/stop_task_manager",
                        json={"force": False},
                    )

                    # Should succeed or return appropriate status
                    assert response.status_code in [200, 202, 404]

                    if response.status_code in [200, 202]:
                        data = response.json()
                        assert "status" in data or "message" in data

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_get_agent_status(self, app: FastAPI) -> None:
        """Test GET /api/agents/{agent_name}/status with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Get detailed status
                    response = await client.get(
                        f"/api/agents/{agent_info.name}/status"
                    )

                    assert response.status_code == 200
                    data = response.json()

                    # Should have status information
                    assert "status" in data
                    # Valid statuses: idle, running, waiting, error
                    assert data["status"] in ["idle", "running", "waiting", "error"]

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_agent_not_found_handling(self, app: FastAPI) -> None:
        """Test that API properly handles non-existent agents."""
        registry = ProcessRegistry()
        service = AgentService(registry=registry)
        asyncio.run(service.initialize())
        app.state.agent_service = service

        transport = ASGITransport(app=app)

        async def run_test() -> None:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Try to get non-existent agent
                response = await client.get("/api/agents/nonexistent_agent_xyz")
                assert response.status_code == 404

                # Try to send message to non-existent agent
                response = await client.post(
                    "/api/agents/nonexistent_agent_xyz/message",
                    json={"message": "Hello"},
                )
                assert response.status_code == 404

                # Try to get history of non-existent agent
                response = await client.get("/api/agents/nonexistent_agent_xyz/history")
                assert response.status_code == 404

        asyncio.run(run_test())
        asyncio.run(service.shutdown())

    def test_concurrent_requests_to_agent(self, app: FastAPI) -> None:
        """Test handling concurrent API requests to same agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Make multiple concurrent requests
                    tasks = [
                        client.get(f"/api/agents/{agent_info.name}"),
                        client.get(f"/api/agents/{agent_info.name}/status"),
                        client.get(f"/api/agents/{agent_info.name}/history"),
                    ]

                    responses = await asyncio.gather(*tasks)

                    # All should succeed
                    for response in responses:
                        assert response.status_code == 200

            asyncio.run(run_test())
            asyncio.run(service.shutdown())
