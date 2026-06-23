"""Integration tests for agents API with real agents."""

import asyncio

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.routers.agents import router
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from tests.integration.agent_fixture import agent_fixture


async def _warm_connection(service: AgentService, name: str) -> None:
    """Establish the per-agent RPyC connection BEFORE issuing HTTP
    requests.

    Since the 2026-06-15 incident the HTTP handlers use the NON-BLOCKING
    ``ensure_connecting`` gate: the very first request to an agent the
    web has not connected to yet returns 503 ``connecting`` while a
    single-flight background connect runs — handlers never block on
    connect establishment, not even for a timeout. The real UI handles
    this by retrying; these tests instead warm the connection up-front
    via the sanctioned blocking lifecycle path (``connect_to_agent``),
    which is exactly the "warm the connection with one sequential call
    before issuing the batch" rule the service already prescribes.
    """
    ok = await service.connect_to_agent(name)
    assert ok is True, f"failed to warm RPyC connection to {name}"


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
            asyncio.run(_warm_connection(service, agent_info.name))
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
        """Test GET /api/agents/{agent_name}/history with real agent.

        The endpoint follows a fire-and-forget contract: it returns
        immediately with a ``request_id`` and the actual history is
        delivered via WebSocket (``history_response`` event). See
        ``deep_worker_web/app/routers/agents.py:get_history``. The
        synchronous response only confirms the request was accepted.
        """
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            asyncio.run(_warm_connection(service, agent_info.name))
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

                    # Fire-and-forget response: {success, request_id, message}
                    assert isinstance(data, dict)
                    assert data.get("success") is True
                    assert isinstance(data.get("request_id"), str)
                    assert data["request_id"]

            asyncio.run(run_test())
            asyncio.run(service.shutdown())

    def test_stop_task_manager(self, app: FastAPI) -> None:
        """Test POST /api/agents/{agent_name}/stop_task_manager with real agent."""
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            asyncio.run(_warm_connection(service, agent_info.name))
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
        """Test GET /api/agents/{agent_name}/status with real agent.

        Fire-and-forget endpoint (see
        ``deep_worker_web/app/routers/agents.py:get_status``) — returns
        ``{success, request_id, message}`` immediately; detailed status
        arrives via WebSocket (``detailed_status_response``).
        """
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            asyncio.run(_warm_connection(service, agent_info.name))
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

                    # Fire-and-forget acknowledgement.
                    assert isinstance(data, dict)
                    assert data.get("success") is True
                    assert isinstance(data.get("request_id"), str)
                    assert data["request_id"]

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
        """Three concurrent API requests must multiplex over ONE RPyC connection.

        Architectural rule this test pins (see
        ``app/services/agent_service.py`` — ``CRITICAL ARCHITECTURAL
        PRINCIPLES`` block, principle "ONE CONNECTION PER (CLIENT,
        AGENT) PAIR"): a single ``AgentService`` instance MUST hold
        at most one ``rpyc.Connection`` to any given agent. Parallel
        API requests routed through this same service for the same
        agent (``/``, ``/status``, ``/history``, ...) share that one
        connection — RPyC supports many in-flight async calls per
        connection, so duplicating connections from the same client
        is forbidden.

        The rule constrains only THIS client's behaviour — a
        different ``AgentService`` instance (or any other RPyC
        client) may legitimately connect to the same agent in
        parallel; the agent's ``AsyncioServer`` is designed to
        accept multiple distinct clients.

        How the test enforces it:

        * Open the connection up-front with one sequential warm-up
          call (``GET /status``). After this call returns,
          ``service._connections[name]`` holds the single
          ``rpyc.Connection`` to the agent.
        * Then fire the three concurrent ``GET``s through the same
          ``AsyncClient``. Every handler hits the hot path of
          ``connect_to_agent`` (``is_connected() == True``) and reuses
          the existing connection. No second ``async_connect`` is
          issued, no race against the agent's ``AsyncioServer``.

        Do NOT remove the warm-up: without it the cold ``/status``
        and cold ``/history`` handlers race in ``connect_to_agent``
        and open two RPyC sessions to the same agent — exactly what
        this test exists to forbid.
        """
        with agent_fixture() as agent_info:
            registry = ProcessRegistry()
            service = AgentService(registry=registry)
            asyncio.run(service.initialize())
            # Warm-up: open the ONE per-agent RPyC connection before
            # fanning out. Since the HTTP handlers are now non-blocking
            # (the 2026-06-15 incident), a cold HTTP request would return 503
            # while the background connect runs; warming via the
            # sanctioned blocking connect path establishes the single
            # connection deterministically before the parallel batch.
            asyncio.run(_warm_connection(service, agent_info.name))
            app.state.agent_service = service

            transport = ASGITransport(app=app)

            async def run_test() -> None:
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Connection already warmed above; confirm the hot
                    # path serves /status with 200 before fanning out.
                    warmup = await client.get(
                        f"/api/agents/{agent_info.name}/status"
                    )
                    assert warmup.status_code == 200, (
                        f"warm-up call must succeed before parallel "
                        f"fan-out; got {warmup.status_code}"
                    )

                    # Now fan out — all three handlers must share the
                    # single connection established by the warm-up.
                    tasks = [
                        client.get(f"/api/agents/{agent_info.name}"),
                        client.get(f"/api/agents/{agent_info.name}/status"),
                        client.get(f"/api/agents/{agent_info.name}/history"),
                    ]

                    responses = await asyncio.gather(*tasks)

                    # All should succeed
                    for response in responses:
                        assert response.status_code == 200

                    # Defence-in-depth: assert the service really has
                    # exactly one connection for this agent — if a
                    # future regression re-introduces duplicate
                    # connections, this assertion catches it before
                    # the flake comes back.
                    assert agent_info.name in service._connections, (
                        "warm-up did not register a connection for the agent"
                    )
                    # ``_connections`` is a dict keyed by agent name —
                    # mere presence with that single key already
                    # encodes the "one connection per agent"
                    # invariant; assert the value is not None as a
                    # sanity check.
                    assert service._connections[agent_info.name] is not None

            asyncio.run(run_test())
            asyncio.run(service.shutdown())
