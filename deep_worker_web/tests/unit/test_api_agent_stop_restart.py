"""Tests for fire-and-forget agent stop/restart/batch endpoints.

The contract changed in the ``DESIGN_FIRE_AND_FORGET_LIFECYCLE`` rewrite:
every lifecycle endpoint now returns 202 + ``request_id`` and the actual
work happens in a background task. These tests pin the HTTP surface;
domain behavior is covered in ``test_lifecycle.py``.
"""

from __future__ import annotations

import time
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.agents import router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(router)
    return a


@pytest.fixture
def mock_agent_service() -> AgentService:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return AgentService(registry=registry)


@pytest.fixture
async def client(
    app: FastAPI, mock_agent_service: AgentService
) -> AsyncIterator[AsyncClient]:
    await mock_agent_service.initialize()
    app.state.agent_service = mock_agent_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await mock_agent_service.shutdown()


# --------------------------------------------------------------------------- #
# POST /api/agents/{name}/stop
# --------------------------------------------------------------------------- #


class TestStopAgent:
    async def test_stop_returns_202_with_request_id(self, client: AsyncClient) -> None:
        """Any click must return fast with a correlation id — even if the agent
        is already gone."""
        with (
            patch("app.services.lifecycle.os.kill"),
            patch(
                "app.services.lifecycle.AgentLifecycleService._attach_pidfd_watcher",
                new=AsyncMock(),
            ),
        ):
            t0 = time.monotonic()
            resp = await client.post("/api/agents/test-agent/stop")
            elapsed = time.monotonic() - t0

        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        assert body["request_id"]
        assert body["agent_name"] == "test-agent"
        assert elapsed < 0.2, f"stop must be instant, took {elapsed:.3f}s"

    async def test_stop_unknown_agent_still_returns_202(
        self, client: AsyncClient
    ) -> None:
        """Unknown agents are reported via WS ``agent_stop_failed``, NOT via
        HTTP 404. Keeps the HTTP contract uniform for the UI."""
        resp = await client.post("/api/agents/no-such/stop")
        assert resp.status_code == 202
        assert resp.json()["request_id"]

    async def test_stop_force_true_is_accepted(self, client: AsyncClient) -> None:
        with (
            patch("app.services.lifecycle.os.kill"),
            patch(
                "app.services.lifecycle.AgentLifecycleService._attach_pidfd_watcher",
                new=AsyncMock(),
            ),
            patch(
                "app.services.lifecycle.AgentLifecycleService._verify_process_gone",
                return_value=True,
            ),
        ):
            resp = await client.post(
                "/api/agents/test-agent/stop",
                json={"grace_seconds": 1.0, "force": True},
            )
        assert resp.status_code == 202
        assert resp.json()["request_id"]


# --------------------------------------------------------------------------- #
# POST /api/agents/{name}/restart
# --------------------------------------------------------------------------- #


class TestRestartAgent:
    async def test_restart_returns_202(self, client: AsyncClient) -> None:
        with (
            patch("app.services.lifecycle.os.kill"),
            patch(
                "app.services.lifecycle.AgentLifecycleService._attach_pidfd_watcher",
                new=AsyncMock(),
            ),
            patch(
                "app.routers.agents.load_launch_config",
                return_value={
                    "provider": "claude-code-sdk",
                    "model": "opus",
                    "working_dir": "/tmp/test",
                },
            ),
        ):
            resp = await client.post("/api/agents/test-agent/restart")
        assert resp.status_code == 202
        assert resp.json()["request_id"]


# --------------------------------------------------------------------------- #
# POST /api/agents/batch
# --------------------------------------------------------------------------- #


class TestBatchOperations:
    async def test_batch_stop_returns_per_agent_jobs(
        self, client: AsyncClient
    ) -> None:
        with (
            patch("app.services.lifecycle.os.kill"),
            patch(
                "app.services.lifecycle.AgentLifecycleService._attach_pidfd_watcher",
                new=AsyncMock(),
            ),
        ):
            t0 = time.monotonic()
            resp = await client.post(
                "/api/agents/batch",
                json={"action": "stop", "agent_names": ["test-agent"]},
            )
            elapsed = time.monotonic() - t0
        assert resp.status_code == 202
        body = resp.json()
        assert body["request_id"]
        assert body["action"] == "stop"
        assert len(body["jobs"]) == 1
        assert body["jobs"][0]["agent_name"] == "test-agent"
        assert body["jobs"][0]["request_id"]
        assert elapsed < 0.2, f"batch must be instant, took {elapsed:.3f}s"

    async def test_batch_invalid_action_returns_400(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/agents/batch",
            json={"action": "invalid", "agent_names": ["test-agent"]},
        )
        assert resp.status_code == 400

    async def test_batch_empty_list_returns_400(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/agents/batch",
            json={"action": "stop", "agent_names": []},
        )
        assert resp.status_code == 400
