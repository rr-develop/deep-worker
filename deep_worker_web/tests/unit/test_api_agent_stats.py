"""Unit tests for GET /api/agents/{agent_name}/stats (§6.1).

Contract (DESIGN_AGENT_TOKEN_STATS.md §6.1):
- Unknown agent → 404.
- Known agent with no stats file → 200 with zero-filled aggregates
  (no file/dir is created by the GET).
- Invalid agent name → 400 (ValueError from validate_agent_name is
  translated to a client error, not 500).
- The response shape matches ``deep_worker_cli.stats.token_stats``
  (single reader/aggregator implementation — MAJOR-1 v2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.agents import router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
async def client(app: FastAPI, monkeypatch) -> AsyncClient:
    """Create async test client with an isolated fake HOME.

    ``stats_for_agent`` resolves the stats file under the agent's
    data_dir, which defaults to ``~/.deep_worker/<name>``. We point
    ``Path.home()`` at a tmp dir so the test never touches the real
    ``~/.deep_worker``.
    """
    from deep_worker_cli.storage_paths import deep_worker_home

    tmp_home = Path("/tmp/dwweb_test_stats_home")
    tmp_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_home)

    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    service = AgentService(registry=registry)
    await service.initialize()
    app.state.agent_service = service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await service.shutdown()
    import shutil

    shutil.rmtree(tmp_home, ignore_errors=True)


class TestGetAgentStats:
    """Test GET /api/agents/{agent_name}/stats."""

    async def test_unknown_agent_returns_404(self, client: AsyncClient) -> None:
        """Unknown agent (not in the web registry) → 404."""
        response = await client.get("/api/agents/ghost-agent/stats")
        assert response.status_code == 404
        assert "ghost-agent" in response.json()["detail"]

    async def test_known_agent_no_file_returns_200_zeros(
        self, client: AsyncClient
    ) -> None:
        """Known agent with no token_stats.jsonl → 200 with zeroed aggregates."""
        response = await client.get("/api/agents/test-agent/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "test-agent"
        assert data["totals"]["requests"] == 0
        assert data["totals"]["prompt_tokens"] == 0
        assert data["totals"]["completion_tokens"] == 0
        assert data["totals"]["total_tokens"] == 0
        assert data["totals"]["cache_hit_tokens"] == 0
        assert data["totals"]["cache_miss_tokens"] == 0
        assert data["totals"]["has_cache"] is False
        assert data["by_provider"] == []
        assert data["skipped_records"] == 0
        assert data["duplicate_records"] == 0
        assert data["partial_tail"] is False

    async def test_no_file_created_by_get(self, client: AsyncClient) -> None:
        """GET must NOT create the stats file or the data dir."""
        await client.get("/api/agents/test-agent/stats")
        data_dir = Path("/tmp/dwweb_test_stats_home") / Path("~/.deep_worker").name / "test-agent"
        assert not (data_dir / "token_stats.jsonl").exists()
        assert not data_dir.exists()

    async def test_invalid_name_returns_400(self, client: AsyncClient) -> None:
        """Invalid agent name → 400, not 500."""
        # '.' is not in the allowed agent-name charset → 400.
        response = await client.get("/api/agents/bad.name/stats")
        assert response.status_code == 400

    async def test_with_records_returns_aggregates(
        self, client: AsyncClient, monkeypatch
    ) -> None:
        """Known agent WITH a stats file → aggregated numbers from the
        single shared reader implementation."""
        from deep_worker_cli.stats.token_stats import (
            STATS_FILE_NAME,
            build_record,
            normalize_usage,
            KIND_AGENT,
            STATUS_OK,
        )
        from deep_worker_cli.stats.token_stats import TokenUsageRecorder

        data_dir = Path("/tmp/dwweb_test_stats_home") / Path("~/.deep_worker").name / "test-agent"
        data_dir.mkdir(parents=True, exist_ok=True)

        recorder = TokenUsageRecorder(data_dir)
        await recorder.record(
            build_record(
                provider="deepseek",
                model="deepseek-chat",
                kind=KIND_AGENT,
                status=STATUS_OK,
                usage=normalize_usage(
                    "deepseek",
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "prompt_cache_hit_tokens": 70,
                        "prompt_cache_miss_tokens": 30,
                    },
                ),
            )
        )

        response = await client.get("/api/agents/test-agent/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["totals"]["requests"] == 1
        assert data["totals"]["requests_with_usage"] == 1
        assert data["totals"]["prompt_tokens"] == 100
        assert data["totals"]["completion_tokens"] == 20
        assert data["totals"]["total_tokens"] == 120
        assert data["totals"]["cache_hit_tokens"] == 70
        assert data["totals"]["cache_miss_tokens"] == 30
        assert data["totals"]["has_cache"] is True
        providers = {p["provider"] for p in data["by_provider"]}
        assert providers == {"deepseek"}

    async def test_context_snapshot_in_provider_totals(
        self, client: AsyncClient
    ) -> None:
        """A grok-fallback context_snapshot record appears in the provider's
        context section, never in token aggregates (MINOR v2/v3)."""
        from deep_worker_cli.stats.token_stats import (
            build_record,
            KIND_AGENT,
            STATUS_OK,
        )
        from deep_worker_cli.stats.token_stats import TokenUsageRecorder

        data_dir = Path("/tmp/dwweb_test_stats_home") / Path("~/.deep_worker").name / "test-agent"
        data_dir.mkdir(parents=True, exist_ok=True)
        recorder = TokenUsageRecorder(data_dir)
        await recorder.record(
            build_record(
                provider="grok",
                model="grok-4.5",
                kind=KIND_AGENT,
                status=STATUS_OK,
                context_used=53000,
                context_size=200000,
            )
        )

        response = await client.get("/api/agents/test-agent/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["totals"]["requests"] == 1
        assert data["totals"]["requests_with_usage"] == 0
        assert data["totals"]["total_tokens"] == 0
        assert data["totals"]["has_cache"] is False
        assert data["totals"]["context"]["snapshots"] == 1
        assert data["totals"]["context"]["max_context_used"] == 53000
        assert data["totals"]["context"]["max_context_size"] == 200000
        grok = next(p for p in data["by_provider"] if p["provider"] == "grok")
        assert grok["totals"]["context"]["snapshots"] == 1

    async def test_malformed_line_counts_skipped_records(
        self, client: AsyncClient
    ) -> None:
        """A malformed line is counted in skipped_records, not dropped."""
        data_dir = Path("/tmp/dwweb_test_stats_home") / Path("~/.deep_worker").name / "test-agent"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "token_stats.jsonl").write_text("{not-json}\n")

        response = await client.get("/api/agents/test-agent/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["skipped_records"] == 1
        assert data["totals"]["requests"] == 0
