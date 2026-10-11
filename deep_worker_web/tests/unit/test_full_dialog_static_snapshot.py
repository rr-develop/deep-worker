"""Full Dialog = static, click-time snapshot (DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md).

HTTP-envelope contract of ``GET /api/agents/{name}/dialog-text``:

* ``Cache-Control: no-store`` + ``X-Content-Type-Options: nosniff`` +
  ``Content-Length`` (one static ``text/plain`` document, never streamed,
  never served from cache) — §3.2 / AC2.
* Body starts with the two-line snapshot banner (agent, verbosity, ISO
  timestamp with offset; "never updates") followed by the rendered text
  verbatim — §3.3 / AC3, for all five verbosity modes.
* ``asyncio.TimeoutError`` from the service → **504** naming the agent and
  the 30 s bound (not a generic 500) — §3.4 / AC4.
* A slow fetch (> threshold) emits exactly one WARNING breadcrumb — §3.4.

The formatter output itself is covered elsewhere (test_dialog_formatting.py);
here the service method is stubbed.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.routers.agents as agents_router
from app.routers.agents import router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry

VERBOSITIES = ["final", "text", "full", "system_prompt", "tools"]

# ISO-8601 with seconds and a numeric UTC offset, e.g. 2026-08-16T00:41:07+02:00
_ISO_WITH_OFFSET = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


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
async def client(app: FastAPI, mock_agent_service: AgentService) -> AsyncIterator[AsyncClient]:
    await mock_agent_service.initialize()
    app.state.agent_service = mock_agent_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await mock_agent_service.shutdown()


def _stub(
    mock_agent_service: AgentService,
    monkeypatch: pytest.MonkeyPatch,
    body: str = "--- User ---\nhello\n\n--- Assistant ---\nworld\n",
) -> dict[str, Any]:
    """Stub connectivity + formatter; return a dict capturing the call."""
    captured: dict[str, Any] = {}

    async def fake_dialog(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
        captured["agent_name"] = agent_name
        captured["verbosity"] = verbosity
        return body

    monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
    monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", fake_dialog)
    return captured


class TestSnapshotHeaders:
    async def test_no_store_nosniff_content_length(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 200
        assert r.headers.get("cache-control") == "no-store"
        assert r.headers.get("x-content-type-options") == "nosniff"
        # Static, non-streamed document: an explicit length, no chunking.
        assert "content-length" in r.headers
        assert "transfer-encoding" not in r.headers
        ct = r.headers["content-type"].lower()
        assert "text/plain" in ct and "charset=utf-8" in ct

    @pytest.mark.parametrize("verbosity", VERBOSITIES)
    async def test_headers_for_every_verbosity(
        self,
        verbosity: str,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub(mock_agent_service, monkeypatch, body="x")
        r = await client.get(f"/api/agents/test-agent/dialog-text?verbosity={verbosity}")
        assert r.status_code == 200
        assert r.headers.get("cache-control") == "no-store"
        assert r.headers.get("x-content-type-options") == "nosniff"


class TestSnapshotBanner:
    @pytest.mark.parametrize("verbosity", VERBOSITIES)
    async def test_banner_then_verbatim_body(
        self,
        verbosity: str,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        body = f"--- Marker ({verbosity}) ---\nline 1\nline 2\n"
        _stub(mock_agent_service, monkeypatch, body=body)
        r = await client.get(f"/api/agents/test-agent/dialog-text?verbosity={verbosity}")
        assert r.status_code == 200
        lines = r.text.split("\n")
        # Line 1: identity + click-time stamp.
        assert lines[0].startswith("=== Full Dialog snapshot:"), lines[0]
        assert "agent=test-agent" in lines[0]
        assert f"verbosity={verbosity}" in lines[0]
        assert "taken=" in lines[0]
        assert _ISO_WITH_OFFSET.search(lines[0]), lines[0]
        # Line 2: the contract, in words the operator can read.
        assert lines[1].startswith("=== Static snapshot"), lines[1]
        assert "never updates" in lines[1]
        # Blank separator, then the rendered text VERBATIM.
        assert lines[2] == ""
        assert r.text.split("\n", 3)[3] == body

    async def test_stamp_is_per_request(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch, body="x")
        t0 = datetime(2026, 8, 16, 0, 41, 7, tzinfo=timezone(timedelta(hours=2)))
        clock = [t0, t0 + timedelta(seconds=90)]
        monkeypatch.setattr(agents_router, "_snapshot_now", lambda: clock.pop(0))
        r1 = await client.get("/api/agents/test-agent/dialog-text")
        r2 = await client.get("/api/agents/test-agent/dialog-text")
        l1 = r1.text.split("\n", 1)[0]
        l2 = r2.text.split("\n", 1)[0]
        assert "taken=2026-08-16T00:41:07+02:00" in l1, l1
        assert "taken=2026-08-16T00:42:37+02:00" in l2, l2
        assert l1 != l2

    async def test_empty_body_still_gets_banner(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch, body="")
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 200
        assert r.text.startswith("=== Full Dialog snapshot:")


class TestBoundedWaitVisibility:
    async def test_timeout_maps_to_504_with_reason(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def slow(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
            raise asyncio.TimeoutError(f"Agent '{agent_name}' did not respond within 30s")

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", slow)
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 504, (r.status_code, r.text)
        detail = r.json()["detail"]
        assert "test-agent" in detail
        assert "30" in detail
        # The operator must learn the main UI is NOT affected.
        assert "main UI" in detail

    async def test_generic_error_still_500(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
            raise ValueError("Agent returned error: nope")

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", boom)
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 500

    async def test_agent_refusal_maps_to_502_not_500(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal BY THE AGENT is an upstream failure, not a server fault.

        Returning 500 here sent operators hunting a bug in the web process
        when the web process had worked correctly; 502 names the real culprit
        and the detail carries the agent's own wording.
        """
        from app.services.agent_service import AgentRefusedError

        async def refused(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
            raise AgentRefusedError("Agent returned error: Agent not initialized")

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", refused)
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 502, (r.status_code, r.text)
        assert "Agent not initialized" in r.json()["detail"]

    async def test_slow_fetch_emits_one_warning(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _stub(mock_agent_service, monkeypatch, body="x")
        # Threshold 0 ⇒ every fetch counts as slow (deterministic, no sleeping).
        monkeypatch.setattr(agents_router, "SLOW_DIALOG_WARN_SECONDS", 0.0)
        with caplog.at_level(logging.WARNING, logger="app.routers.agents"):
            r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 200
        warns = [rec for rec in caplog.records if rec.levelno == logging.WARNING and "dialog" in rec.getMessage().lower()]
        assert len(warns) == 1, [w.getMessage() for w in warns]
        msg = warns[0].getMessage()
        assert "test-agent" in msg
        # Bounded breadcrumb: no dialog content in the log record.
        assert "--- " not in msg

    async def test_fast_fetch_no_warning(
        self,
        client: AsyncClient,
        mock_agent_service: AgentService,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _stub(mock_agent_service, monkeypatch, body="x")
        with caplog.at_level(logging.WARNING, logger="app.routers.agents"):
            r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 200
        assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
