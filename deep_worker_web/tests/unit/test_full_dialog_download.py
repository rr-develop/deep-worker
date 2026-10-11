"""Full Dialog ``download=1`` — the same snapshot as a file attachment.

DESIGN_FULL_DIALOG_ACTION_MODE.md §3.3. A screen-reader user must be able to get
the dialog WITHOUT rendering it into a tab (Chrome builds a full accessibility
tree for a plain-text tab and serves the screen reader on the browser UI thread
— measured 57 s / 152 812 nodes for 150 KB). ``download=1`` adds only a
``Content-Disposition: attachment`` header with a sanitised filename; without
the parameter the response must stay byte-identical to the tab snapshot.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.routers.agents as agents_router
from app.routers.agents import router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry

pytestmark = pytest.mark.asyncio

_FIXED = datetime(2026, 8, 16, 12, 34, 56, tzinfo=timezone.utc)


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(router)
    return a


@pytest.fixture
def mock_agent_service() -> AgentService:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    # A name with characters that must never reach a header verbatim.
    registry.add_mock_agent('we ird"na;me\r\n', 12346, 18862, "/tmp/test2")
    return AgentService(registry=registry)


@pytest.fixture
async def client(app: FastAPI, mock_agent_service: AgentService) -> AsyncIterator[AsyncClient]:
    await mock_agent_service.initialize()
    app.state.agent_service = mock_agent_service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await mock_agent_service.shutdown()


def _stub(service: AgentService, monkeypatch: pytest.MonkeyPatch, body: str = "--- User ---\nhi\n") -> None:
    async def fake_dialog(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
        return body

    monkeypatch.setattr(service, "is_connected", AsyncMock(return_value=True))
    monkeypatch.setattr(service, "get_agent_dialog_formatted", fake_dialog)
    monkeypatch.setattr(agents_router, "_snapshot_now", lambda: _FIXED)


class TestDownloadHeader:
    async def test_download_sets_attachment_with_txt_filename(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        r = await client.get("/api/agents/test-agent/dialog-text?verbosity=full&download=1")
        assert r.status_code == 200
        cd = r.headers.get("content-disposition")
        assert cd is not None, "download=1 must add Content-Disposition"
        assert cd.startswith("attachment;"), cd
        m = re.search(r'filename="([^"]+)"', cd)
        assert m, cd
        name = m.group(1)
        assert name.startswith("dialog_test-agent_full_"), name
        assert name.endswith(".txt"), name
        # Stamp is part of the name (click-time snapshot identity).
        assert "20260816" in name, name

    async def test_download_keeps_snapshot_headers_and_banner(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        plain = await client.get("/api/agents/test-agent/dialog-text?verbosity=text")
        dl = await client.get("/api/agents/test-agent/dialog-text?verbosity=text&download=1")
        assert dl.status_code == 200
        assert dl.headers.get("cache-control") == "no-store"
        assert dl.headers.get("x-content-type-options") == "nosniff"
        assert dl.headers.get("content-type", "").startswith("text/plain")
        # Same bytes as the tab snapshot — only the disposition differs.
        assert dl.content == plain.content
        assert dl.text.startswith("=== Full Dialog snapshot: agent=test-agent  verbosity=text")

    async def test_without_param_no_disposition(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        r = await client.get("/api/agents/test-agent/dialog-text")
        assert r.status_code == 200
        assert "content-disposition" not in r.headers

    async def test_download_false_is_default(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        r = await client.get("/api/agents/test-agent/dialog-text?download=0")
        assert r.status_code == 200
        assert "content-disposition" not in r.headers

    async def test_filename_is_sanitised(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub(mock_agent_service, monkeypatch)
        # httpx would reject a raw CR/LF in the URL path; percent-encode it.
        r = await client.get("/api/agents/we%20ird%22na%3Bme%0D%0A/dialog-text?download=1")
        assert r.status_code == 200, r.text
        cd = r.headers["content-disposition"]
        m = re.search(r'filename="([^"]+)"', cd)
        assert m, cd
        name = m.group(1)
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), name
        assert '"' not in cd.split("filename=", 1)[1][1:-1]
        assert "\r" not in cd and "\n" not in cd and ";" not in name

    async def test_download_on_timeout_still_504(
        self, client: AsyncClient, mock_agent_service: AgentService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def slow(agent_name: str, verbosity: str = "final", timestamps: str = "local") -> str:
            raise asyncio.TimeoutError("nope")

        monkeypatch.setattr(mock_agent_service, "is_connected", AsyncMock(return_value=True))
        monkeypatch.setattr(mock_agent_service, "get_agent_dialog_formatted", slow)
        r = await client.get("/api/agents/test-agent/dialog-text?download=1")
        assert r.status_code == 504
        assert "main UI" in r.json()["detail"]
