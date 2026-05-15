"""Wave 2 — REST endpoints must surface disconnected-agent state as
503 Service Unavailable rather than a generic 500.

The agent process may die or have its RPyC connection drop between
requests. Today the router proceeds blindly and either:

* returns 500 with a generic error, or worse,
* calls a method on ``None`` and surfaces AttributeError through
  the global exception handler — still 500, with no hint for the
  client that the agent is just not reachable right now.

A 503 with ``{"error": "agent_disconnected", "agent_name": ...}``
is the accurate answer and lets the UI show a transient banner
instead of a scary "internal server error".

Tested endpoints
----------------
We cover the two most-hit endpoints that fan out to ``service.*``
after ``is_connected`` / ``get_root`` checks:

* ``POST /api/agents/{name}/message`` — when the service itself
  returns ``{"success": False, "error": "Failed to connect ..."}``
  (this mirrors what happens when the RPyC connect throws).
* ``POST /api/agents/{name}/stop_task_manager`` — when the agent
  exists in the registry but no live RPyC root is available.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers import agents_router
from app.schemas.agent_schemas import AgentInfo, AgentStatus


def _agent_info(name: str = "vc_x") -> AgentInfo:
    from datetime import datetime, timezone

    return AgentInfo(
        name=name,
        pid=12345,
        status=AgentStatus.IDLE,
        provider="claude-code",
        model="opus",
        rpyc_port=18861,
        working_dir="/tmp",
        started_at=datetime.now(timezone.utc),
    )


def _make_app(svc: object) -> FastAPI:
    """Assemble a lightweight app carrying ``svc`` under
    ``app.state.agent_service`` — bypasses real lifespan so tests
    don't trigger the monitor task or the registry scan."""
    # Install the catch-all so 500s don't mask the test's intent.
    from app.main import install_exception_handlers

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(agents_router)
    app.state.agent_service = svc
    return app


@pytest.fixture
async def client_factory() -> AsyncIterator[object]:
    """Return a factory that builds an AsyncClient around a fresh
    app configured with the caller-supplied service mock."""
    opened: list[AsyncClient] = []

    async def _make(svc: object) -> AsyncClient:
        app = _make_app(svc)
        transport = ASGITransport(app=app)
        cli = AsyncClient(transport=transport, base_url="http://test")
        opened.append(cli)
        return cli

    yield _make

    for cli in opened:
        await cli.aclose()


async def test_send_message_returns_503_when_agent_disconnected(
    client_factory,
) -> None:
    """``service.send_message`` signals disconnect by returning
    ``{"success": False, "error": "Failed to connect to agent"}``.
    The router must translate that into 503, not 500."""
    svc = MagicMock()
    svc.get_agent = AsyncMock(return_value=_agent_info("vc_x"))
    svc.send_message = AsyncMock(
        return_value={
            "success": False,
            "error": "Failed to connect to agent",
        }
    )

    cli = await client_factory(svc)
    resp = await cli.post(
        "/api/agents/vc_x/message", json={"message": "hi"}
    )

    assert resp.status_code == 503, (
        f"disconnected agent must yield 503; got {resp.status_code}: "
        + resp.text
    )
    body = resp.json()
    detail = body.get("detail")
    # Either structured form or a string that mentions the marker.
    assert detail is not None
    if isinstance(detail, dict):
        assert detail.get("error") == "agent_disconnected"
        assert detail.get("agent_name") == "vc_x"
    else:
        assert "agent_disconnected" in str(detail).lower()


async def test_stop_task_manager_returns_503_when_root_is_none(
    client_factory,
) -> None:
    """When the agent exists in the registry but there's no live
    RPyC root (``is_connected`` False AND ``connect_to_agent``
    can't recover), the router must answer 503 — not blindly
    return success like today."""
    svc = MagicMock()
    svc.get_agent = AsyncMock(return_value=_agent_info("vc_x"))
    svc.is_connected = AsyncMock(return_value=False)
    svc.connect_to_agent = AsyncMock(return_value=False)
    svc.get_root = MagicMock(return_value=None)

    cli = await client_factory(svc)
    resp = await cli.post(
        "/api/agents/vc_x/stop_task_manager", json={"force": False}
    )

    assert resp.status_code == 503, (
        f"stop_task_manager on disconnected agent must yield 503; "
        f"got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    detail = body.get("detail")
    assert detail is not None
    detail_str = str(detail).lower()
    assert (
        "agent_disconnected" in detail_str
        or "disconnected" in detail_str
    )


async def test_stop_task_manager_returns_200_when_connected_without_tm_feature(
    client_factory,
) -> None:
    """Regression guard: an agent built WITHOUT ``--task-manager``
    has a live RPyC root but no ``stop_task_manager`` method.
    That is a "nothing to do" state, not a failure — the router
    must keep its pre-existing 200 contract. This is the exact
    case integration test
    ``test_api_agents_integration.py::test_stop_task_manager``
    exercises against a real agent, and a previous version of
    this wave broke it by returning 503 unconditionally."""
    # A root object that deliberately has NO ``stop_task_manager``
    # attribute. Using a plain object() ensures hasattr returns False.
    bare_root = object()

    svc = MagicMock()
    svc.get_agent = AsyncMock(return_value=_agent_info("vc_x"))
    svc.is_connected = AsyncMock(return_value=True)
    svc.connect_to_agent = AsyncMock(return_value=True)
    svc.get_root = MagicMock(return_value=bare_root)

    cli = await client_factory(svc)
    resp = await cli.post(
        "/api/agents/vc_x/stop_task_manager", json={"force": False}
    )

    assert resp.status_code == 200, (
        f"connected-but-no-TM-feature case must keep 200; got "
        f"{resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body.get("success") is True
    assert "message" in body
