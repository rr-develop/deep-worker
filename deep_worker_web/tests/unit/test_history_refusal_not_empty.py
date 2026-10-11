"""An agent refusal must never look like an empty history.

Defect (2026-08-29): ``on_success`` in ``get_agent_history`` /
``get_agent_last_response`` did not check ``response["success"]``. A response
``{"success": false, "error": "Agent not initialized"}`` was parsed, fell into
the success branch and went to the UI as a ``history_response`` with an empty
list — the frontend rendered an empty dialog and "No response yet", with no
notification at all.

A silent lie is worse than a visible error: the operator cannot tell "there is
no conversation" from "we could not read it". This test pins that a refusal is
sent as ``history_error`` / ``last_response_error`` and carries the reason text.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


class _Recorder:
    """Intercepts broadcasts instead of a real WebSocket."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


def _service_with_recorder(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Recorder]:
    registry = MockProcessRegistry()
    registry.add_mock_agent("a", 12345, 18861, "/tmp/a")
    svc = AgentService(registry=registry)
    rec = _Recorder()
    svc._broadcast_message = rec  # type: ignore[assignment]
    # Treat the agent as connected: this test is about handling the RESPONSE, not the connection.
    svc._connections["a"] = object()
    return svc, rec


def _capture_callbacks(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Captures ``on_success``/``on_error`` passed to fire_and_forget_call."""
    captured: dict[str, Any] = {}

    def _fake_fire(*args: Any, success_callback=None, error_callback=None, **kw: Any):
        captured["on_success"] = success_callback
        captured["on_error"] = error_callback

    # The import is local (inside the method), so patch the source in rpyc_async.
    monkeypatch.setattr(
        "rpyc_async.utils.helpers.fire_and_forget_call", _fake_fire
    )
    return captured


@pytest.mark.asyncio
async def test_history_refusal_becomes_history_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, rec = _service_with_recorder(monkeypatch)
    captured = _capture_callbacks(monkeypatch)
    svc._roots = {"a": object()}

    await svc.get_agent_history("a", limit=5, offset=0, request_id="req-1")
    await captured["on_success"](
        json.dumps({"success": False, "error": "Agent not initialized"})
    )

    assert rec.sent, "the response must reach the UI"
    msg = rec.sent[-1]
    assert msg["type"] == "history_error", f"refusal was sent as {msg['type']}"
    assert msg["success"] is False
    assert "Agent not initialized" in msg["error"]


@pytest.mark.asyncio
async def test_history_success_still_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety net: a normal response must not be affected by the new check."""
    svc, rec = _service_with_recorder(monkeypatch)
    captured = _capture_callbacks(monkeypatch)
    svc._roots = {"a": object()}

    await svc.get_agent_history("a", limit=5, offset=0, request_id="req-2")
    await captured["on_success"](
        json.dumps({"success": True, "history": [{"content": "hi"}]})
    )

    msg = rec.sent[-1]
    assert msg["type"] == "history_response"
    assert msg["history"] == [{"content": "hi"}]


@pytest.mark.asyncio
async def test_last_response_refusal_becomes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, rec = _service_with_recorder(monkeypatch)
    captured = _capture_callbacks(monkeypatch)
    svc._roots = {"a": object()}

    await svc.get_agent_last_response("a", request_id="req-3")
    await captured["on_success"](
        json.dumps({"success": False, "error": "Agent not initialized"})
    )

    msg = rec.sent[-1]
    assert msg["type"] == "last_response_error", f"refusal was sent as {msg['type']}"
    assert "Agent not initialized" in msg["error"]
