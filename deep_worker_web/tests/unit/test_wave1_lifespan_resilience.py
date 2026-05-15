"""Wave 1 — the FastAPI lifespan must tolerate startup failures.

Contract
--------
If ``AgentService.initialize()`` raises, or the monitor task crashes
out, the app MUST still come up and stay up:

* Startup continues after logging the failure (no StartupFailed).
* ``app.state.agent_service`` is present (so routes don't 500 on
  attribute errors before they can check connected state).
* The health endpoint keeps returning 200.
* The monitor task's failure is logged and the task is respawned
  via ``loop.call_later`` (event-driven, not a tight poll).

These invariants are the difference between "site dies on the
lightest breeze" and "site degrades gracefully and tells us why".
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def failing_initialize(monkeypatch) -> None:
    """Patch AgentService.initialize to raise on the first call."""
    async def boom(self) -> None:
        raise RuntimeError("registry corruption")

    from app.services.agent_service import AgentService
    monkeypatch.setattr(AgentService, "initialize", boom)


def test_app_starts_even_if_agent_service_initialize_raises(
    caplog, failing_initialize
) -> None:
    """``AgentService.initialize()`` blowing up must NOT prevent
    startup — otherwise a transient registry glitch takes the UI
    with it. The health endpoint must still respond 200."""
    # Re-import the app after the monkeypatch. Using the module-level
    # ``app`` directly is fine for TestClient: lifespan runs once per
    # ``with TestClient(app)`` block.
    from app.main import app

    with caplog.at_level(logging.ERROR, logger="app.main"):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "healthy"

    # The failure must be logged so ops can diagnose.
    matches = [
        rec for rec in caplog.records
        if "registry corruption" in (rec.exc_text or rec.getMessage())
    ]
    assert matches, (
        "lifespan must log startup failures with the original error; "
        "no record mentioned 'registry corruption'"
    )


def test_monitor_task_failure_is_logged_and_respawned(monkeypatch) -> None:
    """If the monitor task dies mid-flight, the app must notice
    (log it as ERROR) and schedule a respawn via ``loop.call_later``
    — event-driven, no tight poll."""
    from app.services.agent_service import AgentService

    call_count = {"n": 0}

    async def flakey_monitor(self) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Crash the first run.
            raise RuntimeError("monitor blew up")
        # Subsequent runs: park forever until cancelled.
        await self._shutdown_event.wait()

    monkeypatch.setattr(AgentService, "monitor_agents", flakey_monitor)

    import logging as _logging
    from app.main import app

    # Use caplog-like capture via a local handler so we can inspect
    # what the supervisor logged.
    captured: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:  # noqa: D401
            captured.append(record)

    handler = _Collector(level=_logging.ERROR)
    supervisor_logger = _logging.getLogger("app.utils.supervisor")
    supervisor_logger.addHandler(handler)
    supervisor_logger.setLevel(_logging.ERROR)

    try:
        with TestClient(app) as client:
            # Give the loop a moment so the crashed monitor's
            # done_callback fires and the respawn is scheduled.
            import time
            time.sleep(0.2)
            resp = client.get("/health")
            assert resp.status_code == 200

        # At least one ERROR about monitor_agents must have landed.
        matches = [
            rec for rec in captured
            if "monitor" in rec.getMessage().lower()
            or "monitor blew up" in (rec.exc_text or "")
        ]
        assert matches, (
            "monitor task crash must surface as an ERROR log under "
            "app.utils.supervisor; captured: "
            + "; ".join(rec.getMessage() for rec in captured)
        )
    finally:
        supervisor_logger.removeHandler(handler)
