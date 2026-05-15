"""Wave 1 — unhandled exceptions from REST endpoints must land in a
structured 500 JSON response, not a bare stack trace.

Contract
--------
The app registers a top-level ``Exception`` handler that:

* returns HTTP 500 with a JSON body shaped
  ``{"error": str(exc), "error_type": exc.__class__.__name__}``;
* logs the exception with full traceback under the ``app.main``
  logger, so operators have a durable record;
* keeps the server alive — the exception must not propagate out of
  the request handler and take down the worker.

Why
---
Before the handler, any bug in a route surfaced to the client as an
HTML debug page (in debug mode) or a bare 500 with no body — both
made the UI look "dead" even though the server was fine. Catching
the tail here is the last line of defence after the per-endpoint
try/except work done in Wave 2.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app as real_app
from app.main import install_exception_handlers


def _install_probe(app: FastAPI, exc: BaseException) -> None:
    """Register a route that always raises ``exc``."""

    @app.get("/__probe_raise")
    async def probe() -> dict[str, str]:  # pragma: no cover — asserted on error
        raise exc


def test_unhandled_route_exception_returns_structured_500(caplog) -> None:
    # Use a fresh app so the probe route is local to this test.
    app = FastAPI()
    install_exception_handlers(app)
    _install_probe(app, RuntimeError("synthetic boom"))

    with TestClient(app, raise_server_exceptions=False) as client:
        with caplog.at_level(logging.ERROR, logger="app.main"):
            resp = client.get("/__probe_raise")

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["error"] == "synthetic boom"
    assert body["error_type"] == "RuntimeError"
    # Log must carry the traceback so operators can diagnose.
    matches = [
        rec for rec in caplog.records
        if "synthetic boom" in (rec.exc_text or "")
    ]
    assert matches, (
        "the exception handler must log with exc_info=True — found no "
        "log record with traceback"
    )


def test_real_app_has_exception_handler_installed() -> None:
    """The real app (imported at module level) must wire the handler
    too — otherwise production never gets the protection."""
    # FastAPI stores handlers keyed by exception class.
    assert Exception in real_app.exception_handlers, (
        "install_exception_handlers(app) must be called on the real app"
    )
