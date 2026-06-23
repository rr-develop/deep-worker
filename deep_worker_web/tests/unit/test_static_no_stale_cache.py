"""Static assets must be served with a revalidate-always Cache-Control so the
operator's browser can never run a stale ``index.html`` / ``main.js``.

Root cause of the Full Dialog introspection bug: a bare ``StaticFiles`` mount
serves HTML/JS with an ETag but no ``Cache-Control``. Browsers then apply
heuristic caching and keep running the OLD UI (a 3-option dropdown that cannot
emit ``system_prompt``/``tools``), so clicking Full Dialog shows the
conversation. See ``docs/DESIGN_STATIC_NO_STALE_CACHE.md``.

This is the RED-first unit gate: it fails before the fix (header absent) and
passes after the ``NoStaleStaticFiles`` mount stamps ``no-cache``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    "path",
    [
        "/static/index.html",
        "/static/js/main.js",
    ],
)
def test_static_assets_revalidate_always(path: str) -> None:
    with TestClient(app) as client:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} not served: {resp.status_code}"
        cc = resp.headers.get("cache-control", "")
        # ``no-cache`` = "cache but revalidate before every reuse". Combined
        # with the existing ETag this yields cheap 304s while guaranteeing the
        # browser never serves a stale asset without checking the server.
        assert "no-cache" in cc.lower(), (
            f"{path} must carry Cache-Control: no-cache to prevent stale UI; "
            f"got {cc!r}"
        )
        assert "must-revalidate" in cc.lower(), (
            f"{path} must carry must-revalidate; got {cc!r}"
        )
