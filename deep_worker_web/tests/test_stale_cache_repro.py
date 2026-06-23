"""Reproduction of the USER-REPORTED Full Dialog introspection symptom.

The user sees the conversation (not the system prompt / tool list) when picking
the 4th/5th verbosity entry and clicking Full Dialog. The current SOURCE is
correct (see ``test_introspection_bug_repro.py``), so the symptom only appears
when the browser runs a STALE cached ``index.html`` — a pre-feature page whose
verbosity dropdown has only ``final``/``text``/``full`` and therefore cannot
emit ``system_prompt``/``tools``.

This test reconstructs that stale browser deterministically: it serves the
pre-feature ``index.html`` (from git ``6ae6b5f^``) as the document and asserts
that clicking Full Dialog opens the CONVERSATION — proving the symptom — and
that the stale dropdown genuinely lacks the new options. The durable fix
(``NoStaleStaticFiles`` in ``app/main.py``, ``Cache-Control: no-cache``) stops
browsers from ever holding such a stale document. See
``docs/DESIGN_STATIC_NO_STALE_CACHE.md``.

Run against a live dev server (default 127.0.0.1:7887):

    python3 -m pytest tests/test_stale_cache_repro.py -m ui -s
"""

from __future__ import annotations

import glob
import os
import subprocess

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")
REPO = "/path/to/production"
# Commit that introduced the system_prompt/tools dropdown options. Its parent
# is the last revision whose index.html had only 3 options.
PRE_FEATURE_REV = "6ae6b5f^:deep_worker_web/static/index.html"

HISTORY_MARKERS = ("--- User", "--- Assistant", "--- Summ", "--- TaskMan", "--- From agent")


def _find_chrome() -> str | None:
    base = os.path.expanduser("~/.cache/ms-playwright")
    for pat in (
        f"{base}/chromium-*/chrome-linux64/chrome",
        f"{base}/chromium-*/chrome-linux/chrome",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


@pytest.mark.ui
def test_stale_cached_index_reproduces_conversation_symptom() -> None:
    try:
        stale_html = subprocess.check_output(
            ["git", "-C", REPO, "show", PRE_FEATURE_REV], text=True
        )
    except subprocess.CalledProcessError:
        pytest.skip("pre-feature index.html revision not available")

    # The stale document is the crux: only 3 verbosity options exist.
    assert stale_html.count("<option") == 3, "fixture is not the pre-feature page"

    chrome = _find_chrome()
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if chrome:
            launch_kwargs["executable_path"] = chrome
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context()
        page = context.new_page()

        body = stale_html.replace(
            'src="/static/js/main.js"', f'src="{SERVER}/static/js/main.js"'
        )

        def _serve_stale(route):
            route.fulfill(status=200, content_type="text/html", body=body)

        page.route(f"{SERVER}/static/index.html", _serve_stale)
        page.route(f"{SERVER}/", _serve_stale)

        page.goto(f"{SERVER}/static/index.html")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        items = page.locator(".agent-item")
        if items.count() == 0:
            pytest.skip("no agents available on the running server")
        items.first.click()
        page.wait_for_timeout(800)

        sel = page.locator("#full-dialog-verbosity")
        values = sel.evaluate("el => Array.from(el.options).map(o => o.value)")
        # The stale page CANNOT offer the introspection modes.
        assert "system_prompt" not in values
        assert "tools" not in values

        # The user picks "System prompt" → impossible on the stale dropdown,
        # so the value stays on a history verbosity and Full Dialog opens the
        # conversation. That IS the reported symptom.
        with context.expect_page() as np_info:
            page.locator("#full-dialog-btn").click()
        np = np_info.value
        np.wait_for_load_state("networkidle")
        text = np.locator("body").text_content() or ""
        np.close()
        browser.close()

        shows_conversation = any(m in text for m in HISTORY_MARKERS)
        assert shows_conversation, (
            "expected the stale page to show the conversation (the reported "
            f"symptom); got: {text[:160]!r}"
        )
