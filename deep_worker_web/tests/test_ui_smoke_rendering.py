"""Playwright smoke test: the page must execute main.js without errors
and render *something* in the agent list (either real agents or the
"No agents running" placeholder).

History: a duplicate ``const pendingRequests`` declaration caused a
global ``SyntaxError`` that made main.js silently fail to load. The UI
stayed stuck on "Loading agents…" forever while /api/agents returned
valid data and the WebSocket never reconnected. No existing test
caught it because the whole suite dials the backend without a real JS
engine.

This smoke test:

1. Collects *all* page errors and console errors into lists.
2. Navigates to /static/index.html and waits for ``#agent-list`` to
   leave the "Loading…" state.
3. Asserts no SyntaxError / ReferenceError / TypeError surfaced in the
   browser.

The test picks up a running server on 127.0.0.1:7887 (it is how the
project normally runs — see README security constraints). It is
skipped when the port is closed, so it does not fail CI on hosts
without a live server.
"""

from __future__ import annotations

import socket

import pytest
from playwright.sync_api import Page

BASE_URL = "http://127.0.0.1:7887"


def _server_is_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 7887), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.mark.ui
def test_page_loads_and_renders_agent_list(page: Page) -> None:
    if not _server_is_up():
        pytest.skip("deep_worker_web server not running on 127.0.0.1:7887")

    page_errors: list[str] = []
    console_errors: list[str] = []

    page.on("pageerror", lambda exc: page_errors.append(str(exc)))

    def _on_console(msg: object) -> None:
        # Playwright ConsoleMessage: use .type and .text defensively.
        try:
            if msg.type == "error":  # type: ignore[attr-defined]
                console_errors.append(msg.text)  # type: ignore[attr-defined]
        except Exception:
            pass

    page.on("console", _on_console)

    page.goto(f"{BASE_URL}/static/index.html")

    # ``#agent-list`` starts with a "Loading…" placeholder. After
    # main.js runs successfully, either real ``.agent-item`` entries
    # appear, or (if registry is empty) ``<p class="loading">No agents
    # running</p>`` replaces the initial one. Either way, the text
    # must change from "Loading agents..." within a few seconds.
    #
    # Playwright's function-expression API does the polling for us in
    # a single async wait — no sleep-polling in test code.
    page.wait_for_function(
        """() => {
            const el = document.getElementById('agent-list');
            if (!el) return false;
            const t = el.textContent || '';
            // The initial static HTML says "Loading agents...". Success
            // is ANY other steady-state text.
            return !t.includes('Loading agents...');
        }""",
        timeout=10_000,
    )

    # Fatal errors first: anything from the JS engine is a real bug.
    fatal_page = [
        e for e in page_errors
        if "SyntaxError" in e or "ReferenceError" in e or "TypeError" in e
    ]
    assert not fatal_page, f"fatal pageerror(s): {fatal_page}"

    fatal_console = [
        e for e in console_errors
        if "SyntaxError" in e or "ReferenceError" in e or "has already been declared" in e
    ]
    assert not fatal_console, f"fatal console error(s): {fatal_console}"

    # And the list itself must render meaningfully.
    agent_list = page.locator("#agent-list")
    text = agent_list.inner_text()
    has_agents = page.locator("#agent-list .agent-item").count() > 0
    assert has_agents or "No agents running" in text, (
        f"#agent-list is empty without the 'No agents running' placeholder. "
        f"Rendered: {text!r}"
    )
