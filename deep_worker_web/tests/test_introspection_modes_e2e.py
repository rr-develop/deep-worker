"""Real browser e2e for the system_prompt / tools Full Dialog modes.

Drives the ACTUAL UI: select an agent, pick the verbosity option in the
dropdown, click the Full Dialog button, read the opened tab. This is the
flow the user exercises — curl against the endpoint is NOT a substitute
(a stale cached main.js, a wrong dropdown read, or a second render path
would all pass a curl check but fail the real UI).

Requires a running dev server on 127.0.0.1:7887 with at least one agent
selectable, and a Playwright chromium on disk. Marked ``ui`` so it is
excluded from the default hermetic unit run; invoke explicitly:

    python3 -m pytest tests/test_introspection_modes_e2e.py -m ui -s
"""

from __future__ import annotations

import glob
import os

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")


def _find_chrome() -> str | None:
    # Prefer whatever chromium build is actually present on disk.
    base = os.path.expanduser("~/.cache/ms-playwright")
    for pat in (
        f"{base}/chromium-*/chrome-linux64/chrome",
        f"{base}/chromium-*/chrome-linux/chrome",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _open_with_verbosity(page, context, verbosity: str) -> str:
    sel = page.locator("#full-dialog-verbosity")
    sel.select_option(verbosity)
    page.wait_for_timeout(200)
    with context.expect_page() as np_info:
        page.locator("#full-dialog-btn").click()
    np = np_info.value
    np.wait_for_load_state("networkidle")
    text = np.locator("body").text_content() or ""
    np.close()
    return text


@pytest.mark.ui
def test_introspection_modes_through_real_ui() -> None:
    chrome = _find_chrome()
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if chrome:
            launch_kwargs["executable_path"] = chrome
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"{SERVER}/")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        items = page.locator(".agent-item")
        if items.count() == 0:
            pytest.skip("no agents available on the running server")
        items.first.click()
        page.wait_for_timeout(1000)

        sp = _open_with_verbosity(page, context, "system_prompt")
        assert "--- System Prompt" in sp, f"not a system-prompt view: {sp[:160]!r}"
        assert "--- Summarization ---" not in sp, "BUG: summarization leaked"
        assert "--- User ---" not in sp, "BUG: user history leaked"

        tools = _open_with_verbosity(page, context, "tools")
        assert "--- Available Tools" in tools, f"not a tools view: {tools[:160]!r}"
        assert "--- Summarization ---" not in tools
        assert "--- User ---" not in tools

        browser.close()
