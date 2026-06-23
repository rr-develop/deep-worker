"""HONEST reproduction: drive the REAL UI to verify the Full Dialog
introspection modes (system_prompt / tools).

The user reports: selecting the 4th (system_prompt) or 5th (tools) option in
the verbosity combo next to the "Full Dialog" button and clicking the button
shows the USER/AGENT CONVERSATION instead of the invariant system prompt or
the tool list.

This test reproduces EXACTLY that flow through a real chromium:
  1. open the UI,
  2. select an agent,
  3. pick the verbosity option in the dropdown,
  4. click the "Full Dialog" button,
  5. read the text of the opened tab.

If the opened tab does not show a system-prompt view (mode 4) or a tools view
(mode 5) — e.g. it shows ``--- User ---`` / ``--- Summarization ---`` /
``--- Assistant ---`` conversation turns — the test FAILS.

Run against a live dev server (default 127.0.0.1:7887):

    python3 -m pytest tests/test_introspection_bug_repro.py -m ui -s
"""

from __future__ import annotations

import glob
import os

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")

# Markers that mean "this is conversation history" — they MUST NOT appear in
# the system_prompt / tools views.
HISTORY_MARKERS = (
    "--- User ---",
    "--- Assistant ---",
    "--- TaskManager ---",
    "--- Summarization ---",
    "--- From agent:",
)


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


def _open_with_verbosity(page, context, verbosity: str) -> str:
    """Select ``verbosity`` in the combo, click Full Dialog, return tab text."""
    sel = page.locator("#full-dialog-verbosity")
    sel.select_option(verbosity)
    page.wait_for_timeout(200)
    # Sanity: the dropdown actually committed the value the user picked.
    assert sel.input_value() == verbosity, (
        f"dropdown did not commit {verbosity!r}, got {sel.input_value()!r}"
    )
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

        # ---- Mode 4: system_prompt ----
        sp = _open_with_verbosity(page, context, "system_prompt")
        print(f"\n[system_prompt] first 200 chars:\n{sp[:200]!r}\n")
        assert "--- System Prompt" in sp, (
            f"BUG: system_prompt view is NOT a system prompt: {sp[:200]!r}"
        )
        for m in HISTORY_MARKERS:
            assert m not in sp, f"BUG: history marker {m!r} leaked into system_prompt view"

        # ---- Mode 5: tools ----
        tools = _open_with_verbosity(page, context, "tools")
        print(f"\n[tools] first 200 chars:\n{tools[:200]!r}\n")
        assert "--- Available Tools" in tools, (
            f"BUG: tools view is NOT a tools list: {tools[:200]!r}"
        )
        for m in HISTORY_MARKERS:
            assert m not in tools, f"BUG: history marker {m!r} leaked into tools view"

        browser.close()
