"""Playwright UI tests for the token-stats feature (§6.2).

Covers:
- The "📊 Stats" button is visible next to "Full Dialog" when an agent
  is selected and opens ``stats.html?agent=<name>`` in a new tab.
- ``stats.html`` renders the stats JSON via textContent (no innerHTML
  with server data), shows a zeroed snapshot for an agent without a
  stats file, and shows an error panel for an unknown agent.

The server is started from the dev checkout on
port 7888 (approved binding for this checkout — see app/config.py).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

_TREE = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dev_tree_pythonpath() -> str:
    """PYTHONPATH pointing at THIS checkout's own packages.

    The ambient shell may export a prod-tree PYTHONPATH
    (e.g. a production checkout of deep_worker_cli), which would
    make the dev-checkout server import prod's deep_worker_cli — where
    ``deep_worker_cli.stats`` does not exist. Mirror the spawn PYTHONPATH
    built in app/routers/agents.py so the server under test always resolves
    its own tree. See docs/DESIGN_DEV_FLAG.md.
    """
    paths = [
        str(_REPO_ROOT / "deep_worker" / "deep_worker_cli"),
        str(_REPO_ROOT / "deep_worker" / "deepagents"),
        "/path/to/rpyc_async",
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return ":".join(paths)


@pytest.fixture(scope="module")
def server():
    """Start the FastAPI server (dev tree, port 7888)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _dev_tree_pythonpath()
    process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "7888"],
        cwd=str(_TREE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    yield "http://127.0.0.1:7888"
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _select_first_agent(page: Page, base: str) -> bool:
    """Select the first agent in the list; return False if none exist."""
    page.goto(f"{base}/", wait_until="networkidle")
    page.wait_for_timeout(1500)
    agent_items = page.locator(".agent-item")
    count = agent_items.count()
    if count == 0:
        return False
    agent_items.first.click()
    page.wait_for_timeout(500)
    return True


@pytest.mark.ui
def test_stats_button_opens_stats_page(page: Page, server: str) -> None:
    """The '📊 Stats' button is visible and opens stats.html?agent=…."""
    if not _select_first_agent(page, server):
        pytest.skip("No agents available to test")

    stats_btn = page.locator("#stats-btn")
    expect(stats_btn).to_be_visible()

    with page.context.expect_page() as new_page_info:
        stats_btn.click()
    new_page = new_page_info.value
    new_page.wait_for_load_state("networkidle")
    assert "/static/stats.html?agent=" in new_page.url, (
        f"stats button must open stats.html with agent param; got {new_page.url}"
    )


@pytest.mark.ui
def test_stats_page_renders_zero_snapshot(page: Page, server: str) -> None:
    """stats.html renders a zeroed snapshot for a known agent without a
    stats file (via textContent — no innerHTML with data)."""
    if not _select_first_agent(page, server):
        pytest.skip("No agents available to test")

    # Fetch the current agent name from the UI header.
    agent_name = page.locator("#selected-agent-name").text_content()
    if not agent_name:
        pytest.skip("Could not read selected agent name")

    page.goto(f"{server}/static/stats.html?agent={agent_name}", wait_until="networkidle")
    page.wait_for_timeout(500)

    # The page must render the totals table (even a zeroed one).
    content = page.locator("#stats-content")
    expect(content.locator("table").first).to_be_visible()
    # Zeroed snapshot: "Requests" row shows 0.
    body_text = content.text_content()
    assert "Requests" in body_text
    # No page errors (XSS / broken JS).
    page_errors: list[str] = []
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.wait_for_timeout(300)
    assert page_errors == [], f"page errors: {page_errors}"


@pytest.mark.ui
def test_stats_page_shows_error_for_unknown_agent(page: Page, server: str) -> None:
    """stats.html shows an error panel for an unknown agent (404)."""
    page.goto(f"{server}/static/stats.html?agent=definitely-not-an-agent-xyz", wait_until="networkidle")
    page.wait_for_timeout(500)
    panel = page.locator(".error-panel")
    expect(panel).to_be_visible()
    text = panel.text_content()
    assert "404" in text or "not found" in text.lower(), (
        f"error panel should mention 404/not found; got: {text}"
    )


@pytest.mark.ui
def test_stats_copy_button_copies_text(page: Page, server: str) -> None:
    """The '📋 Copy' button writes a human-readable stats text to the
    clipboard (spied via navigator.clipboard.writeText)."""
    if not _select_first_agent(page, server):
        pytest.skip("No agents available to test")

    agent_name = page.locator("#selected-agent-name").text_content()
    if not agent_name:
        pytest.skip("Could not read selected agent name")

    page.goto(f"{server}/static/stats.html?agent={agent_name}", wait_until="networkidle")
    page.wait_for_timeout(500)

    # Spy on the clipboard API: capture what the page tries to copy.
    page.evaluate(
        """() => {
            window.__copiedText = null;
            navigator.clipboard.writeText = (text) => {
                window.__copiedText = text;
                return Promise.resolve();
            };
        }"""
    )

    copy_btn = page.locator("#copy-btn")
    expect(copy_btn).to_be_visible()
    copy_btn.click()
    page.wait_for_timeout(200)

    copied = page.evaluate("window.__copiedText")
    assert copied is not None, "copy button must call navigator.clipboard.writeText"
    # Human-readable text: header, totals, per-provider sections.
    assert agent_name in copied, f"copied text should mention agent; got: {copied}"
    assert "Totals" in copied
    assert "Requests" in copied
    assert "By provider" in copied


@pytest.mark.ui
def test_stats_copy_button_disabled_on_error(page: Page, server: str) -> None:
    """The copy button is disabled when stats failed to load (nothing to copy)."""
    page.goto(f"{server}/static/stats.html?agent=definitely-not-an-agent-xyz", wait_until="networkidle")
    page.wait_for_timeout(500)
    expect(page.locator(".error-panel")).to_be_visible()
    expect(page.locator("#copy-btn")).to_be_disabled()
