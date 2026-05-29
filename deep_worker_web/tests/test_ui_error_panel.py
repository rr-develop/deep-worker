"""Playwright smoke: the persistent error panel shows the agent's error
detail — including on PAGE LOAD via the snapshot (not only a live push).

User report (2026-05-20): an agent was in ``error`` but the web UI
showed no error detail. Root cause: ``error_message`` was never carried
through the ``/api/agents`` / ``agent_discovered`` snapshot the page
loads, so ``selectAgent`` (which reads ``agent.error_message`` to show
the panel) got ``undefined``.

These tests drive the SAME handlers a real page load + WebSocket
snapshot use, so they exercise the page-load path that was broken.
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


_ERR = (
    "Agent invocation failed: Claude emitted a proxy MCP ToolUseBlock "
    "but the matching MCP handler did not register a pending call within "
    "30s — Claude CLI subprocess was hung."
)


@pytest.mark.ui
def test_error_panel_shows_on_select_for_errored_agent(page: Page) -> None:
    """An agent already in ``error`` at page-load time (its snapshot
    carries ``error_message``) must show the persistent error panel
    with the full text when selected."""
    if not _server_is_up():
        pytest.skip("the web app server not running on 127.0.0.1:7887")

    page.goto(f"{BASE_URL}/static/index.html")
    page.wait_for_function(
        """() => {
            const el = document.getElementById('agent-list');
            return el && !(el.textContent || '').includes('Loading agents...');
        }""",
        timeout=10_000,
    )

    # Discover an agent that is ALREADY in error, with error_message in
    # the snapshot — exactly what /api/agents + agent_discovered now
    # deliver on a fresh load.
    page.evaluate(
        """(err) => {
            handleAgentDiscovered({type: 'agent_discovered', agent: {
                name: 'test_err_agent', pid: 7777, status: 'error',
                provider: 'claude-code', model: 'opus', rpyc_port: 1,
                working_dir: '/tmp', started_at: new Date().toISOString(),
                error_message: err,
            }});
            selectAgent('test_err_agent');
        }""",
        _ERR,
    )
    page.wait_for_timeout(300)

    panel = page.locator("#last-error-panel")
    display = panel.evaluate("el => window.getComputedStyle(el).display")
    assert display != "none", (
        "the persistent error panel must be visible for an agent in error "
        "(error_message arrived via the snapshot — incident observed 2026-05-20)"
    )
    text = page.locator("#last-error-content").evaluate("el => el.textContent")
    assert "Claude CLI subprocess was hung" in text, (
        f"error panel must show the full cause; got: {text[:120]!r}"
    )

    page.evaluate(
        """() => handleAgentTerminated({type:'agent_terminated',
                                        agent_name:'test_err_agent'})"""
    )


@pytest.mark.ui
def test_error_panel_hidden_for_healthy_agent(page: Page) -> None:
    """A healthy agent (no error_message) must NOT show the error panel."""
    if not _server_is_up():
        pytest.skip("the web app server not running on 127.0.0.1:7887")

    page.goto(f"{BASE_URL}/static/index.html")
    page.wait_for_function(
        """() => {
            const el = document.getElementById('agent-list');
            return el && !(el.textContent || '').includes('Loading agents...');
        }""",
        timeout=10_000,
    )

    page.evaluate(
        """() => {
            handleAgentDiscovered({type: 'agent_discovered', agent: {
                name: 'test_ok_agent', pid: 6666, status: 'idle',
                provider: 'codex', model: 'gpt-5.5', rpyc_port: 1,
                working_dir: '/tmp', started_at: new Date().toISOString(),
            }});
            selectAgent('test_ok_agent');
        }"""
    )
    page.wait_for_timeout(300)

    display = page.locator("#last-error-panel").evaluate(
        "el => window.getComputedStyle(el).display"
    )
    assert display == "none", (
        f"error panel must stay hidden for a healthy agent; got display={display!r}"
    )

    page.evaluate(
        """() => handleAgentTerminated({type:'agent_terminated',
                                        agent_name:'test_ok_agent'})"""
    )
