"""Playwright smoke: TaskManager row becomes visible when the agent
reports ``task_manager_active``.

This pins the contract that the TM control row (⏹️ Stop TM / Reset TM /
on-off toggle) is shown for every agent that has a TaskManager
configured — including agents discovered after page load. Individual
buttons may be disabled depending on state (per user's clarification),
but the row itself must not stay hidden.

The test injects a synthetic ``agent_discovered`` + ``agent_status``
WebSocket message via JavaScript so it is independent of real RPyC
traffic.
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
def test_tm_row_visible_when_agent_has_task_manager(page: Page) -> None:
    if not _server_is_up():
        pytest.skip("deep_worker_web server not running on 127.0.0.1:7887")

    page.goto(f"{BASE_URL}/static/index.html")
    page.wait_for_function(
        """() => {
            const el = document.getElementById('agent-list');
            return el && !(el.textContent || '').includes('Loading agents...');
        }""",
        timeout=10_000,
    )

    # Inject a fake agent + drive it through the very handlers that run
    # on real WebSocket messages. This exercises the same code paths
    # the live subscribe channel uses.
    page.evaluate(
        """() => {
            const agent = {
                name: 'test_tm_fake',
                pid: 99999,
                status: 'idle',
                provider: 'deepseek',
                model: 'deepseek-chat',
                rpyc_port: 12345,
                working_dir: '/tmp',
                started_at: new Date().toISOString(),
            };
            handleAgentDiscovered({type: 'agent_discovered', agent});
            selectAgent('test_tm_fake');
            handleAgentStatus({
                type: 'agent_status',
                agent_name: 'test_tm_fake',
                status: 'running',
                task_manager_active: true,
                task_manager_disabled: false,
            });
        }"""
    )

    # The TM row (tm-toggle-label) must become visible, and the Stop
    # TM button must be visible too (enabled/disabled based on state,
    # but not hidden).
    page.wait_for_function(
        """() => {
            const row = document.getElementById('tm-toggle-label');
            if (!row) return false;
            const style = window.getComputedStyle(row);
            return style.display !== 'none';
        }""",
        timeout=2_000,
    )

    stop_btn = page.locator("#stop-taskmanager-btn")
    assert stop_btn.evaluate("el => window.getComputedStyle(el).display") != "none"
    reset_btn = page.locator("#reset-tm-btn")
    assert reset_btn.evaluate("el => window.getComputedStyle(el).display") != "none"

    # Cleanup the synthetic agent so the list is back to live state.
    page.evaluate(
        """() => handleAgentTerminated({type: 'agent_terminated',
                                         agent_name: 'test_tm_fake'})"""
    )


@pytest.mark.ui
def test_tm_row_hidden_when_agent_has_no_task_manager(page: Page) -> None:
    if not _server_is_up():
        pytest.skip("deep_worker_web server not running on 127.0.0.1:7887")

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
            const agent = {
                name: 'test_no_tm',
                pid: 88888,
                status: 'idle',
                provider: 'deepseek',
                model: 'deepseek-chat',
                rpyc_port: 54321,
                working_dir: '/tmp',
                started_at: new Date().toISOString(),
            };
            handleAgentDiscovered({type: 'agent_discovered', agent});
            selectAgent('test_no_tm');
            // status_change without task_manager_* fields
            handleAgentStatus({
                type: 'agent_status',
                agent_name: 'test_no_tm',
                status: 'idle',
            });
        }"""
    )

    # Give the DOM a tick to settle.
    page.wait_for_timeout(300)

    row_display = page.locator("#tm-toggle-label").evaluate(
        "el => window.getComputedStyle(el).display"
    )
    assert row_display == "none", (
        f"TM row must stay hidden for an agent without TaskManager; "
        f"got display={row_display!r}"
    )

    page.evaluate(
        """() => handleAgentTerminated({type: 'agent_terminated',
                                         agent_name: 'test_no_tm'})"""
    )


@pytest.mark.ui
def test_tm_disable_persists_across_agent_reselect(page: Page) -> None:
    """Regression (user report 2026-05-20): unchecking the TM box must
    STICK when the user switches to another agent and back.

    Before the fix, ``toggleTaskManager`` POSTed the new state but never
    updated the local ``agents[name].task_manager_disabled`` cache; a
    later ``selectAgent`` then re-read the stale cache and re-checked the
    box. This test drives the exact UX flow through the real handlers:
    select A → uncheck → select B → select A → assert box still
    unchecked.
    """
    if not _server_is_up():
        pytest.skip("deep_worker_web server not running on 127.0.0.1:7887")

    page.goto(f"{BASE_URL}/static/index.html")
    page.wait_for_function(
        """() => {
            const el = document.getElementById('agent-list');
            return el && !(el.textContent || '').includes('Loading agents...');
        }""",
        timeout=10_000,
    )

    # Two synthetic agents, both WITH a TaskManager. Stub fetch so the
    # toggle POST does not depend on a live RPyC agent.
    page.evaluate(
        """() => {
            window.__origFetch = window.fetch;
            window.fetch = async (url, opts) => {
                if (String(url).includes('/task-manager/toggle')) {
                    return { ok: true, json: async () => ({success: true}) };
                }
                return window.__origFetch(url, opts);
            };
            for (const nm of ['test_tm_persist_A', 'test_tm_persist_B']) {
                handleAgentDiscovered({type: 'agent_discovered', agent: {
                    name: nm, pid: 1, status: 'idle', provider: 'codex',
                    model: 'gpt-5.5', rpyc_port: 1, working_dir: '/tmp',
                    started_at: new Date().toISOString(),
                    has_task_manager: true, task_manager_active: false,
                    task_manager_disabled: false,
                }});
                handleAgentStatus({type: 'agent_status', agent_name: nm,
                    status: 'idle', task_manager_active: false,
                    task_manager_disabled: false});
            }
        }"""
    )
    page.wait_for_timeout(200)

    # Select A; box starts checked (TM enabled).
    page.evaluate("selectAgent('test_tm_persist_A')")
    page.wait_for_timeout(150)
    assert page.locator("#tm-enabled-checkbox").evaluate("el => el.checked") is True

    # User unchecks → toggleTaskManager(false).
    page.evaluate(
        """() => {
            const cb = document.getElementById('tm-enabled-checkbox');
            cb.checked = false;
            toggleTaskManager(false);
        }"""
    )
    page.wait_for_timeout(200)

    # Switch to B, then back to A.
    page.evaluate("selectAgent('test_tm_persist_B')")
    page.wait_for_timeout(150)
    page.evaluate("selectAgent('test_tm_persist_A')")
    page.wait_for_timeout(200)

    # THE ASSERTION: the box must still be unchecked.
    checked = page.locator("#tm-enabled-checkbox").evaluate("el => el.checked")
    assert checked is False, (
        "TM disable must persist across agent re-select; the box was "
        "re-checked (the 2026-05-20 regression)"
    )
    # And the local cache must carry the disabled flag.
    cached = page.evaluate("agents['test_tm_persist_A'].task_manager_disabled")
    assert cached is True

    # Cleanup synthetic agents + restore fetch.
    page.evaluate(
        """() => {
            handleAgentTerminated({type:'agent_terminated', agent_name:'test_tm_persist_A'});
            handleAgentTerminated({type:'agent_terminated', agent_name:'test_tm_persist_B'});
            if (window.__origFetch) window.fetch = window.__origFetch;
        }"""
    )
