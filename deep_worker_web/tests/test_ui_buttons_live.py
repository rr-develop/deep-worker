"""Live UI tests for button visibility when agent is selected.

Tests that buttons appear correctly after selecting an agent.
"""

import pytest
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def server():
    """Start the FastAPI server for testing."""
    # Start server in background
    process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "7887"],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(2)

    yield "http://127.0.0.1:7887"

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.ui
def test_live_agent_controls_visible_when_agent_selected(page: Page, server: str):
    """Test that agent control buttons appear when an agent is selected."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Initially, agent-selected div should be hidden
    agent_selected = page.locator("#agent-selected")
    no_agent_selected = page.locator("#no-agent-selected")

    # Check initial state - no agent selected
    expect(no_agent_selected).to_be_visible()
    # agent_selected might be display:none so we can't check is_visible

    # Wait a bit for agents to load
    page.wait_for_timeout(1000)

    # Check if there are any agents in the list
    agent_items = page.locator(".agent-item")
    agent_count = agent_items.count()

    if agent_count > 0:
        print(f"Found {agent_count} agents, clicking first one")

        # Click first agent
        agent_items.first.click()
        page.wait_for_timeout(500)

        # Now agent-selected should be visible
        expect(agent_selected).to_be_visible()

        # Check that control buttons are visible
        full_dialog_btn = page.locator("#full-dialog-btn")
        stop_taskmanager_btn = page.locator("#stop-taskmanager-btn")

        expect(full_dialog_btn).to_be_visible()
        expect(stop_taskmanager_btn).to_be_visible()

        # Check button dimensions
        full_dialog_box = full_dialog_btn.bounding_box()
        stop_box = stop_taskmanager_btn.bounding_box()

        assert full_dialog_box is not None, "Full Dialog button has no bounding box"
        assert stop_box is not None, "Stop TaskManager button has no bounding box"

        assert full_dialog_box["width"] > 0, f"Full Dialog button has zero width"
        assert full_dialog_box["height"] > 0, f"Full Dialog button has zero height"
        assert stop_box["width"] > 0, f"Stop TaskManager button has zero width"
        assert stop_box["height"] > 0, f"Stop TaskManager button has zero height"

        print(f"✓ Full Dialog button: {full_dialog_box['width']}x{full_dialog_box['height']}px")
        print(f"✓ Stop TaskManager button: {stop_box['width']}x{stop_box['height']}px")
    else:
        print("No agents found - test skipped (this is expected if no agents are running)")
        pytest.skip("No agents available to test")


@pytest.mark.ui
def test_live_buttons_in_viewport(page: Page, server: str):
    """Test that buttons are reachable (not cut off).

    A button is considered cut off only if it lies outside the viewport
    AND cannot be brought into view by scrolling (either the page or a
    scrollable ancestor container). Buttons inside a scrollable panel
    (e.g. .agent-interaction-panel) are reachable and therefore fine.
    """
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    viewport_height = page.viewport_size["height"]
    viewport_width = page.viewport_size["width"]

    # Wait for agents to load
    page.wait_for_timeout(1000)

    agent_items = page.locator(".agent-item")
    agent_count = agent_items.count()

    if agent_count > 0:
        # Click first agent
        agent_items.first.click()
        page.wait_for_timeout(500)

        # Get all visible buttons
        buttons = page.locator(".btn:visible")
        button_count = buttons.count()

        print(f"Checking {button_count} visible buttons are reachable ({viewport_width}x{viewport_height})")

        for i in range(button_count):
            button = buttons.nth(i)
            box = button.bounding_box()

            if box:
                button_text = button.text_content()[:30]

                reachable = button.evaluate(
                    """(el) => {
                        const vw = window.innerWidth, vh = window.innerHeight;
                        const r = el.getBoundingClientRect();
                        if (r.left >= -1 && r.top >= -1 && r.right <= vw + 1 && r.bottom <= vh + 1) return true;
                        // Outside viewport: OK only if a scrollable ancestor can bring it into view.
                        let cur = el.parentElement;
                        while (cur) {
                            const s = window.getComputedStyle(cur);
                            const scrollY = (s.overflowY === 'auto' || s.overflowY === 'scroll' || s.overflowY === 'overlay') && cur.scrollHeight > cur.clientHeight + 1;
                            const scrollX = (s.overflowX === 'auto' || s.overflowX === 'scroll' || s.overflowX === 'overlay') && cur.scrollWidth > cur.clientWidth + 1;
                            if (scrollY || scrollX) return true;
                            cur = cur.parentElement;
                        }
                        return false;
                    }"""
                )

                assert reachable, (
                    f"Button '{button_text}' is cut off and not reachable by scrolling "
                    f"(x={box['x']}, y={box['y']}) size {box['width']}x{box['height']}"
                )

                print(f"✓ Button '{button_text}' reachable at ({box['x']}, {box['y']}) size {box['width']}x{box['height']}")
    else:
        print("No agents found - test skipped (this is expected if no agents are running)")
        pytest.skip("No agents available to test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
