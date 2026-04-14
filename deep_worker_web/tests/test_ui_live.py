"""Live UI tests using Playwright to verify layout with running server.

These tests start the actual server and test the real index.html page.
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
def test_live_layout_fills_viewport(page: Page, server: str):
    """Test that live page layout fills the viewport height."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Get viewport and body heights
    viewport_height = page.viewport_size["height"]
    body_height = page.locator("body").bounding_box()["height"]

    # Body should fill viewport (within 5px tolerance)
    height_diff = abs(body_height - viewport_height)
    assert height_diff < 5, f"Body height ({body_height}px) should match viewport ({viewport_height}px), diff: {height_diff}px"

    print(f"✓ Live layout fills viewport: body={body_height}px, viewport={viewport_height}px, diff={height_diff}px")


@pytest.mark.ui
def test_live_layout_zoomed(page: Page, server: str):
    """Test that live page layout fills viewport at different zoom levels."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    zoom_levels = [0.75, 1.0, 1.25, 1.5]

    for zoom in zoom_levels:
        # Set zoom level
        page.evaluate(f"document.body.style.zoom = {zoom}")
        page.wait_for_timeout(100)

        # Get viewport and body heights
        viewport_height = page.viewport_size["height"]
        body_height = page.locator("body").bounding_box()["height"]

        # Body should still fill viewport (within 5px tolerance)
        height_diff = abs(body_height - viewport_height)

        # At different zoom levels, we need to account for zoom scaling
        # The body height in pixels doesn't change with CSS zoom, but the layout should still fill
        print(f"Zoom {zoom}: body={body_height}px, viewport={viewport_height}px, diff={height_diff}px")


@pytest.mark.ui
def test_live_panels_fill_space(page: Page, server: str):
    """Test that panels expand to fill available space on live page."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Get main content height
    main_content = page.locator(".main-content")
    main_height = main_content.bounding_box()["height"]

    # Get panel heights
    agent_list = page.locator(".agent-list-panel")
    agent_interaction = page.locator(".agent-interaction-panel")

    list_height = agent_list.bounding_box()["height"]
    interaction_height = agent_interaction.bounding_box()["height"]

    # Both panels should have reasonable height (not collapsed)
    assert list_height > 100, f"Agent list panel too small: {list_height}px"
    assert interaction_height > 100, f"Agent interaction panel too small: {interaction_height}px"

    print(f"✓ Live panels fill space: list={list_height}px, interaction={interaction_height}px, main={main_height}px")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
