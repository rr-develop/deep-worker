"""Visual UI tests using Playwright to verify layout and button visibility.

These tests use Playwright to actually render the page in a browser and verify:
1. All buttons are visible and clickable
2. Layout fills the viewport
3. Layout is independent of zoom level
"""

import pytest
from pathlib import Path

from playwright.sync_api import Page, expect

_TEST_PAGE_URL = (Path(__file__).resolve().parents[1] / "test_ui.html").as_uri()


@pytest.mark.ui
def test_buttons_are_visible(page: Page):
    """Test that all buttons are visible when agent is selected."""
    # Navigate to test page
    page.goto(_TEST_PAGE_URL)

    # Wait for page to load
    page.wait_for_load_state("networkidle")

    # Check all buttons are visible
    buttons = page.locator(".btn")
    count = buttons.count()

    assert count >= 3, f"Expected at least 3 buttons, found {count}"

    # Check each button is visible
    for i in range(count):
        button = buttons.nth(i)
        expect(button).to_be_visible()

        # Get button dimensions
        box = button.bounding_box()
        assert box is not None, f"Button {i} has no bounding box"
        assert box["width"] > 0, f"Button {i} has zero width"
        assert box["height"] > 0, f"Button {i} has zero height"

        print(f"✓ Button {i} ({button.text_content()[:20]}): {box['width']}x{box['height']}px")


@pytest.mark.ui
def test_buttons_visible_with_zoom(page: Page):
    """Test that buttons remain visible when page is zoomed."""
    page.goto(_TEST_PAGE_URL)
    page.wait_for_load_state("networkidle")

    # Test at different zoom levels
    zoom_levels = [0.75, 1.0, 1.25, 1.5]

    for zoom in zoom_levels:
        # Set zoom level
        page.evaluate(f"document.body.style.zoom = {zoom}")
        page.wait_for_timeout(100)

        # Check buttons are still visible
        buttons = page.locator(".btn")
        count = buttons.count()

        all_visible = True
        for i in range(count):
            button = buttons.nth(i)
            if not button.is_visible():
                all_visible = False
                print(f"✗ Button {i} NOT visible at zoom {zoom}")

        assert all_visible, f"Some buttons not visible at zoom level {zoom}"
        print(f"✓ All {count} buttons visible at zoom {zoom}")


@pytest.mark.ui
def test_layout_fills_viewport(page: Page):
    """Test that layout fills the viewport height."""
    page.goto(_TEST_PAGE_URL)
    page.wait_for_load_state("networkidle")

    # Get viewport and body heights
    viewport_height = page.viewport_size["height"]
    body_height = page.locator("body").bounding_box()["height"]

    # Body should fill viewport (within 5px tolerance)
    height_diff = abs(body_height - viewport_height)
    assert height_diff < 5, f"Body height ({body_height}px) should match viewport ({viewport_height}px), diff: {height_diff}px"

    print(f"✓ Layout fills viewport: body={body_height}px, viewport={viewport_height}px, diff={height_diff}px")


@pytest.mark.ui
def test_disabled_button_styling(page: Page):
    """Test that disabled buttons have proper styling."""
    page.goto(_TEST_PAGE_URL)
    page.wait_for_load_state("networkidle")

    # Find disabled button
    disabled_btn = page.locator("#stop-taskmanager-disabled")
    expect(disabled_btn).to_be_visible()
    expect(disabled_btn).to_be_disabled()

    # Check disabled styling
    opacity = disabled_btn.evaluate("el => window.getComputedStyle(el).opacity")
    cursor = disabled_btn.evaluate("el => window.getComputedStyle(el).cursor")

    # Disabled buttons should have reduced opacity
    assert float(opacity) < 1.0, f"Disabled button should have opacity < 1.0, got {opacity}"
    # Disabled buttons should have not-allowed cursor
    assert cursor == "not-allowed", f"Disabled button should have cursor: not-allowed, got {cursor}"

    print(f"✓ Disabled button styling correct: opacity={opacity}, cursor={cursor}")


@pytest.mark.ui
def test_panels_fill_available_space(page: Page):
    """Test that panels expand to fill available space."""
    page.goto(_TEST_PAGE_URL)
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

    # Interaction panel should expand to fill space (flex: 1)
    assert interaction_height >= list_height, "Interaction panel should be at least as tall as list panel"

    print(f"✓ Panels fill space: list={list_height}px, interaction={interaction_height}px, main={main_height}px")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
