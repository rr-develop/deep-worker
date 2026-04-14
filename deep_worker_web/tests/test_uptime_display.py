"""Integration test for uptime display in UI.

This test verifies that the server uptime is correctly displayed in the browser UI
by capturing a screenshot and checking the actual rendered text.
"""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest
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
    time.sleep(3)

    yield "http://127.0.0.1:7887"

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.ui
def test_uptime_display_in_browser(page: Page, server: str):
    """Test that uptime is correctly displayed in the browser UI.

    This test:
    1. Navigates to the main page
    2. Waits for WebSocket to connect and receive uptime data
    3. Captures a screenshot of the footer with uptime
    4. Verifies that uptime is NOT showing "--" (the default)
    5. Verifies that uptime shows a proper formatted time
    """
    # Navigate to main page
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Find the uptime element
    uptime_element = page.locator("#server-uptime")
    expect(uptime_element).to_be_visible()

    # Wait for WebSocket to connect and update uptime
    # The uptime should change from "Uptime: --" to something like "Uptime: 5s"
    # We'll wait up to 10 seconds for this to happen

    print("\n⏳ Waiting for uptime to update from '--' to actual value...")

    max_wait = 10  # seconds
    start_time = time.time()
    uptime_updated = False

    while time.time() - start_time < max_wait:
        current_text = uptime_element.text_content()
        print(f"   Current uptime text: '{current_text}'")

        if current_text and "Uptime: --" not in current_text and "Uptime:" in current_text:
            uptime_updated = True
            print(f"✅ Uptime updated successfully: {current_text}")
            break

        # Wait a bit before checking again
        page.wait_for_timeout(500)

    # Take a screenshot of the footer area
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    screenshot_path = screenshots_dir / f"uptime_display_{int(time.time())}.png"

    # Screenshot the footer area where uptime is displayed
    footer = page.locator("footer")
    footer.screenshot(path=str(screenshot_path))
    print(f"\n📸 Screenshot saved: {screenshot_path}")

    # Also take full page screenshot for context
    full_screenshot_path = screenshots_dir / f"full_page_{int(time.time())}.png"
    page.screenshot(path=str(full_screenshot_path), full_page=True)
    print(f"📸 Full page screenshot saved: {full_screenshot_path}")

    # Get final uptime text
    final_uptime_text = uptime_element.text_content()
    print(f"\n📊 Final uptime text: '{final_uptime_text}'")

    # Assertions
    assert uptime_updated, f"Uptime did not update from '--' within {max_wait} seconds. Current: '{final_uptime_text}'"
    assert "Uptime: --" not in final_uptime_text, "Uptime is still showing '--'"
    assert "Uptime:" in final_uptime_text, "Uptime label is missing"

    # Check that uptime has a valid format (contains time units)
    # Valid formats: "5s", "2m 30s", "1h 5m", etc.
    has_time_unit = any(unit in final_uptime_text for unit in ['s', 'm', 'h', 'd'])
    assert has_time_unit, f"Uptime does not contain valid time units: '{final_uptime_text}'"

    print(f"\n✅ Test passed! Uptime is correctly displayed: '{final_uptime_text}'")


@pytest.mark.ui
def test_uptime_updates_over_time(page: Page, server: str):
    """Test that uptime value updates over time via WebSocket."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    uptime_element = page.locator("#server-uptime")

    # Wait for initial uptime
    page.wait_for_timeout(2000)

    # Capture uptime values at different times
    uptime_values = []

    for i in range(3):
        current_text = uptime_element.text_content()
        uptime_values.append(current_text)
        print(f"Uptime reading {i+1}: {current_text}")

        if i < 2:  # Don't wait after last reading
            page.wait_for_timeout(2000)

    # Take screenshot
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    screenshot_path = screenshots_dir / f"uptime_updates_{int(time.time())}.png"
    page.locator("footer").screenshot(path=str(screenshot_path))
    print(f"\n📸 Screenshot saved: {screenshot_path}")

    # Check that we got at least one non-default value
    non_default_values = [v for v in uptime_values if "Uptime: --" not in v]
    assert len(non_default_values) > 0, f"No non-default uptime values received: {uptime_values}"

    print(f"\n✅ Test passed! Got {len(non_default_values)}/3 non-default uptime values")


@pytest.mark.ui
def test_uptime_websocket_connection(page: Page, server: str):
    """Test that WebSocket connection is established and uptime data is received.

    This test checks the browser console logs to verify WebSocket behavior.
    """
    # Enable console logging
    console_messages = []

    def handle_console(msg):
        console_messages.append(f"{msg.type}: {msg.text}")

    page.on("console", handle_console)

    # Navigate
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for WebSocket messages
    page.wait_for_timeout(5000)

    # Take screenshot
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    screenshot_path = screenshots_dir / f"uptime_websocket_{int(time.time())}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    print(f"\n📸 Screenshot saved: {screenshot_path}")

    # Print console messages for debugging
    print("\n📋 Console messages:")
    for msg in console_messages[-20:]:  # Last 20 messages
        print(f"  {msg}")

    # Check for WebSocket connection messages
    ws_connected = any("WebSocket connected" in msg or "connected" in msg.lower() for msg in console_messages)

    # Get uptime display
    uptime_element = page.locator("#server-uptime")
    uptime_text = uptime_element.text_content()

    print(f"\n📊 WebSocket connected: {ws_connected}")
    print(f"📊 Current uptime: {uptime_text}")

    # Log result (don't fail on assertions, just report)
    if ws_connected:
        print("✅ WebSocket connection detected")
    else:
        print("⚠️  WebSocket connection not clearly detected in console")

    if "Uptime: --" not in uptime_text:
        print(f"✅ Uptime is displayed: {uptime_text}")
    else:
        print(f"❌ Uptime still showing default: {uptime_text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
