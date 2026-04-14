"""Integration test for Full Dialog functionality.

This test verifies that the Full Dialog button opens a plain text dialog
in a new tab, showing the conversation history in a readable format.
"""

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
def test_full_dialog_plain_text(page: Page, server: str):
    """Test that Full Dialog opens plain text in a new tab."""
    # Navigate to main page
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for agents to load
    page.wait_for_timeout(2000)

    # Select an agent
    agent_item = page.locator(".agent-item").first
    if agent_item.count() > 0:
        agent_item.click()
        page.wait_for_timeout(1000)

        # Wait for "Full Dialog" button to be visible
        full_dialog_btn = page.locator("#full-dialog-btn")
        expect(full_dialog_btn).to_be_visible()

        # Click Full Dialog button and wait for new page
        with page.context.expect_page() as new_page_info:
            full_dialog_btn.click()

        # Get the new page
        new_page = new_page_info.value
        new_page.wait_for_load_state("networkidle")

        # Verify it's plain text (not HTML with buttons)
        content = new_page.content()
        print(f"\n📄 Dialog page content preview:\n{content[:500]}\n")

        # Check that it's plain text, not HTML dialog.html
        # Plain text should contain "---" markers for user/assistant
        page_text = new_page.locator("body").text_content()

        # Take screenshot
        screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        screenshot_path = screenshots_dir / f"full_dialog_{int(time.time())}.png"
        new_page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"📸 Screenshot saved: {screenshot_path}")

        # Verify it's plain text format
        # Should contain either:
        # - "--- User ---" / "--- Assistant ---" markers
        # - Or "(No conversation history available)"
        has_markers = "---" in page_text or "No conversation history" in page_text

        # Should NOT contain HTML UI elements from dialog.html
        has_html_ui = "dialog-container" in content or "message message-" in content or "btn btn-" in content

        print(f"\n✅ Plain text markers found: {has_markers}")
        print(f"❌ HTML UI elements found: {has_html_ui}")
        print(f"📝 Page text preview (first 500 chars):\n{page_text[:500]}\n")

        # Assertions
        assert has_markers, "Dialog should contain plain text markers or 'No conversation history' message"
        assert not has_html_ui, "Dialog should NOT contain HTML UI elements (buttons, containers, etc.)"

        print("✅ Test passed! Full Dialog shows plain text, not HTML UI")

        # Close the new page
        new_page.close()

    else:
        print("⚠️ No agents found, skipping test")
        pytest.skip("No agents available for testing")


@pytest.mark.ui
def test_full_dialog_api_endpoint(page: Page, server: str):
    """Test that the /api/agents/{name}/dialog-text endpoint returns plain text."""
    # Navigate to main page to get an agent name
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)

    # Get first agent name
    agent_item = page.locator(".agent-item").first
    if agent_item.count() > 0:
        agent_name_el = agent_item.locator(".agent-item-name")
        agent_name = agent_name_el.text_content()

        print(f"\n🔍 Testing dialog-text endpoint for agent: {agent_name}")

        # Navigate directly to the dialog-text endpoint
        dialog_url = f"{server}/api/agents/{agent_name}/dialog-text"
        page.goto(dialog_url)
        page.wait_for_load_state("networkidle")

        # Get the page content
        body_text = page.locator("body").text_content()

        # Take screenshot
        screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        screenshot_path = screenshots_dir / f"dialog_endpoint_{int(time.time())}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"📸 Screenshot saved: {screenshot_path}")

        print(f"\n📝 Dialog text preview (first 500 chars):\n{body_text[:500]}\n")

        # Verify it's plain text
        has_text_markers = "---" in body_text or "No conversation history" in body_text

        # Verify response headers indicate plain text
        # Note: Playwright doesn't easily expose response headers, so we check content

        assert has_text_markers, f"Endpoint should return plain text with markers. Got: {body_text[:200]}"

        print(f"✅ Endpoint returns plain text: {len(body_text)} characters")

    else:
        print("⚠️ No agents found, skipping test")
        pytest.skip("No agents available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
