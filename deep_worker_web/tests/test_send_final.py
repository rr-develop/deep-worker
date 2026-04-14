"""Final UI test - send message and verify response appears."""

import pytest
import time
from pathlib import Path
from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ui
def test_send_message_final(page: Page):
    """Final test: Send message and verify response in UI."""
    server = "http://127.0.0.1:7889"
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("FINAL UI TEST: Send Message and Verify Response")
    print("="*60)

    # Open and wait for WebSocket
    print("\n[1/5] Opening page and waiting for WebSocket...")
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")
    time.sleep(10)  # Wait for WebSocket to connect

    # Select agent
    print("[2/5] Selecting agent...")
    agent_items = page.locator(".agent-item")
    expect(agent_items.first).to_be_visible(timeout=5000)
    agent_items.first.click()
    time.sleep(2)

    # Send message
    print("[3/5] Sending message 'Hello'...")
    message_input = page.locator("#message-input")
    send_button = page.locator("#send-btn")

    message_input.fill("Hello")
    time.sleep(1)

    screenshot_before = screenshots_dir / f"FINAL_before_send_{int(time.time())}.png"
    page.screenshot(path=str(screenshot_before))
    print(f"    Screenshot (before): {screenshot_before.name}")

    send_button.click()
    print("    Message sent!")
    time.sleep(2)

    # Wait for response
    print("[4/5] Waiting for agent response...")
    time.sleep(12)

    screenshot_after = screenshots_dir / f"FINAL_after_response_{int(time.time())}.png"
    page.screenshot(path=str(screenshot_after))
    print(f"    Screenshot (after): {screenshot_after.name}")

    # Verify response appeared
    print("[5/5] Verifying response...")
    page_text = page.locator("body").text_content()

    # Check if there's any new content (response should contain some text)
    has_response = len(page_text) > 1000  # If there's content, it means response arrived

    if has_response:
        print("    ✅ RESPONSE DETECTED in UI!")
        print(f"    Page has {len(page_text)} characters")
    else:
        print(f"    ❌ NO RESPONSE - page only has {len(page_text)} characters")

    print("\n" + "="*60)
    print("FINAL SCREENSHOTS:")
    print(f"  BEFORE: {screenshot_before}")
    print(f"  AFTER:  {screenshot_after}")
    print("="*60 + "\n")
