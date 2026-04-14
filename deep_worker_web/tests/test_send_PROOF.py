"""PROOF test - send unique message to see response change."""

import pytest
import time
from pathlib import Path
from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ui
def test_send_unique_message_proof(page: Page):
    """Send UNIQUE message to prove response actually updates."""
    server = "http://127.0.0.1:7889"
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    unique_msg = f"What is 100 + {timestamp % 100}?"

    print(f"\n{'='*70}")
    print(f"PROOF TEST: Sending UNIQUE message to verify response updates")
    print(f"Message: {unique_msg}")
    print(f"{'='*70}\n")

    # Open and wait
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")
    time.sleep(10)

    # Select agent
    agent_items = page.locator(".agent-item")
    expect(agent_items.first).to_be_visible(timeout=5000)
    agent_items.first.click()
    time.sleep(2)

    # Capture BEFORE
    screenshot_before = screenshots_dir / f"PROOF_BEFORE_{timestamp}.png"
    page.screenshot(path=str(screenshot_before))

    # Get current Last Response text
    last_resp_el = page.locator("#last-response-content")
    text_before = ""
    if last_resp_el.count() > 0:
        text_before = last_resp_el.text_content() or ""

    print(f"[BEFORE] Last Response: {text_before[:100]}")

    # Send unique message
    message_input = page.locator("#message-input")
    send_button = page.locator("#send-btn")
    message_input.fill(unique_msg)
    time.sleep(1)
    send_button.click()
    print(f"[SENT] Message: {unique_msg}")
    time.sleep(2)

    # Wait for response
    print("[WAITING] For agent response (15 seconds)...")
    time.sleep(15)

    # Capture AFTER
    screenshot_after = screenshots_dir / f"PROOF_AFTER_{timestamp}.png"
    page.screenshot(path=str(screenshot_after))

    # Get new Last Response text
    text_after = ""
    if last_resp_el.count() > 0:
        text_after = last_resp_el.text_content() or ""

    print(f"[AFTER] Last Response: {text_after[:100]}")

    # Check if response changed
    response_changed = text_after != text_before and len(text_after) > 0

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Before text: {text_before[:80]}")
    print(f"  After text:  {text_after[:80]}")
    print(f"  Changed: {response_changed}")
    print(f"\nSCREENSHOTS:")
    print(f"  BEFORE: {screenshot_before}")
    print(f"  AFTER:  {screenshot_after}")
    print(f"{'='*70}\n")

    assert response_changed, f"Response did not change! Before='{text_before[:50]}' After='{text_after[:50]}'"
