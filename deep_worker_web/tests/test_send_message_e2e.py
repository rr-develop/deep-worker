"""E2E test for send message functionality.

Tests the complete flow:
1. Open browser
2. Select agent
3. Send message "5+5"
4. Verify response "10" appears
5. Take screenshots at each step
"""

import pytest
import time
from pathlib import Path
from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ui
def test_send_message_e2e(page: Page):
    """E2E test: send message and verify response appears."""
    server = "http://127.0.0.1:7889"
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Step 1: Open page
    print("\n📱 Step 1: Opening page...")
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    screenshot1 = screenshots_dir / f"e2e_step1_opened_{int(time.time())}.png"
    page.screenshot(path=str(screenshot1))
    print(f"   Screenshot: {screenshot1}")

    # Step 2: Wait for agent to appear
    print("\n📱 Step 2: Waiting for agents...")
    agent_items = page.locator(".agent-item")
    expect(agent_items.first).to_be_visible(timeout=10000)
    time.sleep(1)

    screenshot2 = screenshots_dir / f"e2e_step2_agents_{int(time.time())}.png"
    page.screenshot(path=str(screenshot2))
    print(f"   Screenshot: {screenshot2}")

    # Step 3: Select agent
    print("\n📱 Step 3: Selecting agent...")
    agent_items.first.click()
    time.sleep(2)

    screenshot3 = screenshots_dir / f"e2e_step3_selected_{int(time.time())}.png"
    page.screenshot(path=str(screenshot3))
    print(f"   Screenshot: {screenshot3}")

    # Step 4: Send message
    print("\n📱 Step 4: Sending message '5+5'...")
    message_input = page.locator("#message-input")
    send_button = page.locator("#send-btn")

    expect(message_input).to_be_visible()
    expect(send_button).to_be_visible()

    message_input.fill("5+5")
    screenshot4 = screenshots_dir / f"e2e_step4_typed_{int(time.time())}.png"
    page.screenshot(path=str(screenshot4))
    print(f"   Screenshot: {screenshot4}")

    send_button.click()
    time.sleep(1)

    screenshot5 = screenshots_dir / f"e2e_step5_sent_{int(time.time())}.png"
    page.screenshot(path=str(screenshot5))
    print(f"   Screenshot: {screenshot5}")

    # Step 5: Wait for response
    print("\n📱 Step 5: Waiting for response...")
    time.sleep(8)  # Wait for agent to process and response to arrive

    screenshot6 = screenshots_dir / f"e2e_step6_response_{int(time.time())}.png"
    page.screenshot(path=str(screenshot6))
    print(f"   Screenshot: {screenshot6}")

    # Step 6: Verify response
    print("\n📱 Step 6: Verifying response...")
    conversation = page.locator("#conversation")
    expect(conversation).to_be_visible()

    # Get all text content
    conv_text = conversation.text_content()
    print(f"\n📝 Conversation content (last 500 chars):\n{conv_text[-500:]}")

    # Check for user message
    assert "5+5" in conv_text, "User message '5+5' not found in conversation"

    # Check for response "10"
    assistant_messages = page.locator(".message.message-assistant")
    found_response = False
    for i in range(assistant_messages.count()):
        msg_text = assistant_messages.nth(i).text_content()
        if "10" in msg_text:
            found_response = True
            print(f"✅ Found response containing '10': {msg_text[:100]}")
            break

    assert found_response, "Response '10' not found in conversation"

    print(f"\n✅ E2E TEST PASSED!")
    print(f"\n📸 All screenshots saved to: {screenshots_dir}")
    print(f"   - Step 1 (opened): {screenshot1.name}")
    print(f"   - Step 2 (agents): {screenshot2.name}")
    print(f"   - Step 3 (selected): {screenshot3.name}")
    print(f"   - Step 4 (typed): {screenshot4.name}")
    print(f"   - Step 5 (sent): {screenshot5.name}")
    print(f"   - Step 6 (response): {screenshot6.name}")
