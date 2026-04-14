"""Real UI test with WebSocket connection."""

import pytest
import time
from pathlib import Path
from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ui
def test_send_message_real_ui(page: Page):
    """Send message through REAL UI with WebSocket and verify response appears."""
    server = "http://127.0.0.1:7889"
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    print("\n🌐 Opening page...")
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for WebSocket to connect and agents to load
    print("⏳ Waiting for WebSocket and agents (10s)...")
    time.sleep(10)

    screenshot1 = screenshots_dir / f"real_ui_step1_{int(time.time())}.png"
    page.screenshot(path=str(screenshot1))
    print(f"📸 {screenshot1}")

    # Select agent
    print("\n🤖 Selecting agent...")
    agent_items = page.locator(".agent-item")
    expect(agent_items.first).to_be_visible(timeout=5000)
    agent_items.first.click()
    time.sleep(2)

    screenshot2 = screenshots_dir / f"real_ui_step2_{int(time.time())}.png"
    page.screenshot(path=str(screenshot2))
    print(f"📸 {screenshot2}")

    # Type and send message
    print("\n✍️ Typing message '9+9'...")
    message_input = page.locator("#message-input")
    send_button = page.locator("#send-btn")

    message_input.fill("9+9")
    time.sleep(1)

    screenshot3 = screenshots_dir / f"real_ui_step3_{int(time.time())}.png"
    page.screenshot(path=str(screenshot3))
    print(f"📸 {screenshot3}")

    print("\n📤 Clicking Send...")
    send_button.click()
    time.sleep(2)

    screenshot4 = screenshots_dir / f"real_ui_step4_{int(time.time())}.png"
    page.screenshot(path=str(screenshot4))
    print(f"📸 {screenshot4}")

    # Wait for response (agent processes + WebSocket delivers + UI updates)
    print("\n⏳ Waiting for agent response (15s)...")
    time.sleep(15)

    screenshot5 = screenshots_dir / f"real_ui_step5_{int(time.time())}.png"
    page.screenshot(path=str(screenshot5))
    print(f"📸 {screenshot5}")

    # Check Last Response section
    last_response = page.locator(".last-response-content")
    if last_response.count() > 0:
        response_text = last_response.text_content()
        print(f"\n📝 Last Response: {response_text[:200]}")

        if "18" in response_text:
            print("✅ FOUND '18' in Last Response!")
        else:
            print(f"❌ '18' NOT found. Content: {response_text}")
    else:
        print("❌ .last-response-content not found")

    # Also check full page text
    page_text = page.locator("body").text_content()
    if "18" in page_text:
        print("✅ FOUND '18' somewhere on page")
    else:
        print("❌ '18' NOT found anywhere on page")

    print(f"\n📸 All screenshots:")
    print(f"  1. {screenshot1.name}")
    print(f"  2. {screenshot2.name}")
    print(f"  3. {screenshot3.name}")
    print(f"  4. {screenshot4.name}")
    print(f"  5. {screenshot5.name}")
