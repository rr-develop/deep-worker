"""Integration test for sending messages to agents via UI.

This test verifies that:
1. User can send a message to an agent via the UI
2. Agent processes the message and responds
3. Response appears in the conversation history
4. UI updates to show the new message
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
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "7888"],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    yield "http://127.0.0.1:7888"

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.ui
def test_send_message_via_ui(page: Page, server: str):
    """Test that sending a message via UI works and shows response."""
    # Navigate to the app
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for agents to be discovered (can take up to 5s)
    agent_cards = page.locator(".agent-card")
    expect(agent_cards.first).to_be_visible(timeout=15000)

    # Wait a bit more for all agents to appear
    time.sleep(2)

    # Try to find test_dw agent, otherwise use first available
    test_dw_card = agent_cards.filter(has_text="test_dw").first
    if test_dw_card.count() > 0:
        test_dw_card.click()
        agent_name = "test_dw"
    else:
        # Use first available agent
        agent_cards.first.click()
        agent_name = agent_cards.first.text_content().split('\n')[0]

    print(f"\n🤖 Selected agent: {agent_name}")

    # Wait for agent to be selected
    time.sleep(1)

    # Get the message input and send button
    message_input = page.locator("#message-input")
    send_button = page.locator("#send-button")

    expect(message_input).to_be_visible()
    expect(send_button).to_be_visible()

    # Type a simple math question
    test_message = "10+10"
    message_input.fill(test_message)

    # Click send button
    send_button.click()

    # Input should be cleared after sending
    expect(message_input).to_have_value("")

    # Wait for response to arrive (agent should respond quickly to simple math)
    # The response will trigger history reload via WebSocket
    time.sleep(5)

    # Check that the message appears in the conversation history
    conversation_panel = page.locator("#conversation")
    expect(conversation_panel).to_be_visible()

    # Look for the user message in conversation
    user_messages = page.locator(".message.message-user")
    found_message = False
    for i in range(user_messages.count()):
        msg_text = user_messages.nth(i).text_content()
        if test_message in msg_text:
            found_message = True
            break

    assert found_message, f"User message '{test_message}' not found in conversation history"

    # Look for assistant response
    assistant_messages = page.locator(".message.message-assistant")
    found_response = False
    for i in range(assistant_messages.count()):
        msg_text = assistant_messages.nth(i).text_content()
        if "20" in msg_text:  # Expected answer to 10+10
            found_response = True
            break

    assert found_response, f"Assistant response containing '20' not found in conversation history"

    # Take screenshot for verification
    screenshots_dir = _PROJECT_ROOT / "tests" / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    screenshot_path = screenshots_dir / f"send_message_test_{int(time.time())}.png"
    page.screenshot(path=str(screenshot_path))
    print(f"\n📸 Screenshot saved: {screenshot_path}")

    print(f"\n✅ Message sent and response received successfully")
    print(f"   User message: {test_message}")
    print(f"   Agent response contains: 20")


@pytest.mark.ui
def test_send_message_api_endpoint(page: Page, server: str):
    """Test the send message API endpoint directly."""
    # Navigate to the app
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for agents to be discovered
    time.sleep(2)

    # Send message via API using JavaScript
    result = page.evaluate("""
        async () => {
            const response = await fetch('/api/agents/test_dw/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: '7+8' })
            });
            return await response.json();
        }
    """)

    # Check that API returned success
    assert result.get("success") == True, f"API call failed: {result}"
    print(f"\n✅ API endpoint returned success: {result.get('message')}")

    # Wait for response to arrive via WebSocket
    time.sleep(5)

    # Select the agent to view conversation
    agent_cards = page.locator(".agent-card")
    test_dw_card = agent_cards.filter(has_text="test_dw").first
    test_dw_card.click()
    time.sleep(2)

    # Check that response appears in conversation
    conversation_panel = page.locator("#conversation")
    expect(conversation_panel).to_be_visible()

    # Look for the response "15"
    assistant_messages = page.locator(".message.message-assistant")
    found_response = False
    for i in range(assistant_messages.count()):
        msg_text = assistant_messages.nth(i).text_content()
        if "15" in msg_text:
            found_response = True
            break

    assert found_response, "Assistant response '15' not found after API call"

    print(f"✅ Message sent via API and response received")
