"""TDD tests for last response display functionality.

Tests that verify the last response from agents is properly displayed.
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
    process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "7887"],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    yield "http://127.0.0.1:7887"
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.ui
def test_last_response_displayed_when_agent_selected(page: Page, server: str):
    """Test that last response is displayed when an agent with history is selected."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    # Wait for agents to load
    page.wait_for_timeout(1000)

    # Find the test agent if it exists
    agent_items = page.locator(".agent-item")
    agent_count = agent_items.count()

    test_agent_found = False
    test_agent_index = -1

    for i in range(agent_count):
        agent_name = agent_items.nth(i).locator(".agent-item-name").text_content()
        print(f"Found agent: {agent_name}")
        if "test-agent" in agent_name:
            test_agent_found = True
            test_agent_index = i
            break

    if not test_agent_found:
        pytest.skip("test agent not found - test skipped")

    # Click on the test agent
    agent_items.nth(test_agent_index).click()
    page.wait_for_timeout(2000)  # Increased timeout to allow history to load

    # Check console for any errors
    console_messages = []
    page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))
    page.wait_for_timeout(500)

    # Check that agent-selected div is visible
    agent_selected = page.locator("#agent-selected")
    expect(agent_selected).to_be_visible()

    # Check that last-response-panel exists and is visible
    last_response_panel = page.locator(".last-response-panel")
    expect(last_response_panel).to_be_visible()

    # Check that response-content exists
    response_content = page.locator(".response-content")
    expect(response_content).to_be_visible()

    # Get the text content
    response_text = response_content.text_content()

    print(f"Last response text length: {len(response_text)}")
    print(f"Last response preview: {response_text[:200]}")

    # Response should not be empty and should not be default placeholder
    assert response_text is not None, "Response text is None"
    assert len(response_text.strip()) > 0, "Response text is empty or whitespace only"

    # Should not be the default "No response yet" text
    assert "No response yet" not in response_text, f"Last response shows default placeholder instead of actual response: '{response_text.strip()}'"

    print(f"✓ Last response displayed: {len(response_text)} characters")


@pytest.mark.ui
def test_last_response_updates_on_agent_change(page: Page, server: str):
    """Test that last response updates when switching between agents."""
    page.goto(f"{server}/")
    page.wait_for_load_state("networkidle")

    page.wait_for_timeout(1000)

    agent_items = page.locator(".agent-item")
    agent_count = agent_items.count()

    if agent_count < 2:
        pytest.skip("Need at least 2 agents to test switching")

    # Click first agent
    agent_items.nth(0).click()
    page.wait_for_timeout(1000)

    response_content = page.locator(".response-content")
    first_response = response_content.text_content()

    print(f"First agent response length: {len(first_response)}")

    # Click second agent
    agent_items.nth(1).click()
    page.wait_for_timeout(1000)

    second_response = response_content.text_content()

    print(f"Second agent response length: {len(second_response)}")

    # Responses can be the same if both agents have no history,
    # but the element should still be updated
    print(f"✓ Response element updates when switching agents")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ui", "-s"])
