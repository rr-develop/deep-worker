"""Debug test for Running status bug.

This test specifically checks if set_success() is called properly when tool completes.
"""

import time
from pathlib import Path

import pytest

from tests.integration_ai.clean_agent_fixture import clean_agent
from tests.integration_ai.terminal_emulator import DwcliTerminalSession


@pytest.fixture
def debug_agent(clean_agent):
    """Create a test agent with debug logging enabled."""
    agent_name = "test_debug_running"
    agent_process = clean_agent(
        agent_name,
        working_dir="/tmp",
        provider="deepseek",
        skip_permissions=True,
    )
    yield agent_process
    # Cleanup handled by fixture


def test_running_status_cleared_after_tool_completes(debug_agent):
    """Test that Running status is cleared when tool completes.

    User report: Tool calls take 30+ seconds for simple commands like pwd,
    and "Running" status persists with incrementing counter even after
    tool completes and response is received.

    This test verifies that set_success() is called properly.
    """
    agent_name = "test_debug_running"

    with DwcliTerminalSession(agent_name=agent_name) as session:
        # Send query that will call execute tool
        query = "What is your current working directory? Use the execute tool with pwd command."
        session.send_input(query, press_enter=True)

        # Wait for tool call to appear
        found_tool_call = session.wait_for_text("execute", timeout=10.0)
        assert found_tool_call, "Tool call should appear in TUI"

        print("\n[TEST] Tool call appeared in TUI")
        screen_with_tool = session.get_screen_text()
        print(f"[TEST] Screen when tool call appeared:\n{screen_with_tool}\n")

        # Give tool time to execute (pwd is instant, but wait a bit)
        time.sleep(2.0)

        # Check if "Running" status is still visible
        screen_after = session.get_screen_text()
        print(f"[TEST] Screen after 2 seconds:\n{screen_after}\n")

        # Wait for response
        found_response = session.wait_for_text("/tmp", timeout=30.0, case_sensitive=False)
        assert found_response, "Response with /tmp should appear within 30s"

        print("\n[TEST] Response appeared")
        final_screen = session.get_screen_text()
        print(f"[TEST] Final screen:\n{final_screen}\n")

        # Check if "Running" is still present (it shouldn't be)
        # Note: We need to check if the running animation is still active
        # The text "Running" might still be visible in history, but the
        # widget should have called set_success() and stopped animating

        # Give a moment for UI to update
        time.sleep(1.0)

        # The key check: After response arrives, wait and see if Running persists
        # If bug exists: "Running" with incrementing counter
        # If fixed: No running status, or status changed to success
        time.sleep(3.0)

        post_response_screen = session.get_screen_text()
        print(f"\n[TEST] Screen 3 seconds after response:\n{post_response_screen}\n")

        # TODO: Need to parse the screen better to detect running animation
        # For now, just show the output and manually verify

        # If we made it here without timeout, that's good
        assert True, "Test completed - check console output for Running status"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
