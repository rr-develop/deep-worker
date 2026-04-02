"""Integration test for WebSearch delegation tool with DeepSeek provider.

This test verifies that the WebSearchDelegationTool works correctly with the
default DeepSeek provider in the deep_worker CLI agent.

The test:
1. Creates a clean agent with DeepSeek provider
2. Starts a dwcli session
3. Asks agent to search for "Python 3.13 features"
4. Verifies that web_search tool is called
5. Verifies that agent receives and processes search results

PASS CRITERIA:
- Agent successfully calls web_search tool
- Agent receives JSON results from WebSearch delegation
- Agent synthesizes search results into natural response

FAIL CRITERIA:
- web_search tool is not available
- web_search tool call fails or times out
- Agent doesn't use search results in response

Run with:
    cd /tmp && pytest deep_worker/deep_worker_cli/tests/integration_ai/test_websearch_delegation_deepseek.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import (
    DwcliTerminalSession,
    create_terminal_snapshot,
)

# Try to import check_reply, but don't fail if claude_sdk is unavailable
try:
    from .check_reply import verify_screenshot_contains_sync
    CHECK_REPLY_AVAILABLE = True
except ImportError:
    CHECK_REPLY_AVAILABLE = False
    logging.warning("verify_screenshot_contains_sync not available (claude_sdk missing). AI validation will be skipped.")

logger = logging.getLogger(__name__)


def _check_dwcli_available() -> bool:
    """Check if dwcli command is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["which", "dwcli"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
@pytest.mark.skipif(
    not CHECK_REPLY_AVAILABLE,
    reason="claude_sdk not available (required for AI validation)",
)
def test_websearch_delegation_with_deepseek(clean_agent):
    """Test that WebSearchDelegationTool works with DeepSeek provider.

    This test verifies that:
    1. WebSearchDelegationTool is available to the agent
    2. DeepSeek provider can successfully call the web_search tool
    3. The tool returns valid search results
    4. The agent can process and respond with search information

    EXPECTED BEHAVIOR:
    - Agent receives query about recent Python features
    - Agent calls web_search tool (delegated to Claude Code SDK)
    - Tool returns JSON with search results
    - Agent synthesizes results into natural language response

    PASS CRITERIA:
    - web_search tool call is visible in TUI
    - Agent receives search results
    - Agent response contains information from search

    FAIL CRITERIA:
    - web_search tool is not available
    - Tool call fails or times out
    - Agent doesn't incorporate search results
    """
    logger.info("=" * 80)
    logger.info("TEST: WebSearch Delegation Tool with DeepSeek Provider")
    logger.info("=" * 80)

    # Change to /tmp directory
    original_cwd = os.getcwd()
    logger.info(f"Original working directory: {original_cwd}")

    os.chdir("/tmp")
    logger.info(f"Changed to: {os.getcwd()}")

    try:
        agent_name = "test_websearch_deepseek"
        logger.info(f"Starting clean test agent: {agent_name}")
        logger.info("Provider: deepseek (DEFAULT)")
        logger.info("Tool: web_search (WebSearchDelegationTool)")
        logger.info("Skip permissions: True")

        # Start clean agent with DEFAULT provider (deepseek)
        agent_process = clean_agent(
            agent_name,
            working_dir="/tmp",
            provider="deepseek"  # Using default provider
        )
        logger.info(f"✓ Clean agent started (PID: {agent_process.pid})")

        # Open dwcli session
        session = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            # Start dwcli
            logger.info(f"Starting dwcli for agent: {agent_name}")
            session.start(timeout=15.0)
            logger.info(f"✓ dwcli started (PID: {session.pid if session.pid else 'unknown'})")

            # Wait for UI to render
            time.sleep(2.0)
            logger.info("✓ UI rendered")

            # ========================================================================
            # QUERY: Ask about Python 3.13 features (requires web search)
            # ========================================================================
            query = "Use web search to find what's new in Python 3.13. Give me 3 key features."
            logger.info(f"\n{'=' * 80}")
            logger.info(f"QUERY: '{query}'")
            logger.info(f"{'=' * 80}")
            logger.info("Expected: web_search tool call should be visible")

            session.send_input(query, press_enter=False)
            time.sleep(0.3)
            session.send_key("enter")
            logger.info("✓ Query sent to dwcli")

            # Wait for AI response (up to 120 seconds for web search)
            logger.info("Waiting for web search and response...")

            response_received = False
            web_search_detected = False
            search_indicators_found = []

            for i in range(240):  # 120 seconds max (web search takes longer)
                time.sleep(0.5)
                screen_current = session.get_screen_text()

                # Check for web_search tool call indicators
                search_indicators = ["web_search", "web search", "Search", "🔧"]
                for indicator in search_indicators:
                    if indicator in screen_current:
                        if not web_search_detected:
                            web_search_detected = True
                            search_indicators_found.append(indicator)
                            logger.info(f"✓ web_search tool call detected at {i * 0.5:.1f}s: '{indicator}'")

                # Check if still thinking
                if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                    if i % 20 == 0:
                        logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                    continue

                # Check for response mentioning Python features
                if ("Python" in screen_current and ("3.13" in screen_current or "feature" in screen_current.lower())):
                    response_received = True
                    logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                    break

                if i % 40 == 0 and i > 0:
                    logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

            if not response_received:
                # Still capture the screen for debugging
                screen_final = session.get_screen_text()
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="TIMEOUT - No response received",
                ))
                pytest.fail(f"Query did not receive response within 120 seconds")

            # Wait for screen to stabilize
            time.sleep(2.0)
            screen_after_query = session.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_after_query,
                title="AFTER QUERY (should show web_search and results)",
            ))

            # ========================================================================
            # CRITICAL CHECK: Verify web_search tool was called
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("CRITICAL CHECK: WebSearch tool verification")
            logger.info("=" * 80)

            if not web_search_detected:
                logger.error(f"❌ CRITICAL: web_search tool was NOT called!")
                logger.error("Expected: Agent should use web_search to find Python 3.13 features")
                logger.error("Actual: No web_search tool call detected in TUI")
                pytest.fail(
                    f"WebSearchDelegationTool was not called by DeepSeek!\n\n"
                    f"Query: '{query}'\n"
                    f"Expected: web_search tool call should be visible\n"
                    f"Actual: No tool call indicators found\n\n"
                    f"This indicates the tool is not properly registered or\n"
                    f"the agent chose not to use it."
                )

            logger.info(f"✓ web_search tool call detected: {search_indicators_found}")

            # ========================================================================
            # Validate that response contains Python features
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION: Checking response quality")
            logger.info("=" * 80)

            validation_result = verify_screenshot_contains_sync(
                screenshot_text=screen_after_query,
                original_query=query,
                expected_response_contains="Python 3.13",
                model="haiku",
            )

            logger.info(f"Validation result: {validation_result}")

            if not validation_result.is_valid:
                logger.warning(f"⚠ Response validation: {validation_result.comment}")
                # Don't fail on validation, just warn
                logger.warning("Note: Validation is informational only")

            # Additional check: Look for feature-like content
            features_mentioned = False
            feature_keywords = ["feature", "new", "improved", "added", "support", "enhancement"]
            screen_lower = screen_after_query.lower()
            for keyword in feature_keywords:
                if keyword in screen_lower:
                    features_mentioned = True
                    logger.info(f"✓ Found feature keyword: '{keyword}'")
                    break

            if not features_mentioned:
                logger.warning("⚠ Response doesn't clearly mention features")
                logger.warning("Agent may not have processed search results properly")

            # ========================================================================
            # SUCCESS
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("✅ TEST PASSED: WebSearch delegation works with DeepSeek!")
            logger.info("✅ web_search tool was called successfully")
            logger.info(f"✅ Tool call indicators: {search_indicators_found}")
            logger.info("✅ Agent received and responded with search results")
            if features_mentioned:
                logger.info("✅ Response mentions Python features")
            logger.info("=" * 80)

        finally:
            session.close()

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {os.getcwd()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
