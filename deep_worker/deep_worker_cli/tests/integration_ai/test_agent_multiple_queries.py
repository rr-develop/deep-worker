"""Test to verify that agent correctly handles multiple sequential queries.

This test verifies CRITICAL behavior:
- Agent must respond to EACH new query independently
- Agent must NOT reuse answers from previous queries
- Each response must be relevant to its specific query

REGRESSION TEST for bug where agent with claude-code provider:
- First query: "2+2" → correctly answers "4"
- Second query: "3+3" → incorrectly answers "4" (repeats first answer!)

The agent "sticks" to the first query and ignores subsequent ones.

Run with:
    pytest deep_worker/deep_worker_cli/tests/integration_ai/test_agent_multiple_queries.py -v -s
"""

from __future__ import annotations

import logging
import subprocess
import time

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401

logger = logging.getLogger(__name__)


def _check_dwcli_available() -> bool:
    """Check if dwcli command is available."""
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
def test_agent_handles_multiple_queries_independently(clean_agent):
    """Test that agent responds to each query independently.

    CRITICAL BUG TEST: With claude-code provider, agent "sticks" to first query
    and keeps answering it, ignoring subsequent queries.

    Test sequence:
    1. Start agent with claude-code provider
    2. Send first query: "2+2"
    3. Verify response contains "4"
    4. Send second query: "3+3"
    5. Verify response contains "6" (NOT "4"!)
    6. Send third query: "5+5"
    7. Verify response contains "10" (NOT "4" or "6"!)

    PASS CRITERIA:
    - First response contains "4"
    - Second response contains "6" (NOT "4")
    - Third response contains "10" (NOT "4" or "6")
    - Each response is relevant to its specific query

    FAIL CRITERIA:
    - Second response contains "4" instead of "6"
    - Third response contains previous answer instead of "10"
    - Agent reuses answers from earlier queries
    """
    logger.info("=" * 80)
    logger.info("TEST: Agent multiple queries independence verification")
    logger.info("=" * 80)

    agent_name = "test_multi_query"

    # Step 1: Start agent with claude-code provider using fixture
    logger.info(f"Starting agent '{agent_name}' with claude-code provider...")

    agent_process = clean_agent(agent_name, working_dir="/tmp", provider="claude-code")
    logger.info(f"✓ Agent '{agent_name}' started (PID: {agent_process.pid})")

    # Additional wait for agent to be fully ready
    time.sleep(2.0)

    # Step 2: Send first query
    query1 = "2+2"
    logger.info(f"\nQuery 1: '{query1}'")

    result1 = subprocess.run(
        ["dwcli", agent_name, "-p", query1],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result1.returncode != 0:
        pytest.fail(f"Query 1 failed: {result1.stderr}")

    response1 = result1.stdout.strip()
    logger.info(f"Response 1: '{response1}'")

    # Step 3: Verify first response contains "4"
    if "4" not in response1:
        pytest.fail(
            f"Query 1 validation failed!\n"
            f"Query: '{query1}'\n"
            f"Response: '{response1}'\n"
            f"Expected: Response containing '4'\n"
            f"Actual: '4' not found in response"
        )

    logger.info(f"✓ Query 1 validation passed: Response contains '4'")

    # Step 4: Send second query
    time.sleep(2.0)  # Brief pause between queries
    query2 = "3+3"
    logger.info(f"\nQuery 2: '{query2}'")

    result2 = subprocess.run(
        ["dwcli", agent_name, "-p", query2],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result2.returncode != 0:
        pytest.fail(f"Query 2 failed: {result2.stderr}")

    response2 = result2.stdout.strip()
    logger.info(f"Response 2: '{response2}'")

    # Step 5: CRITICAL CHECK - Verify second response contains "6" (NOT "4"!)
    # First check if it contains "4" (the bug)
    if "4" in response2 and "6" not in response2:
        logger.error("\n" + "=" * 80)
        logger.error("❌ BUG DETECTED: Agent repeating first answer!")
        logger.error("=" * 80)
        logger.error(f"Query 1: '{query1}' → Response: '{response1}'")
        logger.error(f"Query 2: '{query2}' → Response: '{response2}' (WRONG! Contains '4' instead of '6')")
        logger.error("\nAgent is 'stuck' on first query and ignoring subsequent queries!")

        pytest.fail(
            f"Agent multiple queries bug detected!\\n\\n"
            f"Query 1: '{query1}'\\n"
            f"Response 1: '{response1}' ✓ Correct (contains '4')\\n\\n"
            f"Query 2: '{query2}'\\n"
            f"Response 2: '{response2}' ✗ WRONG!\\n\\n"
            f"Expected: Response containing '6'\\n"
            f"Actual: Response contains '4' (answer from Query 1)\\n\\n"
            f"The agent is repeating the first answer instead of responding to the new query.\\n"
            f"This indicates the agent is 'stuck' on the first query."
        )

    if "6" not in response2:
        logger.error("\n" + "=" * 80)
        logger.error("❌ Query 2 validation failed!")
        logger.error("=" * 80)
        logger.error(f"Query 2: '{query2}'")
        logger.error(f"Response 2: '{response2}'")

        # Additional check: does it contain the wrong answer?
        if "4" in response2:
            logger.error("\n⚠ Response contains '4' (answer from Query 1) instead of '6'")
            logger.error("This confirms the agent is stuck on the first query!")

        pytest.fail(
            f"Query 2 validation failed!\\n"
            f"Query: '{query2}'\\n"
            f"Response: '{response2}'\\n"
            f"Expected: Response containing '6'\\n"
            f"Actual: '6' not found in response\\n"
            f"Agent may be repeating answer from previous query."
        )

    logger.info(f"✓ Query 2 validation passed: Response contains '6'")

    # Step 6: Send third query (additional verification)
    time.sleep(2.0)
    query3 = "5+5"
    logger.info(f"\nQuery 3: '{query3}'")

    result3 = subprocess.run(
        ["dwcli", agent_name, "-p", query3],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result3.returncode != 0:
        pytest.fail(f"Query 3 failed: {result3.stderr}")

    response3 = result3.stdout.strip()
    logger.info(f"Response 3: '{response3}'")

    # Step 7: Verify third response contains "10" (NOT "4" or "6"!)
    # Check for previous answers (the bug)
    if ("4" in response3 or "6" in response3) and "10" not in response3:
        logger.error("\n" + "=" * 80)
        logger.error("❌ BUG DETECTED: Agent repeating previous answer!")
        logger.error("=" * 80)
        logger.error(f"Query 1: '{query1}' → '{response1}'")
        logger.error(f"Query 2: '{query2}' → '{response2}'")
        logger.error(f"Query 3: '{query3}' → '{response3}' (WRONG! Contains previous answer)")

        pytest.fail(
            f"Agent stuck on previous query!\\n\\n"
            f"Query 3: '{query3}'\\n"
            f"Response 3: '{response3}' ✗ WRONG!\\n\\n"
            f"Expected: Response containing '10'\\n"
            f"Actual: Response contains previous answer ('4' or '6')\\n\\n"
            f"The agent continues to reuse answers from earlier queries."
        )

    if "10" not in response3:
        pytest.fail(
            f"Query 3 validation failed!\\n"
            f"Query: '{query3}'\\n"
            f"Response: '{response3}'\\n"
            f"Expected: Response containing '10'\\n"
            f"Actual: '10' not found in response"
        )

    logger.info(f"✓ Query 3 validation passed: Response contains '10'")

    # All queries passed!
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST PASSED: Agent correctly handles multiple queries independently")
    logger.info("=" * 80)
    logger.info(f"Query 1: '{query1}' → Response contains '4' ✓")
    logger.info(f"Query 2: '{query2}' → Response contains '6' ✓")
    logger.info(f"Query 3: '{query3}' → Response contains '10' ✓")
    logger.info("Each query received its own independent response.")

    # Note: Agent cleanup is handled by clean_agent fixture


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
