"""
Integration test to reproduce RPyC KeyError with TaskManager.

This test demonstrates the version conflict where daemon agent processes
use old rpyc instead of rpyc_async, causing KeyError during
async callback deserialization.

REPRODUCTION SCENARIO:
1. Start agent with TaskManager enabled (-tm flag)
2. Send query that triggers TaskManager corrections
3. TaskManager sends messages via await subscriber.on_message()
4. KeyError occurs in _unbox() on CLIENT side (dwcli)
5. Error appears in agent logs, but system continues to work

EXPECTED BEHAVIOR:
- No KeyError in logs
- Messages delivered successfully via async callbacks

ACTUAL BEHAVIOR (BUG):
- KeyError: ('builtins.method', 10665440, <id>) in protocol.py:607
- Occurs because agent process uses OLD rpyc from pip
- rpyc_async (line 612) not used in daemon process

ROOT CAUSE:
- sys.path.insert(0, "rpyc_async") in agent_process.py:22
- NOT inherited by daemon subprocess
- Agent imports old system rpyc instead

FIX NEEDED:
- Set PYTHONPATH=rpyc_async in agent environment
- OR ensure rpyc_async loaded BEFORE daemonization
"""
import pytest
import asyncio
import time
from pathlib import Path

from deep_worker_cli.client.agent_client import AgentClient
from tests.integration_ai.agent_fixture import agent_fixture


@pytest.mark.asyncio
async def test_rpyc_keyerror_with_taskmanager(agent_fixture):
    """
    Reproduce RPyC KeyError when using TaskManager with daemon agent.

    This test FAILS (reproduces bug) when agent uses old rpyc.
    This test PASSES when agent uses rpyc_async correctly.
    """
    agent_name = "test_keyerror_repro"

    # Start agent with TaskManager enabled via fixture
    agent_info = await agent_fixture(
        agent_name=agent_name,
        enable_task_manager=True,  # ← This triggers the bug
    )

    try:
        print("\n" + "="*80)
        print("TESTING: RPyC KeyError reproduction with TaskManager")
        print("="*80)

        # Create client to communicate with agent
        client = AgentClient(agent_name=agent_name)
        await client.connect()

        # Send query that will trigger TaskManager corrections
        # (subordinate won't provide full answer, forcing corrections)
        test_query = (
            "Write a Python function called `calculate_sum` that adds two numbers. "
            "Your response MUST include ALL of the following:\n"
            "1. The complete function code\n"
            "2. Example usage\n"
            "3. Test cases"
        )

        print(f"\n1. Sending query that triggers TaskManager...")
        print(f"   Query: {test_query[:80]}...")

        # Send via RPyC to agent
        response = await client.send_message(
            message=test_query,
            session_id="test_session",
        )

        print(f"\n2. Response received: {response[:200] if response else 'None'}...")

        # Wait for TaskManager to process and send corrections
        print("\n3. Waiting for TaskManager iterations...")
        await asyncio.sleep(3.0)

        # Check agent error log for KeyError
        log_file = Path("~/.deep_worker/logs").expanduser() / "test_keyerror_repro.err.log"

        if log_file.exists():
            log_content = log_file.read_text()

            # Look for the specific KeyError pattern
            has_keyerror = "KeyError: ('builtins.method'" in log_content
            has_unbox_error = "in _unbox" in log_content
            has_old_rpyc_line = "protocol.py\", line 607" in log_content  # OLD rpyc
            has_new_rpyc_line = "protocol.py\", line 612" in log_content  # NEW rpyc

            print("\n" + "="*80)
            print("DIAGNOSTIC RESULTS:")
            print("="*80)
            print(f"KeyError present: {has_keyerror}")
            print(f"_unbox error: {has_unbox_error}")
            print(f"Old rpyc (line 607): {has_old_rpyc_line}")
            print(f"New rpyc (line 612): {has_new_rpyc_line}")

            if has_keyerror and has_old_rpyc_line:
                print("\n❌ BUG REPRODUCED!")
                print("   Agent is using OLD rpyc from pip (line 607)")
                print("   Expected: rpyc_async (line 612)")
                print("\nRelevant log excerpt:")
                print("-" * 80)

                # Extract relevant error lines
                lines = log_content.split('\n')
                for i, line in enumerate(lines):
                    if 'KeyError' in line or 'protocol.py' in line or '_unbox' in line:
                        # Print context (2 lines before and after)
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        for context_line in lines[start:end]:
                            print(context_line)
                        break
                print("-" * 80)

                # FAIL the test to document the bug
                pytest.fail(
                    "KeyError reproduced: Agent daemon process uses old rpyc "
                    "(line 607) instead of rpyc_async (line 612). "
                    "sys.path modification not inherited by subprocess."
                )

            elif has_keyerror and has_new_rpyc_line:
                print("\n⚠️  KeyError with NEW rpyc (unexpected)")
                print("   This might be a different bug in rpyc_async itself")
                pytest.fail(f"KeyError with new rpyc_async - unexpected! Log:\n{log_content[-500:]}")

            elif not has_keyerror:
                print("\n✅ NO KeyError detected!")
                print("   Agent is using correct rpyc_async")
                print("   Bug is FIXED!")

        else:
            print(f"\n⚠️  Log file not found: {log_file}")
            print("   Cannot verify bug reproduction")

        print("="*80)

    finally:
        await client.close()
        print("\n4. Client closed")
