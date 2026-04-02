"""
Simplified integration test to reproduce RPyC KeyError with TaskManager.

This test DIRECTLY starts an agent process and connects a client,
reproducing the real-world scenario where daemon agents use old rpyc.

BUG DESCRIPTION:
- Daemon agent processes use old rpyc from pip (line 607)
- Expected: rpyc_async (line 612)
- sys.path modification not inherited by subprocess

EXPECTED BEHAVIOR: No KeyError
ACTUAL BEHAVIOR: KeyError in _unbox() on CLIENT side
"""
import pytest
import asyncio
import subprocess
import time
from pathlib import Path

from deep_worker_cli.client.agent_client import AgentClient


@pytest.mark.asyncio
async def test_rpyc_keyerror_with_taskmanager_simple():
    """
    Reproduce KeyError by starting real daemon agent with TaskManager.

    This test documents the bug and verifies it's fixed.
    """
    agent_name = "test_keyerror"

    # Clean up old logs
    log_dir = Path("~/.deep_worker/logs").expanduser()
    for log_file in log_dir.glob(f"{agent_name}.*"):
        log_file.unlink(missing_ok=True)

    print("\n" + "="*80)
    print("REPRODUCING: RPyC KeyError with TaskManager")
    print("="*80)

    # Start agent with TaskManager using dwcli command
    print("\n1. Starting agent with TaskManager (-tm flag)...")

    # Using subprocess to ensure real daemon behavior
    start_cmd = [
        "python3", "-m", "deep_worker_cli.cli.main",
        agent_name, "-tm", "--daemon",
    ]

    try:
        result = subprocess.run(
            start_cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            timeout=10,
        )

        print(f"   Agent start output: {result.stdout[:200]}")
        if result.returncode != 0:
            print(f"   Error: {result.stderr}")
            pytest.fail(f"Failed to start agent: {result.stderr}")

    except FileNotFoundError:
        pytest.skip("dwcli not available as module")

    # Wait for agent to fully start
    await asyncio.sleep(3.0)

    try:
        # Connect client
        print("\n2. Connecting client to agent...")
        client = AgentClient(agent_name=agent_name)
        await client.connect()

        # Send query that triggers TaskManager corrections
        test_query = (
            "Write Python function `add_numbers` that adds two numbers. "
            "Your response MUST include:\n"
            "1. Complete function code\n"
            "2. Example usage\n"
            "3. Test cases"
        )

        print(f"\n3. Sending query (triggers TaskManager)...")
        print(f"   Query: {test_query[:80]}...")

        response = await client.send_message(
            message=test_query,
            session_id="test_session",
        )

        print(f"\n4. Response: {response[:150] if response else 'None'}...")

        # Wait for TaskManager iterations
        print("\n5. Waiting for TaskManager to process...")
        await asyncio.sleep(5.0)

        # Check logs for KeyError
        err_log = log_dir / f"{agent_name}.err.log"

        print("\n6. Checking logs for KeyError...")

        if err_log.exists():
            log_content = err_log.read_text()

            has_keyerror = "KeyError: ('builtins.method'" in log_content
            has_old_rpyc = "protocol.py\", line 607" in log_content
            has_new_rpyc = "protocol.py\", line 612" in log_content

            print("\n" + "="*80)
            print("DIAGNOSTIC RESULTS:")
            print("="*80)
            print(f"  KeyError present: {has_keyerror}")
            print(f"  Old rpyc (line 607): {has_old_rpyc}")
            print(f"  New rpyc (line 612): {has_new_rpyc}")

            if has_keyerror and has_old_rpyc:
                print("\n❌ BUG REPRODUCED!")
                print("   Daemon agent uses OLD rpyc (line 607)")
                print("   Expected: rpyc_async (line 612)")
                print("\n  Log excerpt:")
                print("  " + "-" * 76)

                # Show relevant lines
                for i, line in enumerate(log_content.split('\n')):
                    if 'KeyError' in line or 'protocol.py' in line:
                        print(f"    {line[:76]}")
                        if i < len(log_content.split('\n')) - 1:
                            print(f"    {log_content.split(chr(10))[i+1][:76]}")
                        break

                print("  " + "-" * 76)

                pytest.fail(
                    "KeyError reproduced: Daemon agent uses old rpyc. "
                    "sys.path not inherited by subprocess. "
                    "Fix: Set PYTHONPATH=rpyc_async before agent start."
                )

            elif has_keyerror and has_new_rpyc:
                print("\n⚠️  KeyError with NEW rpyc")
                print("   Different bug in rpyc_async itself!")
                pytest.fail(f"Unexpected KeyError with new rpyc:\n{log_content[-300:]}")

            elif not has_keyerror:
                print("\n✅ NO KeyError!")
                print("   Agent uses correct rpyc_async")
                print("   Bug is FIXED!")

        else:
            print(f"\n⚠️  No error log: {err_log}")

        print("="*80)

    finally:
        # Cleanup
        try:
            await client.close()
        except:
            pass

        # Stop agent
        print("\n7. Stopping agent...")
        subprocess.run(
            ["python3", "-m", "deep_worker_cli.cli.main", "stop", agent_name],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
        )

        print("8. Test complete")


if __name__ == "__main__":
    asyncio.run(test_rpyc_keyerror_with_taskmanager_simple())
