"""Integration test for detecting RPyC refcount errors in agent logs.

This test verifies that:
1. Agent is created with clean logs via deep-worker command (daemonized)
2. Simple prompt "2+2" is sent via dwcli -p
3. Agent ERROR LOG FILE is checked for RPyC refcount errors
4. Test fails if errors like "[REFCOUNT] DECREF on missing key" or "Failed to delete remote object" are found

The test writes logs to ~/.deep_worker/logs/ so they can be inspected after test run.
"""

import asyncio
import re
import subprocess
import time
from pathlib import Path

import pytest


def count_refcount_errors_in_file(log_file: Path, start_position: int = 0) -> dict:
    """Count RPyC refcount errors in log file from given position.

    Args:
        log_file: Path to log file
        start_position: Byte position to start reading from

    Returns:
        Dict with error counts and sample errors
    """
    if not log_file.exists():
        return {
            'refcount_decref': 0,
            'failed_delete': 0,
            'total': 0,
            'samples': []
        }

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_position)
            content = f.read()

        refcount_matches = list(re.finditer(r'\[REFCOUNT\]\s+DECREF on missing key[^\n]*', content))
        failed_delete_matches = list(re.finditer(r'Failed to delete remote object[^\n]*', content))

        samples = []
        # Get first 10 of each type
        for i, match in enumerate(refcount_matches[:10], 1):
            samples.append(f"REFCOUNT #{i}: {match.group(0)}")
        for i, match in enumerate(failed_delete_matches[:10], 1):
            samples.append(f"DELETE #{i}: {match.group(0)}")

        return {
            'refcount_decref': len(refcount_matches),
            'failed_delete': len(failed_delete_matches),
            'total': len(refcount_matches) + len(failed_delete_matches),
            'samples': samples
        }
    except Exception as e:
        return {
            'refcount_decref': 0,
            'failed_delete': 0,
            'total': 0,
            'samples': [],
            'error': str(e)
        }


@pytest.mark.asyncio
async def test_agent_rpyc_refcount_errors_on_simple_prompt():
    """Test that simple prompt execution doesn't produce RPyC refcount errors.

    This test:
    1. Kills any pre-existing test agent to ensure clean start
    2. Removes old log files
    3. Starts agent via deep-worker command (creates daemon with file logging)
    4. Records initial error log size
    5. Sends simple prompt "2+2" via dwcli -p
    6. Waits for response
    7. Checks error log file for NEW refcount errors
    8. Stops agent
    9. Fails if errors found, providing detailed analysis with log file path
    """
    agent_name = "test_refcount_check"

    # Kill any existing agent
    subprocess.run(
        ["pkill", "-9", "-f", f"deep-worker {agent_name}"],
        capture_output=True
    )
    await asyncio.sleep(1.0)

    # Clean up registry
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()
    registry.unregister(agent_name)

    # Remove old log files
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    out_log = log_dir / f"{agent_name}.out.log"
    err_log = log_dir / f"{agent_name}.err.log"

    for log_file in [out_log, err_log]:
        if log_file.exists():
            log_file.unlink()

    print(f"\n🧹 Cleaned up old agent and logs")

    # Start agent via deep-worker command (creates daemon with file logging)
    print(f"\n🚀 Starting agent '{agent_name}' via deep-worker command...")

    subprocess.run(
        ["deep-worker", agent_name],
        capture_output=False,
        check=False
    )

    # Wait for agent to start and verify in registry
    max_wait = 30.0
    start_time = time.time()
    agent_started = False

    while time.time() - start_time < max_wait:
        list_result = subprocess.run(
            ["dwcli", "--list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if agent_name in list_result.stdout:
            agent_started = True
            print(f"✅ Agent '{agent_name}' started successfully")
            break

        await asyncio.sleep(1.0)

    if not agent_started:
        # Check logs for startup errors
        if err_log.exists():
            with open(err_log, 'r') as f:
                err_content = f.read()
            print(f"\n❌ Agent startup error log:\n{err_content[:500]}")
        pytest.fail(f"Failed to start agent '{agent_name}' via deep-worker command (waited {max_wait}s)")

    print(f"✅ Agent '{agent_name}' started successfully")

    # Record initial error log size (should be ~0 for fresh agent)
    err_log_size_before = err_log.stat().st_size if err_log.exists() else 0

    print(f"\n📊 Initial error log size: {err_log_size_before} bytes")
    print(f"📁 Error log path: {err_log}")

    try:
        # Send simple prompt via dwcli
        prompt = "2+2"

        print(f"\n📤 Sending prompt: '{prompt}'")

        send_process = await asyncio.create_subprocess_exec(
            "dwcli",
            agent_name,
            "-p",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await send_process.communicate()

        # Verify command succeeded
        assert send_process.returncode == 0, \
            f"dwcli command failed (exit code {send_process.returncode}): {stderr.decode()}"

        response = stdout.decode().strip()
        assert len(response) > 0, "Should get non-empty response from agent"

        print(f"✅ Agent responded: '{response}' ({len(response)} chars)")

        # Wait for log writes to complete
        await asyncio.sleep(2.0)

        # Check error log for new errors
        err_log_size_after = err_log.stat().st_size if err_log.exists() else 0
        new_bytes = err_log_size_after - err_log_size_before

        print(f"\n📊 Error log after test:")
        print(f"  Size before: {err_log_size_before} bytes")
        print(f"  Size after:  {err_log_size_after} bytes")
        print(f"  New bytes:   {new_bytes} bytes")

        # Count errors in NEW log entries only
        result = count_refcount_errors_in_file(err_log, err_log_size_before)

        print(f"\n📊 RPyC refcount errors found in log file:")
        print(f"  [REFCOUNT] DECREF on missing key: {result['refcount_decref']}")
        print(f"  Failed to delete remote object: {result['failed_delete']}")
        print(f"  Total: {result['total']}")

        if result['total'] > 0:
            print(f"\n" + "="*80)
            print("RPyC REFCOUNT ERROR ANALYSIS")
            print("="*80)
            print(f"\n📁 Error log file: {err_log}")
            print(f"📏 Log file size: {err_log_size_after} bytes")
            print(f"📝 New errors added: {result['total']}")

            if result['samples']:
                print(f"\n📄 Sample errors (first 10 of each type):")
                print("-" * 80)
                for sample in result['samples']:
                    print(f"  {sample}")

            print("\n" + "="*80)
            print("ANALYSIS SUMMARY")
            print("="*80)

            print("\n🔍 What these errors mean:")
            print("  - RPyC reference counting errors indicate issues with proxy object lifecycle")
            print("  - '[REFCOUNT] DECREF on missing key' means RPyC tried to decrease reference")
            print("    count for an object that was already deleted or never existed")
            print("  - 'Failed to delete remote object' indicates cleanup failure during proxy")
            print("    object destruction, leading to memory leaks")

            print("\n🔍 Impact:")
            print("  - Memory leaks on server side (15 objects per request)")
            print("  - Log file pollution (agent generated {} bytes of errors for 1 request)".format(new_bytes))
            print("  - Performance degradation over time")
            print("  - Potential crashes after thousands of requests")

            print("\n🔍 Recommended investigation areas:")
            print("  1. Check arpyc session lifecycle management")
            print("  2. Review proxy object cleanup in agent_client.py")
            print("  3. Investigate exposed_get_status return value handling")
            print("  4. Review RPyC connection close/cleanup sequence")

            print("\n" + "="*80)
            print(f"\n💡 You can inspect the full error log at: {err_log}")
            print("="*80)

            # Fail test
            pytest.fail(
                f"❌ TEST FAILED: Found {result['total']} RPyC refcount error(s) in agent error log.\n"
                f"  - [REFCOUNT] DECREF on missing key: {result['refcount_decref']}\n"
                f"  - Failed to delete remote object: {result['failed_delete']}\n"
                f"\n"
                f"Error log file: {err_log}\n"
                f"Log file size: {err_log_size_after} bytes\n"
                f"New errors added: {new_bytes} bytes\n"
                f"\n"
                f"See analysis above for details."
            )
        else:
            print(f"\n✅ No RPyC refcount errors found in agent error log")
            print(f"✅ Test passed!")

    finally:
        # Stop agent
        print(f"\n🛑 Stopping agent '{agent_name}'...")
        subprocess.run(
            ["dwcli", "--stop", agent_name],
            capture_output=True,
            timeout=10
        )
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
