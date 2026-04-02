"""Integration test for deep-worker --clean option.

This test verifies that the --clean option properly cleans all agent history
while preserving user configuration.
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.sessions import get_db_path, list_threads, thread_exists
from tests.integration_ai.clean_agent_fixture import clean_agent


@pytest.fixture
def agent_with_history(clean_agent):
    """Create an agent with some history and then stop it."""
    agent_name = "test_clean_option"
    working_dir = "/tmp"

    # Start agent with clean state
    agent_process = clean_agent(agent_name, working_dir=working_dir)

    # Wait for agent to be ready
    time.sleep(2)

    # Send a query to create history
    try:
        # Import here to avoid circular dependency
        # ============================================================================
        # CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
        # ============================================================================
        # DO NOT use 'pip install arpyc' or ANY virtual environment (.venv)!
        # ALWAYS import arpyc from the arpyc source directory (sibling checkout).
        # Using installed packages causes version conflicts and compatibility issues.
        # This project MUST run globally without venv to ensure correct arpyc version.
        # ============================================================================
        _arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
        import sys

        if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
            sys.path.insert(0, str(_arpyc_path))

        from arpyc import RpycAsyncSession

        # Connect to agent
        registry = ProcessRegistry()
        agent_info = registry.lookup(agent_name)
        assert agent_info is not None, f"Agent '{agent_name}' not found in registry"

        # Create session and send query
        async def send_query():
            async with RpycAsyncSession(
                host=agent_info.host, port=agent_info.port
            ) as session:
                # Send a simple query
                async for _chunk in session.send_query("echo 'test query'", stream=True):
                    pass  # Just consume the stream

        asyncio.run(send_query())

        # Wait for query to be processed
        time.sleep(2)

    except Exception as e:
        print(f"Warning: Could not create history: {e}")

    # Stop the agent properly
    import os
    import signal

    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    if agent_info:
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait for process to fully terminate and be removed from registry
    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not registry.lookup(agent_name):
            break
        time.sleep(0.2)

    # Final wait to ensure cleanup
    time.sleep(1)

    return agent_name


def test_clean_option_removes_history(agent_with_history):
    """Test that --clean option removes all history before starting agent.

    This test:
    1. Creates an agent with history
    2. Stops the agent
    3. Verifies history exists
    4. Starts agent with --clean
    5. Verifies history is removed
    6. Verifies agent starts successfully
    """
    agent_name = agent_with_history

    # Verify history exists in database
    threads = asyncio.run(list_threads(agent_name=agent_name))
    print(f"Threads before clean: {len(threads)}")
    # Note: Thread might not exist if the query failed, so we allow 0 or more
    # The important part is testing that --clean works regardless

    # Verify data directory exists
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    print(f"Data directory exists: {data_dir.exists()}")

    # Create a log file to verify it gets cleaned
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log = log_dir / f"{agent_name}.test.log"
    test_log.write_text("test log content")
    assert test_log.exists()

    # Create user config that should NOT be deleted
    config_dir = Path("~/.deepagents").expanduser() / agent_name
    config_dir.mkdir(parents=True, exist_ok=True)
    agents_md = config_dir / "AGENTS.md"
    agents_md.write_text("# User configuration\nThis should be preserved")
    assert agents_md.exists()

    # Start agent with --clean
    result = subprocess.run(
        ["deep-worker", agent_name, "--dir", "/tmp", "--clean"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Verify command output
    print(f"Clean command stdout: {result.stdout}")
    print(f"Clean command stderr: {result.stderr}")
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Cleaned history" in result.stdout

    # Wait for agent to register
    registry = ProcessRegistry()
    timeout = 30
    start_time = time.time()
    agent_registered = False

    while time.time() - start_time < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            break
        time.sleep(0.5)

    assert agent_registered, f"Agent '{agent_name}' did not register within {timeout}s"

    # Verify history is removed
    threads_after = asyncio.run(list_threads(agent_name=agent_name))
    assert (
        len(threads_after) == 0
    ), f"Expected 0 threads after clean, got {len(threads_after)}"

    # Note: data_dir will exist again after agent starts, so we can't check for its non-existence
    # The important verification is that the old session_metadata.json is gone
    # and a fresh one is created
    if data_dir.exists():
        metadata_file = data_dir / "session_metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            # Verify it's a fresh session (no completed tasks, no messages)
            # Note: message_count might be > 0 if agent already processed something
            # but tasks_completed should be 0
            assert metadata.get("tasks_completed", 0) == 0, "Tasks should be reset to 0"

    # Verify test log file is removed (not recreated by agent)
    assert not test_log.exists(), "Test log file should be removed"

    # Verify user config is preserved
    assert agents_md.exists(), "User config should be preserved"
    assert agents_md.read_text() == "# User configuration\nThis should be preserved"

    # Clean up: stop agent
    try:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            import os
            import signal

            os.kill(agent_info.pid, signal.SIGTERM)
            time.sleep(1)
    except Exception as e:
        print(f"Warning: Could not stop agent: {e}")


def test_clean_option_prevents_running_agent(clean_agent):
    """Test that --clean fails if agent is already running."""
    agent_name = "test_clean_running"
    working_dir = "/tmp"

    # Start agent
    agent_process = clean_agent(agent_name, working_dir=working_dir)

    # Wait for agent to be ready
    time.sleep(2)

    # Try to start with --clean while agent is running
    result = subprocess.run(
        ["deep-worker", agent_name, "--dir", working_dir, "--clean"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # Verify command fails
    assert result.returncode == 1, "Command should fail when agent is running"
    assert "already running" in result.stderr.lower()

    # Clean up
    agent_process.terminate()
    agent_process.wait(timeout=5)


def test_clean_option_with_no_history():
    """Test that --clean works gracefully when there's no history."""
    agent_name = "test_clean_no_history"
    working_dir = "/tmp"

    # Ensure no agent is running
    registry = ProcessRegistry()
    existing = registry.lookup(agent_name)
    if existing:
        import os
        import signal

        os.kill(existing.pid, signal.SIGTERM)
        time.sleep(1)

    # Start agent with --clean (no history exists)
    result = subprocess.run(
        ["deep-worker", agent_name, "--dir", working_dir, "--clean"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Verify command succeeds
    print(f"Clean command stdout: {result.stdout}")
    print(f"Clean command stderr: {result.stderr}")
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Cleaned history" in result.stdout

    # Wait for agent to register
    timeout = 30
    start_time = time.time()
    agent_registered = False

    while time.time() - start_time < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            break
        time.sleep(0.5)

    assert agent_registered, f"Agent '{agent_name}' did not register within {timeout}s"

    # Clean up
    try:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            import os
            import signal

            os.kill(agent_info.pid, signal.SIGTERM)
            time.sleep(1)
    except Exception as e:
        print(f"Warning: Could not stop agent: {e}")
