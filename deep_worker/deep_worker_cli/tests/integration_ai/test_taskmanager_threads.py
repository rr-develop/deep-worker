"""Integration test for TaskManager persistence and metadata.

This test verifies that TaskManager creates checkpoint entries with correct naming
and persists them to the same checkpointer as the main agent.

NOTE: TaskManager no longer uses threads - it runs in the main event loop.
The ':tm' suffix in thread_id is retained for compatibility and identification.
"""

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.sessions import get_db_path


def test_taskmanager_checkpoint_creation_with_sqlite():
    """Test TaskManager creates checkpoint with 'agent:tm' naming in SQLite.

    This test verifies:
    1. Main agent checkpoint 'test_tm_agent' exists in SQLite
    2. TaskManager checkpoint 'test_tm_agent:tm' exists in SQLite (with colon)
    3. Both checkpoints have proper metadata (agent_name, updated_at)
    4. TaskManager uses AsyncSqliteSaver (same as main agent)
    5. TaskManager runs in main event loop (no separate threads)
    """
    agent_name = "test_tm_agent"
    working_dir = "/tmp"

    # Clean up any existing agent
    registry = ProcessRegistry()
    existing = registry.lookup(agent_name)
    if existing:
        try:
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass

    # Start agent with TaskManager and SQLite
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", working_dir,
            "--provider", "deepseek",
            "--use-sqlite",  # Force SQLite
            "--skip-permissions",
            "--task-manager",  # Enable TaskManager
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for agent to register
    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            break
        time.sleep(0.5)

    if not agent_registered:
        agent_process.kill()
        raise RuntimeError(f"Agent '{agent_name}' failed to register within {timeout}s")

    # Wait for TaskManager to initialize
    time.sleep(3)

    # Import arpyc from source
    _arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
    if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
        sys.path.insert(0, str(_arpyc_path))

    from arpyc import RpycAsyncSession

    # Get agent connection info
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None

    # Send a message to trigger TaskManager evaluation
    # TaskManager only activates when a task is evaluated
    async def send_query():
        async with RpycAsyncSession(
            host="127.0.0.1", port=agent_info.rpyc_port
        ) as session:
            # Send query that will trigger task evaluation
            async for _chunk in session.send_query(
                "Create a file called test.txt with content 'hello'",
                stream=True
            ):
                pass

    asyncio.run(send_query())

    # Wait for checkpoint to be written
    time.sleep(3)

    # Check SQLite database directly
    db_path = get_db_path()
    assert db_path.exists(), f"Database not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Query all threads for this agent
        cursor.execute(
            """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id = ? OR thread_id = ?
            ORDER BY thread_id
            """,
            (agent_name, f"{agent_name}:tm")
        )

        rows = cursor.fetchall()
        thread_ids = [row[0] for row in rows]

        print(f"\n✓ Found threads in database: {thread_ids}")

        # CRITICAL: Verify both checkpoints exist
        assert agent_name in thread_ids, \
            f"Main agent checkpoint '{agent_name}' not found in database"

        assert f"{agent_name}:tm" in thread_ids, \
            f"TaskManager checkpoint '{agent_name}:tm' not found in database. " \
            f"Expected checkpoint with COLON, not underscore. Found: {thread_ids}"

        # Verify metadata for main agent thread
        cursor.execute(
            """
            SELECT metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (agent_name,)
        )

        row = cursor.fetchone()
        assert row is not None, f"No checkpoint found for thread_id={agent_name}"

        metadata_json = row[0]
        metadata = json.loads(metadata_json)

        assert "agent_name" in metadata, \
            f"Main agent metadata missing 'agent_name': {metadata}"
        assert metadata["agent_name"] == agent_name
        assert "updated_at" in metadata

        print(f"✓ Main agent checkpoint metadata: agent_name={metadata['agent_name']}, "
              f"updated_at={metadata.get('updated_at', 'N/A')}")

        # Verify metadata for TaskManager checkpoint
        cursor.execute(
            """
            SELECT metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (f"{agent_name}:tm",)
        )

        row = cursor.fetchone()
        assert row is not None, \
            f"No checkpoint found for thread_id={agent_name}:tm"

        tm_metadata_json = row[0]
        tm_metadata = json.loads(tm_metadata_json)

        assert "agent_name" in tm_metadata, \
            f"TaskManager metadata missing 'agent_name': {tm_metadata}"
        assert tm_metadata["agent_name"] == agent_name, \
            f"Expected agent_name='{agent_name}' (parent agent), got '{tm_metadata['agent_name']}'"
        assert "assistant_id" in tm_metadata
        assert tm_metadata["assistant_id"] == f"{agent_name}:tm"
        assert "updated_at" in tm_metadata

        print(f"✓ TaskManager checkpoint metadata: agent_name={tm_metadata['agent_name']}, "
              f"assistant_id={tm_metadata['assistant_id']}, "
              f"updated_at={tm_metadata.get('updated_at', 'N/A')}")

    finally:
        conn.close()

    # Cleanup agent
    agent_info = registry.lookup(agent_name)
    if agent_info:
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
            time.sleep(1)
        except ProcessLookupError:
            pass

    print(f"\n✅ All checks passed!")


def test_taskmanager_checkpoint_appears_in_list_command():
    """Test that TaskManager checkpoint appears in deep-worker --list output."""
    agent_name = "test_tm_list_agent"
    working_dir = "/tmp"

    # Clean up any existing agent
    registry = ProcessRegistry()
    existing = registry.lookup(agent_name)
    if existing:
        try:
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass

    # Start agent with TaskManager and SQLite
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", working_dir,
            "--provider", "deepseek",
            "--use-sqlite",
            "--skip-permissions",
            "--task-manager",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for agent to register
    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            break
        time.sleep(0.5)

    if not agent_registered:
        agent_process.kill()
        raise RuntimeError(f"Agent failed to register")

    time.sleep(3)

    # Import arpyc
    _arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
    if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
        sys.path.insert(0, str(_arpyc_path))

    from arpyc import RpycAsyncSession

    agent_info = registry.lookup(agent_name)

    # Send message to trigger TaskManager
    async def send_query():
        async with RpycAsyncSession(
            host="127.0.0.1", port=agent_info.rpyc_port
        ) as session:
            async for _chunk in session.send_query("echo hello", stream=True):
                pass

    asyncio.run(send_query())
    time.sleep(3)

    # Run deep-worker --list --format json
    result = subprocess.run(
        ["deep-worker", "--list", "--format", "json"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"deep-worker --list failed: {result.stderr}"

    output = json.loads(result.stdout)
    threads = output.get("threads", [])

    # Find threads for this agent
    main_thread = None
    tm_thread = None

    for thread in threads:
        if thread["thread_id"] == agent_name:
            main_thread = thread
        elif thread["thread_id"] == f"{agent_name}:tm":
            tm_thread = thread

    # Verify both checkpoints exist in --list output
    assert main_thread is not None, \
        f"Main agent checkpoint '{agent_name}' not found in --list output"

    assert tm_thread is not None, \
        f"TaskManager checkpoint '{agent_name}:tm' not found in --list output. " \
        f"Available threads: {[t['thread_id'] for t in threads]}"

    # Verify updated_at is NOT null
    assert main_thread.get("updated_at") is not None, \
        f"Main agent updated_at should not be null"

    assert tm_thread.get("updated_at") is not None, \
        f"TaskManager updated_at should not be null"

    # Verify agent_name (should be parent agent name, not with :tm suffix)
    assert tm_thread.get("agent_name") == agent_name, \
        f"Expected agent_name='{agent_name}' (parent agent), got '{tm_thread.get('agent_name')}'"

    print(f"\n✓ --list output verification passed:")
    print(f"  Main thread: {main_thread['thread_id']} - {main_thread['updated_at']}")
    print(f"  TM thread: {tm_thread['thread_id']} - {tm_thread['updated_at']}")

    # Cleanup
    agent_info = registry.lookup(agent_name)
    if agent_info:
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    print(f"\n✅ All checks passed!")


if __name__ == "__main__":
    # Allow running directly for debugging
    print("Running test_taskmanager_checkpoint_creation_with_sqlite...")
    test_taskmanager_checkpoint_creation_with_sqlite()

    print("\n" + "="*80 + "\n")

    print("Running test_taskmanager_checkpoint_appears_in_list_command...")
    test_taskmanager_checkpoint_appears_in_list_command()
