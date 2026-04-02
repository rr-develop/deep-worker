"""Integration test to verify metadata is written to SQLite checkpoints.

This test verifies that the fix for missing updated_at timestamps works
by actually starting an agent, sending a message, and checking the database.
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.sessions import get_db_path


def test_metadata_written_to_checkpoints():
    """Test that agent_name and updated_at metadata are written to database."""
    agent_name = "test_metadata_agent"
    working_dir = "/tmp"

    # Clean up any existing agent
    registry = ProcessRegistry()
    existing = registry.lookup(agent_name)
    if existing:
        import os, signal
        try:
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass

    # Start agent with SQLite persistence (--use-sqlite)
    import subprocess
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", working_dir,
            "--provider", "deepseek",
            "--use-sqlite",
            "--skip-permissions",
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
        raise RuntimeError(f"Agent '{agent_name}' failed to register within {timeout}s")

    # Import arpyc from source
    # ============================================================================
    # CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
    # ============================================================================
    _arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
    if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
        sys.path.insert(0, str(_arpyc_path))

    from arpyc import RpycAsyncSession

    # Get agent connection info
    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None, f"Agent '{agent_name}' not found in registry"

    # Send a simple message to create checkpoint
    async def send_query():
        async with RpycAsyncSession(
            host="127.0.0.1", port=agent_info.rpyc_port
        ) as session:
            # Send query and consume stream
            async for _chunk in session.send_query("echo 'test'", stream=True):
                pass

    asyncio.run(send_query())

    # Wait for checkpoint to be written
    time.sleep(2)

    # Check SQLite database directly
    db_path = get_db_path()
    assert db_path.exists(), f"Database not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Query metadata from checkpoints table
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

        # CRITICAL: Verify metadata contains required fields
        assert "agent_name" in metadata, f"metadata missing 'agent_name': {metadata}"
        assert metadata["agent_name"] == agent_name, \
            f"Expected agent_name={agent_name}, got {metadata['agent_name']}"

        assert "assistant_id" in metadata, f"metadata missing 'assistant_id': {metadata}"
        assert metadata["assistant_id"] == agent_name, \
            f"Expected assistant_id={agent_name}, got {metadata['assistant_id']}"

        assert "updated_at" in metadata, f"metadata missing 'updated_at': {metadata}"

        # Verify updated_at is a valid ISO timestamp
        updated_at = metadata["updated_at"]
        try:
            dt = datetime.fromisoformat(updated_at)
            assert dt is not None, "updated_at should parse as datetime"
        except (ValueError, TypeError) as e:
            pytest.fail(f"updated_at is not valid ISO timestamp: {updated_at} - {e}")

        print(f"\n✓ Metadata verification passed:")
        print(f"  agent_name: {metadata['agent_name']}")
        print(f"  assistant_id: {metadata['assistant_id']}")
        print(f"  updated_at: {metadata['updated_at']}")

    finally:
        conn.close()

    # Cleanup agent
    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    if agent_info:
        import os, signal
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_metadata_visible_in_list_command():
    """Test that metadata makes timestamps visible in --list output."""
    agent_name = "test_list_metadata_agent"
    working_dir = "/tmp"

    # Clean up any existing agent
    registry = ProcessRegistry()
    existing = registry.lookup(agent_name)
    if existing:
        import os, signal
        try:
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass

    # Start agent with SQLite
    import subprocess
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", working_dir,
            "--provider", "deepseek",
            "--use-sqlite",
            "--skip-permissions",
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
        raise RuntimeError(f"Agent '{agent_name}' failed to register within {timeout}s")

    # Import arpyc from source
    _arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
    if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
        sys.path.insert(0, str(_arpyc_path))

    from arpyc import RpycAsyncSession

    # Get agent connection info
    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None, f"Agent '{agent_name}' not found in registry"

    # Send message to create checkpoint
    async def send_query():
        async with RpycAsyncSession(
            host="127.0.0.1", port=agent_info.rpyc_port
        ) as session:
            async for _chunk in session.send_query("Say hello", stream=True):
                pass

    asyncio.run(send_query())
    time.sleep(2)

    # Run deep-worker --list --format json
    import subprocess
    result = subprocess.run(
        ["deep-worker", "--list", "--format", "json", "--agent", agent_name],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"deep-worker --list failed: {result.stderr}"

    output = json.loads(result.stdout)
    threads = output.get("threads", [])

    # Find our thread
    our_thread = None
    for thread in threads:
        if thread["thread_id"] == agent_name:
            our_thread = thread
            break

    assert our_thread is not None, f"Thread {agent_name} not found in --list output"

    # Verify updated_at is NOT null
    assert our_thread.get("updated_at") is not None, \
        f"updated_at should not be null for {agent_name}"

    assert our_thread.get("agent_name") == agent_name, \
        f"agent_name should be {agent_name}"

    print(f"\n✓ --list output verification passed:")
    print(f"  thread_id: {our_thread['thread_id']}")
    print(f"  agent_name: {our_thread['agent_name']}")
    print(f"  updated_at: {our_thread['updated_at']}")

    # Cleanup agent
    agent_info = registry.lookup(agent_name)
    if agent_info:
        import os, signal
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
