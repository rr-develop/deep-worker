"""Integration tests for thread management CLI commands.

This test file verifies the CLI commands work end-to-end:
- deep-worker -rm <thread-id>
- deep-worker --history <thread-id>
- deep-worker --remove-before <date>
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from deep_worker_cli.sessions import get_db_path


def test_remove_thread_cli():
    """Test deep-worker -rm command."""
    import aiosqlite

    # Create test thread in database
    db_path = get_db_path()

    async def setup():
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, metadata)
                VALUES (?, ?, ?)
            """, ("test_rm_cli", "cp1", '{"updated_at": "2026-03-29T10:00:00Z"}'))
            await conn.commit()

    asyncio.run(setup())

    # Test dry run
    result = subprocess.run(
        ["deep-worker", "-rm", "test_rm_cli", "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Dry run" in result.stdout
    assert "test_rm_cli" in result.stdout

    # Verify thread still exists
    async def check_exists():
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_rm_cli",)
            ) as cursor:
                count = (await cursor.fetchone())[0]
        return count > 0

    assert asyncio.run(check_exists())

    # Actually remove
    result = subprocess.run(
        ["deep-worker", "-rm", "test_rm_cli"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Removed thread" in result.stdout or "✓" in result.stdout

    # Verify thread is gone
    assert not asyncio.run(check_exists())

    print("✓ Test passed: -rm command works correctly")


def test_history_cli():
    """Test deep-worker --history command."""
    import aiosqlite

    db_path = get_db_path()

    # Create test thread with messages
    async def setup():
        async with aiosqlite.connect(str(db_path)) as conn:
            checkpoint_data = json.dumps({
                "channel_values": {
                    "messages": [
                        {"type": "human", "content": "Hello"},
                        {"type": "ai", "content": "Hi there!"}
                    ]
                }
            })

            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata, parent_checkpoint_id)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_history_cli", "cp1", checkpoint_data, '{}', None))
            await conn.commit()

    asyncio.run(setup())

    # Test text format
    result = subprocess.run(
        ["deep-worker", "--history", "test_history_cli"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "test_history_cli" in result.stdout
    assert "Hello" in result.stdout
    assert "Hi there" in result.stdout

    # Test JSON format
    result = subprocess.run(
        ["deep-worker", "--history", "test_history_cli", "--format", "json"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["thread_id"] == "test_history_cli"
    assert len(data["messages"]) == 2

    # Cleanup
    async def cleanup():
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", ("test_history_cli",))
            await conn.commit()

    asyncio.run(cleanup())

    print("✓ Test passed: --history command works correctly")


def test_remove_before_cli():
    """Test deep-worker --remove-before command."""
    import aiosqlite

    db_path = get_db_path()

    # Create old and recent threads
    async def setup():
        async with aiosqlite.connect(str(db_path)) as conn:
            # Old thread
            old_metadata = json.dumps({"updated_at": "2025-01-01T10:00:00Z"})
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, metadata)
                VALUES (?, ?, ?)
            """, ("test_old_cli", "cp1", old_metadata))

            # Recent thread
            recent_metadata = json.dumps({"updated_at": datetime.now(UTC).isoformat()})
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, metadata)
                VALUES (?, ?, ?)
            """, ("test_recent_cli", "cp2", recent_metadata))

            await conn.commit()

    asyncio.run(setup())

    # Test dry run
    result = subprocess.run(
        ["deep-worker", "--remove-before", "2026-03-01", "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Dry run" in result.stdout
    assert "threads" in result.stdout.lower()

    # Verify both threads still exist
    async def check_counts():
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id IN (?, ?)",
                ("test_old_cli", "test_recent_cli")
            ) as cursor:
                count = (await cursor.fetchone())[0]
        return count

    assert asyncio.run(check_counts()) == 2

    # Actually remove old threads
    result = subprocess.run(
        ["deep-worker", "--remove-before", "2026-03-01"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Removed" in result.stdout or "✓" in result.stdout

    # Verify old thread is gone, recent remains
    async def check_specific():
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_old_cli",)
            ) as cursor:
                old_count = (await cursor.fetchone())[0]

            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_recent_cli",)
            ) as cursor:
                recent_count = (await cursor.fetchone())[0]

        return old_count, recent_count

    old_count, recent_count = asyncio.run(check_specific())
    assert old_count == 0
    assert recent_count > 0

    # Cleanup
    async def cleanup():
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("DELETE FROM checkpoints WHERE thread_id IN (?, ?)",
                             ("test_old_cli", "test_recent_cli"))
            await conn.commit()

    asyncio.run(cleanup())

    print("✓ Test passed: --remove-before command works correctly")


def test_remove_before_relative_date_cli():
    """Test deep-worker --remove-before with relative date format."""
    import aiosqlite

    db_path = get_db_path()

    # Create very old thread
    async def setup():
        old_date = datetime.now(UTC) - timedelta(days=100)
        old_metadata = json.dumps({"updated_at": old_date.isoformat()})

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, metadata)
                VALUES (?, ?, ?)
            """, ("test_very_old_cli", "cp1", old_metadata))
            await conn.commit()

    asyncio.run(setup())

    # Remove threads older than 30 days
    result = subprocess.run(
        ["deep-worker", "--remove-before", "30d"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # Verify thread is gone
    async def check_gone():
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_very_old_cli",)
            ) as cursor:
                count = (await cursor.fetchone())[0]
        return count == 0

    assert asyncio.run(check_gone())

    print("✓ Test passed: --remove-before with relative date works correctly")


if __name__ == "__main__":
    # Run tests sequentially
    print("Running integration tests for thread management CLI...\n")

    test_remove_thread_cli()
    test_history_cli()
    test_remove_before_cli()
    test_remove_before_relative_date_cli()

    print("\n✅ All integration tests passed!")
