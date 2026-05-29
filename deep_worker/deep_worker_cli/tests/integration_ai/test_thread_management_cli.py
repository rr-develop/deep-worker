"""Integration tests for thread management CLI commands.

This test file verifies the CLI commands work end-to-end:
- deep-worker -rm <thread-id>
- deep-worker --history <thread-id>
- deep-worker --remove-before <date>

Tests run against an isolated agent data directory (HOME-override) so they
exercise the per-agent DB layout that is now the primary storage.  They
do not touch the developer's real agent data directory.
"""
from __future__ import annotations

import asyncio
import json
import os
import pwd
import subprocess
from datetime import datetime, timedelta, UTC
from pathlib import Path

import aiosqlite
import pytest


CHECKPOINT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata TEXT,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""

WRITES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL DEFAULT 0,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)
"""


@pytest.fixture
def isolated_home(tmp_path):
    """Give the subprocess a private agent data directory."""
    home = tmp_path / "home"
    (home / ".deepagents").mkdir(parents=True)
    yield home


def _run_dw_cli(*args: str, home: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    env["HOME"] = str(home)
    # Keep user site-packages discoverable despite HOME override.
    env["PYTHONUSERBASE"] = str(real_home / ".local")
    return subprocess.run(
        ["deep-worker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


async def _seed_checkpoint(
    db_path: Path,
    thread_id: str,
    *,
    content: str | None = None,
    ai_content: str | None = None,
    updated_at: str | None = None,
) -> None:
    """Seed one checkpoint row into the given (per-agent) DB."""
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute(CHECKPOINT_SCHEMA_SQL)
        await conn.execute(WRITES_SCHEMA_SQL)

        if content is not None or ai_content is not None:
            messages = []
            if content is not None:
                messages.append({"type": "human", "content": content})
            if ai_content is not None:
                messages.append({"type": "ai", "content": ai_content})
            checkpoint_id = f"cp-{thread_id}"
            checkpoint_data = json.dumps(
                {
                    "id": checkpoint_id,
                    "ts": updated_at or datetime.now(UTC).isoformat(),
                    "channel_values": {"messages": messages},
                }
            )
        else:
            checkpoint_data = None

        metadata = {}
        if updated_at is not None:
            metadata["updated_at"] = updated_at
        metadata_str = json.dumps(metadata) if metadata else "{}"

        checkpoint_id = f"cp-{thread_id}"
        await conn.execute(
            """
            INSERT INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata, parent_checkpoint_id)
            VALUES (?, '', ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                checkpoint_id,
                "json" if checkpoint_data else None,
                checkpoint_data,
                metadata_str,
                None,
            ),
        )
        await conn.commit()


def test_remove_thread_cli(isolated_home):
    """Test deep-worker -rm command against a per-agent DB file."""
    thread_id = "test_rm_cli"
    per_agent_db = isolated_home / ".deepagents" / f"sessions_{thread_id}.db"

    asyncio.run(
        _seed_checkpoint(
            per_agent_db,
            thread_id,
            updated_at="2026-03-29T10:00:00Z",
        )
    )

    # Dry run
    result = _run_dw_cli("-rm", thread_id, "--dry-run", home=isolated_home)
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert thread_id in result.stdout

    async def count() -> int:
        async with aiosqlite.connect(str(per_agent_db)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                return (await cursor.fetchone())[0]

    assert asyncio.run(count()) > 0

    result = _run_dw_cli("-rm", thread_id, home=isolated_home)
    assert result.returncode == 0, result.stderr
    assert ("Removed thread" in result.stdout) or ("\u2713" in result.stdout)

    assert asyncio.run(count()) == 0


def test_history_cli(isolated_home):
    """Test deep-worker --history command."""
    thread_id = "test_history_cli"
    per_agent_db = isolated_home / ".deepagents" / f"sessions_{thread_id}.db"

    asyncio.run(
        _seed_checkpoint(
            per_agent_db,
            thread_id,
            content="Hello",
            ai_content="Hi there!",
        )
    )

    result = _run_dw_cli("--history", thread_id, home=isolated_home)
    assert result.returncode == 0, result.stderr
    assert thread_id in result.stdout
    assert "Hello" in result.stdout
    assert "Hi there" in result.stdout

    result = _run_dw_cli(
        "--history", thread_id, "--format", "json", home=isolated_home
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["thread_id"] == thread_id
    assert len(data["messages"]) == 2


def test_remove_before_cli(isolated_home):
    """Test deep-worker --remove-before command."""
    old_thread = "test_old_cli"
    recent_thread = "test_recent_cli"
    old_db = isolated_home / ".deepagents" / f"sessions_{old_thread}.db"
    recent_db = isolated_home / ".deepagents" / f"sessions_{recent_thread}.db"

    asyncio.run(
        _seed_checkpoint(
            old_db, old_thread, updated_at="2025-01-01T10:00:00Z"
        )
    )
    asyncio.run(
        _seed_checkpoint(
            recent_db,
            recent_thread,
            updated_at=datetime.now(UTC).isoformat(),
        )
    )

    result = _run_dw_cli(
        "--remove-before", "2026-03-01", "--dry-run", home=isolated_home
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert "threads" in result.stdout.lower()

    async def count(db: Path, tid: str) -> int:
        async with aiosqlite.connect(str(db)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (tid,)
            ) as cursor:
                return (await cursor.fetchone())[0]

    assert asyncio.run(count(old_db, old_thread)) == 1
    assert asyncio.run(count(recent_db, recent_thread)) == 1

    result = _run_dw_cli(
        "--remove-before", "2026-03-01", home=isolated_home
    )
    assert result.returncode == 0, result.stderr
    assert ("Removed" in result.stdout) or ("\u2713" in result.stdout)

    assert asyncio.run(count(old_db, old_thread)) == 0
    assert asyncio.run(count(recent_db, recent_thread)) == 1


def test_remove_before_relative_date_cli(isolated_home):
    """Test deep-worker --remove-before with relative date format."""
    thread_id = "test_very_old_cli"
    per_agent_db = isolated_home / ".deepagents" / f"sessions_{thread_id}.db"

    old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    asyncio.run(_seed_checkpoint(per_agent_db, thread_id, updated_at=old_ts))

    result = _run_dw_cli("--remove-before", "30d", home=isolated_home)
    assert result.returncode == 0, result.stderr

    async def check_gone() -> bool:
        async with aiosqlite.connect(str(per_agent_db)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                count = (await cursor.fetchone())[0]
        return count == 0

    assert asyncio.run(check_gone())
