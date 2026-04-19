"""Integration tests for deep-worker offline commands against per-agent DBs.

These tests drive the real ``deep-worker`` CLI binary against real
``sessions_<agent>.db`` files on disk, reproducing the exact scenario
that caused the reported failure:

    $ deep-worker --history lc4_test
    Error: no such table: checkpoints

The tests use a temporary ``HOME`` so they do not touch the developer's
real ``~/.deepagents``.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, UTC
from pathlib import Path

import aiosqlite
import pytest


def _schema_sql() -> tuple[str, str]:
    return (
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT,
            checkpoint_id TEXT,
            parent_checkpoint_id TEXT,
            checkpoint_ns TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS writes (
            thread_id TEXT,
            task_id TEXT
        )
        """,
    )


async def _seed_per_agent_db(
    db_path: Path,
    thread_id: str,
    content: str,
    agent_name_meta: str | None = None,
    updated_at: str | None = None,
) -> None:
    checkpoints_sql, writes_sql = _schema_sql()
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute(checkpoints_sql)
        await conn.execute(writes_sql)

        ts = updated_at or datetime.now(UTC).isoformat()
        checkpoint = json.dumps(
            {
                "ts": ts,
                "channel_values": {
                    "messages": [{"type": "human", "content": content}]
                },
            }
        )
        meta = {"updated_at": ts}
        if agent_name_meta is not None:
            meta["agent_name"] = agent_name_meta

        await conn.execute(
            """
            INSERT INTO checkpoints
            (thread_id, checkpoint_id, type, checkpoint, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, f"cp-{thread_id}", "json", checkpoint, json.dumps(meta)),
        )
        await conn.commit()


@pytest.fixture
def isolated_home(tmp_path):
    """Run deep-worker with a temp HOME so tests don't touch ~/.deepagents."""
    home = tmp_path / "home"
    (home / ".deepagents").mkdir(parents=True)

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    try:
        yield home
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


def _run_dw_cli(*args: str, home: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    """Run deep-worker with an overridden HOME env.

    We always use ``env=`` to guarantee deep-worker sees our fake HOME even
    if the caller's shell inherited a different one.  ``PYTHONUSERBASE``
    is pinned so user-site-installed dependencies (aiosqlite, etc.) remain
    importable even though HOME has changed.
    """
    env = os.environ.copy()
    # Resolve the real home from /etc/passwd-style lookup BEFORE the pytest
    # fixture mutated HOME via os.environ (which expanduser would follow).
    import pwd
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    env["HOME"] = str(home)
    # Keep user site-packages discoverable despite HOME override:
    # Python's site.py derives USER_BASE from HOME otherwise.
    env["PYTHONUSERBASE"] = str(real_home / ".local")
    return subprocess.run(
        ["deep-worker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.asyncio
async def test_dw_history_reads_per_agent_db_file(isolated_home):
    """`deep-worker --history <agent>` reads sessions_<agent>.db and
    prints the stored messages (used to fail with 'no such table:
    checkpoints').
    """
    agent = "integ_agent_hist"
    per_agent_db = isolated_home / ".deepagents" / f"sessions_{agent}.db"
    await _seed_per_agent_db(
        per_agent_db,
        thread_id=agent,
        content="hello from integ test",
        agent_name_meta=agent,
    )

    result = _run_dw_cli("--history", agent, home=isolated_home)

    assert result.returncode == 0, (
        f"deep-worker --history exited non-zero.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "hello from integ test" in result.stdout, result.stdout
    # Must NOT contain the old failure signature.
    assert "no such table" not in (result.stdout + result.stderr)


@pytest.mark.asyncio
async def test_dw_history_still_works_when_global_db_is_empty(
    isolated_home,
):
    """Exact reproducer of the bug: empty legacy global + populated
    per-agent file."""
    agent = "integ_lc4"
    # Empty 0-byte legacy file, like on the dev machine.
    (isolated_home / ".deepagents" / "sessions.db").write_bytes(b"")

    per_agent_db = isolated_home / ".deepagents" / f"sessions_{agent}.db"
    await _seed_per_agent_db(
        per_agent_db,
        thread_id=agent,
        content="integ marker message",
        agent_name_meta=agent,
    )

    result = _run_dw_cli("--history", agent, home=isolated_home)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "integ marker message" in result.stdout


@pytest.mark.asyncio
async def test_dw_history_works_for_tm_thread(isolated_home):
    """`deep-worker --history <agent>:tm` reads sessions_<agent>.db
    (derives agent from thread_id prefix)."""
    agent = "integ_tm_agent"
    tm_thread = f"{agent}:tm"
    per_agent_db = isolated_home / ".deepagents" / f"sessions_{agent}.db"
    await _seed_per_agent_db(
        per_agent_db,
        thread_id=tm_thread,
        content="tm integ message",
        agent_name_meta=agent,
    )

    result = _run_dw_cli("--history", tm_thread, home=isolated_home)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "tm integ message" in result.stdout


@pytest.mark.asyncio
async def test_dw_list_shows_threads_from_per_agent_db(isolated_home):
    """`deep-worker --list` aggregates threads across all
    sessions_*.db files, not just the legacy global."""
    now = datetime.now(UTC).isoformat()

    # Two per-agent DBs with different threads.
    await _seed_per_agent_db(
        isolated_home / ".deepagents" / "sessions_list_a.db",
        thread_id="list_a",
        content="a",
        agent_name_meta="list_a",
        updated_at=now,
    )
    await _seed_per_agent_db(
        isolated_home / ".deepagents" / "sessions_list_b.db",
        thread_id="list_b",
        content="b",
        agent_name_meta="list_b",
        updated_at=now,
    )

    result = _run_dw_cli("--list", home=isolated_home)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "list_a" in result.stdout, (
        f"list_a not present in --list output:\n{result.stdout}"
    )
    assert "list_b" in result.stdout, (
        f"list_b not present in --list output:\n{result.stdout}"
    )


@pytest.mark.asyncio
async def test_dw_remove_before_cleans_per_agent_dbs(isolated_home):
    """`deep-worker --remove-before <cutoff>` sweeps every
    sessions_<agent>.db file, not only the legacy global."""
    old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    recent_ts = datetime.now(UTC).isoformat()

    old_db = isolated_home / ".deepagents" / "sessions_old_one.db"
    recent_db = isolated_home / ".deepagents" / "sessions_new_one.db"

    await _seed_per_agent_db(
        old_db,
        thread_id="old_one",
        content="old",
        agent_name_meta="old_one",
        updated_at=old_ts,
    )
    await _seed_per_agent_db(
        recent_db,
        thread_id="new_one",
        content="new",
        agent_name_meta="new_one",
        updated_at=recent_ts,
    )

    # Dry run first.
    result = _run_dw_cli(
        "--remove-before", "30d", "--dry-run", home=isolated_home
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "old_one" in result.stdout or "1" in result.stdout  # count visible

    # Real run.
    result = _run_dw_cli(
        "--remove-before", "30d", home=isolated_home
    )
    assert result.returncode == 0, result.stderr or result.stdout

    # Old thread gone.
    async with aiosqlite.connect(str(old_db)) as conn:
        async with conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("old_one",)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
    # Recent thread preserved.
    async with aiosqlite.connect(str(recent_db)) as conn:
        async with conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("new_one",)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1


def test_dw_history_missing_thread_has_clean_error(isolated_home):
    """Missing thread → non-zero exit with a human-readable error
    (not a raw SQLite traceback)."""
    result = _run_dw_cli("--history", "definitely_missing", home=isolated_home)

    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    # No raw SQLite traceback leaks.
    assert "no such table" not in combined
    assert "not found" in combined
