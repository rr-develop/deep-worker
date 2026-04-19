"""Unit tests for per-agent DB resolution in offline session helpers.

Covers the bug where `deep-worker --history <agent>` (and friends)
read the legacy global ~/.deepagents/sessions.db instead of the
per-agent file ~/.deepagents/sessions_<agent>.db.

See: docs/DESIGN_FIX_DW_HISTORY_PER_AGENT_DB.md
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from pathlib import Path

import aiosqlite
import pytest


def _ensure_checkpoints_schema_sql() -> str:
    """Schema compatible with the columns referenced by sessions.py."""
    return """
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT,
            checkpoint_id TEXT,
            parent_checkpoint_id TEXT,
            checkpoint_ns TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata TEXT
        )
    """


def _ensure_writes_schema_sql() -> str:
    return """
        CREATE TABLE IF NOT EXISTS writes (
            thread_id TEXT,
            task_id TEXT
        )
    """


async def _seed_agent_db(
    db_path: Path,
    thread_id: str,
    content: str = "Hello",
    ts: str | None = None,
    metadata_updated_at: str | None = None,
    agent_name_meta: str | None = None,
) -> None:
    """Create a minimal per-agent DB with a single checkpoint for thread_id."""
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute(_ensure_checkpoints_schema_sql())
        await conn.execute(_ensure_writes_schema_sql())

        if ts is None:
            ts = datetime.now(UTC).isoformat()

        checkpoint_data = json.dumps(
            {
                "ts": ts,
                "channel_values": {
                    "messages": [
                        {"type": "human", "content": content},
                    ]
                },
            }
        )

        meta: dict = {}
        if metadata_updated_at is not None:
            meta["updated_at"] = metadata_updated_at
        if agent_name_meta is not None:
            meta["agent_name"] = agent_name_meta
        metadata_str = json.dumps(meta)

        await conn.execute(
            """
            INSERT INTO checkpoints
            (thread_id, checkpoint_id, type, checkpoint, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, f"cp-{thread_id}", "json", checkpoint_data, metadata_str),
        )
        await conn.commit()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ~/.deepagents at tmp_path/.deepagents for the test."""
    home = tmp_path / "home"
    (home / ".deepagents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


# ---------------------------------------------------------------------------
# get_thread_history
# ---------------------------------------------------------------------------


class TestGetThreadHistoryPerAgentDB:
    @pytest.mark.asyncio
    async def test_reads_per_agent_db_when_global_missing(self, fake_home):
        """Reproduces the exact reported bug: per-agent file has data, global
        file has no checkpoints table. --history must still succeed."""
        from deep_worker_cli.sessions import get_thread_history

        agent = "myagent"
        per_agent_db = fake_home / ".deepagents" / f"sessions_{agent}.db"
        await _seed_agent_db(per_agent_db, thread_id=agent, content="hi from myagent")

        # Legacy global file does NOT exist. In the real-world failure it
        # existed but was empty — both cases must work.

        result = await get_thread_history(agent)

        assert result["success"] is True, result
        assert result["thread_id"] == agent
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "hi from myagent"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_reads_per_agent_db_when_global_empty(self, fake_home):
        """Global sessions.db exists but has no checkpoints table — exact
        scenario on the dev machine that produced the bug report."""
        from deep_worker_cli.sessions import get_thread_history

        # Empty legacy file (0 bytes is the shape seen in the bug report).
        (fake_home / ".deepagents" / "sessions.db").write_bytes(b"")

        agent = "lc4_test"
        per_agent_db = fake_home / ".deepagents" / f"sessions_{agent}.db"
        await _seed_agent_db(per_agent_db, thread_id=agent, content="lc4 ping")

        result = await get_thread_history(agent)

        assert result["success"] is True, result
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "lc4 ping"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_derives_agent_from_tm_thread_id(self, fake_home):
        """thread_id '<agent>:tm' → agent DB 'sessions_<agent>.db'."""
        from deep_worker_cli.sessions import get_thread_history

        agent = "myagent"
        per_agent_db = fake_home / ".deepagents" / f"sessions_{agent}.db"
        await _seed_agent_db(per_agent_db, thread_id=f"{agent}:tm", content="tm msg")

        result = await get_thread_history(f"{agent}:tm")

        assert result["success"] is True, result
        assert result["thread_id"] == f"{agent}:tm"
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "tm msg"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_global_when_per_agent_missing(self, fake_home):
        """Legacy installs with only the global DB still work."""
        from deep_worker_cli.sessions import get_thread_history

        global_db = fake_home / ".deepagents" / "sessions.db"
        await _seed_agent_db(global_db, thread_id="legacy_agent", content="legacy")

        result = await get_thread_history("legacy_agent")

        assert result["success"] is True, result
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "legacy"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_explicit_db_path_still_wins(self, fake_home, tmp_path):
        """Passing db_path explicitly must override auto-resolution — keeps
        existing unit tests behaving identically."""
        from deep_worker_cli.sessions import get_thread_history

        # Seed per-agent DB with one content and explicit DB with another —
        # the explicit one must be the one that's read.
        agent = "myagent"
        await _seed_agent_db(
            fake_home / ".deepagents" / f"sessions_{agent}.db",
            thread_id=agent,
            content="from per-agent",
        )
        explicit_db = tmp_path / "explicit.db"
        await _seed_agent_db(explicit_db, thread_id=agent, content="from explicit")

        result = await get_thread_history(agent, db_path=str(explicit_db))

        assert result["success"] is True
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "from explicit"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_returns_error_when_no_db_anywhere(self, fake_home):
        """Clear error message when neither per-agent nor global DB exist."""
        from deep_worker_cli.sessions import get_thread_history

        result = await get_thread_history("ghost_agent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# list_threads
# ---------------------------------------------------------------------------


class TestListThreadsPerAgentDB:
    @pytest.mark.asyncio
    async def test_list_threads_aggregates_per_agent_files(self, fake_home):
        """When agent_name is None, list_threads iterates over every
        sessions_*.db file and merges results."""
        from deep_worker_cli.sessions import list_threads

        now = datetime.now(UTC).isoformat()
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_a.db",
            thread_id="agent_a",
            metadata_updated_at=now,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_b.db",
            thread_id="agent_b",
            metadata_updated_at=now,
            agent_name_meta="agent_b",
        )

        threads = await list_threads(limit=20)
        tids = {t["thread_id"] for t in threads}
        assert "agent_a" in tids
        assert "agent_b" in tids

    @pytest.mark.asyncio
    async def test_list_threads_filter_reads_only_that_agent_db(self, fake_home):
        """agent_name filter reads only sessions_<agent>.db."""
        from deep_worker_cli.sessions import list_threads

        now = datetime.now(UTC).isoformat()
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_a.db",
            thread_id="agent_a",
            metadata_updated_at=now,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_b.db",
            thread_id="agent_b",
            metadata_updated_at=now,
            agent_name_meta="agent_b",
        )

        threads = await list_threads(agent_name="agent_a", limit=20)
        tids = {t["thread_id"] for t in threads}
        assert tids == {"agent_a"}

    @pytest.mark.asyncio
    async def test_list_threads_includes_legacy_global(self, fake_home):
        """Legacy global sessions.db still contributes when it has data."""
        from deep_worker_cli.sessions import list_threads

        now = datetime.now(UTC).isoformat()
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions.db",
            thread_id="legacy",
            metadata_updated_at=now,
            agent_name_meta="legacy",
        )

        threads = await list_threads(limit=20)
        tids = {t["thread_id"] for t in threads}
        assert "legacy" in tids

    @pytest.mark.asyncio
    async def test_list_threads_empty_when_no_dbs(self, fake_home):
        from deep_worker_cli.sessions import list_threads

        threads = await list_threads(limit=20)
        assert threads == []


# ---------------------------------------------------------------------------
# clean_agent_history
# ---------------------------------------------------------------------------


class TestCleanAgentHistoryPerAgentDB:
    @pytest.mark.asyncio
    async def test_clean_deletes_per_agent_db_file(self, fake_home):
        """`deep-worker --clean <agent>` must actually remove the per-agent DB."""
        from deep_worker_cli.sessions import clean_agent_history

        agent = "cleanme"
        per_agent_db = fake_home / ".deepagents" / f"sessions_{agent}.db"
        await _seed_agent_db(per_agent_db, thread_id=agent)
        assert per_agent_db.exists()

        await clean_agent_history(agent, _home_dir=fake_home)

        assert not per_agent_db.exists(), (
            f"sessions_{agent}.db must be removed after clean"
        )

    @pytest.mark.asyncio
    async def test_clean_also_removes_rows_from_legacy_global(self, fake_home):
        """Backward compat: if the legacy global still has stale rows for this
        agent, they should be cleaned up too."""
        from deep_worker_cli.sessions import clean_agent_history

        agent = "bothplaces"
        # Global still has rows from before migration
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions.db",
            thread_id=agent,
            agent_name_meta=agent,
        )
        # Per-agent file also has rows
        await _seed_agent_db(
            fake_home / ".deepagents" / f"sessions_{agent}.db",
            thread_id=agent,
            agent_name_meta=agent,
        )

        await clean_agent_history(agent, _home_dir=fake_home)

        # Per-agent file gone
        assert not (fake_home / ".deepagents" / f"sessions_{agent}.db").exists()

        # Global rows for this agent gone
        async with aiosqlite.connect(
            str(fake_home / ".deepagents" / "sessions.db")
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (agent,)
            ) as cursor:
                count = (await cursor.fetchone())[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_clean_db_only_deletes_per_agent_db_file(self, fake_home):
        from deep_worker_cli.sessions import clean_agent_history_db_only

        agent = "quickclean"
        per_agent_db = fake_home / ".deepagents" / f"sessions_{agent}.db"
        await _seed_agent_db(per_agent_db, thread_id=agent)
        assert per_agent_db.exists()

        checkpoints_deleted, _writes_deleted = await clean_agent_history_db_only(agent)

        assert checkpoints_deleted >= 1
        assert not per_agent_db.exists()


# ---------------------------------------------------------------------------
# remove_threads_before
# ---------------------------------------------------------------------------


class TestRemoveThreadsBeforePerAgentDB:
    @pytest.mark.asyncio
    async def test_removes_old_threads_across_per_agent_dbs(self, fake_home):
        """Without explicit db_path, function must iterate every sessions_*.db."""
        from deep_worker_cli.sessions import remove_threads_before

        old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        recent_ts = datetime.now(UTC).isoformat()

        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_a.db",
            thread_id="agent_a_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_b.db",
            thread_id="agent_b_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_b",
        )
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_c.db",
            thread_id="agent_c_recent",
            metadata_updated_at=recent_ts,
            agent_name_meta="agent_c",
        )

        result = await remove_threads_before("30d", dry_run=False)

        assert result["success"] is True
        # Both old threads from two different per-agent DBs deleted.
        assert result["threads_deleted"] >= 2

        # Confirm physically deleted
        async with aiosqlite.connect(
            str(fake_home / ".deepagents" / "sessions_agent_a.db")
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("agent_a_old",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0

        async with aiosqlite.connect(
            str(fake_home / ".deepagents" / "sessions_agent_c.db")
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("agent_c_recent",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_dry_run_does_not_delete_anything(self, fake_home):
        from deep_worker_cli.sessions import remove_threads_before

        old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        await _seed_agent_db(
            fake_home / ".deepagents" / "sessions_agent_a.db",
            thread_id="agent_a_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_a",
        )

        result = await remove_threads_before("30d", dry_run=True)
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["threads_found"] >= 1

        async with aiosqlite.connect(
            str(fake_home / ".deepagents" / "sessions_agent_a.db")
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("agent_a_old",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
