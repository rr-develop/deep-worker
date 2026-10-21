"""Unit tests for per-agent DB resolution in offline session helpers.

Covers the bug where `deep-worker --history <agent>` (and friends)
read a global sessions DB instead of the agent's own file, which now
lives at ``~/.deep_worker/agents/<agent>/threads.db``.

The v1/v2 layouts these helpers used to fall back to lived in the
ancestor's ``~/.deepagents`` tree. POLICIES §16 bans that tree for reads
and writes alike, so the tests naming it here pin the *ban*: seeded data
there must stay invisible and untouched.

See: docs/DESIGN_FIX_DEEPWORKER_HISTORY_PER_AGENT_DB.md
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
    """Create a per-agent DB with a single checkpoint for thread_id.

    Uses the real `DeltaSqliteCheckpointer` rather than a hand-rolled
    `CREATE TABLE checkpoints (...)`. The delta-aware reader
    (`aget_tuple`) requires schema columns that the hand-rolled
    variant didn't populate — `checkpoint_ns`, `parent_checkpoint_id`,
    `metadata_blob`, etc. Using the real checkpointer keeps the
    schema in sync with production no matter how it evolves.
    """
    from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import empty_checkpoint

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if ts is None:
        ts = datetime.now(UTC).isoformat()

    meta: dict = {}
    if metadata_updated_at is not None:
        meta["updated_at"] = metadata_updated_at
    if agent_name_meta is not None:
        meta["agent_name"] = agent_name_meta

    # NB: use UUID7 from langgraph rather than `uuid4().hex`. The
    # delta checkpointer resolves "latest" via
    # `ORDER BY checkpoint_id DESC` which is lexicographic — UUID4
    # is not monotonic and makes the result random; UUID7 is
    # time-prefixed and matches the production id allocator.
    checkpoint = {
        "v": 1,
        "id": empty_checkpoint()["id"],
        "ts": ts,
        "channel_values": {
            "messages": [HumanMessage(content=content)],
            "working_dir": "/tmp",
        },
        "channel_versions": {"messages": "00000000000000000000000000000001.0"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["messages"],
    }
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with DeltaSqliteCheckpointer.from_conn_string(str(db_path)) as cp:
        await cp.aput(config, checkpoint, meta, {})


def _agent_db(home: Path, agent_name: str) -> Path:
    """The per-agent DB under the v3 layout: ``agents/<name>/threads.db``."""
    return home / ".deep_worker" / "agents" / agent_name / "threads.db"


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ~/.deep_worker at tmp_path/home/.deep_worker for the test."""
    home = tmp_path / "home"
    (home / ".deep_worker" / "agents").mkdir(parents=True)
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
        per_agent_db = _agent_db(fake_home, agent)
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
    async def test_reads_per_agent_db_when_foreign_db_is_empty(self, fake_home):
        """A stray 0-byte DB in the ancestor's tree must not disturb lookup.

        This is the shape seen on the dev machine that produced the original
        bug report. Under POLICIES §16 that file is never consulted at all,
        so the per-agent DB must be served regardless.
        """
        from deep_worker_cli.sessions import get_thread_history

        foreign = fake_home / ".deepagents"
        foreign.mkdir(parents=True, exist_ok=True)
        (foreign / "sessions.db").write_bytes(b"")

        agent = "agent_1"
        per_agent_db = _agent_db(fake_home, agent)
        await _seed_agent_db(per_agent_db, thread_id=agent, content="agent_2 ping")

        result = await get_thread_history(agent)

        assert result["success"] is True, result
        assert any(
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            == "agent_2 ping"
            for m in result["messages"]
        )

    @pytest.mark.asyncio
    async def test_derives_agent_from_tm_thread_id(self, fake_home):
        """thread_id '<agent>:tm' → agent DB 'sessions_<agent>.db'."""
        from deep_worker_cli.sessions import get_thread_history

        agent = "myagent"
        per_agent_db = _agent_db(fake_home, agent)
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
    async def test_does_not_fall_back_to_the_ancestors_tree(self, fake_home):
        """POLICIES §16: a DB in ``~/.deepagents`` must never be consulted.

        Previously this asserted the opposite — that a legacy global
        ``sessions.db`` was read when the per-agent file was missing. Reading
        the ancestor's tree is now banned on the same terms as writing, so
        seeded data there must stay invisible: an absent per-agent DB reports
        as absent rather than silently making foreign state authoritative.
        """
        from deep_worker_cli.sessions import get_thread_history

        foreign_db = fake_home / ".deepagents" / "sessions.db"
        await _seed_agent_db(foreign_db, thread_id="legacy_agent", content="legacy")

        result = await get_thread_history("legacy_agent")

        contents = [
            (m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in result.get("messages") or []
        ]
        assert "legacy" not in contents, (
            "history was served out of ~/.deepagents, which POLICIES §16 bans"
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
            _agent_db(fake_home, agent),
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
            _agent_db(fake_home, "agent_a"),
            thread_id="agent_a",
            metadata_updated_at=now,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            _agent_db(fake_home, "agent_b"),
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
            _agent_db(fake_home, "agent_a"),
            thread_id="agent_a",
            metadata_updated_at=now,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            _agent_db(fake_home, "agent_b"),
            thread_id="agent_b",
            metadata_updated_at=now,
            agent_name_meta="agent_b",
        )

        threads = await list_threads(agent_name="agent_a", limit=20)
        tids = {t["thread_id"] for t in threads}
        assert tids == {"agent_a"}

    @pytest.mark.asyncio
    async def test_list_threads_excludes_the_ancestors_tree(self, fake_home):
        """POLICIES §16: ``--list`` must not surface threads from ``.deepagents``.

        This asserted the opposite before the ban: that a legacy global
        ``sessions.db`` still contributed rows. Aggregating the ancestor's
        tree would put foreign threads in our listing.
        """
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
        assert "legacy" not in tids, (
            "list_threads aggregated ~/.deepagents, which POLICIES §16 bans"
        )

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
        per_agent_db = _agent_db(fake_home, agent)
        await _seed_agent_db(per_agent_db, thread_id=agent)
        assert per_agent_db.exists()

        await clean_agent_history(agent, _home_dir=fake_home)

        assert not per_agent_db.exists(), (
            f"sessions_{agent}.db must be removed after clean"
        )

    @pytest.mark.asyncio
    async def test_clean_leaves_the_ancestors_tree_untouched(self, fake_home):
        """POLICIES §16: ``--clean`` must not write into ``.deepagents``.

        This previously asserted the opposite — that stale rows in the legacy
        global DB were scrubbed too. Deleting rows there is a write to a
        foreign project's namespace, which the policy bans outright, so the
        rows must survive while our own per-agent DB is removed.
        """
        from deep_worker_cli.sessions import clean_agent_history

        agent = "bothplaces"
        foreign_db = fake_home / ".deepagents" / "sessions.db"
        await _seed_agent_db(foreign_db, thread_id=agent, agent_name_meta=agent)
        await _seed_agent_db(
            _agent_db(fake_home, agent),
            thread_id=agent,
            agent_name_meta=agent,
        )

        await clean_agent_history(agent, _home_dir=fake_home)

        # Our own per-agent file is gone.
        assert not (_agent_db(fake_home, agent)).exists()

        # The foreign tree is left exactly as it was.
        async with aiosqlite.connect(str(foreign_db)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (agent,)
            ) as cursor:
                count = (await cursor.fetchone())[0]
        assert count >= 1, "clean scrubbed ~/.deepagents, which POLICIES §16 bans"

    @pytest.mark.asyncio
    async def test_clean_db_only_deletes_per_agent_db_file(self, fake_home):
        from deep_worker_cli.sessions import clean_agent_history_db_only

        agent = "quickclean"
        per_agent_db = _agent_db(fake_home, agent)
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
            _agent_db(fake_home, "agent_a"),
            thread_id="agent_a_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_a",
        )
        await _seed_agent_db(
            _agent_db(fake_home, "agent_b"),
            thread_id="agent_b_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_b",
        )
        await _seed_agent_db(
            _agent_db(fake_home, "agent_c"),
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
            str(_agent_db(fake_home, "agent_a"))
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("agent_a_old",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0

        async with aiosqlite.connect(
            str(_agent_db(fake_home, "agent_c"))
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
            _agent_db(fake_home, "agent_a"),
            thread_id="agent_a_old",
            metadata_updated_at=old_ts,
            agent_name_meta="agent_a",
        )

        result = await remove_threads_before("30d", dry_run=True)
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["threads_found"] >= 1

        async with aiosqlite.connect(
            str(_agent_db(fake_home, "agent_a"))
        ) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("agent_a_old",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
