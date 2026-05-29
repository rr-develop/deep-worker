"""Tests for per-agent SQLite database isolation.

BUG: All agents share a single sessions.db file. With multiple concurrent
agents writing large checkpoints, SQLite "database is locked" errors occur.
The shared file also grows to tens of GB without any per-agent cleanup.

Fix: Each agent gets its own DB file: sessions_{agent_name}.db
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.sessions import _migrate_from_global_db, get_db_path


class TestPerAgentDb:
    """Verify that get_db_path returns per-agent paths."""

    def test_get_db_path_with_agent_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_db_path(agent_name) must return agent-specific path."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path = get_db_path(agent_name="agent_a")
        assert path.name == "threads.db", (
            f"Expected threads.db, got {path.name}"
        )

    def test_get_db_path_different_agents_different_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different agents must get different DB files."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path_a = get_db_path(agent_name="agent_a")
        path_b = get_db_path(agent_name="agent_b")
        assert path_a != path_b
        assert "agent_a" in str(path_a)
        assert "agent_b" in str(path_b)

    def test_get_db_path_without_agent_name_returns_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compat: no agent_name returns the global sessions.db."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path = get_db_path()
        assert path.name == "sessions.db"

    def test_get_db_path_same_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Global legacy and v3 agent DB files use different roots."""
        monkeypatch.setenv("HOME", str(tmp_path))
        global_path = get_db_path()
        agent_path = get_db_path(agent_name="test_agent")
        assert global_path.parent.name == ".deepagents"
        assert agent_path.parent.name == "test_agent"


class TestMigrateFromGlobalDb:
    """Verify automatic migration from global sessions.db to per-agent DB."""

    def test_migrates_checkpoints_and_writes(self, tmp_path: Path) -> None:
        """Data for agent should be copied from global to per-agent DB."""
        # Create global DB with test data
        global_db = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(global_db))
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, parent_checkpoint_id TEXT, type TEXT, "
            "checkpoint BLOB, metadata BLOB, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
        )
        conn.execute(
            "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, task_id TEXT, idx INTEGER, "
            "channel TEXT, type TEXT, value BLOB, "
            "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))"
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            ("myagent", "", "cp1", None, "json", b"data1", b"meta1"),
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            ("myagent:tm", "", "cp2", None, "json", b"data2", b"meta2"),
        )
        conn.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            ("other_agent", "", "cp3", None, "json", b"data3", b"meta3"),
        )
        conn.execute(
            "INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
            ("myagent", "", "cp1", "t1", 0, "ch", "json", b"val1"),
        )
        conn.commit()
        conn.close()

        # Migrate
        agent_db = tmp_path / "sessions_myagent.db"
        with patch("deep_worker_cli.sessions.get_db_path", side_effect=lambda a=None: global_db if a is None else agent_db):
            _migrate_from_global_db("myagent", agent_db)

        # Verify
        dst = sqlite3.connect(str(agent_db))
        cps = dst.execute("SELECT thread_id FROM checkpoints ORDER BY thread_id").fetchall()
        assert len(cps) == 2
        assert cps[0][0] == "myagent"
        assert cps[1][0] == "myagent:tm"

        writes = dst.execute("SELECT thread_id FROM writes").fetchall()
        assert len(writes) == 1
        assert writes[0][0] == "myagent"

        dst.close()

    def test_skips_if_agent_db_already_has_data(self, tmp_path: Path) -> None:
        """Migration should not run twice."""
        global_db = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(global_db))
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY, data TEXT)"
        )
        conn.execute("INSERT INTO checkpoints VALUES ('agent', 'data')")
        conn.commit()
        conn.close()

        # Create per-agent DB with existing data (> 8192 bytes)
        agent_db = tmp_path / "sessions_agent.db"
        dst = sqlite3.connect(str(agent_db))
        dst.execute("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY, data TEXT)")
        # Insert enough data to exceed 8192 bytes
        for i in range(100):
            dst.execute(f"INSERT INTO checkpoints VALUES ('t{i}', '{'x' * 100}')")
        dst.commit()
        dst.close()

        original_size = agent_db.stat().st_size
        assert original_size > 8192

        with patch("deep_worker_cli.sessions.get_db_path", return_value=global_db):
            _migrate_from_global_db("agent", agent_db)

        # Size should not change (migration skipped)
        assert agent_db.stat().st_size == original_size

    def test_no_crash_if_global_db_missing(self, tmp_path: Path) -> None:
        """Migration should not crash if global DB doesn't exist."""
        agent_db = tmp_path / "sessions_agent.db"
        with patch("deep_worker_cli.sessions.get_db_path", return_value=tmp_path / "nonexistent.db"):
            _migrate_from_global_db("agent", agent_db)
        # Should not crash
