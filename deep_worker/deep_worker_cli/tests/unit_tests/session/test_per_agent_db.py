"""Tests for per-agent SQLite database isolation.

BUG: All agents share a single sessions.db file. With multiple concurrent
agents writing large checkpoints, SQLite "database is locked" errors occur.
The shared file also grows to tens of GB without any per-agent cleanup.

Fix: Each agent gets its own DB file under its own root:
``~/.deep_worker/agents/<agent_name>/threads.db``.
"""

from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import DeepagentsPathForbiddenError
from deep_worker_cli.sessions import get_db_path


class TestPerAgentDb:
    """Verify that get_db_path returns per-agent paths."""

    def test_get_db_path_with_agent_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_db_path(agent_name) must return agent-specific path."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path = get_db_path(agent_name="agent_1")
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

    def test_global_db_path_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The v1 global DB lived in the banned tree, so it must raise.

        Per-agent resolution is unaffected and stays in the product tree.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(DeepagentsPathForbiddenError):
            get_db_path()
        assert get_db_path(agent_name="test_agent").parent.name == "test_agent"


class TestNoMigrationFromTheAncestorsTree:
    """POLICIES §16: no import path out of ``~/.deepagents`` exists.

    This class replaced ``TestMigrateFromGlobalDb``, which asserted that
    creating a per-agent DB copied checkpoints and writes out of the legacy
    global ``sessions.db``. That DB lived in the ancestor's tree, which is
    banned for reads and writes alike, so the shim was removed rather than
    repointed: migrating an existing install is a one-time, out-of-band
    action, never a branch inside the product's own lookup path.
    """

    def test_migration_helper_no_longer_exists(self) -> None:
        """The shim must stay gone — re-adding it reintroduces the read."""
        import deep_worker_cli.sessions as sessions

        assert not hasattr(sessions, "_migrate_from_global_db"), (
            "_migrate_from_global_db reads ~/.deepagents (POLICIES §16); "
            "it must not be reintroduced"
        )

    @pytest.mark.asyncio
    async def test_checkpointer_creates_a_fresh_per_agent_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opening a checkpointer must not consult any global DB."""
        from deep_worker_cli.sessions import get_checkpointer

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        async with get_checkpointer("freshagent"):
            pass

        db = tmp_path / ".deep_worker" / "agents" / "freshagent" / "threads.db"
        assert db.exists(), "per-agent DB was not created"
        assert not (tmp_path / ".deepagents").exists(), (
            "opening a checkpointer touched ~/.deepagents (POLICIES §16)"
        )
