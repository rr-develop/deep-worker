"""Code revision must be readable from session_metadata.json at startup.

An agent's metadata has to answer "which code is this running?" immediately,
without grepping the log line and without waiting for the agent to file an
infrastructure report. These tests pin that contract.
"""

import json
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for session data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _read(data_dir: Path) -> dict:
    return json.loads((data_dir / "session_metadata.json").read_text())


class TestProvenanceWrittenAtStartup:
    """The revision lands on disk as soon as the SessionManager exists."""

    def test_commit_and_dirty_are_persisted_immediately(self, temp_data_dir):
        sha = "a" * 40
        SessionManager(data_dir=temp_data_dir, commit=sha, dirty=False)

        meta = _read(temp_data_dir)
        assert meta["commit"] == sha
        assert meta["dirty"] is False

    def test_provenance_is_top_level_not_inside_launch_config(self, temp_data_dir):
        """launch_config is replayed on --restart and would pin a stale SHA.

        Storing the revision there would reintroduce exactly the bug the
        capture path avoids: reporting the PREVIOUS process's commit after a
        restart taken to pick up new code.
        """
        sha = "b" * 40
        SessionManager(
            data_dir=temp_data_dir,
            launch_config={"provider": "claude-code", "tree_root": "/path/to/tree"},
            commit=sha,
            dirty=True,
        )

        meta = _read(temp_data_dir)
        assert meta["commit"] == sha
        assert meta["dirty"] is True
        assert "commit" not in meta["launch_config"]
        assert "dirty" not in meta["launch_config"]

    def test_fields_always_present_even_when_unknown(self, temp_data_dir):
        """Consumers must never have to special-case a missing key."""
        SessionManager(data_dir=temp_data_dir, launch_config={"provider": "x"})

        meta = _read(temp_data_dir)
        assert "commit" in meta and "dirty" in meta
        assert meta["commit"] is None
        assert meta["dirty"] is None

    def test_unknown_provenance_is_null_not_a_fake_clean_tree(self, temp_data_dir):
        """dirty=None must not be collapsed to False.

        Reporting "clean" on evidence we never had would be a fabricated
        claim about the running code.
        """
        SessionManager(data_dir=temp_data_dir, commit=None, dirty=None)

        meta = _read(temp_data_dir)
        assert meta["dirty"] is not False
        assert meta["dirty"] is None


class TestProvenanceSurvivesUpdates:
    """Later persists must not blank the revision recorded at startup."""

    @pytest.mark.asyncio
    async def test_status_change_preserves_provenance(self, temp_data_dir):
        sha = "c" * 40
        manager = SessionManager(data_dir=temp_data_dir, commit=sha, dirty=False)

        await manager.set_status(AgentStatus.RUNNING)

        meta = _read(temp_data_dir)
        assert meta["status"] == AgentStatus.RUNNING.value
        assert meta["commit"] == sha, "set_status must not wipe the revision"
        assert meta["dirty"] is False

    @pytest.mark.asyncio
    async def test_error_and_counters_preserve_provenance(self, temp_data_dir):
        sha = "d" * 40
        manager = SessionManager(data_dir=temp_data_dir, commit=sha, dirty=True)

        await manager.set_error("boom")
        await manager.increment_message_count()

        meta = _read(temp_data_dir)
        assert meta["commit"] == sha
        assert meta["dirty"] is True

    def test_reload_carries_provenance_forward(self, temp_data_dir):
        """A manager reopened without provenance keeps what is on disk."""
        sha = "e" * 40
        SessionManager(data_dir=temp_data_dir, commit=sha, dirty=False)

        reopened = SessionManager(data_dir=temp_data_dir)
        assert reopened.commit == sha
        assert reopened.dirty is False

    def test_new_process_revision_overrides_the_recorded_one(self, temp_data_dir):
        """A restart onto new code must overwrite the previous SHA."""
        SessionManager(data_dir=temp_data_dir, commit="f" * 40, dirty=True)

        fresh = "0" * 40
        SessionManager(data_dir=temp_data_dir, commit=fresh, dirty=False)

        meta = _read(temp_data_dir)
        assert meta["commit"] == fresh
        assert meta["dirty"] is False

    def test_omitted_provenance_leaves_the_recorded_file_untouched(self, temp_data_dir):
        """Bare tooling makes no claim about the running code.

        A caller that never mentions provenance (dwcli inspecting a
        directory, a test opening it) must leave the owning process's
        revision on DISK intact — not merely in memory.
        """
        sha = "1" * 40
        SessionManager(data_dir=temp_data_dir, commit=sha, dirty=False)

        SessionManager(data_dir=temp_data_dir)

        meta = _read(temp_data_dir)
        assert meta["commit"] == sha, "bare SessionManager must not wipe recorded provenance"
        assert meta["dirty"] is False

    def test_explicit_none_clears_a_stale_recorded_revision(self, temp_data_dir):
        """Explicit None means "unknown NOW" and must overwrite the old SHA.

        The stale-provenance trap: a previous process recorded a known
        commit; this process could not determine its tree and passes None.
        Inheriting the predecessor's SHA would make the agent report code
        it is not running — worse than reporting nothing.
        """
        SessionManager(data_dir=temp_data_dir, commit="2" * 40, dirty=False)

        SessionManager(
            data_dir=temp_data_dir,
            launch_config={"provider": "claude-code"},
            commit=None,
            dirty=None,
        )

        meta = _read(temp_data_dir)
        assert meta["commit"] is None, "explicit None must clear the predecessor's SHA"
        assert meta["dirty"] is None

    def test_legacy_metadata_without_provenance_loads_as_null(self, temp_data_dir):
        """Files written before this field existed must not crash or lie."""
        legacy = {
            "status": "idle",
            "tasks_completed": 0,
            "message_count": 0,
            "error_message": None,
            "last_activity": "2026-08-28T10:00:00",
            "storage_version": 6,
        }
        (temp_data_dir / "session_metadata.json").write_text(json.dumps(legacy))

        manager = SessionManager(data_dir=temp_data_dir)
        assert manager.commit is None
        assert manager.dirty is None
