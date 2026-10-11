"""Tests for agent databases under ``~/.deep_worker/agents/<name>/``.

DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR: per-agent directories live under
``agents/<name>/`` instead of the top-level ``~/.deep_worker/<name>/``.
The legacy location is kept as a read-only fallback for backward
compatibility and is walked by the migration CLI.

These tests use the ``deep_worker_home`` accessor monkeypatch per the
design doc §6.5 (avoids the ``HOME`` env var, which async tasks can
cache and which would leak between tests).
"""

from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli import storage_paths
from deep_worker_cli.storage_paths import (
    CURRENT_STORAGE_VERSION,
    HOME_DIR_NAME,
    STORAGE_VERSION_AGENTS_SUBDIR,
    STORAGE_VERSION_AGENT_ROOT,
    STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    STORAGE_VERSION_GLOBAL_DB,
    STORAGE_VERSION_WITH_PREFERRED_PROVIDER,
    agent_root,
    deep_worker_home,
    detect_storage_version,
    iter_legacy_thread_db_paths,
    iter_thread_db_paths,
    legacy_agent_root,
    migrate_agent_storage_to_current,
    migrate_agent_storage_to_v6,
    read_agent_file,
)


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect storage home resolution to a per-test tmp dir (design §6.5).

    ``deep_worker_home`` is re-pointed so all path helpers follow the
    test's tmp directory. There is no legacy-tree equivalent: reading
    ``~/.deepagents`` is banned (POLICIES §16).
    """
    home = tmp_path / ("." + HOME_DIR_NAME)
    monkeypatch.setattr(storage_paths, "deep_worker_home", lambda h=None: home)
    return home


# ---------------------------------------------------------------------------
# agent_root / legacy_agent_root / read_agent_file
# ---------------------------------------------------------------------------


def test_agent_root_creates_path_under_agents_subdir(fake_home: Path) -> None:
    root = agent_root("foo")
    assert root == fake_home / "agents" / "foo"
    assert root.is_dir()
    assert oct(root.stat().st_mode & 0o777) == "0o700"


def test_agent_root_create_false_does_not_create(fake_home: Path) -> None:
    root = agent_root("foo", create=False)
    assert root == fake_home / "agents" / "foo"
    assert not root.exists()


def test_legacy_agent_root_returns_top_level_path(fake_home: Path) -> None:
    assert legacy_agent_root("foo") == fake_home / "foo"


def test_read_agent_file_prefers_new_path(fake_home: Path) -> None:
    new_dir = agent_root("foo", create=True)
    (new_dir / "x.json").write_bytes(b"new")

    assert read_agent_file("foo", "x.json") == b"new"


def test_read_agent_file_falls_back_to_legacy(fake_home: Path) -> None:
    legacy = legacy_agent_root("foo")
    legacy.mkdir(parents=True)
    (legacy / "x.json").write_bytes(b"legacy")

    assert read_agent_file("foo", "x.json") == b"legacy"


def test_read_agent_file_new_wins_when_both_exist(fake_home: Path) -> None:
    """Both new and legacy files exist → new wins (no merge)."""
    new_dir = agent_root("foo", create=True)
    (new_dir / "x.json").write_bytes(b"new")
    legacy = legacy_agent_root("foo")
    legacy.mkdir(parents=True)
    (legacy / "x.json").write_bytes(b"legacy")

    assert read_agent_file("foo", "x.json") == b"new"


def test_read_agent_file_returns_none_when_missing(fake_home: Path) -> None:
    assert read_agent_file("foo", "x.json") is None


def test_iter_thread_db_paths_returns_only_new_layout(fake_home: Path) -> None:
    """Main API does NOT return legacy thread DBs (design §4.1)."""
    new_dir = agent_root("new_agent", create=True)
    (new_dir / "threads.db").write_bytes(b"")
    legacy = legacy_agent_root("legacy_agent")
    legacy.mkdir(parents=True)
    (legacy / "threads.db").write_bytes(b"")

    paths = iter_thread_db_paths()
    names = [p.parent.name for p in paths]
    assert "new_agent" in names
    assert "legacy_agent" not in names


def test_iter_legacy_thread_db_paths_returns_only_legacy(fake_home: Path) -> None:
    """Walk the legacy top-level layout: return the agent DIR itself
    (not the threads.db inside it), so dirs without threads.db — test
    dirs whose DB was already cleaned up, etc. — are still picked up by
    the migration CLI classifier. The new ``agents/`` subdir must be
    excluded so the legacy walker never double-counts an already-migrated
    agent."""
    new_dir = agent_root("new_agent", create=True)
    (new_dir / "threads.db").write_bytes(b"")
    legacy = legacy_agent_root("legacy_agent")
    legacy.mkdir(parents=True)
    (legacy / "threads.db").write_bytes(b"")

    paths = iter_legacy_thread_db_paths()
    names = [p.name for p in paths]
    assert "legacy_agent" in names
    assert "new_agent" not in names
    # And the returned path IS the agent dir (legacy location), not the
    # threads.db inside it.
    assert paths[names.index("legacy_agent")] == legacy


# ---------------------------------------------------------------------------
# detect_storage_version: v6 chain
# ---------------------------------------------------------------------------


def test_detect_storage_version_v6_for_new_layout(fake_home: Path) -> None:
    new_dir = agent_root("foo", create=True)
    (new_dir / "session_metadata.json").write_text(
        json.dumps({"storage_version": STORAGE_VERSION_AGENTS_SUBDIR})
    )

    assert detect_storage_version("foo") == STORAGE_VERSION_AGENTS_SUBDIR
    assert CURRENT_STORAGE_VERSION == STORAGE_VERSION_AGENTS_SUBDIR


def test_detect_storage_version_legacy_v5(fake_home: Path) -> None:
    legacy = legacy_agent_root("foo")
    legacy.mkdir(parents=True)
    (legacy / "session_metadata.json").write_text(
        json.dumps({"storage_version": STORAGE_VERSION_WITH_PREFERRED_PROVIDER})
    )

    assert detect_storage_version("foo") == STORAGE_VERSION_WITH_PREFERRED_PROVIDER


def test_detect_storage_version_ignores_foreign_v2_db(fake_home: Path) -> None:
    """A v2 DB under ``.deepagents`` must not be detected (POLICIES §16)."""
    db = fake_home / ".deepagents" / "sessions_foo.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")

    assert detect_storage_version("foo") == STORAGE_VERSION_AGENTS_SUBDIR


def test_detect_storage_version_no_data_returns_v6(fake_home: Path) -> None:
    """Fresh agent with no legacy data → starts at current (v6)."""
    assert detect_storage_version("foo") == STORAGE_VERSION_AGENTS_SUBDIR


# ---------------------------------------------------------------------------
# migrate_agent_storage_to_v6
# ---------------------------------------------------------------------------


def _make_legacy_agent(fake_home: Path, name: str, files: dict[str, bytes]) -> None:
    legacy = legacy_agent_root(name)
    legacy.mkdir(parents=True)
    for rel, data in files.items():
        target = legacy / rel
        if data == b"<DIR>":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def test_migrate_v6_moves_legacy_to_agents_subdir(fake_home: Path) -> None:
    _make_legacy_agent(
        fake_home,
        "foo",
        {
            "threads.db": b"sqlite_blob",
            "session_metadata.json": json.dumps(
                {"storage_version": STORAGE_VERSION_WITH_PREFERRED_PROVIDER}
            ).encode(),
            "token_stats.jsonl": b"{}\n",
        },
    )

    result = migrate_agent_storage_to_v6("foo")

    assert result["migrated"] is True
    assert not legacy_agent_root("foo").exists()
    new_dir = agent_root("foo")
    assert new_dir.is_dir()
    assert (new_dir / "threads.db").read_bytes() == b"sqlite_blob"
    assert (new_dir / "token_stats.jsonl").read_bytes() == b"{}\n"

    meta = json.loads((new_dir / "session_metadata.json").read_text())
    assert meta["storage_version"] == STORAGE_VERSION_AGENTS_SUBDIR


def test_migrate_v6_idempotent_when_already_at_new_layout(fake_home: Path) -> None:
    new_dir = agent_root("foo", create=True)
    (new_dir / "threads.db").write_bytes(b"")
    (new_dir / "session_metadata.json").write_text(
        json.dumps({"storage_version": STORAGE_VERSION_AGENTS_SUBDIR})
    )

    result = migrate_agent_storage_to_v6("foo")
    assert result["skipped"] == "already-migrated"
    # dst still intact
    assert (new_dir / "threads.db").exists()


def test_migrate_v6_skips_when_no_legacy(fake_home: Path) -> None:
    result = migrate_agent_storage_to_v6("foo")
    assert result["skipped"] == "no-legacy"


def test_migrate_v6_preserves_permissions(fake_home: Path) -> None:
    legacy = legacy_agent_root("foo")
    legacy.mkdir(parents=True, mode=0o755)  # intentionally permissive
    (legacy / "threads.db").write_bytes(b"x")

    migrate_agent_storage_to_v6("foo")

    new_dir = agent_root("foo")
    assert oct(new_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((new_dir / "threads.db").stat().st_mode & 0o777) == "0o600"


def test_migrate_v6_agents_subdir_created_with_0700(fake_home: Path) -> None:
    """The ``agents/`` parent is created (mode 0o700) BEFORE first move."""
    assert not (fake_home / "agents").exists()
    _make_legacy_agent(fake_home, "foo", {"threads.db": b"x"})

    migrate_agent_storage_to_v6("foo")

    parent = fake_home / "agents"
    assert parent.is_dir()
    assert oct(parent.stat().st_mode & 0o777) == "0o700"


def test_migrate_v6_cross_device_falls_back_to_copy_rmtree(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXDEV → copytree + rmtree (design §5.1)."""
    _make_legacy_agent(fake_home, "foo", {"threads.db": b"x"})

    real_move = storage_paths.shutil.move

    def fake_move(src: str, dst: str) -> str:
        raise OSError(errno.EXDEV, "Cross-device link not permitted")

    monkeypatch.setattr(storage_paths.shutil, "move", fake_move)

    result = migrate_agent_storage_to_v6("foo")

    assert result["migrated"] is True
    assert not legacy_agent_root("foo").exists()
    new_dir = agent_root("foo")
    assert (new_dir / "threads.db").read_bytes() == b"x"
    # Restore real shutil.move to avoid affecting other tests.
    storage_paths.shutil.move = real_move


def test_migrate_v6_rmtree_failure_records_mv_failed_flag(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If copytree succeeds but rmtree fails, metadata records mv_failed for retry."""
    _make_legacy_agent(fake_home, "foo", {"threads.db": b"x"})

    def fake_move(src: str, dst: str) -> str:
        raise OSError(errno.EXDEV, "Cross-device link not permitted")

    monkeypatch.setattr(storage_paths.shutil, "move", fake_move)

    # copytree succeeds, rmtree fails → dst exists, src still exists.
    def fake_rmtree(path: str, *args: object, **kwargs: object) -> None:
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(storage_paths.shutil, "rmtree", fake_rmtree)

    result = migrate_agent_storage_to_v6("foo")
    assert result["migrated"] is True
    assert result["partial"] is True

    new_dir = agent_root("foo")
    meta = json.loads((new_dir / "session_metadata.json").read_text())
    assert meta["storage_version"] == STORAGE_VERSION_AGENTS_SUBDIR
    assert meta.get("mv_failed") is True


# ---------------------------------------------------------------------------
# migrate_agent_storage_to_current (chain v2 → v6)
# ---------------------------------------------------------------------------


def test_migrate_current_v5_to_v6(fake_home: Path) -> None:
    _make_legacy_agent(
        fake_home,
        "v5_agent",
        {
            "threads.db": b"x",
            "session_metadata.json": json.dumps(
                {"storage_version": STORAGE_VERSION_WITH_PREFERRED_PROVIDER}
            ).encode(),
        },
    )

    assert detect_storage_version("v5_agent") == STORAGE_VERSION_WITH_PREFERRED_PROVIDER
    result = migrate_agent_storage_to_current("v5_agent")
    assert result["migrated"] is True
    assert detect_storage_version("v5_agent") == CURRENT_STORAGE_VERSION


def test_migrate_current_never_imports_a_foreign_v2_db(fake_home: Path) -> None:
    """The v2 chain read out of ``.deepagents``; it must not run at all.

    The agent is reported as already-current, and the foreign DB is left
    untouched rather than copied into the product tree (POLICIES §16).
    """
    db = fake_home / ".deepagents" / "sessions_chain_agent.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"sqlite_v2")

    assert detect_storage_version("chain_agent") == CURRENT_STORAGE_VERSION
    result = migrate_agent_storage_to_current("chain_agent")
    assert result["skipped"] == "already-current"

    assert not (agent_root("chain_agent") / "threads.db").exists()
    assert db.read_bytes() == b"sqlite_v2"


def test_migrate_current_idempotent(fake_home: Path) -> None:
    """Calling migrate_agent_storage_to_current twice is a no-op the second time."""
    _make_legacy_agent(
        fake_home,
        "idem",
        {
            "threads.db": b"x",
            "session_metadata.json": json.dumps(
                {"storage_version": STORAGE_VERSION_WITH_PREFERRED_PROVIDER}
            ).encode(),
        },
    )
    migrate_agent_storage_to_current("idem")
    result2 = migrate_agent_storage_to_current("idem")
    assert result2["skipped"] == "already-current"


# ---------------------------------------------------------------------------
# Integration with the broader storage system (smoke)
# ---------------------------------------------------------------------------


def test_threads_db_path_uses_new_layout(fake_home: Path) -> None:
    from deep_worker_cli.storage_paths import threads_db_path

    assert threads_db_path("foo") == fake_home / "agents" / "foo" / "threads.db"


def test_metadata_path_uses_new_layout(fake_home: Path) -> None:
    from deep_worker_cli.storage_paths import metadata_path

    assert metadata_path("foo") == fake_home / "agents" / "foo" / "session_metadata.json"


def test_tm_root_uses_new_layout(fake_home: Path) -> None:
    from deep_worker_cli.storage_paths import tm_root

    tm = tm_root("foo")
    assert tm == fake_home / "agents" / "foo" / "tm"
    assert tm.is_dir()


def test_agent_root_creates_agents_parent_with_0o700_mode(fake_home: Path) -> None:
    """Mutation-1 guard (review round-1 observation).

    Without this test, a mutation that skips
    ``ensure_private_dir(root.parent)`` in ``agent_root(create=True)``
    would NOT be caught — because migration tests create the parent
    via ``migrate_agent_storage_to_v6``'s own
    ``ensure_private_dir(root.parent)`` call (DESIGN §5.1 step 1).

    This test exercises ``agent_root(create=True)`` DIRECTLY on a
    fresh ``fake_home`` where neither ``agents/`` nor the leaf exists,
    and asserts the parent is created with mode 0o700 (privacy
    contract, DESIGN §3.3 step 1).
    """
    from deep_worker_cli.storage_paths import agent_root

    assert not (fake_home / "agents").exists()
    root = agent_root("foo", home=fake_home, create=True)
    assert root == fake_home / "agents" / "foo"
    parent = fake_home / "agents"
    assert parent.is_dir()
    assert oct(parent.stat().st_mode & 0o777) == "0o700"
    assert root.is_dir()
    assert oct(root.stat().st_mode & 0o777) == "0o700"


def test_migrate_v6_both_exist_returns_skipped_without_overwriting(
    fake_home: Path,
) -> None:
    """Mutation-3 guard (review round-1 observation).

    Race condition: ``shutil.move`` when both ``src`` and ``dst`` exist
    drops ``src`` INSIDE ``dst`` (it silently creates ``dst/src/...``).
    The DESIGN §5.1 step 2 explicit ``dst.exists()`` guard prevents this
    by returning ``{"skipped": "both-exist"}`` instead of moving.

    Setup: create BOTH the legacy ``<name>/`` AND the new
    ``agents/<name>/`` directories with DISTINGUISHABLE markers. Run
    ``migrate_agent_storage_to_v6(name)``. Assert it returns
    ``{"skipped": "both-exist"}`` AND both files are still where they
    were (no overwrite, no nesting).
    """
    # Pre-existing legacy dir with marker file
    legacy = fake_home / "foo"
    legacy.mkdir(parents=True)
    (legacy / "LEGACY_MARKER.txt").write_text("legacy")
    (legacy / "threads.db").write_bytes(b"legacy-db")

    # Pre-existing new dir with different marker
    new = fake_home / "agents" / "foo"
    new.mkdir(parents=True)
    (new / "NEW_MARKER.txt").write_text("new")
    (new / "threads.db").write_bytes(b"new-db")

    result = migrate_agent_storage_to_v6("foo", home=fake_home)

    # Side-effect FIRST: this catches the race even if the function's
    # return signature changes (mutation 3 catches via filesystem).
    assert not (new / "foo").exists(), "shutil.move dropped legacy INSIDE new!"
    assert (legacy / "LEGACY_MARKER.txt").read_text() == "legacy"
    assert (new / "NEW_MARKER.txt").read_text() == "new"
    assert (legacy / "threads.db").read_bytes() == b"legacy-db"
    assert (new / "threads.db").read_bytes() == b"new-db"

    # Return value confirms the early-skip path was taken
    assert result["skipped"] == "both-exist"
    assert result["src"] == str(legacy)
    assert result["dst"] == str(new)


def test_migrate_v6_already_migrated_bumps_metadata(fake_home: Path) -> None:
    """If only the new layout exists (canonical), bump metadata to v6
    if it lags, and clear any stale ``mv_failed`` flag from a prior
    EXDEV-recovery attempt (DESIGN §5.1 step 3).

    This is the third branch of the migrate-guard: no legacy, dst
    exists, metadata stale → still safe (no-op for files, metadata
    bump only).
    """
    new = fake_home / "agents" / "foo"
    new.mkdir(parents=True)
    meta = new / "session_metadata.json"
    meta.write_text(
        json.dumps({"storage_version": 5, "mv_failed": True})
    )

    result = migrate_agent_storage_to_v6("foo", home=fake_home)

    assert result["skipped"] == "already-migrated"
    loaded = json.loads(meta.read_text())
    assert loaded["storage_version"] == STORAGE_VERSION_AGENTS_SUBDIR
    assert "mv_failed" not in loaded


def test_migrate_v6_proceeds_when_canonical_dir_empty_but_legacy_exists(
    fake_home: Path,
) -> None:
    """Regression test: when ``agents/<name>/`` exists as an empty
    directory (left over from a previous failed migration or partial
    startup), the v6 migration MUST proceed with the move rather than
    refusing as ``both-exist``. The migration auto-removes the empty
    target before ``shutil.move`` runs.

    Production incident 2026-08-18: ``_ensure_agent_layout`` creates
    ``agent_root(create=True)`` which mkdir's ``agents/<name>/`` even
    before the move runs. If the agent process then dies between
    mkdir and move, the canonical dir is left empty while the legacy
    is still populated. A subsequent migration would refuse to
    proceed and the agent would crash with
    ``storage_version 5 != current 6`` on restart.
    """
    fake_home.mkdir(parents=True, exist_ok=True)
    legacy = fake_home / "foo"
    legacy.mkdir()
    (legacy / "session_metadata.json").write_text(
        json.dumps({"storage_version": 5, "launch_config": {"provider": "deepseek"}})
    )
    (legacy / "threads.db").write_bytes(b"\x00")

    new = fake_home / "agents" / "foo"
    new.mkdir(parents=True)  # empty — the partial-state bug

    result = migrate_agent_storage_to_v6("foo", home=fake_home)

    assert result.get("migrated") is True
    assert result.get("skipped") != "both-exist"
    assert not legacy.exists()
    assert (new / "threads.db").exists()