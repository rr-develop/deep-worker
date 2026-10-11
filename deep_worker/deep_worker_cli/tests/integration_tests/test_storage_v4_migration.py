"""Integration tests for storage v4 auto-migration.

These tests exercise the real filesystem (no mocks) to verify:
- v3 agent → auto-migrate to v4 on restart
- v2 agent → auto-migrate to v4 on restart
- New agent starts on the CURRENT storage version (no migration needed)
"""
import json
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import (
    CURRENT_STORAGE_VERSION,
    DeepagentsPathForbiddenError,
    STORAGE_VERSION_AGENT_ROOT,
    STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    agent_root,
    deep_worker_home,
    detect_storage_version,
    migrate_agent_storage_to_v4,
    migrate_agent_storage_to_v3,
)


def test_new_agent_starts_on_the_current_version(
    tmp_path: Path, monkeypatch
) -> None:
    """A brand-new agent with no prior data starts on the CURRENT version.

    This used to also assert ``== STORAGE_VERSION_AGENT_ROOT_WITH_JSONL``
    (v4), which made the test a hostage of whatever the current version
    happened to be when it was written: shipping v5
    (STORAGE_VERSION_WITH_PREFERRED_PROVIDER) turned it red even though
    nothing was broken.

    The invariant worth pinning is RELATIVE — a fresh agent needs no
    migration — so it is stated against ``CURRENT_STORAGE_VERSION`` alone
    and survives every future bump. The absolute-version claims belong to
    the migration tests below, where the specific old layout is the point.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    version = detect_storage_version("brand_new")
    assert version == CURRENT_STORAGE_VERSION


def test_v3_agent_migrates_to_v4(tmp_path: Path, monkeypatch) -> None:
    """A v3 agent should auto-migrate to v4 without losing data."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = agent_root("v3_test", home=tmp_path, create=False)
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))
    (root / "threads.db").write_bytes(b"real_checkpoint_data")

    version_before = detect_storage_version("v3_test")
    assert version_before == STORAGE_VERSION_AGENT_ROOT

    result = migrate_agent_storage_to_v4("v3_test")

    version_after = detect_storage_version("v3_test")
    assert version_after == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL

    # v3→v4 is no-op — threads.db is untouched
    assert (root / "threads.db").read_bytes() == b"real_checkpoint_data"
    assert result["migrated_from"] == STORAGE_VERSION_AGENT_ROOT


def test_a_db_in_the_ancestors_tree_is_invisible(tmp_path: Path, monkeypatch) -> None:
    """POLICIES §16: a v2 DB in ``~/.deepagents`` is not detected or migrated.

    This asserted the opposite before the ban — that the legacy per-agent DB
    was detected as v2 and copied into the new agent root. The probe that
    found it looked inside the ancestor's tree, which is banned for reads, so
    such an agent now reads as brand-new and the foreign file is left alone.
    Installs still on v2 are migrated out-of-band.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_v2_test.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"legacy_sqlite")

    assert detect_storage_version("v2_test") != STORAGE_VERSION_DEEPAGENTS_PER_AGENT, (
        "storage detection probed ~/.deepagents, which POLICIES §16 bans"
    )

    assert old_db.read_bytes() == b"legacy_sqlite", (
        "the foreign tree must be left untouched"
    )
    new_root = agent_root("v2_test", home=tmp_path, create=False)
    assert not (new_root / "threads.db").exists(), (
        "history was copied out of ~/.deepagents, which POLICIES §16 bans"
    )


def test_idempotent_migration(tmp_path: Path, monkeypatch) -> None:
    """Calling migrate twice should be safe (second call is no-op)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = agent_root("idem", home=tmp_path, create=False)
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))

    migrate_agent_storage_to_v4("idem")
    result2 = migrate_agent_storage_to_v4("idem")

    assert result2["migrated_from"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    metadata = json.loads(meta.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL


def test_v3_migration_out_of_the_ancestors_tree_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """POLICIES §16: v2→v3 copied out of ``.deepagents``; that read is banned.

    Previously this asserted the copy succeeded. The source tree is now
    unreachable, so the call must raise rather than fabricate an empty agent.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_legacy.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"legacy")

    with pytest.raises(DeepagentsPathForbiddenError):
        migrate_agent_storage_to_v3("legacy_agent")

    assert old_db.read_bytes() == b"legacy", (
        "the foreign tree must be left untouched"
    )
