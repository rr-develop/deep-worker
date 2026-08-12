"""Integration tests for storage v4 auto-migration.

These tests exercise the real filesystem (no mocks) to verify:
- v3 agent → auto-migrate to v4 on restart
- v2 agent → auto-migrate to v4 on restart
- New agent starts on v4
"""
import json
from pathlib import Path

from deep_worker_cli.storage_paths import (
    CURRENT_STORAGE_VERSION,
    STORAGE_VERSION_AGENT_ROOT,
    STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    deep_worker_home,
    detect_storage_version,
    migrate_agent_storage_to_v4,
    migrate_agent_storage_to_v3,
)


def test_new_agent_starts_on_v4(tmp_path: Path, monkeypatch) -> None:
    """A brand-new agent with no prior data should detect as v4."""
    monkeypatch.setenv("HOME", str(tmp_path))
    version = detect_storage_version("brand_new")
    assert version == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    assert version == CURRENT_STORAGE_VERSION


def test_v3_agent_migrates_to_v4(tmp_path: Path, monkeypatch) -> None:
    """A v3 agent should auto-migrate to v4 without losing data."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = deep_worker_home(tmp_path) / "v3_test"
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


def test_v2_agent_migrates_to_v4(tmp_path: Path, monkeypatch) -> None:
    """A v2 agent (legacy per-agent DB) should migrate to v4 with DB copy."""
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_v2_test.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"legacy_sqlite")

    version_before = detect_storage_version("v2_test")
    assert version_before == STORAGE_VERSION_DEEPAGENTS_PER_AGENT

    result = migrate_agent_storage_to_v4("v2_test")

    new_root = deep_worker_home(tmp_path) / "v2_test"
    assert (new_root / "threads.db").read_bytes() == b"legacy_sqlite"
    metadata = json.loads((new_root / "session_metadata.json").read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    assert result["migrated_from"] == STORAGE_VERSION_DEEPAGENTS_PER_AGENT


def test_idempotent_migration(tmp_path: Path, monkeypatch) -> None:
    """Calling migrate twice should be safe (second call is no-op)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = deep_worker_home(tmp_path) / "idem"
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))

    migrate_agent_storage_to_v4("idem")
    result2 = migrate_agent_storage_to_v4("idem")

    assert result2["migrated_from"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    metadata = json.loads(meta.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL


def test_v3_migration_still_works(tmp_path: Path, monkeypatch) -> None:
    """Backward compat: v3 migration function still works."""
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_legacy.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"legacy")

    result = migrate_agent_storage_to_v3("legacy_agent")

    root = deep_worker_home(tmp_path) / "legacy_agent"
    metadata = json.loads((root / "session_metadata.json").read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT
    assert Path(result["threads_db"]) == root / "threads.db"
