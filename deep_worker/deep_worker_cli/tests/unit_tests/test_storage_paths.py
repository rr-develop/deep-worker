import json
import os
from pathlib import Path

from deep_worker_cli.storage_paths import (
    STORAGE_VERSION_AGENT_ROOT,
    STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
    CURRENT_STORAGE_VERSION,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    agent_root,
    deepagents_data_dir,
    detect_storage_version,
    ensure_private_file,
    migrate_agent_storage_to_v3,
    migrate_agent_storage_to_v4,
    threads_db_path,
    tm_root,
)


def test_new_agent_starts_on_v4(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert CURRENT_STORAGE_VERSION == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    assert detect_storage_version("new_agent") == CURRENT_STORAGE_VERSION
    assert threads_db_path("new_agent") == (
        tmp_path / Path("~/.deep_worker").name / "new_agent" / "threads.db"
    )
    assert deepagents_data_dir("new_agent") == (
        tmp_path / Path("~/.deep_worker").name / "new_agent" / "data"
    )


def test_existing_metadata_without_version_stays_v2(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / Path("~/.deep_worker").name / "old_agent"
    root.mkdir(parents=True)
    (root / "session_metadata.json").write_text(json.dumps({"status": "idle"}))

    assert detect_storage_version("old_agent") == STORAGE_VERSION_DEEPAGENTS_PER_AGENT
    assert threads_db_path("old_agent") == (
        tmp_path / ".deepagents" / "sessions_old_agent.db"
    )


def test_existing_v2_db_without_metadata_stays_v2(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / ".deepagents" / "sessions_old_agent.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")

    assert detect_storage_version("old_agent") == STORAGE_VERSION_DEEPAGENTS_PER_AGENT


def test_v3_agent_auto_migrate_to_v4(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / Path("~/.deep_worker").name / "v3_agent"
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))
    (root / "threads.db").write_bytes(b"v3_data")

    assert detect_storage_version("v3_agent") == STORAGE_VERSION_AGENT_ROOT

    result = migrate_agent_storage_to_v4("v3_agent")

    assert result["migrated_from"] == STORAGE_VERSION_AGENT_ROOT
    metadata = json.loads(meta.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    # v3→v4 is no-op physically; threads.db stays
    assert (root / "threads.db").read_bytes() == b"v3_data"


def test_v2_agent_auto_migrate_to_v4(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_mig_agent.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"sqlite")
    old_data = tmp_path / ".deepagents" / "mig_agent"
    old_data.mkdir()
    (old_data / "AGENTS.md").write_text("config")

    assert detect_storage_version("mig_agent") == STORAGE_VERSION_DEEPAGENTS_PER_AGENT

    result = migrate_agent_storage_to_v4("mig_agent")

    root = tmp_path / Path("~/.deep_worker").name / "mig_agent"
    assert (root / "threads.db").read_bytes() == b"sqlite"
    assert (root / "data" / "AGENTS.md").read_text() == "config"
    metadata = json.loads((root / "session_metadata.json").read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert oct((root / "threads.db").stat().st_mode & 0o777) == "0o600"
    assert result["migrated_from"] == STORAGE_VERSION_DEEPAGENTS_PER_AGENT


def test_v2_to_v4_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Second call to migrate_agent_storage_to_v4 should be a no-op."""
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_idem_agent.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"sqlite")

    migrate_agent_storage_to_v4("idem_agent")
    result2 = migrate_agent_storage_to_v4("idem_agent")

    assert result2["migrated_from"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    root = tmp_path / Path("~/.deep_worker").name / "idem_agent"
    metadata = json.loads((root / "session_metadata.json").read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL


def test_v3_to_v4_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Second call to migrate_agent_storage_to_v4 should be a no-op."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / Path("~/.deep_worker").name / "v3_idem"
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))

    migrate_agent_storage_to_v4("v3_idem")
    result2 = migrate_agent_storage_to_v4("v3_idem")

    assert result2["migrated_from"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    metadata = json.loads(meta.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL


def test_private_permissions_for_agent_dirs_and_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    root = agent_root("secure_agent")
    data = deepagents_data_dir("secure_agent")
    tm = tm_root("secure_agent")
    db = threads_db_path("secure_agent")
    db.write_bytes(b"")
    ensure_private_file(db)

    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert oct(data.stat().st_mode & 0o777) == "0o700"
    assert oct(tm.stat().st_mode & 0o777) == "0o700"
    assert oct(db.stat().st_mode & 0o777) == "0o600"
    assert not (tmp_path / Path("~/.deep_worker").name / "secure_agent:tm").exists()


def test_migrate_agent_storage_to_v3_still_works(tmp_path: Path, monkeypatch) -> None:
    """v3 migration function still works for backward compat."""
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_mig_v3.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"sqlite")

    result = migrate_agent_storage_to_v3("mig_v3_agent")

    root = tmp_path / Path("~/.deep_worker").name / "mig_v3_agent"
    assert Path(result["threads_db"]) == root / "threads.db"
    metadata = json.loads((root / "session_metadata.json").read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT
