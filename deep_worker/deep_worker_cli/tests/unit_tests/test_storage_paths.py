import json
import os
from pathlib import Path

import pytest

from deep_worker_cli import storage_paths
from deep_worker_cli.storage_paths import (
    DeepagentsPathForbiddenError,
    STORAGE_VERSION_AGENT_ROOT,
    STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
    STORAGE_VERSION_AGENTS_SUBDIR,
    CURRENT_STORAGE_VERSION,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    HOME_DIR_NAME,
    agent_root,
    deepagents_data_dir,
    detect_storage_version,
    UnsafeStorageRootError,
    ensure_private_file,
    ensure_private_home,
    migrate_agent_storage_to_v3,
    migrate_agent_storage_to_v4,
    threads_db_path,
    tm_root,
)


def test_new_agent_starts_on_v6(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert CURRENT_STORAGE_VERSION == STORAGE_VERSION_AGENTS_SUBDIR
    assert detect_storage_version("new_agent") == CURRENT_STORAGE_VERSION
    assert threads_db_path("new_agent") == (
        tmp_path / ("." + HOME_DIR_NAME) / "agents" / "new_agent" / "threads.db"
    )
    assert deepagents_data_dir("new_agent") == (
        tmp_path / ("." + HOME_DIR_NAME) / "agents" / "new_agent" / "data"
    )


def test_existing_metadata_without_version_defaults_to_v6(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME) / "agents" / "old_agent"
    root.mkdir(parents=True)
    (root / "session_metadata.json").write_text(json.dumps({"status": "idle"}))

    assert detect_storage_version("old_agent") == STORAGE_VERSION_AGENTS_SUBDIR
    # Default for a freshly-encountered metadata file under ``agents/``
    # is the new layout → threads.db path is the new canonical one.
    assert threads_db_path("old_agent") == (
        tmp_path / ("." + HOME_DIR_NAME) / "agents" / "old_agent" / "threads.db"
    )


def test_v2_db_in_foreign_tree_is_never_detected(tmp_path: Path, monkeypatch) -> None:
    """A v2 DB inside ``~/.deepagents`` must be invisible (POLICIES §16).

    Detecting it would resurrect the foreign tree as authoritative storage,
    which is exactly what the ban forbids. An agent with no data in the
    product tree is a FRESH agent at the current version.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".deepagents" / "sessions_old_agent.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")

    assert detect_storage_version("old_agent") == STORAGE_VERSION_AGENTS_SUBDIR


def test_v3_agent_auto_migrate_to_v4(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # v3 (and v4) live under the new ``agents/`` layout (post-migration
    # would have moved them here). The test exercises the metadata-only
    # bump from v3 → v4 — physical path is already in the new layout.
    root = tmp_path / ("." + HOME_DIR_NAME) / "agents" / "v3_agent"
    root.mkdir(parents=True)
    meta = root / "session_metadata.json"
    meta.write_text(json.dumps({"storage_version": STORAGE_VERSION_AGENT_ROOT}))
    (root / "threads.db").write_bytes(b"v3_data")

    # Newly created agents in v6 layout no longer have legacy data —
    # detecting v3 here simulates a chain-walk during migration.
    assert detect_storage_version("v3_agent") == STORAGE_VERSION_AGENT_ROOT

    result = migrate_agent_storage_to_v4("v3_agent")

    assert result["migrated_from"] == STORAGE_VERSION_AGENT_ROOT
    metadata = json.loads(meta.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
    # v3→v4 is no-op physically; threads.db stays
    assert (root / "threads.db").read_bytes() == b"v3_data"


def test_v2_data_in_foreign_tree_is_never_copied(tmp_path: Path, monkeypatch) -> None:
    """POLICIES §16: v2→v4 must not import anything out of ``~/.deepagents``.

    The old chain copied ``sessions_<name>.db`` and the per-agent config
    dir out of the ancestor's tree. Reading it is now banned, so the agent
    is treated as fresh and the foreign files are left untouched.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_mig_agent.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"sqlite")
    old_data = tmp_path / ".deepagents" / "mig_agent"
    old_data.mkdir()
    (old_data / "AGENTS.md").write_text("config")

    assert detect_storage_version("mig_agent") == STORAGE_VERSION_AGENTS_SUBDIR

    root = tmp_path / ("." + HOME_DIR_NAME) / "agents" / "mig_agent"
    assert not (root / "threads.db").exists()
    assert not (root / "data" / "AGENTS.md").exists()
    # The foreign tree is left exactly as it was — never read, never moved.
    assert old_db.read_bytes() == b"sqlite"
    assert (old_data / "AGENTS.md").read_text() == "config"


def test_v3_to_v4_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Second call to migrate_agent_storage_to_v4 should be a no-op."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME) / "agents" / "v3_idem"
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
    assert not (tmp_path / ("." + HOME_DIR_NAME) / "agents" / "secure_agent:tm").exists()


def test_fresh_storage_root_is_private(tmp_path: Path, monkeypatch) -> None:
    """A freshly created runtime root must not be group/other-accessible.

    The root was previously created implicitly by ``mkdir(parents=True)``,
    which applies the process umask (0o755 by default) and leaves the whole
    tree world-readable.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    agent_root("fresh_agent")

    root = tmp_path / ("." + HOME_DIR_NAME)
    assert oct(root.stat().st_mode & 0o777) == "0o700"


def test_existing_permissive_storage_root_is_normalized(
    tmp_path: Path, monkeypatch
) -> None:
    """An already-existing owned root is tightened back to 0o700."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o755)

    agent_root("existing_agent")

    assert oct(root.stat().st_mode & 0o777) == "0o700"


def test_symlinked_storage_root_raises_and_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """A symlinked root must stop every caller, not just skip normalization.

    Returning quietly would let ``ensure_private_dir`` go on to create and
    chmod children inside the attacker-controlled target.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.symlink_to(target)

    with pytest.raises(UnsafeStorageRootError):
        ensure_private_home()
    with pytest.raises(UnsafeStorageRootError):
        agent_root("victim")

    assert root.is_symlink()
    assert oct(target.stat().st_mode & 0o777) == "0o755"
    assert not (target / "agents").exists()


def test_non_directory_storage_root_raises(tmp_path: Path, monkeypatch) -> None:
    """A regular file where the root belongs is rejected, not overwritten."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.write_text("not a directory")

    with pytest.raises(UnsafeStorageRootError):
        ensure_private_home()

    assert root.read_text() == "not a directory"


def test_foreign_owned_storage_root_raises_without_chmod(
    tmp_path: Path, monkeypatch
) -> None:
    """A root owned by another uid is rejected and never chmod-ed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o755)
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 12345)

    fchmod_calls: list[int] = []
    monkeypatch.setattr(os, "fchmod", lambda _fd, mode: fchmod_calls.append(mode))

    with pytest.raises(UnsafeStorageRootError):
        ensure_private_home()
    with pytest.raises(UnsafeStorageRootError):
        agent_root("victim")

    assert fchmod_calls == []
    assert oct(root.stat().st_mode & 0o777) == "0o755"
    assert not (root / "agents").exists()


def test_creation_swap_to_symlink_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A symlink swapped in just after ``mkdir`` must not be followed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    root = tmp_path / ("." + HOME_DIR_NAME)

    real_mkdir = os.mkdir

    def swapping_mkdir(name, mode=0o777, *, dir_fd=None):  # noqa: ANN202
        real_mkdir(name, mode, dir_fd=dir_fd)
        # Attacker replaces the just-created root with a symlink.
        os.rmdir(name, dir_fd=dir_fd)
        os.symlink(target, name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swapping_mkdir)

    with pytest.raises(UnsafeStorageRootError):
        ensure_private_home()

    assert root.is_symlink()
    assert oct(target.stat().st_mode & 0o777) == "0o755"


def test_symlinked_component_below_root_raises_and_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """A swapped component BELOW a valid root must also fail closed.

    Verifying only the root is not enough: ``mkdir(parents=True)`` walks the
    remaining components by pathname and would happily follow a symlinked
    ``agents/`` into an attacker-controlled directory.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o700)
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    (root / "agents").symlink_to(target)

    with pytest.raises(UnsafeStorageRootError):
        agent_root("victim")

    assert not (target / "victim").exists()
    assert oct(target.stat().st_mode & 0o777) == "0o755"


def test_root_swapped_after_verification_creates_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """A root swapped AFTER verification must not be followed by the walk.

    ``ensure_private_home`` verifies the root on a descriptor and closes it, so
    reopening by pathname would follow a symlink planted in that window.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o700)
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)

    original = storage_paths._open_verified_root

    def swapping_open_verified_root(path: Path | None = None):  # noqa: ANN202
        verified_root, fd = original(path)
        # Attacker swaps the root the instant verification finishes.
        root.rmdir()
        root.symlink_to(target)
        return verified_root, fd

    monkeypatch.setattr(
        storage_paths, "_open_verified_root", swapping_open_verified_root
    )

    # The walk holds the verified descriptor, so the swapped-in symlink is
    # never consulted; work fails closed on the detached directory instead.
    with pytest.raises(OSError):  # noqa: PT011
        agent_root("victim")

    assert not (target / "agents").exists()
    assert oct(target.stat().st_mode & 0o777) == "0o755"


def test_non_directory_component_below_root_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """A regular file standing in for an intermediate component is rejected."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o700)
    (root / "agents").write_text("not a directory")

    with pytest.raises(UnsafeStorageRootError):
        agent_root("victim")

    assert (root / "agents").read_text() == "not a directory"


def test_explicit_home_root_is_normalized(tmp_path: Path, monkeypatch) -> None:
    """An explicit ``home=`` root is normalized, not just the default HOME."""
    monkeypatch.setenv("HOME", str(tmp_path / "default_home"))
    (tmp_path / "default_home").mkdir(mode=0o700)
    explicit = tmp_path / "explicit_home"
    explicit.mkdir(mode=0o700)
    root = explicit / ("." + HOME_DIR_NAME)
    root.mkdir(mode=0o755)

    agent_root("a", home=explicit)

    assert oct(root.stat().st_mode & 0o777) == "0o700"


def test_explicit_home_symlinked_root_raises(tmp_path: Path, monkeypatch) -> None:
    """Fail-closed applies to explicit ``home=`` roots too."""
    monkeypatch.setenv("HOME", str(tmp_path / "default_home"))
    (tmp_path / "default_home").mkdir(mode=0o700)
    explicit = tmp_path / "explicit_home"
    explicit.mkdir(mode=0o700)
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    (explicit / ("." + HOME_DIR_NAME)).symlink_to(target)

    with pytest.raises(UnsafeStorageRootError):
        agent_root("a", home=explicit)

    assert not (target / "agents").exists()


def test_migrate_agent_storage_to_v3_is_refused(tmp_path: Path, monkeypatch) -> None:
    """POLICIES §16: v2→v3 read out of ``~/.deepagents``, so it must refuse loudly.

    Returning a path (or silently doing nothing) would let a caller believe
    legacy data had been imported when it had not.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    old_db = tmp_path / ".deepagents" / "sessions_mig_v3.db"
    old_db.parent.mkdir(parents=True)
    old_db.write_bytes(b"sqlite")

    with pytest.raises(DeepagentsPathForbiddenError):
        migrate_agent_storage_to_v3("mig_v3_agent")
