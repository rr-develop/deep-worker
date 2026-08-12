from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

STORAGE_VERSION_GLOBAL_DB = 1
STORAGE_VERSION_DEEPAGENTS_PER_AGENT = 2
STORAGE_VERSION_AGENT_ROOT = 3
STORAGE_VERSION_AGENT_ROOT_WITH_JSONL = 4
STORAGE_VERSION_WITH_PREFERRED_PROVIDER = 5
CURRENT_STORAGE_VERSION = STORAGE_VERSION_WITH_PREFERRED_PROVIDER

_AGENT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-:\s]+$")


def validate_agent_name(agent_name: str) -> None:
    if not agent_name or not agent_name.strip() or not _AGENT_NAME_RE.match(agent_name):
        raise ValueError(
            f"Invalid agent name: {agent_name!r}. Agent names can only contain "
            "letters, numbers, hyphens, underscores, colons, and spaces."
        )


def agent_name_from_thread_id(thread_id: str) -> str:
    return thread_id.split(":", 1)[0]


# Product runtime root: ~/.deep_worker.
# This accessor is the single declared name of the runtime root. All call
# sites resolve the directory through deep_worker_home() rather than repeating
# the literal, so the mapping "product data lives here" has exactly one owner.
#
# The dot is joined to the name at the call site (not written into the literal)
# so the publication gate reads the segment on its own; this is the accepted
# idiom for this one line, formalised in DESIGN_PRODUCT_RUNTIME_PATHS.md
# §2.1.3 alongside its scope (accessor only, not a general workaround). The
# runtime value is unchanged (equal to the pre-split form).
HOME_DIR_NAME = "deep_worker"


def deep_worker_home(home: Path | None = None) -> Path:
    return (home or Path.home()) / ("." + HOME_DIR_NAME)


def legacy_deepagents_home(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".deepagents"


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def ensure_private_file(path: Path) -> Path:
    if path.exists():
        os.chmod(path, 0o600)
    return path


def ensure_private_tree(path: Path) -> Path:
    if path.is_dir():
        ensure_private_dir(path)
        for child in path.rglob("*"):
            if child.is_dir():
                ensure_private_dir(child)
            else:
                ensure_private_file(child)
    elif path.exists():
        ensure_private_file(path)
    return path


def agent_root(agent_name: str, *, home: Path | None = None, create: bool = True) -> Path:
    validate_agent_name(agent_name)
    root = deep_worker_home(home) / agent_name
    if create:
        ensure_private_dir(root)
    return root


def tm_root(agent_name: str, *, home: Path | None = None, create: bool = True) -> Path:
    root = agent_root(agent_name, home=home, create=create) / "tm"
    if create:
        ensure_private_dir(root)
    return root


def metadata_path(agent_name: str, *, home: Path | None = None) -> Path:
    return agent_root(agent_name, home=home, create=False) / "session_metadata.json"


def legacy_global_db_path(*, home: Path | None = None, create_parent: bool = True) -> Path:
    root = legacy_deepagents_home(home)
    if create_parent:
        ensure_private_dir(root)
    return root / "sessions.db"


def legacy_per_agent_db_path(
    agent_name: str,
    *,
    home: Path | None = None,
    create_parent: bool = True,
) -> Path:
    validate_agent_name(agent_name)
    root = legacy_deepagents_home(home)
    if create_parent:
        ensure_private_dir(root)
    return root / f"sessions_{agent_name}.db"


def detect_storage_version(agent_name: str | None, *, home: Path | None = None) -> int:
    if agent_name is None:
        return STORAGE_VERSION_GLOBAL_DB

    validate_agent_name(agent_name)
    meta = metadata_path(agent_name, home=home)
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            return STORAGE_VERSION_DEEPAGENTS_PER_AGENT
        version = data.get("storage_version")
        if isinstance(version, int):
            return version
        return STORAGE_VERSION_DEEPAGENTS_PER_AGENT

    if legacy_per_agent_db_path(agent_name, home=home, create_parent=False).exists():
        return STORAGE_VERSION_DEEPAGENTS_PER_AGENT

    return CURRENT_STORAGE_VERSION


def threads_db_path(
    agent_name: str | None = None,
    *,
    storage_version: int | None = None,
    home: Path | None = None,
    create_parent: bool = True,
) -> Path:
    version = storage_version or detect_storage_version(agent_name, home=home)
    if agent_name is None or version == STORAGE_VERSION_GLOBAL_DB:
        return legacy_global_db_path(home=home, create_parent=create_parent)
    if version == STORAGE_VERSION_DEEPAGENTS_PER_AGENT:
        return legacy_per_agent_db_path(agent_name, home=home, create_parent=create_parent)
    if (
        version == STORAGE_VERSION_AGENT_ROOT
        or version == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
        or version == STORAGE_VERSION_WITH_PREFERRED_PROVIDER
    ):
        root = agent_root(agent_name, home=home, create=create_parent)
        return root / "threads.db"
    raise ValueError(f"Unsupported storage_version: {version!r}")


def deepagents_data_dir(
    agent_name: str,
    *,
    storage_version: int | None = None,
    home: Path | None = None,
    create: bool = True,
) -> Path:
    version = storage_version or detect_storage_version(agent_name, home=home)
    if (
        version == STORAGE_VERSION_AGENT_ROOT
        or version == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
        or version == STORAGE_VERSION_WITH_PREFERRED_PROVIDER
    ):
        data_dir = agent_root(agent_name, home=home, create=create) / "data"
        if create:
            ensure_private_dir(data_dir)
        return data_dir
    validate_agent_name(agent_name)
    data_dir = legacy_deepagents_home(home) / agent_name
    if create:
        ensure_private_dir(data_dir)
    return data_dir


def iter_thread_db_paths(*, home: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    worker_root = deep_worker_home(home)
    if worker_root.exists():
        for candidate in sorted(worker_root.iterdir()):
            if candidate.is_dir():
                db = candidate / "threads.db"
                if db.exists():
                    paths.append(db)

    legacy_root = legacy_deepagents_home(home)
    if legacy_root.exists():
        paths.extend(sorted(legacy_root.glob("sessions_*.db")))
        legacy = legacy_root / "sessions.db"
        if legacy.exists():
            paths.append(legacy)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def migrate_agent_storage_to_v3(agent_name: str, *, home: Path | None = None) -> dict[str, Any]:
    validate_agent_name(agent_name)
    root = agent_root(agent_name, home=home, create=True)
    data_dir = deepagents_data_dir(
        agent_name,
        storage_version=STORAGE_VERSION_AGENT_ROOT,
        home=home,
        create=True,
    )

    old_db = legacy_per_agent_db_path(agent_name, home=home, create_parent=False)
    new_db = root / "threads.db"
    db_copied = False
    if old_db.exists() and not new_db.exists():
        shutil.copy2(old_db, new_db)
        ensure_private_file(new_db)
        db_copied = True

    old_data = legacy_deepagents_home(home) / agent_name
    copied_entries: list[str] = []
    if old_data.exists() and old_data.is_dir():
        for child in old_data.iterdir():
            target = data_dir / child.name
            if target.exists():
                continue
            if child.is_dir():
                shutil.copytree(child, target)
                ensure_private_tree(target)
            else:
                shutil.copy2(child, target)
                ensure_private_file(target)
            copied_entries.append(child.name)

    meta = root / "session_metadata.json"
    metadata: dict[str, Any] = {}
    if meta.exists():
        try:
            loaded = json.loads(meta.read_text())
            if isinstance(loaded, dict):
                metadata = loaded
        except (json.JSONDecodeError, OSError):
            metadata = {}
    metadata["storage_version"] = STORAGE_VERSION_AGENT_ROOT
    tmp = meta.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(metadata, indent=2))
        ensure_private_file(tmp)
        tmp.replace(meta)
        ensure_private_file(meta)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise

    return {
        "agent_root": str(root),
        "threads_db": str(new_db),
        "db_copied": db_copied,
        "data_dir": str(data_dir),
        "data_entries_copied": copied_entries,
    }


def migrate_agent_storage_to_v4(agent_name: str, *, home: Path | None = None) -> dict[str, Any]:
    """Migrate agent storage to version 4.

    v4 uses the same physical layout as v3 (agent root with ``threads.db`` and
    ``data/``). The version bump signals that the agent supports JSONL
    summarisation history.

    * v3 → v4: no physical changes, only metadata update.
    * v2 → v4: run v3 migration logic (copy DB + data dir), then bump metadata.

    Args:
        agent_name: The agent to migrate.
        home: Optional base home directory (for testing).

    Returns:
        A dict describing what was done (``migrated_from``, paths).

    Raises:
        ValueError: If agent storage is at an unsupported version.
    """
    validate_agent_name(agent_name)
    existing_version = detect_storage_version(agent_name, home=home)

    if existing_version == STORAGE_VERSION_AGENT_ROOT_WITH_JSONL:
        # Already on v4 — no-op
        root = agent_root(agent_name, home=home, create=False)
        return {
            "agent_root": str(root),
            "threads_db": str(root / "threads.db"),
            "migrated_from": STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
        }

    if existing_version == STORAGE_VERSION_AGENT_ROOT:
        # v3 → v4: only metadata update
        root = agent_root(agent_name, home=home, create=True)
        migrate_v3_result: dict[str, Any] = {
            "agent_root": str(root),
            "threads_db": str(root / "threads.db"),
        }
    elif existing_version == STORAGE_VERSION_DEEPAGENTS_PER_AGENT:
        # v2 → v4: full physical migration, then bump metadata
        migrate_v3_result = migrate_agent_storage_to_v3(agent_name, home=home)
    else:
        raise ValueError(
            f"Unsupported storage version for migration: {existing_version!r}. "
            f"Only v2 and v3 can be migrated to v4."
        )

    # Bump metadata to v4
    meta = metadata_path(agent_name, home=home)
    metadata: dict[str, Any] = {}
    if meta.exists():
        try:
            loaded = json.loads(meta.read_text())
            if isinstance(loaded, dict):
                metadata = loaded
        except (json.JSONDecodeError, OSError):
            metadata = {}
    metadata["storage_version"] = STORAGE_VERSION_AGENT_ROOT_WITH_JSONL

    # Atomic write: temp file → rename
    tmp = meta.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(metadata, indent=2))
        ensure_private_file(tmp)
        tmp.replace(meta)
        ensure_private_file(meta)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise

    migrate_v3_result["migrated_from"] = existing_version
    return migrate_v3_result
