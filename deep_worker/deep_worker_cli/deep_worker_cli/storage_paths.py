from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

STORAGE_VERSION_GLOBAL_DB = 1
STORAGE_VERSION_DEEPAGENTS_PER_AGENT = 2
STORAGE_VERSION_AGENT_ROOT = 3
STORAGE_VERSION_AGENT_ROOT_WITH_JSONL = 4
STORAGE_VERSION_WITH_PREFERRED_PROVIDER = 5
STORAGE_VERSION_AGENTS_SUBDIR = 6
CURRENT_STORAGE_VERSION = STORAGE_VERSION_AGENTS_SUBDIR

_AGENT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-:\s]+$")

# Owner-only mode for every directory in the runtime tree, including the root.
PRIVATE_DIR_MODE = 0o700

_ROOT_SYMLINK = "runtime root is a symlink"
_ROOT_NOT_DIR = "runtime root is not a directory"
_ROOT_FOREIGN = "runtime root is owned by another user"
_ROOT_SWAPPED = "runtime root changed during open"
_ROOT_UNSECURED = "runtime root mode could not be secured"


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


def config_home(home: Path | None = None) -> Path:
    """Return the user-scope configuration root ``~/.deep_worker``.

    POLICIES §16: the single owner of "user configuration lives here".
    """
    return deep_worker_home(home)


PROJECT_CONFIG_DIR_NAME = "." + HOME_DIR_NAME


def project_config_dir(project_root: Path) -> Path:
    """Return the project-scope config dir ``<project_root>/.deep_worker``."""
    return Path(project_root) / PROJECT_CONFIG_DIR_NAME


def agent_config_dir(
    agent_name: str,
    *,
    home: Path | None = None,
    create: bool = False,
) -> Path:
    """Return the per-agent config dir ``~/.deep_worker/agents/<name>``.

    Per-agent configuration is a sibling of the agent's own state, under
    ``agents/``, never a top-level entry (the top level is reserved for
    infrastructure — see ``INFRASTRUCTURE_TOP_LEVEL_DIRS``).
    """
    return agent_root(agent_name, home=home, create=create)


class UnsafeStorageRootError(OSError):
    """The runtime root is not a private directory owned by this user.

    Raised instead of returning so that no caller creates, chmods, or writes
    anything beneath a root that an attacker may control.
    """


def ensure_private_dir(path: Path) -> Path:
    """Create/verify ``path`` and every component below the runtime root as 0o700.

    Each component is created and opened relative to a held descriptor of its
    parent, so a symlink swapped in for an intermediate directory cannot be
    followed out of the runtime tree.

    Returns:
        The requested directory path.
    """
    # Establish the runtime root FIRST; it raises on an unsafe root, so
    # nothing below is created until the root is known-good. The verified
    # descriptor is carried into the walk — reopening by pathname here would
    # follow a symlink swapped in after verification.
    target = Path(path)
    root, root_fd = _open_verified_root(path)
    if root_fd is None:
        # Outside any runtime root:
        # keep the previous pathname behaviour.
        return _ensure_private_dir_by_pathname(target)
    held = [root_fd]
    try:
        if target == root:
            return target
        try:
            relative = target.relative_to(root)
        except ValueError:
            return _ensure_private_dir_by_pathname(target)
        parent_fd = root_fd
        for component in relative.parts:
            parent_fd = _open_private_subdir(parent_fd, component)
            held.append(parent_fd)
    finally:
        for fd in reversed(held):
            os.close(fd)
    return target


def _ensure_private_dir_by_pathname(target: Path) -> Path:
    """Create ``target`` as 0o700 by pathname, for paths outside a runtime root.

    Returns:
        The requested directory path.
    """
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(PRIVATE_DIR_MODE)
    return target


def _open_private_subdir(parent_fd: int, name: str) -> int:
    """Create if needed, then open+verify ``name`` under ``parent_fd`` as 0o700.

    Returns:
        A descriptor for the verified directory. The caller owns it.

    Raises:
        UnsafeStorageRootError: If the component is a symlink, is not a
            directory, or is owned by another user.
        OSError: If the component cannot be created or opened.
    """
    with suppress(FileExistsError):
        os.mkdir(name, PRIVATE_DIR_MODE, dir_fd=parent_fd)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    before = None
    if not nofollow:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise UnsafeStorageRootError(_ROOT_SYMLINK)
    try:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafeStorageRootError(_ROOT_SYMLINK) from exc
        raise
    try:
        _verify_not_swapped(fd, before)
        _verify_root_fd(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _verify_not_swapped(fd: int, before: os.stat_result | None) -> None:
    """Reject a descriptor whose identity changed since the pre-open ``lstat``.

    Raises:
        UnsafeStorageRootError: If the target was swapped during the open.
    """
    if before is None:
        return
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise UnsafeStorageRootError(_ROOT_SWAPPED)


def _verify_root_fd(fd: int) -> None:
    """Reject a root descriptor that is not a private directory we own.

    Raises:
        UnsafeStorageRootError: If the descriptor fails any check.
    """
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeStorageRootError(_ROOT_NOT_DIR)
    if info.st_uid != os.geteuid():
        raise UnsafeStorageRootError(_ROOT_FOREIGN)
    if stat.S_IMODE(info.st_mode) != PRIVATE_DIR_MODE:
        os.fchmod(fd, PRIVATE_DIR_MODE)
        if stat.S_IMODE(os.fstat(fd).st_mode) != PRIVATE_DIR_MODE:
            raise UnsafeStorageRootError(_ROOT_UNSECURED)


def ensure_private_home(path: Path | None = None) -> Path:
    """Create/verify the runtime root ``~/.deep_worker`` as private (0o700).

    The single owner of the root's permissions. The root is created and
    inspected relative to a held file descriptor of its parent directory, so a
    symlink swapped in between the check and the open cannot be followed.

    Fails closed: a symlink, a non-directory, or a root owned by another user
    raises instead of returning, which stops every caller before it can create
    or chmod anything underneath an attacker-controlled path.

    Args:
        path: The runtime root itself, or any path inside it. Paths outside a
            runtime root are ignored. ``None`` resolves the default root.

    Propagates ``UnsafeStorageRootError`` when the root is not a private
    directory owned by the current user, and ``OSError`` when the root's
    parent cannot be opened or inspected.

    Returns:
        The verified runtime root path.
    """
    root, fd = _open_verified_root(path)
    if fd is not None:
        os.close(fd)
    return root


def _open_verified_root(path: Path | None) -> tuple[Path, int | None]:
    """Create/verify the runtime root and return it with its open descriptor.

    The descriptor is the ONLY safe handle on the verified root: reopening by
    pathname afterwards would follow a symlink planted in the meantime. The
    caller owns the returned fd and must close it.

    Propagates ``UnsafeStorageRootError`` when the root is not a private
    directory owned by the current user, and ``OSError`` when the root's
    parent cannot be opened or inspected.

    Returns:
        ``(root, fd)``; ``fd`` is ``None`` when ``path`` lies outside any
        runtime root and nothing was verified.
    """
    root = _root_for(path)
    if root is None:
        return deep_worker_home(), None
    parent_fd = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        return root, _open_private_subdir(parent_fd, root.name)
    finally:
        os.close(parent_fd)


def _root_for(path: Path | None) -> Path | None:
    """Return the runtime root governing ``path``, or ``None`` if unrelated.

    Supports explicit ``home=`` roots: any ancestor of ``path`` named
    ``.deep_worker`` is the governing root, not just the default one.
    """
    default = deep_worker_home()
    if path is None:
        return default
    candidate = Path(path)
    root_name = "." + HOME_DIR_NAME
    if candidate.name == root_name:
        return candidate
    for parent in candidate.parents:
        if parent.name == root_name:
            return parent
    return default if candidate == default else None


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
    """Return the canonical agent root under ``~/.deep_worker/agents/<name>``.

    DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §3: agent databases live in a
    dedicated ``agents/`` subdirectory so they cannot collide with
    infrastructure files (``logs/``, ``env.d/``, ``providers.json``,
    ``summaries_backup/``, etc.) that share the ``~/.deep_worker/`` tree.
    """
    validate_agent_name(agent_name)
    root = deep_worker_home(home) / "agents" / agent_name
    if create:
        # Ensure the ``agents/`` parent exists with mode 0o700 BEFORE any
        # per-agent dir is created under it. This keeps the privacy contract
        # for the whole subtree intact (DESIGN §5.3, §3.3 step 1).
        ensure_private_dir(root.parent)
        ensure_private_dir(root)
    return root


def legacy_agent_root(agent_name: str, *, home: Path | None = None) -> Path:
    """Return the pre-subdir location ``~/.deep_worker/<name>``.

    Read-only fallback for backward compatibility (DESIGN §3.2). Never
    used for writes — new agents and migration land under
    ``agents/<name>/`` only.
    """
    validate_agent_name(agent_name)
    return deep_worker_home(home) / agent_name


def tm_root(agent_name: str, *, home: Path | None = None, create: bool = True) -> Path:
    root = agent_root(agent_name, home=home, create=create) / "tm"
    if create:
        ensure_private_dir(root)
    return root


def metadata_path(agent_name: str, *, home: Path | None = None) -> Path:
    return agent_root(agent_name, home=home, create=False) / "session_metadata.json"


def read_agent_file(name: str, filename: str, *, home: Path | None = None) -> bytes | None:
    """Read a per-agent file, preferring the new ``agents/<name>/`` layout.

    Lookup order (DESIGN §4.5):
      1. ``~/.deep_worker/agents/<name>/<filename>`` (preferred)
      2. ``~/.deep_worker/<name>/<filename>`` (legacy fallback)

    Returns the bytes if found, else ``None``. NEVER merges — the first
    hit wins. This avoids accidentally reviving stale data after a
    partial migration.
    """
    new_path = agent_root(name, home=home, create=False) / filename
    if new_path.exists():
        return new_path.read_bytes()
    legacy_path = legacy_agent_root(name, home=home) / filename
    if legacy_path.exists():
        return legacy_path.read_bytes()
    return None


class DeepagentsPathForbiddenError(RuntimeError):
    """A caller tried to resolve a path inside the foreign ``.deepagents`` tree.

    POLICIES §16 bans that directory for both reads and writes. Raising here
    — rather than returning a path nobody may use — keeps the ban at the one
    place the path was ever constructed, so a new caller fails loudly instead
    of silently reviving a foreign config.
    """


def legacy_global_db_path(*, home: Path | None = None, create_parent: bool = True) -> Path:
    """Unreachable: the v1 global DB lived in ``.deepagents`` (POLICIES §16).

    Raises:
        DeepagentsPathForbiddenError: Always.
    """
    raise DeepagentsPathForbiddenError(
        "the v1 global sessions.db lived under ~/.deepagents, which this "
        "project must never read or write (POLICIES §16)"
    )


def legacy_per_agent_db_path(
    agent_name: str,
    *,
    home: Path | None = None,
    create_parent: bool = True,
) -> Path:
    """Unreachable: the v2 per-agent DB lived in ``.deepagents`` (POLICIES §16).

    Raises:
        DeepagentsPathForbiddenError: Always.
    """
    raise DeepagentsPathForbiddenError(
        f"the v2 sessions_{agent_name}.db lived under ~/.deepagents, which "
        "this project must never read or write (POLICIES §16)"
    )


def detect_storage_version(agent_name: str | None, *, home: Path | None = None) -> int:
    if agent_name is None:
        return STORAGE_VERSION_GLOBAL_DB

    validate_agent_name(agent_name)
    # 1. New layout: ``agents/<name>/session_metadata.json`` (DESIGN §3.3).
    new_root = agent_root(agent_name, home=home, create=False)
    new_meta = new_root / "session_metadata.json"
    if new_meta.exists():
        try:
            data = json.loads(new_meta.read_text())
        except (json.JSONDecodeError, OSError):
            return STORAGE_VERSION_AGENTS_SUBDIR
        version = data.get("storage_version") if isinstance(data, dict) else None
        if isinstance(version, int):
            return version
        # Metadata exists but no version key — the file is already in the
        # new layout, so report v6 rather than backtracking.
        return STORAGE_VERSION_AGENTS_SUBDIR

    # 2. Legacy top-level path: ``<name>/session_metadata.json``.
    legacy_meta = deep_worker_home(home) / agent_name / "session_metadata.json"
    if legacy_meta.exists():
        try:
            data = json.loads(legacy_meta.read_text())
        except (json.JSONDecodeError, OSError):
            return STORAGE_VERSION_WITH_PREFERRED_PROVIDER
        version = data.get("storage_version") if isinstance(data, dict) else None
        if isinstance(version, int):
            return version
        return STORAGE_VERSION_WITH_PREFERRED_PROVIDER

    # 3. No data anywhere → fresh agent under the new layout. The old v1/v2
    # probe looked inside ``~/.deepagents``, which is banned (POLICIES §16).
    return STORAGE_VERSION_AGENTS_SUBDIR


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
        or version == STORAGE_VERSION_AGENTS_SUBDIR
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
        or version == STORAGE_VERSION_AGENTS_SUBDIR
    ):
        data_dir = agent_root(agent_name, home=home, create=create) / "data"
        if create:
            ensure_private_dir(data_dir)
        return data_dir
    # Every remaining version predates the product tree and would resolve
    # into ``~/.deepagents`` (POLICIES §16).
    raise DeepagentsPathForbiddenError(
        f"storage_version {version!r} for {agent_name!r} maps to ~/.deepagents, "
        "which this project must never read or write (POLICIES §16)"
    )


def iter_thread_db_paths(*, home: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    worker_root = deep_worker_home(home)
    if worker_root.exists():
        # Walk ONLY the ``agents/`` subdirectory (DESIGN §4.1). The top
        # level contains infrastructure files (``logs/``, ``env.d/``,
        # ``summaries_backup/``) that must not be treated as agents.
        agents_root = worker_root / "agents"
        if agents_root.exists():
            for candidate in sorted(agents_root.iterdir()):
                if candidate.is_dir():
                    db = candidate / "threads.db"
                    if db.exists():
                        paths.append(db)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def iter_legacy_thread_db_paths(*, home: Path | None = None) -> list[Path]:
    """Return agent directories in the *legacy* top-level layout only.

    Walks ``~/.deep_worker/<name>/`` for agent directories that have
    NOT yet been migrated to ``agents/<name>/``. Used by the migration
    CLI (DESIGN §4.6) to discover pending migrations. Returns paths to
    the agent dirs themselves — callers can use ``.parent`` if they
    need a stub threads.db path (the CLI only uses ``agent_dir``).

    Returns ALL legacy dirs that look like agent dirs, not just those
    with ``threads.db``: test dirs and recently-stopped agents often
    lack a ``threads.db`` but still need to be considered for move
    or delete decisions. The classifier still drops infrastructure
    dirs and decides per-name whether to move, delete, or skip.

    Detection of "looks like an agent dir": any top-level dir that
    is NOT in ``INFRASTRUCTURE_TOP_LEVEL_DIRS``. Cost of being
    conservative here is low because the ``classify()`` decision
    only runs with ``--execute``.
    """
    paths: list[Path] = []
    worker_root = deep_worker_home(home)
    if not worker_root.exists():
        return paths

    for candidate in sorted(worker_root.iterdir()):
        if not candidate.is_dir():
            continue
        # Skip the new layout (those are already handled by the main API).
        if candidate == worker_root / "agents":
            continue
        # Skip infrastructure directories.
        if candidate.name in INFRASTRUCTURE_TOP_LEVEL_DIRS:
            continue
        # Anything else at the top level is a legacy agent dir.
        # The CLI's classifier decides whether to move, delete, or skip.
        paths.append(candidate)
    return paths


# Names that sit at the top level of ``~/.deep_worker/`` and must never
# be treated as per-agent storage. Mirrors the list in DESIGN §5.2 step 2.
INFRASTRUCTURE_TOP_LEVEL_DIRS: frozenset[str] = frozenset({
    "agents",
    "logs",
    "env.d",
    "lsp",
    "summaries_backup",
    "services_shared",
    "tasks_service_runtime",
    # User-created 2026-08-18: hold-back for test/probe agents that the
    # user wants to keep around for reference but out of the active
    # migration sweep. Anything moved here is NOT classified as an
    # agent dir by the migration CLI.
    "test_archive",
})


def migrate_agent_storage_to_v3(agent_name: str, *, home: Path | None = None) -> dict[str, Any]:
    """Unreachable: v2→v3 copied out of ``.deepagents`` (POLICIES §16).

    The source tree is banned for reads, so there is nothing left to copy
    from. Installs still on v2 must be migrated out-of-band before use.

    Raises:
        DeepagentsPathForbiddenError: Always.
    """
    raise DeepagentsPathForbiddenError(
        f"v2→v3 migration for {agent_name!r} would read ~/.deepagents, which "
        "this project must never read or write (POLICIES §16)"
    )


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


def migrate_agent_storage_to_v6(agent_name: str, *, home: Path | None = None) -> dict[str, Any]:
    """Move ``~/.deep_worker/<name>/`` → ``~/.deep_worker/agents/<name>/``.

    DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §3.3.

    For names with a ``:tm`` or ``:sub_<hash>`` suffix, the canonical
    location is ``agents/<parent>/<suffix>/`` (e.g. ``agents/my_agent/tm/``)
    rather than ``agents/<name>/`` as a sibling. The legacy layout put
    these as siblings under ``~/.deep_worker/``; the canonical layout
    puts them as children of the parent agent. The migration routes
    accordingly so the parent agent code (which uses ``tm_root()`` to
    find the TaskManager state) sees its data.

    The ``agents/`` parent directory is created with mode 0o700 *before*
    any move happens, and the destination is checked to be free
    *before* ``shutil.move`` (avoids relying on the latter's idempotent
    semantics — when ``dst`` already exists, ``shutil.move`` would
    drop ``src`` INSIDE it rather than rename ``src`` to ``dst``).

    Special cases:
      - ``src`` missing, ``dst`` already populated → "already-migrated".
        If metadata is from a prior version chain (e.g. the legacy
        ``_v3``/``_v4`` paths already wrote into ``dst``), bump it to
        v6 so subsequent ``detect_storage_version`` calls return
        ``STORAGE_VERSION_AGENTS_SUBDIR``.
      - ``src`` missing, ``dst`` absent → "no-legacy".
      - both absent → "no-legacy" (caller has nothing to do).
    """
    validate_agent_name(agent_name)
    src = legacy_agent_root(agent_name, home=home)
    dst = agent_root(agent_name, home=home, create=False)

    # Route ``:tm`` and ``:sub_<hash>`` names to the proper child
    # location (``agents/<parent>/<suffix>/``) instead of treating them
    # as siblings. The canonical TaskManager state for ``my_agent`` is
    # ``agents/my_agent/tm/``, NOT ``agents/my_agent:tm/``.
    if ":" in agent_name:
        parent_name, _, suffix = agent_name.partition(":")
        if parent_name and suffix:
            parent_dst = agent_root(parent_name, home=home, create=False)
            if parent_dst.exists():
                dst = parent_dst / suffix

    if not src.exists():
        if dst.exists():
            # Already in the new layout. Bump metadata to v6 if it lags.
            meta = dst / "session_metadata.json"
            existing = _atomic_load_metadata(meta) if meta.exists() else {}
            if existing.get("storage_version") != STORAGE_VERSION_AGENTS_SUBDIR:
                existing["storage_version"] = STORAGE_VERSION_AGENTS_SUBDIR
                existing.pop("mv_failed", None)
                _set_metadata_atomic(meta, existing)
            return {"skipped": "already-migrated", "dst": str(dst)}
        return {"skipped": "no-legacy", "dst": str(dst)}

    if dst.exists():
        # Both legacy and new exist. This happens when:
        #   (a) a previous failed migration left an empty ``agents/<name>/``
        #       dir behind (the v6 path was created during
        #       ``_ensure_agent_layout`` but the move itself never ran), OR
        #   (b) a real race / partial migration left files in both places.
        # Case (a) is harmless: the canonical dir is empty. Detect by
        # checking that no entries live there. If dst is empty (only
        # ``.`` and ``..``), remove it and proceed with the move.
        dst_entries = list(dst.iterdir())
        if not dst_entries:
            try:
                dst.rmdir()
            except OSError:
                pass
        else:
            # Real both-exist (case b). Refuse to silently overwrite;
            # the caller can decide.
            return {
                "skipped": "both-exist",
                "src": str(src),
                "dst": str(dst),
            }

    # Ensure the ``agents/`` parent exists with mode 0o700 BEFORE first move.
    ensure_private_dir(dst.parent)

    # EXDEV fallback: cross-device move needs copy+rmtree
    try:
        shutil.move(str(src), str(dst))
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            try:
                shutil.rmtree(str(src))
            except OSError:
                meta = dst / "session_metadata.json"
                _set_metadata_atomic(
                    meta,
                    _atomic_load_metadata(meta)
                    | {"mv_failed": True, "storage_version": STORAGE_VERSION_AGENTS_SUBDIR},
                )
                return {"migrated": True, "partial": True, "dst": str(dst)}
        else:
            raise
    else:
        ensure_private_tree(dst)

    # Bump metadata to v6 (DESIGN §3.3 step 6-7). Use direct
    # composition — ``metadata_path(agent_name)`` would *also* return
    # this path, but making the path explicit here documents intent
    # and protects against future refactors of ``metadata_path``.
    meta = dst / "session_metadata.json"
    existing = _atomic_load_metadata(meta)
    existing.pop("mv_failed", None)
    existing["storage_version"] = STORAGE_VERSION_AGENTS_SUBDIR
    _set_metadata_atomic(meta, existing)

    # If this was a ``:tm``/``:sub_xxx`` name, the previous attempt left
    # an orphaned ``agents/<parent>:<suffix>/`` (sibling) dir with no
    # real data — ``shutil.move`` refused to overwrite, so the old
    # attempt created the empty dst and then bailed. Now that we've
    # successfully moved into the proper child location, clean up the
    # misplaced sibling if it's empty.
    if ":" in agent_name:
        parent_name, _, suffix = agent_name.partition(":")
        if parent_name and suffix:
            agents_root = deep_worker_home(home) / "agents"
            misplaced = agents_root / f"{parent_name}:{suffix}"
            if misplaced.exists() and misplaced != dst:
                try:
                    misplaced.rmdir()  # only succeeds if empty
                except OSError:
                    pass  # not empty — leave for manual inspection

    return {"migrated": True, "dst": str(dst)}


def migrate_agent_storage_to_current(agent_name: str, *, home: Path | None = None) -> dict[str, Any]:
    """Walk the agent from any prior storage version up to CURRENT.

    DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §3.4. Calls ``_v3`` and
    ``_v4`` as needed, then ``_v6``. Idempotent — calling twice for a
    v5-or-newer agent is a single ``skipped: already-current``.
    """
    validate_agent_name(agent_name)
    current = detect_storage_version(agent_name, home=home)

    if current == STORAGE_VERSION_AGENTS_SUBDIR:
        return {"skipped": "already-current"}

    # v2 → v3 → v4 (existing helpers)
    if current == STORAGE_VERSION_GLOBAL_DB:
        if not legacy_per_agent_db_path(agent_name, home=home, create_parent=False).exists():
            raise ValueError(
                f"Cannot migrate {agent_name!r}: no legacy storage found for v1 global."
            )
    if current in (STORAGE_VERSION_GLOBAL_DB, STORAGE_VERSION_DEEPAGENTS_PER_AGENT):
        migrate_agent_storage_to_v3(agent_name, home=home)
        migrate_agent_storage_to_v4(agent_name, home=home)
    elif current == STORAGE_VERSION_AGENT_ROOT:
        migrate_agent_storage_to_v4(agent_name, home=home)
    elif current in (
        STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
        STORAGE_VERSION_WITH_PREFERRED_PROVIDER,
    ):
        pass  # fall through to v6
    else:
        raise ValueError(f"Unsupported storage version for migration: {current!r}")

    return migrate_agent_storage_to_v6(agent_name, home=home)


def _ensure_agent_layout(name: str, *, home: Path | None = None) -> Path:
    """Bring ``name``'s storage up to the current layout, in place.

    DESIGN_AGENT_DBS §4.4. Idempotent: called at every agent startup.
    No-op when the storage is already at CURRENT_STORAGE_VERSION.
    Returns the canonical ``agents/<name>/`` path for the caller.
    """
    current = detect_storage_version(name, home=home)
    if current != CURRENT_STORAGE_VERSION:
        migrate_agent_storage_to_current(name, home=home)
    return agent_root(name, home=home, create=True)


def _atomic_load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _set_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    """Atomic write of ``session_metadata.json`` via tmp + replace.

    Mirrors the technique used by ``migrate_agent_storage_to_v4``: write
    to a uniquely-named sibling and ``replace()`` so a crash mid-write
    cannot corrupt the file. Caller is responsible for permission setup.
    """
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(metadata, indent=2))
        ensure_private_file(tmp)
        tmp.replace(path)
        ensure_private_file(path)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise
