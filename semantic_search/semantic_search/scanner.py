"""File discovery and metadata helpers for semantic_search."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from .config import CODE_EXTENSIONS, EXCLUDED_DIRS, INDEXED_EXTENSIONS, LANGUAGE_MAP


def _has_excluded_component(path: Path) -> bool:
    """Return True if any component of *path* is an excluded directory name."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _scan_files_via_filesystem(repo_path: Path) -> Iterator[Path]:
    """Yield indexable files under *repo_path* using a pure filesystem walk.

    Walks *repo_path* recursively via :meth:`Path.rglob`, skipping any path
    whose components include a directory name listed in
    :data:`semantic_search.config.EXCLUDED_DIRS` (``.git``, ``node_modules``,
    ``__pycache__``, ``.venv``, ``dist``, ``build``, ``.vector_index``, etc.),
    and keeping only files whose suffix is in
    :data:`semantic_search.config.INDEXED_EXTENSIONS`.

    This walker has **no** notion of ``.gitignore`` — callers that want that
    semantics must use the git-aware branch (see :func:`_scan_files_via_git`).
    """
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if _has_excluded_component(path):
            continue
        if path.suffix in INDEXED_EXTENSIONS:
            yield path.resolve()


def _scan_files_via_git(repo_path: Path) -> Iterator[Path]:
    """Yield indexable files under *repo_path* using ``git ls-files``.

    Honours ``.gitignore`` automatically by deferring to git for the file
    list. Combines tracked files with untracked-but-not-ignored files so
    that newly-added (uncommitted) source files are indexed too.

    Raises:
        RuntimeError: If *repo_path* is not a git repository.
    """
    import subprocess  # noqa: PLC0415

    if not (repo_path / ".git").exists():
        msg = (
            f"Not a git repository: {repo_path}. "
            "Pass ``use_git=False`` for a filesystem-only scan, or run "
            "``git init`` first."
        )
        raise RuntimeError(msg)

    # Get tracked files (already committed or staged)
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Get untracked files that are NOT in .gitignore
    # This includes new files that haven't been committed yet but are valid source files
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    untracked_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Combine and filter by extension
    all_files = set(tracked_files + untracked_files)
    for file_path_str in all_files:
        if not file_path_str:  # Skip empty strings
            continue
        path = repo_path / file_path_str
        if not path.exists() or not path.is_file():
            continue
        # Apply additional exclusions (belt-and-suspenders)
        if _has_excluded_component(path):
            continue
        if path.suffix in INDEXED_EXTENSIONS:
            yield path


def scan_files(
    repo_path: Path,
    *,
    use_git: bool | None = None,
) -> Iterator[Path]:
    """Yield all indexable files under *repo_path*.

    Two scanning strategies are supported; the right one is picked from
    *use_git*:

      - ``use_git=True``  — shell out to ``git ls-files`` so ``.gitignore``
        is honoured for free. Raises :class:`RuntimeError` if *repo_path* is
        not a git repository.
      - ``use_git=False`` — pure-Python filesystem walk that ignores
        ``.gitignore`` entirely. Always works on any directory.
      - ``use_git=None`` (default) — **auto-detect**. Use the git-aware
        scanner when ``repo_path/.git`` exists, otherwise the filesystem
        scanner. This is what callers like ``index_repository`` rely on so
        the feature works for both committed-repo and ad-hoc directories.

    Args:
        repo_path: Root directory of the repository to scan.
        use_git: Override auto-detection. See above.

    Yields:
        Absolute ``Path`` objects for each indexable file.

    Raises:
        FileNotFoundError: If *repo_path* does not exist or is not a directory.
        RuntimeError: If *use_git* is ``True`` and *repo_path* is not a
            git repository.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {repo_path}")

    # Decide on the strategy.
    if use_git is None:
        use_git = (repo_path / ".git").exists()

    if use_git:
        yield from _scan_files_via_git(repo_path)
    else:
        yield from _scan_files_via_filesystem(repo_path)


def get_language(path: Path) -> str | None:
    """Return the tree-sitter language name for *path*, or ``None``.

    Only code files (those in ``LANGUAGE_MAP``) return a language string.
    Docs, config files, and unknown extensions return ``None``.

    Args:
        path: Path to the file (only the suffix is examined).

    Returns:
        Language name such as ``"python"`` or ``"typescript"``, or ``None``.
    """
    return LANGUAGE_MAP.get(path.suffix)


def is_code_file(path: Path) -> bool:
    """Return ``True`` if *path* should be split with ``CodeSplitter``.

    Args:
        path: Path to the file.
    """
    return path.suffix in CODE_EXTENSIONS


def compute_file_hash(path: Path) -> str:
    """Return a stable MD5 hex-digest of the file contents.

    Used for incremental indexing: files whose hash is unchanged since the
    last run are skipped entirely.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex string of the MD5 digest.
    """
    md5 = hashlib.md5(usedforsecurity=False)
    md5.update(path.read_bytes())
    return md5.hexdigest()
