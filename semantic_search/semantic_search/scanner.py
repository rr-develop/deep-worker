"""File discovery and metadata helpers for semantic_search."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from .config import CODE_EXTENSIONS, EXCLUDED_DIRS, INDEXED_EXTENSIONS, LANGUAGE_MAP


def _has_excluded_component(path: Path) -> bool:
    """Return True if any component of *path* is an excluded directory name."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def scan_files(repo_path: Path) -> Iterator[Path]:
    """Yield all indexable files under *repo_path*.

    Uses git to determine which files to index, automatically respecting .gitignore.
    Includes both tracked files and untracked files that are not gitignored.

    This ensures that:
    - Downloaded dependencies (e.g., google-cloud-sdk/) are excluded
    - Build artifacts, virtual environments, and other ignored files are excluded
    - Only source files that are (or could be) committed to git are indexed

    Args:
        repo_path: Root directory of the repository to scan (must be a git repo).

    Yields:
        Absolute ``Path`` objects for each indexable file.

    Raises:
        FileNotFoundError: If *repo_path* does not exist or is not a directory.
        RuntimeError: If *repo_path* is not a git repository.
    """
    import subprocess  # noqa: PLC0415

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {repo_path}")

    # Check if it's a git repository
    if not (repo_path / ".git").exists():
        msg = (
            f"Not a git repository: {repo_path}. "
            "Semantic search requires a git repository to respect .gitignore patterns."
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
