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

    Traverses the directory tree recursively, skipping excluded directories
    and returning only files whose extension is in ``INDEXED_EXTENSIONS``.

    Args:
        repo_path: Root directory of the repository to scan.

    Yields:
        Absolute ``Path`` objects for each indexable file.

    Raises:
        FileNotFoundError: If *repo_path* does not exist or is not a directory.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {repo_path}")

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
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
