"""Utilities for checking semantic search index availability."""

from __future__ import annotations

from pathlib import Path


def has_semantic_index(working_dir: Path) -> bool:
    """Check if a semantic search index exists in the given directory.

    Verifies the presence of all required index components:
    - .vector_index/lancedb/ (vector database)
    - .vector_index/docstore/docstore.json (document store)
    - .vector_index/file_hashes.json (incremental indexing metadata)

    Args:
        working_dir: Directory to check for index.

    Returns:
        True if valid index exists, False otherwise.

    Example:
        >>> has_semantic_index(Path("/path/to/my_project"))
        True
        >>> has_semantic_index(Path("/path/to/new_project"))
        False
    """
    index_dir = working_dir / ".vector_index"

    # Check all required components
    required_paths = [
        index_dir / "lancedb",
        index_dir / "docstore" / "docstore.json",
        index_dir / "file_hashes.json",
    ]

    return all(path.exists() for path in required_paths)
