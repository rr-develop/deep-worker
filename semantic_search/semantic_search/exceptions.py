"""Custom exceptions for semantic_search."""

from __future__ import annotations


class IndexNotFoundError(FileNotFoundError):
    """Raised when searching a repository that has not been indexed yet."""


class IndexingError(RuntimeError):
    """Raised when an unrecoverable error occurs during indexing."""
