"""semantic_search  semantic code search using LlamaIndex + LanceDB."""

from __future__ import annotations

from .config import IndexConfig, RepoIndex
from .exceptions import IndexNotFoundError, IndexingError
from .history_indexer import (
    HistoryIndexStats,
    clear_history_index,
    history_index_exists,
    index_all_history_files,
    index_history_file,
)
from .history_retriever import HistorySearchResult, search_history
from .indexer import IndexStats, index_repository
from .retriever import SearchResult, search

__all__ = [
    "IndexConfig",
    "RepoIndex",
    "IndexNotFoundError",
    "IndexingError",
    "IndexStats",
    "index_repository",
    "SearchResult",
    "search",
    "HistoryIndexStats",
    "history_index_exists",
    "index_history_file",
    "index_all_history_files",
    "clear_history_index",
    "HistorySearchResult",
    "search_history",
]
