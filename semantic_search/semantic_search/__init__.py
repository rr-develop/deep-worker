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
from .long_text_embedding import embed_history_messages, embed_long_text
from .message_embedding_store import (
    FLAG_NAMES,
    MessageEmbeddingStats,
    MessageFlag,
    MessageMetadataStats,
    build_message_embeddings,
    build_message_metadata,
    classify_message,
    flag_from_name,
    flag_names,
    load_message_embeddings,
    load_message_metadata,
    message_embeddings_exist,
    message_metadata_exist,
)
from .message_search import MessageSearchResult, search_messages
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
    "embed_long_text",
    "embed_history_messages",
    "MessageEmbeddingStats",
    "MessageMetadataStats",
    "MessageFlag",
    "build_message_embeddings",
    "build_message_metadata",
    "load_message_embeddings",
    "load_message_metadata",
    "message_embeddings_exist",
    "message_metadata_exist",
    "classify_message",
    "FLAG_NAMES",
    "flag_from_name",
    "flag_names",
    "MessageSearchResult",
    "search_messages",
]
