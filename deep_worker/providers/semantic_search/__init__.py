"""semantic_search — semantic code search using LlamaIndex + LanceDB.

LAZY SURFACE: importing this root
package MUST NOT load the heavy submodules (``indexer``, ``retriever``,
``history_indexer``, ``history_retriever``, ``long_text_embedding``) and MUST
NOT pull llama-index. All exports are resolved lazily via PEP 562
``__getattr__`` on first attribute access. This keeps core-startup free of the
heavy extra and lets the root shim alias (§3.2b) import without side effects.
"""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import IndexConfig, RepoIndex  # noqa: F401
    from .exceptions import IndexNotFoundError, IndexingError  # noqa: F401
    from .history_indexer import (  # noqa: F401
        HistoryIndexStats,
        clear_history_index,
        history_index_exists,
        index_all_history_files,
        index_history_file,
    )
    from .history_retriever import HistorySearchResult, search_history  # noqa: F401
    from .indexer import IndexStats, index_repository  # noqa: F401
    from .long_text_embedding import embed_history_messages, embed_long_text  # noqa: F401
    from .message_embedding_store import (  # noqa: F401
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
    from .message_search import MessageSearchResult, search_messages  # noqa: F401
    from .retriever import SearchResult, search  # noqa: F401

# name → submodule (relative to this package). Kept as a plain mapping so the
# lazy ``__getattr__`` below stays import-free and side-effect-free.
_LAZY = {
    "IndexConfig": "config",
    "RepoIndex": "config",
    "IndexNotFoundError": "exceptions",
    "IndexingError": "exceptions",
    "IndexStats": "indexer",
    "index_repository": "indexer",
    "SearchResult": "retriever",
    "search": "retriever",
    "HistoryIndexStats": "history_indexer",
    "history_index_exists": "history_indexer",
    "index_history_file": "history_indexer",
    "index_all_history_files": "history_indexer",
    "clear_history_index": "history_indexer",
    "HistorySearchResult": "history_retriever",
    "search_history": "history_retriever",
    "embed_long_text": "long_text_embedding",
    "embed_history_messages": "long_text_embedding",
    "MessageEmbeddingStats": "message_embedding_store",
    "MessageMetadataStats": "message_embedding_store",
    "MessageFlag": "message_embedding_store",
    "build_message_embeddings": "message_embedding_store",
    "build_message_metadata": "message_embedding_store",
    "load_message_embeddings": "message_embedding_store",
    "load_message_metadata": "message_embedding_store",
    "message_embeddings_exist": "message_embedding_store",
    "message_metadata_exist": "message_embedding_store",
    "classify_message": "message_embedding_store",
    "FLAG_NAMES": "message_embedding_store",
    "flag_from_name": "message_embedding_store",
    "flag_names": "message_embedding_store",
    "MessageSearchResult": "message_search",
    "search_messages": "message_search",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(f".{sub}", __name__)
    value = getattr(module, name)
    # Cache on the package so repeated access does not re-import.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
