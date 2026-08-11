"""DEPRECATED shim — migrate to ``deep_worker.providers.semantic_search``.

This root package exists only for local backward compatibility on this host.
It is NOT part of the published wheel/sdist. Old submodule imports
(``import semantic_search.history_indexer``) are NOT supported: use
``deep_worker.providers.semantic_search.history_indexer`` instead.
"""
import warnings as _warnings

_warned = False
__all__ = [
    "IndexConfig",
    "RepoIndex",
    "IndexNotFoundError",
    "IndexingError",
    "HistoryIndexStats",
    "clear_history_index",
    "history_index_exists",
    "index_all_history_files",
    "index_history_file",
    "HistorySearchResult",
    "search_history",
    "IndexStats",
    "index_repository",
    "embed_history_messages",
    "embed_long_text",
    "FLAG_NAMES",
    "MessageEmbeddingStats",
    "MessageFlag",
    "search_messages",
]


def __getattr__(name):
    global _warned
    if not _warned:
        _warned = True
        _warnings.warn(
            "semantic_search is deprecated; use "
            "deep_worker.providers.semantic_search",
            DeprecationWarning,
            stacklevel=2,
        )
    from deep_worker.providers import semantic_search as _canonical

    return getattr(_canonical, name)
