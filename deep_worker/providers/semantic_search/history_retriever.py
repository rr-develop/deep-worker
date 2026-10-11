"""Retrieval pipeline for indexed history (JSONL summarization) files.

Reuses the same LanceDB-backed VectorStoreIndex infrastructure as
``semantic_search.retriever``, but queries the separately stored history
index under ``{summaries_dir}/.history_index/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore  # type: ignore[import-untyped]

from .config import IndexConfig
from .history_indexer import (
    HISTORY_INDEX_DIR_NAME,
    docstore_path,
    history_index_exists,
    lancedb_uri,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class HistorySearchResult:
    """A single result returned by :func:`search_history`."""

    content: str
    """Text of the retrieved message chunk."""

    score: float
    """Relevance score in ``(0, 1]`` — higher is more similar.

    NOTE: this is NOT the raw cosine similarity. LanceDB ranks with the
    ``L2`` metric (the default; no ``metric=`` is set) and llama-index maps
    the L2 distance ``d`` to ``score = exp(-d)`` (see
    ``llama_index.vector_stores.lancedb`` ``_to_llama_similarities``).

    Because the embeddings are L2-normalized (``HuggingFaceEmbedding`` has
    ``normalize=True`` by default), ranking by L2 is equivalent to ranking by
    cosine, so the ORDER of results is the cosine order. The score NUMBER,
    however, is ``exp(-d)`` with ``d in [0, 2]`` → score in ``[exp(-2), 1] ≈
    [0.135, 1.0]``. It is a monotonic but non-linear transform of cosine
    (``d^2 = 2 - 2*cos`` for unit vectors), never greater than 1.
    """

    thread_id: str
    """Thread ID (stem of the JSONL filename)."""

    line_range: str
    """Line range in the source JSONL file, e.g. ``"42-45"``."""

    timestamp: str
    """ISO-8601 timestamp of the first message in this chunk."""

    message_types: str
    """Comma-separated message type list, e.g. ``"human,ai"``."""

    file_path: str
    """Absolute path to the source JSONL file."""


# ---------------------------------------------------------------------------
# Process-wide caches (same pattern as retriever.py)
# ---------------------------------------------------------------------------

_index_cache: dict[tuple[str, str], tuple[VectorStoreIndex, StorageContext | None]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_embed_model(config: IndexConfig, embed_model: Any | None) -> Any:
    """Resolve embedding model with caching.

    Delegates to the process-wide shared cache in ``._embed_cache`` so
    every semantic-search call site (retriever, this file, history_indexer,
    indexer) shares one instance per model name. See a related internal
    incident analysis (not included here): the write path (history_indexer)
    originally had no cache; hoisting the resolver into a shared module is
    the fix that prevents the paths from drifting apart again.

    KNOWN LIMITATION (unchanged by this delegation): the e5 family
    (``intfloat/multilingual-e5-*``) was trained with mandatory input
    prefixes — ``"query: "`` for queries and ``"passage: "`` for indexed
    text. We currently embed raw text WITHOUT these prefixes here and in
    ``history_indexer``. Search still works and scores are sensible, but
    adding the prefixes would better match the model's training distribution.
    Changing it requires a full re-index, so it is deliberately out of
    scope.
    """
    from ._embed_cache import resolve_embed_model  # noqa: PLC0415

    return resolve_embed_model(config, embed_model)


def _load_index(summaries_dir: Path, config: IndexConfig, embed_model: Any) -> VectorStoreIndex:
    """Load the VectorStoreIndex for history search."""
    index_dir = summaries_dir / HISTORY_INDEX_DIR_NAME
    lancedb_uri_str = lancedb_uri(index_dir)
    docstore_dir = docstore_path(index_dir)

    vector_store = LanceDBVectorStore(uri=lancedb_uri_str, mode="append")
    docstore_persist = str(docstore_dir / "docstore.json")

    from llama_index.core.storage.docstore import SimpleDocumentStore

    try:
        docstore = SimpleDocumentStore.from_persist_path(docstore_persist)
    except (FileNotFoundError, ValueError, KeyError):
        docstore = SimpleDocumentStore()

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_history(
    query: str,
    summaries_dir: str | Path,
    top_k: int = 5,
    config: IndexConfig | None = None,
    embed_model: Any | None = None,
) -> list[HistorySearchResult]:
    """Search indexed history using semantic similarity.

    Args:
        query: Natural language search query.
        summaries_dir: Path to the summaries directory containing
            ``.history_index/``.
        top_k: Maximum number of results to return.
        config: Index configuration. Uses defaults if ``None``.
        embed_model: Override the embedding model (useful for testing).

    Returns:
        List of :class:`HistorySearchResult`, sorted by descending score.

    Raises:
        FileNotFoundError: If no history index exists.
    """
    summaries_dir = Path(summaries_dir)
    if not history_index_exists(summaries_dir):
        raise FileNotFoundError(
            f"No history index found in {summaries_dir / HISTORY_INDEX_DIR_NAME}\n"
            "Run index_all_history_files() first."
        )

    if config is None:
        config = IndexConfig()

    # Resolve embedding model
    embed_model = _ensure_embed_model(config, embed_model)

    Settings.embed_model = embed_model  # type: ignore[assignment]
    Settings.llm = None  # type: ignore[assignment]

    # Load or cache index
    cache_key = (str(summaries_dir), config.embedding_model)
    cached = _index_cache.get(cache_key)
    if cached is not None:
        index, _storage_context = cached
    else:
        index = _load_index(summaries_dir, config, embed_model)
        # We don't need to cache StorageContext for search-only
        _index_cache[cache_key] = (index, None)

    # Retrieve
    retriever = index.as_retriever(similarity_top_k=top_k)
    raw_results = retriever.retrieve(query)

    results: list[HistorySearchResult] = []
    for nws in raw_results[:top_k]:
        node = nws.node
        metadata = node.metadata or {}

        results.append(
            HistorySearchResult(
                content=node.get_content(),
                score=float(nws.score or 0.0),
                thread_id=metadata.get("thread_id", "unknown"),
                line_range=metadata.get("line_range", "0-0"),
                timestamp=metadata.get("timestamp", ""),
                message_types=metadata.get("message_types", ""),
                file_path=metadata.get("file_path", ""),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
