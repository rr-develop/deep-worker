"""Retrieval pipeline for semantic_search.

Uses a LanceDB-backed VectorStoreIndex for semantic search.
Supports both flat structure (single chunk size) and hierarchical structure
(AutoMergingRetriever with parent/child chunks for context expansion).

Process-wide caching
====================
Three things are deliberately cached at module scope so that repeated
``search()`` calls reuse them instead of paying the per-call price every
time:

* ``_embedding_cache`` keyed by model name — a ``HuggingFaceEmbedding``
  loads sentence-transformer weights into a torch storage and ``mmap``-s
  the model blob (~450 MB virtual / ~128 MB resident for
  ``multilingual-e5-small``). Reinstantiating it per call adds one
  fresh ``mmap`` and one fresh weight tensor each time.

* ``_index_cache`` keyed by ``(repo_path, model_name)`` — holds
  ``(VectorStoreIndex, StorageContext)``. The ``StorageContext`` owns a
  ``LanceDBVectorStore`` which spawns a tokio thread pool (visible as
  ``tokio-runtime-w`` threads) and keeps a C++-side handle to the
  on-disk lance dataset; that handle keeps a Python reference to the
  embed_model. Without caching, every search creates a new
  vector_store + new index + new tokio pool, and the previous ones
  cannot be freed by Python GC alone — the lance C++ retainers (and
  ``llama_index.core.Settings.embed_model`` global indirection) keep
  the Python objects alive past the function return.

The original incident (2026-04-25) showed RSS growing
by ~390 MB on every ``project_memory`` call until the agent reached
multiple GB. The regression test
``tests/integration/test_search_no_leak_per_call.py`` enforces the
post-warmup invariant: 5 sequential searches must add < 50 MB total.

The caches are unbounded by design — there are at most a handful of
``(repo_path, model_name)`` combinations per process, and the working
agents live for hours. If we ever need bounded eviction we can swap
the dicts for an LRU; for now the cost-of-eviction (re-creating these
objects has the very leak we're guarding against) outweighs the
benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import NodeRelationship
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore  # type: ignore[import-untyped]

from .config import IndexConfig, RepoIndex
from .exceptions import IndexNotFoundError


# ---------------------------------------------------------------------------
# Process-wide caches (see module docstring for rationale)
# ---------------------------------------------------------------------------

# NOTE: the embedding-model cache used to live here (a module-level
# ``_embedding_cache`` dict). It has been hoisted into
# ``._embed_cache.resolve_embed_model`` so all four semantic-search call
# sites (this file, ``history_retriever``, ``history_indexer``, ``indexer``)
# share one instance per model name. See a related internal incident
# analysis (not included here) for why the duplicated per-file caches
# were dangerous.

# (repo_path_str, model_name) -> (VectorStoreIndex, StorageContext).
# StorageContext is kept alongside the index because AutoMergingRetriever
# needs the docstore reference at retriever-build time.
_index_cache: dict[tuple[str, str], tuple[VectorStoreIndex, StorageContext]] = {}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result returned by :func:`search`."""

    file_path: str
    """Absolute path to the source file."""

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

    content: str
    """Text of the retrieved chunk."""

    language: str | None
    """Tree-sitter language name, or ``None`` for non-code files."""

    node_type: str
    """Either ``"leaf"`` (exact match) or ``"parent"`` (merged context)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_storage_context(repo_index: RepoIndex) -> StorageContext:
    """Load vector store and docstore from an existing index directory."""
    vector_store = LanceDBVectorStore(uri=repo_index.lancedb_uri, mode="append")
    docstore_persist = str(repo_index.docstore_path / "docstore.json")

    try:
        docstore = SimpleDocumentStore.from_persist_path(docstore_persist)
    except (FileNotFoundError, ValueError, KeyError):
        docstore = SimpleDocumentStore()

    return StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )


def _build_retriever(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    config: IndexConfig,
    top_k: int,
) -> BaseRetriever:
    """Build a retriever over *index*.

    Returns AutoMergingRetriever if parent_chunk_size is set (hierarchical),
    or plain VectorIndexRetriever if parent_chunk_size is None (flat structure).
    """
    # Flat structure: use plain VectorIndexRetriever
    if config.parent_chunk_size is None:
        return index.as_retriever(similarity_top_k=top_k)

    # Hierarchical structure: use AutoMergingRetriever
    # Retrieve more candidates than top_k so merging has room to work.
    # VectorStoreIndex.as_retriever() always returns VectorIndexRetriever,
    # but the base class return type is BaseRetriever — assert the real type.
    base_retriever = index.as_retriever(similarity_top_k=top_k * 2)
    if not isinstance(base_retriever, VectorIndexRetriever):
        raise TypeError(
            f"Expected VectorIndexRetriever from VectorStoreIndex.as_retriever(), "
            f"got {type(base_retriever).__name__}"
        )
    return AutoMergingRetriever(
        base_retriever,
        storage_context,  # AutoMergingRetriever expects StorageContext, not docstore
        simple_ratio_thresh=config.simple_ratio_thresh,
        verbose=False,
    )


def _to_search_result(node_with_score: Any, *, node_type: str) -> SearchResult:
    """Convert a LlamaIndex NodeWithScore to a :class:`SearchResult`."""
    node = node_with_score.node
    metadata: dict[str, Any] = node.metadata or {}
    return SearchResult(
        file_path=metadata.get("file_path", ""),
        score=float(node_with_score.score or 0.0),
        content=node.get_content(),
        language=metadata.get("language") or None,
        node_type=node_type,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    query: str,
    repo_path: Path,
    top_k: int = 5,
    config: IndexConfig | None = None,
    embedding_model: Any | None = None,
) -> list[SearchResult]:
    """Search an indexed repository using semantic similarity.

    Args:
        query:           Natural language search query.
        repo_path:       Absolute path to the repository root.
        top_k:           Maximum number of results to return.
        config:          Index configuration. Uses defaults if ``None``.
        embedding_model: Override the embedding model (useful for testing).

    Returns:
        List of :class:`SearchResult`, sorted by descending score.

    Raises:
        IndexNotFoundError: If the repository has not been indexed yet.
    """
    repo_index = RepoIndex(repo_path=repo_path)
    if not repo_index.exists():
        raise IndexNotFoundError(
            f"Repository has not been indexed: {repo_path}\n"
            "Run index_repository() first."
        )

    if config is None:
        config = IndexConfig()

    # Resolve embedding model via the shared process-wide cache in
    # ``._embed_cache``. Caller-supplied models bypass the cache (tests
    # pass e.g. ``MockEmbedding`` with arbitrary lifetimes; storing them
    # would poison the cache). When no model is supplied the FIRST call
    # for a given model name pays the full cost of instantiating
    # ``HuggingFaceEmbedding`` (weight load + mmap of the model blob) —
    # every subsequent call across every semantic-search entry point
    # returns the same instance, so no extra mmap / no extra torch storage
    # is allocated. ``local_files_only=True`` is enforced inside the
    # resolver (offline guarantee, see
    # ``docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md``).
    from ._embed_cache import resolve_embed_model  # noqa: PLC0415

    embedding_model = resolve_embed_model(config, embedding_model)

    # ``Settings.embed_model`` is a global llama_index singleton. We
    # keep updating it so that any code path that reads from
    # ``Settings`` (instead of receiving an explicit embed_model) sees
    # a consistent value. This is just a ref reassignment — it does
    # NOT keep the previous value alive.
    Settings.embed_model = embedding_model  # type: ignore[assignment]  # cache may hold Any
    Settings.llm = None  # type: ignore[assignment]  # LlamaIndex accepts None at runtime

    # Resolve VectorStoreIndex + StorageContext.
    #
    # Cache key is (repo_path, model_name) because:
    #   * Different repos point at different lance datasets — distinct
    #     ``LanceDBVectorStore`` C++ handles and distinct docstores.
    #   * Different embedding models produce vectors of different
    #     dimensionality / semantics, so reusing one index against
    #     another model would silently return garbage scores.
    #
    # When the caller supplies a custom ``embedding_model`` we still
    # cache by ``config.embedding_model`` (the *name*), not by the
    # caller's object identity, so a test passing a MockEmbedding doesn't
    # collide with the production cache entry for the same model name.
    cache_key = (str(repo_path), config.embedding_model)
    cached_pair = _index_cache.get(cache_key)
    if cached_pair is None:
        storage_context = _load_storage_context(repo_index)
        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            storage_context=storage_context,
            embed_model=embedding_model,
        )
        _index_cache[cache_key] = (index, storage_context)
    else:
        index, storage_context = cached_pair

    retriever = _build_retriever(index, storage_context, config, top_k)
    raw_results = retriever.retrieve(query)

    # Determine node_type: check if node has children in docstore (→ parent)
    results: list[SearchResult] = []
    for nws in raw_results[:top_k]:
        node = nws.node
        # A "parent" node is one that has CHILD relationships
        # Use official NodeRelationship enum instead of hardcoded strings
        has_children = bool(
            node.relationships and NodeRelationship.CHILD in node.relationships
        )
        node_type = "parent" if has_children else "leaf"
        results.append(_to_search_result(nws, node_type=node_type))

    # Sort by descending score
    results.sort(key=lambda r: r.score, reverse=True)
    return results
