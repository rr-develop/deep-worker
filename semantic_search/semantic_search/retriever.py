"""Retrieval pipeline for semantic_search.

Uses AutoMergingRetriever on top of a LanceDB-backed VectorStoreIndex.
Leaf nodes are searched by vector similarity; when a sufficient fraction of
a parent's children are retrieved, the parent (larger context) is returned
instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from .config import IndexConfig, RepoIndex
from .exceptions import IndexNotFoundError


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result returned by :func:`search`."""

    file_path: str
    """Absolute path to the source file."""

    score: float
    """Cosine similarity score in [0, 1]."""

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
) -> AutoMergingRetriever:
    """Build an AutoMergingRetriever over *index*."""
    # Retrieve more candidates than top_k so merging has room to work
    base_retriever = index.as_retriever(similarity_top_k=top_k * 2)
    return AutoMergingRetriever(
        base_retriever,
        storage_context.docstore,
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

    # Resolve embedding model
    if embedding_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: PLC0415

        embedding_model = HuggingFaceEmbedding(model_name=config.embedding_model)

    Settings.embed_model = embedding_model
    Settings.llm = None

    storage_context = _load_storage_context(repo_index)
    index = VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        storage_context=storage_context,
        embed_model=embedding_model,
    )

    retriever = _build_retriever(index, storage_context, config, top_k)
    raw_results = retriever.retrieve(query)

    # Determine node_type: check if node has children in docstore (→ parent)
    results: list[SearchResult] = []
    for nws in raw_results[:top_k]:
        node = nws.node
        # A "parent" node is one that has child relationships stored
        has_children = bool(
            node.relationships and
            any(k in ("2", "3") for k in node.relationships)  # CHILD = "2", NEXT = "3"
        )
        node_type = "parent" if has_children else "leaf"
        results.append(_to_search_result(nws, node_type=node_type))

    # Sort by descending score
    results.sort(key=lambda r: r.score, reverse=True)
    return results
