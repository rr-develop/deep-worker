"""Indexing pipeline for semantic_search.

Architecture
------------
Per-file splitting:
  - Code files  → CodeSplitter (tree-sitter, language-aware)
  - Other files → SentenceSplitter (generic text)

Hierarchical retrieval:
  - Leaf nodes (small, from the splitters above) are stored in LanceDB.
  - Parent nodes (large, from HierarchicalNodeParser) are stored only in
    SimpleDocumentStore so AutoMergingRetriever can expand them.

Incremental indexing:
  - A JSON file at .vector_index/file_hashes.json tracks MD5 hashes.
  - On each run, only added / modified files are processed.
  - Deleted files are removed from both LanceDB and the docstore.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    CodeSplitter,
    HierarchicalNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from .config import IndexConfig, RepoIndex
from .scanner import compute_file_hash, get_language, scan_files

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    """Statistics returned by :func:`index_repository`."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    total_nodes: int = 0

    @property
    def total_files(self) -> int:
        """Sum of all file counters."""
        return self.added + self.updated + self.deleted + self.skipped

    def __repr__(self) -> str:
        return (
            f"IndexStats(added={self.added}, updated={self.updated}, "
            f"deleted={self.deleted}, skipped={self.skipped}, "
            f"total_nodes={self.total_nodes})"
        )


# ---------------------------------------------------------------------------
# Hash persistence helpers
# ---------------------------------------------------------------------------


def _load_hashes(path: Path) -> dict[str, str]:
    """Load file hashes from a JSON file, or return empty dict."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_hashes(path: Path, hashes: dict[str, str]) -> None:
    """Persist file hashes to a JSON file."""
    path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Document / node helpers
# ---------------------------------------------------------------------------


def _load_document(file_path: Path, repo_path: Path) -> Document:
    """Read *file_path* and return a LlamaIndex Document with metadata."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""

    language = get_language(file_path)
    return Document(
        text=text,
        doc_id=str(file_path),
        metadata={
            "file_path": str(file_path),
            "rel_path": str(file_path.relative_to(repo_path)),
            "file_name": file_path.name,
            "language": language if language is not None else "text",
            "extension": file_path.suffix,
            "repo_path": str(repo_path),
        },
        excluded_embed_metadata_keys=["repo_path", "rel_path", "extension"],
        excluded_llm_metadata_keys=["repo_path", "extension"],
    )


def _split_document(
    doc: Document,
    config: IndexConfig,
) -> list[BaseNode]:
    """Split a document into leaf nodes using the appropriate splitter.

    Code files use CodeSplitter (tree-sitter). Everything else falls back
    to SentenceSplitter.
    """
    language: str | None = doc.metadata.get("language")

    if language and language != "text":
        try:
            splitter: CodeSplitter | SentenceSplitter = CodeSplitter(
                language=language,
                chunk_lines=config.chunk_lines,
                chunk_lines_overlap=config.chunk_lines_overlap,
                max_chars=config.max_chars,
            )
            nodes = splitter.get_nodes_from_documents([doc])
            if nodes:
                return nodes
        except Exception:  # noqa: BLE001
            pass  # Fall through to sentence splitter

    return SentenceSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.chunk_lines_overlap * 4,
    ).get_nodes_from_documents([doc])


def _make_hierarchy(
    doc: Document,
    leaf_nodes: list[BaseNode],
    config: IndexConfig,
) -> list[BaseNode]:
    """Create parent nodes from *doc* and link them to *leaf_nodes*.

    Parent nodes are built by SentenceSplitter at a larger chunk size.
    Each parent stores references to its children so AutoMergingRetriever
    can group and expand them.
    """
    parent_splitter = SentenceSplitter(chunk_size=config.parent_chunk_size)
    parent_nodes = parent_splitter.get_nodes_from_documents([doc])

    if not parent_nodes or not leaf_nodes:
        return leaf_nodes

    # Assign parent_node reference to each leaf node.
    # We distribute leaves evenly across parents by text position.
    leaves_per_parent = max(1, len(leaf_nodes) // len(parent_nodes))
    for idx, leaf in enumerate(leaf_nodes):
        parent_idx = min(idx // leaves_per_parent, len(parent_nodes) - 1)
        from llama_index.core.schema import RelatedNodeInfo

        leaf.relationships["1"] = RelatedNodeInfo(  # NodeRelationship.PARENT = "1"
            node_id=parent_nodes[parent_idx].node_id,
        )

    return parent_nodes + leaf_nodes


def _process_document(
    doc: Document,
    config: IndexConfig,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """Return (all_nodes, leaf_nodes) for a document.

    *all_nodes* go into the docstore (needed for AutoMergingRetriever).
    *leaf_nodes* go into the vector store (for similarity search).
    """
    leaf_nodes = _split_document(doc, config)
    all_nodes = _make_hierarchy(doc, leaf_nodes, config)
    return all_nodes, leaf_nodes


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _load_or_create_docstore(persist_path: str) -> SimpleDocumentStore:
    """Load an existing SimpleDocumentStore or return a fresh one."""
    try:
        return SimpleDocumentStore.from_persist_path(persist_path)
    except (FileNotFoundError, ValueError, KeyError):
        return SimpleDocumentStore()


def _build_storage_context(
    vector_store: LanceDBVectorStore,
    docstore: SimpleDocumentStore,
) -> StorageContext:
    return StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )


def _add_document(
    doc: Document,
    index: VectorStoreIndex,
    storage_context: StorageContext,
    config: IndexConfig,
    embed_model: Any,
) -> int:
    """Add *doc* to the index. Returns the number of leaf nodes added."""
    all_nodes, leaf_nodes = _process_document(doc, config)

    # Embed leaf nodes
    texts = [n.get_content() for n in leaf_nodes]
    if texts:
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
        for node, emb in zip(leaf_nodes, embeddings):
            node.embedding = emb

    # Store ALL nodes in docstore for hierarchical merging
    storage_context.docstore.add_documents(all_nodes)

    # Index only leaf nodes in the vector store
    index.insert_nodes(leaf_nodes)

    return len(leaf_nodes)


def _remove_document(
    doc_id: str,
    index: VectorStoreIndex,
) -> None:
    """Remove all nodes for *doc_id* from vector store and docstore."""
    try:
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
    except Exception:  # noqa: BLE001
        pass  # May not exist; safe to ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_repository(
    repo_path: Path,
    config: IndexConfig | None = None,
    *,
    reset: bool = False,
    embedding_model: Any | None = None,
) -> IndexStats:
    """Index *repo_path* for semantic search.

    Creates or updates the vector index at ``{repo_path}/.vector_index/``.
    Indexing is incremental by default — only changed files are processed.

    Args:
        repo_path:       Absolute path to the repository root.
        config:          Indexing configuration. Uses defaults if ``None``.
        reset:           Delete existing index and reindex from scratch.
        embedding_model: Override the embedding model (useful for testing).

    Returns:
        :class:`IndexStats` with counts of added / updated / deleted / skipped.

    Raises:
        FileNotFoundError: If *repo_path* does not exist or is not a directory.
    """
    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if config is None:
        config = IndexConfig()

    repo_index = RepoIndex(repo_path=repo_path)

    # Reset: wipe existing index before starting
    if reset and repo_index.exists():
        shutil.rmtree(repo_index.index_dir)

    # Ensure directories exist
    repo_index.docstore_path.mkdir(parents=True, exist_ok=True)

    # Resolve embedding model
    if embedding_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: PLC0415

        embedding_model = HuggingFaceEmbedding(model_name=config.embedding_model)

    Settings.embed_model = embedding_model
    Settings.llm = None  # No LLM required for indexing

    # --- Storage setup ---
    # Load hashes before deciding LanceDB mode.
    # - Fresh index (no hashes yet) or reset → "overwrite" creates the table.
    # - Subsequent incremental run → "append" keeps existing vectors.
    existing_hashes = _load_hashes(repo_index.hashes_path)
    is_fresh = not existing_hashes
    lance_mode = "overwrite" if is_fresh else "append"
    vector_store = LanceDBVectorStore(uri=repo_index.lancedb_uri, mode=lance_mode)
    docstore_persist = str(repo_index.docstore_path / "docstore.json")
    docstore = _load_or_create_docstore(docstore_persist)
    storage_context = _build_storage_context(vector_store, docstore)

    # Load or create the index
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=False,
    )

    # --- Incremental diff ---
    current_hashes: dict[str, str] = {}
    stats = IndexStats()

    for file_path in scan_files(repo_path):
        path_str = str(file_path)
        file_hash = compute_file_hash(file_path)
        current_hashes[path_str] = file_hash
        existing_hash = existing_hashes.get(path_str)

        if existing_hash is None:
            # New file
            doc = _load_document(file_path, repo_path)
            n = _add_document(doc, index, storage_context, config, embedding_model)
            stats.added += 1
            stats.total_nodes += n

        elif existing_hash != file_hash:
            # Modified file: remove old nodes, re-index
            _remove_document(path_str, index)
            doc = _load_document(file_path, repo_path)
            n = _add_document(doc, index, storage_context, config, embedding_model)
            stats.updated += 1
            stats.total_nodes += n

        else:
            # Unchanged
            stats.skipped += 1

    # Handle deleted files
    deleted_paths = set(existing_hashes) - set(current_hashes)
    for path_str in deleted_paths:
        _remove_document(path_str, index)
        stats.deleted += 1

    # Persist state
    _save_hashes(repo_index.hashes_path, current_hashes)
    docstore.persist(persist_path=docstore_persist)

    return stats
