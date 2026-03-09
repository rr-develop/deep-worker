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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    CodeSplitter,
    SentenceSplitter,
)
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.utils import get_tokenizer  # type: ignore[import-untyped]
from llama_index.vector_stores.lancedb import LanceDBVectorStore  # type: ignore[import-untyped]

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


def _create_merged_node(batch: list[BaseNode], merged_text: str) -> BaseNode:  # type: ignore[misc]
    """Create a merged node from a batch of nodes.

    Preserves metadata from the first node in the batch.

    Args:
        batch: List of nodes to merge.
        merged_text: Combined text content.

    Returns:
        Single node (original if batch size is 1, new TextNode if merged).
    """
    if len(batch) == 1:
        return batch[0]

    first = batch[0]
    return TextNode(
        text=merged_text,
        metadata=first.metadata.copy() if first.metadata else {},
        excluded_embed_metadata_keys=first.excluded_embed_metadata_keys or [],
        excluded_llm_metadata_keys=first.excluded_llm_metadata_keys or [],
    )


def _merge_small_chunks(
    nodes: list[BaseNode],
    max_tokens: int,
    min_chars: int = 100,
) -> list[BaseNode]:  # type: ignore[misc]
    """Merge small chunks and filter out files with single tiny chunk.

    Algorithm:
    1. Phase 1 - Merging: Iterate through nodes sequentially and accumulate
       consecutive nodes into merged chunks (up to max_tokens)
    2. Phase 2 - Filtering (applied to Phase 1 results):
       - Single-chunk file < min_chars → skip entire file (return [])
       - Multi-chunk file → discard chunks < min_chars that remain after merging
         (if chunk < min_chars remains, it means it couldn't be merged)

    Args:
        nodes: List of nodes from CodeSplitter or SentenceSplitter.
        max_tokens: Maximum tokens per merged chunk (from config.child_chunk_size).
        min_chars: Minimum characters threshold (100 by default).

    Returns:
        List of merged nodes where chunks are merged as much as possible.
        Small chunks (< min_chars) that couldn't be merged are discarded in multi-chunk files.
        Empty list if single-chunk file < min_chars (skip entire file).
    """
    if not nodes:
        return []

    tokenizer = get_tokenizer()  # type: ignore[no-untyped-call]
    merged_nodes: list[BaseNode] = []
    current_batch: list[BaseNode] = []
    current_text = ""
    current_tokens = 0

    # Phase 1: Merge chunks up to max_tokens
    for node in nodes:
        node_text = node.get_content()
        node_tokens = len(tokenizer(node_text))  # type: ignore[no-untyped-call]

        # Case 1: Node is already too big → keep as is, don't merge
        if node_tokens > max_tokens:
            # Flush current batch first
            if current_batch:
                merged_nodes.append(_create_merged_node(current_batch, current_text))
                current_batch = []
                current_text = ""
                current_tokens = 0

            merged_nodes.append(node)
            continue

        # Case 2: Adding node would exceed max_tokens → flush batch
        if current_tokens + node_tokens > max_tokens and current_batch:
            merged_nodes.append(_create_merged_node(current_batch, current_text))
            current_batch = [node]
            current_text = node_text
            current_tokens = node_tokens
            continue

        # Case 3: Accumulate node into current batch
        current_batch.append(node)
        if current_text:
            current_text += "\n"
        current_text += node_text
        current_tokens += node_tokens

    # Flush remaining batch
    if current_batch:
        merged_nodes.append(_create_merged_node(current_batch, current_text))

    # Phase 2: Filter small chunks based on file context
    if len(merged_nodes) == 1:
        # Single chunk for entire file
        chunk_size = len(merged_nodes[0].get_content())
        if chunk_size < min_chars:
            # File too small → skip entire file
            return []
        # File has single chunk >= min_chars → keep it
        return merged_nodes
    else:
        # Multiple chunks → filter out those < min_chars that remain after merging
        # If a chunk < min_chars remains here, it means it COULDN'T be merged in Phase 1
        # (merging would have exceeded max_tokens), so we discard it
        filtered = [n for n in merged_nodes if len(n.get_content()) >= min_chars]
        return filtered


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

    Code files use CodeSplitter (tree-sitter) with token-based chunking and
    smart merging to eliminate micro-chunks. Everything else falls back to
    SentenceSplitter.
    """
    language: str | None = doc.metadata.get("language")

    if language and language != "text":
        try:
            splitter: CodeSplitter | SentenceSplitter = CodeSplitter(
                language=language,
                chunk_lines=999999,  # Don't limit by lines, use tokens instead
                chunk_lines_overlap=config.chunk_lines_overlap,
                max_chars=config.max_chars,  # Fallback for char mode
                count_mode="token",  # Use token-based chunking
                max_tokens=config.child_chunk_size,  # Read from config (default 1024)
            )
            nodes = splitter.get_nodes_from_documents([doc])
            if nodes:
                # Apply smart merging to combine small chunks
                nodes = _merge_small_chunks(
                    nodes,
                    max_tokens=config.child_chunk_size,
                    min_chars=100,
                )
                return nodes
        except Exception:  # noqa: BLE001
            pass  # Fall through to sentence splitter

    # SentenceSplitter already produces chunks of size child_chunk_size (1024 tokens)
    # No merging needed - it handles size internally
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

    If parent_chunk_size is None, returns only leaf_nodes (flat structure).
    """
    # Flat structure: no hierarchy
    if config.parent_chunk_size is None:
        return leaf_nodes

    parent_splitter = SentenceSplitter(chunk_size=config.parent_chunk_size)
    parent_nodes = parent_splitter.get_nodes_from_documents([doc])

    if not parent_nodes or not leaf_nodes:
        return leaf_nodes

    # Assign parent_node reference to each leaf node.
    # We distribute leaves evenly across parents by text position.
    leaves_per_parent = max(1, len(leaf_nodes) // len(parent_nodes))
    for idx, leaf in enumerate(leaf_nodes):
        parent_idx = min(idx // leaves_per_parent, len(parent_nodes) - 1)
        leaf.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
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
    dry_run: bool = False,
    embedding_model: Any | None = None,
) -> IndexStats:
    """Index *repo_path* for semantic search.

    Creates or updates the vector index at ``{repo_path}/.vector_index/``.
    Indexing is incremental by default — only changed files are processed.

    Args:
        repo_path:       Absolute path to the repository root.
        config:          Indexing configuration. Uses defaults if ``None``.
        reset:           Delete existing index and reindex from scratch.
        dry_run:         Only show which files would be indexed, without actually indexing.
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
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]  # noqa: PLC0415

        embedding_model = HuggingFaceEmbedding(model_name=config.embedding_model)

    Settings.embed_model = embedding_model
    Settings.llm = None  # type: ignore[assignment]  # LlamaIndex accepts None at runtime

    # --- Storage setup ---
    # Load hashes before deciding LanceDB mode.
    # - Fresh index (no hashes yet) or reset → "overwrite" creates the table.
    # - Subsequent incremental run → "append" keeps existing vectors.
    # We batch all inserts into a single operation to avoid creating
    # thousands of version manifests (one per insert).
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
    # Collect files to add/update/delete, then process in batches to minimize
    # LanceDB version manifests (one batch operation = one version).
    current_hashes: dict[str, str] = {}
    stats = IndexStats()

    files_to_add: list[Path] = []
    files_to_update: list[Path] = []
    paths_to_delete: list[str] = []

    # Scan and categorize files
    for file_path in scan_files(repo_path):
        path_str = str(file_path)
        file_hash = compute_file_hash(file_path)
        current_hashes[path_str] = file_hash
        existing_hash = existing_hashes.get(path_str)

        if existing_hash is None:
            files_to_add.append(file_path)
            stats.added += 1
        elif existing_hash != file_hash:
            files_to_update.append(file_path)
            stats.updated += 1
        else:
            stats.skipped += 1

    # Handle deleted files
    deleted_paths = set(existing_hashes) - set(current_hashes)
    paths_to_delete.extend(deleted_paths)
    stats.deleted = len(paths_to_delete)

    # --- Dry run: show what would be indexed without actually indexing ---
    if dry_run:
        print("Dry run: showing files that would be processed")
        print(f"\nFiles to add ({stats.added}):")
        for path in files_to_add:
            print(f"  + {path.relative_to(repo_path)}")
        print(f"\nFiles to update ({stats.updated}):")
        for path in files_to_update:
            print(f"  ~ {path.relative_to(repo_path)}")
        print(f"\nFiles to delete ({stats.deleted}):")
        for path_str in paths_to_delete:
            rel_path = Path(path_str).relative_to(repo_path) if Path(path_str).is_absolute() else path_str
            print(f"  - {rel_path}")
        print(f"\nFiles to skip (unchanged): {stats.skipped}")
        print(f"\nTotal files to process: {stats.added + stats.updated + stats.deleted}")
        return stats

    # --- Batch operations ---
    # 1. Remove deleted and modified files (single batch)
    for path_str in paths_to_delete:
        _remove_document(path_str, index)
    for file_path in files_to_update:
        _remove_document(str(file_path), index)

    # 2. Process all new/updated documents
    all_nodes_to_store: list[BaseNode] = []
    all_leaf_nodes_to_index: list[BaseNode] = []

    for file_path in files_to_add + files_to_update:
        doc = _load_document(file_path, repo_path)
        all_nodes, leaf_nodes = _process_document(doc, config)
        all_nodes_to_store.extend(all_nodes)
        all_leaf_nodes_to_index.extend(leaf_nodes)

    # 3. Batch embed all leaf nodes
    if all_leaf_nodes_to_index:
        texts = [n.get_content() for n in all_leaf_nodes_to_index]
        embeddings = embedding_model.get_text_embedding_batch(texts, show_progress=False)
        for node, emb in zip(all_leaf_nodes_to_index, embeddings):
            node.embedding = emb

    # 4. Store all nodes in docstore (for hierarchical merging)
    if all_nodes_to_store:
        storage_context.docstore.add_documents(all_nodes_to_store)

    # 5. Index all leaf nodes in vector store (SINGLE INSERT = SINGLE VERSION)
    if all_leaf_nodes_to_index:
        index.insert_nodes(all_leaf_nodes_to_index)
        stats.total_nodes = len(all_leaf_nodes_to_index)

    # 6. Cleanup old LanceDB versions to prevent disk space explosion
    # CRITICAL: DO NOT DELETE THIS BLOCK!
    # LanceDB creates a new version manifest (~1-2 MB) for every insert operation.
    # Without cleanup, versions accumulate indefinitely, wasting gigabytes of disk space.
    # This cleanup runs after each indexing operation and removes all old versions,
    # keeping only the latest. This preserves atomicity without keeping history.
    # If cleanup fails, the user MUST be notified immediately.
    from datetime import timedelta  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    try:
        lance_table = vector_store._lance_table  # type: ignore[attr-defined]  # noqa: SLF001
        if lance_table is not None:
            # Delete all versions older than 0 seconds (i.e., everything except current)
            lance_table.cleanup_old_versions(older_than=timedelta(seconds=0))  # type: ignore[attr-defined]
    except AttributeError:
        # LanceDBVectorStore may not expose _lance_table in all versions
        # This is not critical for index correctness, but warn about potential disk usage
        warnings.warn(
            "Could not access LanceDB internal table for version cleanup. "
            "Old version manifests may accumulate over time in "
            f"{repo_index.index_dir / 'lancedb' / 'vectors.lance' / '_versions'}. "
            "Consider periodic manual cleanup or upgrading llama-index-vector-stores-lancedb.",
            stacklevel=2,
        )
    except Exception as e:
        # Any other error is a real problem that must be reported
        raise RuntimeError(
            f"Failed to cleanup old LanceDB versions at {repo_index.lancedb_uri}. "
            f"Error: {e}. This may cause excessive disk space usage over time. "
            "Check LanceDB compatibility and file permissions."
        ) from e

    # Persist state
    _save_hashes(repo_index.hashes_path, current_hashes)
    docstore.persist(persist_path=docstore_persist)

    return stats
