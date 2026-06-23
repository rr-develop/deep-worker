"""Indexing pipeline for history (JSONL summarization) files.

Architecture
------------
Per-file splitting:
  - JSONL lines are grouped by ``summarized_at`` event.
  - Adjacent messages in the same event are merged up to ``child_chunk_size`` tokens.
  - Merged chunks become LanceDB Documents with metadata.

Incremental indexing:
  - A JSON file at ``.history_index/file_hashes.json`` tracks line counts.
  - On each call, only JSONL files with new lines since last index are processed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.utils import get_tokenizer  # type: ignore[import-untyped]
from llama_index.vector_stores.lancedb import LanceDBVectorStore  # type: ignore[import-untyped]

from .config import IndexConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_INDEX_DIR_NAME = ".history_index"
"""Subdirectory under summaries/ where the LanceDB index lives."""

MIN_CHARS_PER_CHUNK = 50
"""Minimum characters for a single chunk; smaller chunks are dropped."""

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class HistoryIndexStats:
    """Statistics returned by :func:`index_history_file`."""

    lines_read: int = 0
    chunks_indexed: int = 0
    files_processed: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_history_index_dir(summaries_dir: Path) -> Path:
    """Return the path to the history index directory under *summaries_dir*."""
    return summaries_dir / HISTORY_INDEX_DIR_NAME


def _line_hashes_path(index_dir: Path) -> Path:
    return index_dir / "line_hashes.json"


def lancedb_uri(index_dir: Path) -> str:
    return str(index_dir / "lancedb")


def docstore_path(index_dir: Path) -> Path:
    return index_dir / "docstore"


# ---------------------------------------------------------------------------
# Hash persistence (line-count tracking for incremental indexing)
# ---------------------------------------------------------------------------


def _load_line_counts(path: Path) -> dict[str, int]:
    """Load per-file line counts from JSON, or return empty dict."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_line_counts(path: Path, counts: dict[str, int]) -> None:
    """Persist per-file line counts to JSON."""
    path.write_text(json.dumps(counts, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSONL parsing and chunking
# ---------------------------------------------------------------------------


def _read_jsonl_lines(file_path: Path) -> list[dict[str, Any]]:
    """Read all lines from a JSONL file, returning parsed JSON objects.

    Skips empty lines and lines that fail to parse.
    """
    results: list[dict[str, Any]] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return results

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj: dict[str, Any] = json.loads(stripped)
            results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def _group_by_summarized_event(
    entries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group consecutive entries by ``summarized_at`` value.

    Returns list of groups, each group being a list of entries sharing
    the same ``summarized_at`` timestamp.
    """
    if not entries:
        return []
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = [entries[0]]
    for entry in entries[1:]:
        if entry.get("summarized_at") == current_group[0].get("summarized_at"):
            current_group.append(entry)
        else:
            groups.append(current_group)
            current_group = [entry]
    groups.append(current_group)
    return groups


def _chunk_messages(
    groups: list[list[dict[str, Any]]],
    max_tokens: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Merge adjacent message groups into chunks up to *max_tokens*.

    Returns list of ``(combined_text, metadata)`` tuples.
    Metadata includes ``line_range`` (1-indexed), ``timestamp``,
    ``message_types``, and ``thread_id``.
    """
    tokenizer = get_tokenizer()  # type: ignore[no-untyped-call]
    chunks: list[tuple[str, dict[str, Any]]] = []
    current_text_parts: list[str] = []
    current_metadata: dict[str, Any] | None = None
    current_tokens = 0
    # Track which original line numbers (1-indexed) this chunk spans
    line_offset = 0  # How many lines we've consumed so far (0-indexed)

    for group in groups:
        # Build text for this group
        group_text_parts: list[str] = []
        group_line_count = 0
        first_entry = group[0]
        thread_id = (
            first_entry.get("message", {})
            .get("metadata", {})
            .get("file_path", "")
        )
        timestamp = first_entry.get("summarized_at", "")
        message_types: list[str] = []

        for entry in group:
            msg = entry.get("message", {})
            msg_type = entry.get("message_type") or "unknown"
            if msg_type not in message_types:
                message_types.append(msg_type)

            # Extract the content from the message
            data: Any = msg.get("data", {})
            content: str = ""
            if isinstance(data, dict):
                _data: dict[str, Any] = cast("dict[str, Any]", data)
                # LangChain message format: { "content": "...", ... }
                raw_content: Any = _data.get("content", "")
                if isinstance(raw_content, list):
                    # Multi-modal content: extract text parts
                    text_parts: list[str] = [
                        str(c.get("text", "")) for c in raw_content if isinstance(c, dict) and c.get("type") == "text"  # type: ignore[union-attr]
                    ]
                    content = "\n".join(text_parts)
                elif isinstance(raw_content, str):
                    content = raw_content
                else:
                    content = str(raw_content)

            if content and content.strip():
                # Prepend role marker
                role_prefix = {"human": "User: ", "ai": "Assistant: ", "tool": "Tool: "}.get(
                    msg_type, f"{msg_type}: "
                )
                content = role_prefix + content.strip()
                group_text_parts.append(content)
                # Each JSONL line is one line in the file; estimate line count
                group_line_count += 1
            else:
                group_line_count += 1

        if not group_text_parts:
            line_offset += group_line_count
            continue

        group_text = "\n".join(group_text_parts)
        group_tokens = len(tokenizer(group_text))  # type: ignore[arg-type]

        # If this group alone exceeds max_tokens, split it with SentenceSplitter
        if group_tokens > max_tokens:
            # Flush current batch first
            if current_text_parts:
                combined = "\n".join(current_text_parts)
                chunks.append((combined, current_metadata))  # type: ignore[arg-type]
                current_text_parts = []
                current_tokens = 0
                current_metadata = None

            # Split the large group using SentenceSplitter
            node = TextNode(text=group_text)
            splitter = SentenceSplitter(chunk_size=max_tokens, chunk_overlap=20)
            sub_nodes = splitter.get_nodes_from_documents([_node_to_doc(node)])
            for i, sub_node in enumerate(sub_nodes):
                sub_text = sub_node.get_content()
                if len(sub_text) < MIN_CHARS_PER_CHUNK and len(sub_nodes) > 1:
                    continue
                start_line = line_offset + 1
                end_line = line_offset + group_line_count
                chunk_meta = {
                    "file_path": thread_id,
                    "thread_id": Path(thread_id).stem if thread_id else "unknown",
                    "line_range": f"{start_line}-{end_line}",
                    "total_lines": str(group_line_count),
                    "timestamp": timestamp,
                    "message_types": ",".join(message_types),
                    "chunk_index": str(i),
                }
                chunks.append((sub_text, chunk_meta))
            line_offset += group_line_count
            continue

        # Case: adding this group would exceed max_tokens -> flush
        if current_tokens + group_tokens > max_tokens and current_text_parts:
            combined = "\n".join(current_text_parts)
            # Calculate line range for the flushed batch
            # This is approximate since we track by group
            chunks.append((combined, current_metadata))  # type: ignore[arg-type]
            current_text_parts = []
            current_tokens = 0
            current_metadata = None

        # Accumulate
        if not current_text_parts:
            # First part: set metadata from this group's first entry
            start_line = line_offset + 1
            end_line = line_offset + group_line_count
            current_metadata = {
                "file_path": thread_id,
                "thread_id": Path(thread_id).stem if thread_id else "unknown",
                "line_range": f"{start_line}-{end_line}",
                "total_lines": str(group_line_count),
                "timestamp": timestamp,
                "message_types": ",".join(message_types),
            }
        elif current_metadata is not None:
            # Update end_line
            end_line = line_offset + group_line_count
            current_metadata["line_range"] = (
                current_metadata["line_range"].split("-")[0] + f"-{end_line}"
            )

        current_text_parts.append(group_text)
        current_tokens += group_tokens
        line_offset += group_line_count

    # Flush remaining
    if current_text_parts:
        combined = "\n".join(current_text_parts)
        chunks.append((combined, current_metadata))  # type: ignore[arg-type]

    return chunks


def _node_to_doc(node: TextNode) -> Document:
    """Convert a TextNode back to a Document for SentenceSplitter."""
    return Document(
        text=node.get_content(),
        metadata=node.metadata or {},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _load_or_create_storage(index_dir: Path) -> tuple[LanceDBVectorStore, SimpleDocumentStore]:
    """Load existing LanceDB + docstore or create fresh ones."""
    index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Try appending first (table exists), fall back to create
    vector_store = LanceDBVectorStore(
        uri=lancedb_uri(index_dir),
        mode="create",
    )
    docstore_dir = docstore_path(index_dir)
    persist_file = str(docstore_dir / "docstore.json")
    try:
        docstore = SimpleDocumentStore.from_persist_path(persist_file)
    except (FileNotFoundError, ValueError, KeyError):
        docstore = SimpleDocumentStore()

    return vector_store, docstore


def _build_index(
    vector_store: LanceDBVectorStore,
    docstore: SimpleDocumentStore,
    embed_model: Any,
) -> VectorStoreIndex:
    """Build or load a VectorStoreIndex from storage."""
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def index_history_file(
    jsonl_path: str | Path,
    summaries_dir: str | Path,
    config: IndexConfig | None = None,
    embed_model: Any | None = None,
) -> HistoryIndexStats:
    """Index a single JSONL history file.

    Incremental: if the file has been indexed before (tracked by line count),
    only new lines are processed.

    Args:
        jsonl_path: Path to the ``.jsonl`` history file.
        summaries_dir: Path to the summaries directory that contains
            ``.history_index/``.
        config: Index configuration. Uses defaults if ``None``.
        embed_model: Override the embedding model (for testing).

    Returns:
        :class:`HistoryIndexStats` with counts of processed lines/chunks.
    """
    jsonl_path = Path(jsonl_path)
    summaries_dir = Path(summaries_dir)
    index_dir = get_history_index_dir(summaries_dir)

    if config is None:
        config = IndexConfig()

    stats = HistoryIndexStats()

    # Resolve embedding model
    if embed_model is None:
        # Lazy import — only when actually indexing
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]  # noqa: PLC0415

        embed_model = HuggingFaceEmbedding(
            model_name=config.embedding_model,
            model_kwargs={"local_files_only": True},
        )

    Settings.embed_model = embed_model  # type: ignore[assignment]
    Settings.llm = None  # type: ignore[assignment]

    # Load existing line counts
    line_counts = _load_line_counts(_line_hashes_path(index_dir))
    last_line_count = line_counts.get(str(jsonl_path), 0)

    # Read the file
    if not jsonl_path.exists():
        return stats

    try:
        current_line_count = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
    except OSError:
        return stats

    stats.lines_read = current_line_count - last_line_count
    if stats.lines_read <= 0:
        return stats  # No new lines

    # Parse entries (read all - we need context for grouping)
    all_entries = _read_jsonl_lines(jsonl_path)
    total_entries = len(all_entries)

    # If we've indexed this file before, only process new entries
    if last_line_count > 0 and total_entries > last_line_count:
        entries = all_entries[last_line_count:]
    elif last_line_count == 0:
        entries = all_entries
    else:
        # File shrank - re-index from scratch wouldn't happen via append-only
        # but handle gracefully
        entries = all_entries

    if not entries:
        return stats

    # Group and chunk
    groups = _group_by_summarized_event(entries)
    chunks = _chunk_messages(groups, max_tokens=config.child_chunk_size)

    if not chunks:
        # Still update line count to avoid re-processing
        line_counts[str(jsonl_path)] = total_entries
        _save_line_counts(_line_hashes_path(index_dir), line_counts)
        return stats

    # Build documents for indexing
    documents: list[Document] = []
    for chunk_text, meta in chunks:
        doc = Document(
            text=chunk_text,
            doc_id=f"{jsonl_path}:L{meta.get('line_range', '0-0')}",
            metadata={
                "file_path": meta.get("file_path") or str(jsonl_path),
                "thread_id": meta.get("thread_id", "unknown"),
                "line_range": meta.get("line_range", "0-0"),
                "total_lines": meta.get("total_lines", "0"),
                "timestamp": meta.get("timestamp", ""),
                "message_types": meta.get("message_types", ""),
                "jsonl_file": str(jsonl_path),
            },
            excluded_embed_metadata_keys=[
                "file_path", "thread_id", "line_range",
                "total_lines", "timestamp", "jsonl_file",
            ],
            excluded_llm_metadata_keys=[
                "file_path", "thread_id", "total_lines", "jsonl_file",
            ],
        )
        documents.append(doc)
        stats.chunks_indexed += 1

    # Load or create vector store and index
    vector_store, docstore = _load_or_create_storage(index_dir)
    index = _build_index(vector_store, docstore, embed_model)

    # Insert new documents
    for doc in documents:
        index.insert(doc)

    # Persist docstore
    persist_path = str(docstore_path(index_dir) / "docstore.json")  # type: ignore[arg-type]
    docstore.persist(persist_path)

    # Update line counts
    line_counts[str(jsonl_path)] = total_entries
    _save_line_counts(_line_hashes_path(index_dir), line_counts)

    stats.files_processed = 1
    return stats


def index_all_history_files(
    summaries_dir: str | Path,
    config: IndexConfig | None = None,
    embed_model: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> HistoryIndexStats:
    """Index all ``.jsonl`` files in *summaries_dir*.

    Incremental: only files with new lines since last index are processed.

    Args:
        summaries_dir: Path to the summaries directory.
        config: Index configuration. Uses defaults if ``None``.
        embed_model: Override the embedding model (for testing).
        on_progress: Optional callback invoked with the filename (stem)
            before each file is indexed.

    Returns:
        Aggregated :class:`HistoryIndexStats`.
    """
    summaries_dir = Path(summaries_dir)
    total = HistoryIndexStats()

    if not summaries_dir.exists():
        return total

    for jsonl_file in sorted(summaries_dir.glob("*.jsonl")):
        if on_progress is not None:
            on_progress(jsonl_file.name)
        try:
            file_stats = index_history_file(
                jsonl_file,
                summaries_dir,
                config=config,
                embed_model=embed_model,
            )
            total.lines_read += file_stats.lines_read
            total.chunks_indexed += file_stats.chunks_indexed
            total.files_processed += file_stats.files_processed
            total.errors += file_stats.errors
        except Exception:  # noqa: BLE001
            total.errors += 1

    return total


def history_index_exists(summaries_dir: str | Path) -> bool:
    """Return ``True`` if a history index exists in *summaries_dir*."""
    index_dir = get_history_index_dir(Path(summaries_dir))
    return index_dir.exists() and (index_dir / "lancedb").exists()


def clear_history_index(summaries_dir: str | Path) -> None:
    """Delete the history index under *summaries_dir*."""
    import shutil

    index_dir = get_history_index_dir(Path(summaries_dir))
    if index_dir.exists():
        shutil.rmtree(index_dir)
