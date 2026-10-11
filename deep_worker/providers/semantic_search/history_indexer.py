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
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore  # type: ignore[import-untyped]

from ._tokenizer import (
    TokenizeFn,
    effective_max_tokens,
    embed_model_tokenizer,
)
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

# In JSON Lines, records are separated ONLY by the newline character ``\n``.
# We deliberately do NOT use ``str.splitlines()`` anywhere in this module:
# ``splitlines()`` also breaks on ``\r``, ``\v``, ``\f``, ``\x1c``-``\x1e``,
# ``U+0085``, ``U+2028`` (LINE SEPARATOR) and ``U+2029`` (PARAGRAPH SEPARATOR).
# Those characters legitimately appear *inside* JSON string values, so
# splitting on them shatters a single valid record into unparseable fragments
# and undercounts records. Splitting on ``\n`` keeps each JSONL record intact
# and makes the record count match a plain physical-line count — the single
# source of truth shared with the startup catch-up check.
JSONL_RECORD_SEPARATOR = "\n"


def _iter_jsonl_lines(text: str) -> list[str]:
    """Split raw JSONL text into records on ``\\n`` only (see note above)."""
    return text.split(JSONL_RECORD_SEPARATOR)


def count_jsonl_lines(file_path: Path) -> int:
    """Return the number of JSONL records in *file_path*.

    Single source of truth for "how many records does this file have" used by
    BOTH the incremental indexer and the startup catch-up cheap-check. Counts
    non-empty physical lines separated by ``\\n`` — no JSON parsing, so it is
    cheap even for large files. A record whose content embeds ``U+2028`` etc.
    still counts as exactly one record (unlike ``str.splitlines()``).

    Returns 0 for a missing or unreadable file.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for line in _iter_jsonl_lines(text) if line.strip())


def _read_jsonl_lines(file_path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file, returning parsed JSON objects.

    Records are separated by ``\\n`` only (NOT ``str.splitlines()`` — see the
    module note on ``JSONL_RECORD_SEPARATOR``), so a value containing
    ``U+2028``/``U+2029``/``\\r`` does not shatter its record. Skips empty
    lines and lines that fail to parse.
    """
    results: list[dict[str, Any]] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return results

    for line in _iter_jsonl_lines(text):
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
    effective_max: int,
    tokenize_fn: TokenizeFn,
    *,
    step: int = 1,
    on_group_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Merge adjacent message groups into chunks up to *effective_max*.

    Returns list of ``(combined_text, metadata)`` tuples.
    Metadata includes ``line_range`` (1-indexed), ``timestamp``,
    ``message_types``, and ``thread_id``.

    *effective_max* and *tokenize_fn* MUST come from
    :func:`semantic_search._tokenizer.embed_model_tokenizer` /
    :func:`effective_max_tokens` so the chunker measures in the same units as
    the embedder. See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.

    Phase 1 "Analyzing history…" progress: pass ``on_group_progress`` to be
    called ``(done_groups, total_groups)`` as the (slow) chunking pass walks
    the groups — this keeps the bar from sitting silent on large histories.
    It fires ``(0, total)`` up front, every ``step`` groups, and a guaranteed
    ``(total, total)`` at the end. Without the callback the chunking is
    identical to before (backward compatible). See
    ``docs/DESIGN_HISTORY_INDEX_ANALYZE_PHASE_PROGRESS.md``.
    """
    tokenizer = tokenize_fn
    max_tokens = effective_max
    chunks: list[tuple[str, dict[str, Any]]] = []
    current_text_parts: list[str] = []
    current_metadata: dict[str, Any] | None = None
    current_tokens = 0
    # Track which original line numbers (1-indexed) this chunk spans
    line_offset = 0  # How many lines we've consumed so far (0-indexed)

    total_groups = len(groups)
    tick_step = max(1, step)
    _last_tick = -1  # last ``done`` value actually reported (dedup guard)

    def _tick(done: int) -> None:
        # Fire on the step boundary and always on the final group, so the bar
        # reaches (total, total) regardless of ``step`` or ``continue`` paths.
        # Never emit the same ``done`` twice (avoids a duplicate final tick).
        nonlocal _last_tick
        if on_group_progress is None:
            return
        if done == _last_tick:
            return
        if done % tick_step == 0 or done == total_groups:
            _last_tick = done
            on_group_progress(done, total_groups)

    _tick(0)

    for group_index, group in enumerate(groups, start=1):
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
            _tick(group_index)
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
            _tick(group_index)
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
        _tick(group_index)

    # Flush remaining
    if current_text_parts:
        combined = "\n".join(current_text_parts)
        chunks.append((combined, current_metadata))  # type: ignore[arg-type]

    # Guaranteed final tick so the "Analyzing…" bar always reaches (total,
    # total), even if the last group hit a step boundary that didn't align or
    # the groups list was empty. ``_tick`` dedups, so no double emit.
    _tick(total_groups)

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


def _resolve_embed_and_tokenizer(
    embed_model: Any | None, config: IndexConfig
) -> tuple[Any, Any, int, int]:
    """Resolve the embedding model + tokenizer and wire them into Settings.

    The model itself is resolved via the process-wide shared cache in
    ``._embed_cache`` — see a related internal incident analysis (not included
    here). Prior to that change this function built a
    fresh 449 MB ``HuggingFaceEmbedding`` on every summarization; the
    shared resolver enforces the single-load invariant across every
    semantic-search call site (retriever, history_retriever, this file,
    and ``indexer``).

    Returns ``(embed_model, tokenize_fn, model_max_length,
    special_tokens_reserve)``.
    """
    from ._embed_cache import resolve_embed_model  # noqa: PLC0415

    embed_model = resolve_embed_model(config, embed_model)

    Settings.embed_model = embed_model  # type: ignore[assignment]
    Settings.llm = None  # type: ignore[assignment]
    tokenize_fn, model_max_length, special_tokens_reserve = (
        embed_model_tokenizer(embed_model)
    )
    Settings.tokenizer = tokenize_fn  # type: ignore[assignment]
    return embed_model, tokenize_fn, model_max_length, special_tokens_reserve


def _pending_chunks_for_file(
    jsonl_path: Path,
    index_dir: Path,
    config: IndexConfig,
    tokenize_fn: Any,
    model_max_length: int,
    special_tokens_reserve: int,
    on_group_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    """Compute the chunks a file will contribute WITHOUT embedding them.

    Returns ``(chunks, total_entries, lines_read)``.

    ``on_group_progress`` (optional) is forwarded to :func:`_chunk_messages`
    for the Phase 1 "Analyzing history…" bar — called ``(done, total)`` by
    message group as the chunking pass runs.
    """
    line_counts = _load_line_counts(_line_hashes_path(index_dir))
    last_line_count = line_counts.get(str(jsonl_path), 0)

    if not jsonl_path.exists():
        return [], 0, 0

    current_line_count = count_jsonl_lines(jsonl_path)

    lines_read = current_line_count - last_line_count
    if lines_read <= 0:
        # Nothing new (or the file shrank): equality is a no-op — do NOT
        # re-index the whole file. ``count_jsonl_lines`` is the single source
        # of truth shared with the startup catch-up check, so at equality
        # ``lines_read == 0`` and we stop here.
        return [], 0, lines_read

    all_entries = _read_jsonl_lines(jsonl_path)
    total_entries = len(all_entries)

    # ``count_jsonl_lines`` and ``len(_read_jsonl_lines())`` agree for a file
    # whose records are all valid JSON, so ``last_line_count`` is a correct
    # index into ``all_entries``. Use ``>=`` so the incremental tail is taken
    # whenever there is genuinely new content; fall back to a full pass only
    # when the recorded count no longer fits the parsed records (e.g. the file
    # shrank, was rewritten, or has unparseable lines skewing the parse).
    if last_line_count > 0 and total_entries >= last_line_count:
        entries = all_entries[last_line_count:]
    else:
        # last_line_count == 0, or parsed records fewer than recorded — handle
        # gracefully by reconsidering the whole (already-bounded) file.
        entries = all_entries

    if not entries:
        return [], total_entries, lines_read

    effective_max = effective_max_tokens(
        config, model_max_length, special_tokens_reserve
    )
    groups = _group_by_summarized_event(entries)
    chunks = _chunk_messages(
        groups,
        effective_max=effective_max,
        tokenize_fn=tokenize_fn,
        step=max(1, config.analyze_progress_step),
        on_group_progress=on_group_progress,
    )
    return chunks, total_entries, lines_read


def index_history_file(
    jsonl_path: str | Path,
    summaries_dir: str | Path,
    config: IndexConfig | None = None,
    embed_model: Any | None = None,
    on_message_progress: Callable[[int, int], None] | None = None,
    on_analyze_progress: Callable[[int, int], None] | None = None,
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
        on_message_progress: Phase 2 callback ``(done, total)`` by message —
            the embedding/write bar.
        on_analyze_progress: Phase 1 callback ``(done, total)`` by message
            group — the "Analyzing history…" (chunking) bar, which runs BEFORE
            any ``on_message_progress`` tick.

    Returns:
        :class:`HistoryIndexStats` with counts of processed lines/chunks.
    """
    jsonl_path = Path(jsonl_path)
    summaries_dir = Path(summaries_dir)
    index_dir = get_history_index_dir(summaries_dir)

    if config is None:
        config = IndexConfig()

    stats = HistoryIndexStats()

    # Resolve embedding model + tokenizer (also wires Settings). e5 KNOWN
    # LIMITATION: indexed text is embedded WITHOUT the ``"passage: "`` prefix
    # (must match the query side in ``history_retriever``). See
    # docs/DESIGN_FIX_TOKENIZER_MISMATCH.md.
    embed_model, tokenize_fn, model_max_length, special_tokens_reserve = (
        _resolve_embed_and_tokenizer(embed_model, config)
    )

    line_counts = _load_line_counts(_line_hashes_path(index_dir))

    # Compute pending chunks (shared with the pre-count path). Per-chunk
    # ceiling is min(config.child_chunk_size, model_max_length -
    # special_tokens), measured by the embedder's own tokenizer.
    chunks, total_entries, lines_read = _pending_chunks_for_file(
        jsonl_path, index_dir, config,
        tokenize_fn, model_max_length, special_tokens_reserve,
        on_group_progress=on_analyze_progress,
    )
    stats.lines_read = lines_read
    if lines_read <= 0:
        return stats  # No new lines

    if not chunks:
        # Still update line count to avoid re-processing
        if total_entries:
            line_counts[str(jsonl_path)] = total_entries
            _save_line_counts(_line_hashes_path(index_dir), line_counts)
        return stats

    # Build documents for indexing
    # The chunks are ALREADY the final units (we chunked them ourselves in
    # _chunk_messages), so build TextNodes directly — no extra SentenceSplitter
    # pass, no per-document transformation. One node == one chunk.
    nodes: list[TextNode] = []
    for chunk_text, meta in chunks:
        node = TextNode(
            text=chunk_text,
            id_=f"{jsonl_path}:L{meta.get('line_range', '0-0')}",
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
        nodes.append(node)
        stats.chunks_indexed += 1

    # Load or create vector store and index
    vector_store, docstore = _load_or_create_storage(index_dir)
    index = _build_index(vector_store, docstore, embed_model)

    # ── Progress is measured in MESSAGES, not chunks ──────────────────────
    # The denominator (number of new messages/lines) is known up front, so
    # the bar can show X/N + %% + ETA immediately — unlike chunk counts,
    # which are only known after the (slow) chunking pass. We map "how far
    # we've embedded" back to messages via each chunk's line_range end.
    #
    # These two axes are INDEPENDENT:
    #   * embedding is done in progress-sized slices so the bar advances
    #     smoothly and ETA is meaningful;
    #   * the LanceDB WRITE stays batched at config.index_batch_size (one
    #     transaction per batch — the fast path), decoupled from progress.
    total_messages = lines_read
    embed_step = max(1, config.embed_progress_step)
    write_batch_size = max(1, config.index_batch_size)

    def _end_line(node: TextNode) -> int:
        rng = str(node.metadata.get("line_range", "0-0"))
        try:
            return int(rng.split("-")[-1])
        except (ValueError, IndexError):
            return 0

    if on_message_progress is not None:
        on_message_progress(0, total_messages)

    # Phase 1 — embed all nodes in progress-sized slices, ticking progress in
    # messages after each slice. Nodes carry their embeddings forward so the
    # write phase does NOT re-embed (insert_nodes skips nodes that already
    # have .embedding set).
    assert embed_model is not None  # resolved above
    for start in range(0, len(nodes), embed_step):
        sl = nodes[start:start + embed_step]
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in sl]
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
        for n, emb in zip(sl, embeddings):
            n.embedding = emb
        if on_message_progress is not None:
            covered = _end_line(sl[-1])
            done_msgs = min(total_messages, max(0, covered))
            on_message_progress(done_msgs, total_messages)

    # Phase 2 — write to LanceDB in batches of write_batch_size (one
    # transaction each). Embeddings are already set, so this is pure I/O.
    for start in range(0, len(nodes), write_batch_size):
        index.insert_nodes(nodes[start:start + write_batch_size])

    if on_message_progress is not None:
        on_message_progress(total_messages, total_messages)

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
    on_message_progress: Callable[[int, int], None] | None = None,
    on_analyze_progress: Callable[[int, int], None] | None = None,
) -> HistoryIndexStats:
    """Index all ``.jsonl`` files in *summaries_dir*.

    Incremental: only files with new lines since last index are processed.

    Args:
        summaries_dir: Path to the summaries directory.
        config: Index configuration. Uses defaults if ``None``.
        embed_model: Override the embedding model (for testing).
        on_progress: Optional callback invoked with the filename (stem)
            before each file is indexed.
        on_message_progress: Optional callback invoked as ``(done, total)``
            in MESSAGES (new jsonl lines), aggregated across ALL files.
            ``total`` is known up front (a cheap line count — no chunking),
            so the bar can show X/N + %% + ETA from the very first tick.
        on_analyze_progress: Optional Phase 1 callback invoked as
            ``(done, total)`` in message GROUPS for the CURRENT file — the
            "Analyzing history…" (chunking) bar. Per-file (not aggregated),
            since chunking is done file-by-file and precedes that file's
            embedding phase.

    Returns:
        Aggregated :class:`HistoryIndexStats`.
    """
    summaries_dir = Path(summaries_dir)
    total = HistoryIndexStats()

    if not summaries_dir.exists():
        return total

    jsonl_files = sorted(summaries_dir.glob("*.jsonl"))

    # Cheap up-front denominator: how many NEW messages (lines) each file
    # contributes. Just a line count vs the previously-indexed count — no
    # chunking, no embedding — so it is effectively instant even for huge
    # files. Only computed when a message-progress consumer is attached.
    index_dir = get_history_index_dir(summaries_dir)
    grand_total_messages = 0
    per_file_new_lines: dict[Path, int] = {}
    if on_message_progress is not None:
        line_counts = _load_line_counts(_line_hashes_path(index_dir))
        for jsonl_file in jsonl_files:
            current = count_jsonl_lines(jsonl_file)
            new_lines = max(0, current - line_counts.get(str(jsonl_file), 0))
            per_file_new_lines[jsonl_file] = new_lines
            grand_total_messages += new_lines

    def _make_file_msg_cb(
        base: int,
        sink: Callable[[int, int], None],
    ) -> Callable[[int, int], None]:
        """Bind a per-file (done, total) message callback to the global bar."""
        def _cb(done: int, _file_total: int) -> None:
            sink(base + done, grand_total_messages)
        return _cb

    if on_message_progress is not None:
        on_message_progress(0, grand_total_messages)

    done_so_far = 0

    for jsonl_file in jsonl_files:
        if on_progress is not None:
            on_progress(jsonl_file.name)

        file_msg_cb: Callable[[int, int], None] | None = (
            _make_file_msg_cb(done_so_far, on_message_progress)
            if on_message_progress is not None
            else None
        )

        try:
            file_stats = index_history_file(
                jsonl_file,
                summaries_dir,
                config=config,
                embed_model=embed_model,
                on_message_progress=file_msg_cb,
                on_analyze_progress=on_analyze_progress,
            )
            total.lines_read += file_stats.lines_read
            total.chunks_indexed += file_stats.chunks_indexed
            total.files_processed += file_stats.files_processed
            total.errors += file_stats.errors
            done_so_far += per_file_new_lines.get(jsonl_file, file_stats.lines_read)
        except Exception:  # noqa: BLE001
            total.errors += 1
            done_so_far += per_file_new_lines.get(jsonl_file, 0)

    if on_message_progress is not None:
        on_message_progress(grand_total_messages, grand_total_messages)

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
