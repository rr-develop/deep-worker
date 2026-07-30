"""Persistent, incremental per-message embedding store.

Computing one embedding per history message (see
:func:`semantic_search.long_text_embedding.embed_history_messages`) is a full
embedder pass over every message and takes minutes for a large history. This
module persists the result next to the existing semantic-search data
(``summaries_dir/.message_embeddings/``) so it is computed once and then only
*extended* as the agent's history grows — append-only, by request.

Layout (``summaries_dir/.message_embeddings/``, mode 0700)
----------------------------------------------------------
* ``embeddings.npy``     — ``float32 (N, dim)``, one L2-normalised row per
  non-empty message, in file-sorted then line order (stable, append-only).
* ``token_lengths.npy``  — ``int64 (N,)``, embedder-tokenizer token count per
  message (full length even when the message spanned multiple 512-token
  windows).
* ``manifest.json``      — schema version, embedding model name, vector dim,
  total ``N``, and per-file consumed-entry counts used to detect new messages.

Incrementality
--------------
For each ``*.jsonl`` under *summaries_dir* we remember how many parsed entries
were already embedded. On update we re-parse the file, take only the entries
beyond that watermark, embed their non-empty messages, and append the new rows.
History files are append-only, so this is correct and cheap. If a file shrank
or the embedding model / dim changed, the store is rebuilt from scratch (a
loud, explicit fallback — never a silent mismatch).

Heavy deps (``numpy``, ``torch``, the embedder) load lazily inside the build
path; :func:`load_message_embeddings` needs only ``numpy``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import IndexConfig
from .long_text_embedding import (
    count_jsonl_lines,
    embed_message_texts,
    entry_message_type,
    is_embeddable_message,
    message_texts_from_file,
    read_jsonl_entries,
    resolve_sentence_transformer,
)

STORE_DIR_NAME = ".message_embeddings"
_SCHEMA_VERSION = 1
_EMB_FILE = "embeddings.npy"
_LEN_FILE = "token_lengths.npy"
_INDEX_FILE = "message_indices.npy"
_FLAGS_FILE = "flags.npy"
_MANIFEST_FILE = "manifest.json"


# ---------------------------------------------------------------------------
# Per-message flags (bitmask)
# ---------------------------------------------------------------------------


class MessageFlag:
    """Bit flags describing each embedded message (stored as int64 bitmask).

    A single message may set several bits at once (e.g. ``USER_ROLE`` together
    with ``TASK_MANAGER``, or ``AGENT`` together with ``TOOL_CALL`` in a future
    build that embeds tool calls). Combine/read with bitwise ``&`` / ``|``.
    """

    USER_ROLE = 1 << 0
    """Formally a ``human``-role message — but NOT a tool result."""

    TASK_MANAGER = 1 << 1
    """A human-role message with a ``TaskManager:>`` prefix."""

    PLAIN_USER = 1 << 2
    """A human-role message with NO known prefix (genuine end-user text)."""

    AGENT = 1 << 3
    """An ``ai``-role message (final or intermediate, incl. with tool calls)."""

    AGENT_TO_AGENT = 1 << 4
    """A message carrying an inter-agent ``<sender> -> <receiver> :>`` prefix."""

    TOOL_RESULT = 1 << 5
    """A ``tool``-role message (the result returned by a tool)."""

    TOOL_CALL = 1 << 6
    """The tool CALL itself — NOT the agent's accompanying comment text.

    Reserved / future: this bit marks an embedding *of a tool call*. The store
    does not (yet) build separate embeddings for tool calls — only for the
    agent's accompanying comment text when present — so on current data this bit
    is **always 0**. It is deliberately NOT set as an annotation on an ``AGENT``
    comment that happens to carry a tool call, because that embedding is of the
    comment, not of the call. The bit is reserved (stable position) for when
    standalone tool-call embeddings are introduced. See
    DESIGN_MESSAGE_EMBEDDING_STORE.md.
    """

    SYSTEM = 1 << 7
    """A ``system``-role message, or a ``System:>``-prefixed user message."""

    SUMMARIZATION = 1 << 8
    """A conversation-summarization ("condensed summary") message.

    Detected by content: an ``ai``-role message whose body starts with the
    summary template heading ``## SESSION INTENT`` (the deepagents summarization
    middleware emits the summary body under this heading). The structural marker
    ``additional_kwargs.lc_source == "summarization"`` is NOT persisted into the
    offloaded JSONL, and the wrapper text "A condensed summary follows" only
    appears inside unrelated tool dumps that quoted history, so the heading is
    the reliable signal on stored data. See DESIGN_MESSAGE_EMBEDDING_STORE.md.
    """

    AGENT_FINAL = 1 << 9
    """A *final* agent message: an ``ai`` message whose next message is a
    user/TaskManager turn (i.e. the agent handed control back).

    Determined with one-message look-ahead: set on an ``ai`` entry when the
    immediately following entry (parsed-entry order, across the sorted file
    sequence) is ``human``-role. ``ai`` messages followed by a ``tool`` result
    (mid tool-loop) or by ``system`` are NOT final. Requires the ``next_entry``
    argument to :func:`classify_message`; without it the bit is never set.
    """


#: Canonical MessageFlag names, in stable bit order. Single source of truth for
#: tallying (:func:`build_message_metadata`), decoding (:func:`flag_names`) and
#: name→bit resolution (:func:`flag_from_name`), so the three never drift.
FLAG_NAMES: tuple[str, ...] = (
    "USER_ROLE", "TASK_MANAGER", "PLAIN_USER", "AGENT", "AGENT_TO_AGENT",
    "TOOL_RESULT", "TOOL_CALL", "SYSTEM", "SUMMARIZATION", "AGENT_FINAL",
)


def flag_from_name(name: str) -> int:
    """Resolve a case-insensitive :class:`MessageFlag` name to its bit value.

    Raises ``KeyError`` for an unknown name (caller maps it to a usage error).
    """
    key = name.strip().upper()
    if key not in FLAG_NAMES:
        raise KeyError(name)
    return int(getattr(MessageFlag, key))


def flag_names(flags: int) -> list[str]:
    """Decode a :class:`MessageFlag` bitmask to its set-flag names (bit order)."""
    return [n for n in FLAG_NAMES if flags & getattr(MessageFlag, n)]


# Inter-agent prefix:  "<name> -> <name> :>"   (names: word chars / hyphen)
_AGENT_TO_AGENT_RE = re.compile(r"^\s*[\w\-]+\s*->\s*[\w\-]+\s*:>")
_TASK_MANAGER_RE = re.compile(r"^\s*TaskManager\s*:>")
_SYSTEM_PREFIX_RE = re.compile(r"^\s*System\s*:>")
# Summarization "condensed summary" body heading (deepagents middleware template).
_SUMMARY_HEADING = "## SESSION INTENT"


def _raw_content_text(entry: dict[str, Any]) -> str:
    """Best-effort raw text of an entry's content for prefix sniffing.

    Unlike ``_message_text`` this does NOT add a role prefix — we need the
    original leading characters to detect ``TaskManager:>`` etc.
    """
    from typing import cast  # noqa: PLC0415

    msg: Any = entry.get("message", {})
    data: Any = cast("dict[str, Any]", msg).get("data", {}) if isinstance(msg, dict) else {}
    if not isinstance(data, dict):
        return ""
    raw: Any = cast("dict[str, Any]", data).get("content", "")
    if isinstance(raw, list):
        for b in cast("list[Any]", raw):
            if isinstance(b, dict):
                bd = cast("dict[str, Any]", b)
                if bd.get("type") == "text":
                    t = str(bd.get("text", ""))
                    if t.strip():
                        return t
        return ""
    return raw if isinstance(raw, str) else str(raw)


def classify_message(
    entry: dict[str, Any], next_entry: dict[str, Any] | None = None
) -> int:
    """Return the :class:`MessageFlag` bitmask for one history entry.

    Pure function of the entry's role, prefix, content heading and tool-call
    presence — does not touch embeddings. *next_entry*, when supplied, is the
    immediately following history entry (parsed-entry order) and is used ONLY to
    decide :attr:`MessageFlag.AGENT_FINAL` (an ``ai`` message whose next message
    is a ``human`` turn). See DESIGN_MESSAGE_EMBEDDING_STORE.md for the taxonomy.
    """
    role = entry_message_type(entry)
    text = _raw_content_text(entry)
    flags = 0

    if role == "system":
        flags |= MessageFlag.SYSTEM
    elif role == "tool":
        flags |= MessageFlag.TOOL_RESULT
    elif role == "ai":
        flags |= MessageFlag.AGENT
        # NOTE: MessageFlag.TOOL_CALL is intentionally NOT set here. It means
        # "this embedding IS of a tool call itself" — and we do not (yet) build
        # separate embeddings for tool calls, only for the agent's accompanying
        # comment text. Annotating an AGENT comment with TOOL_CALL would mislabel
        # a comment embedding as a tool-call embedding. The bit stays reserved
        # (always 0) until standalone tool-call embeddings exist. See
        # DESIGN_MESSAGE_EMBEDDING_STORE.md.
        if text.lstrip().startswith(_SUMMARY_HEADING):
            flags |= MessageFlag.SUMMARIZATION
        # Final agent message: the next turn is a user/TaskManager (human) one.
        if next_entry is not None and entry_message_type(next_entry) == "human":
            flags |= MessageFlag.AGENT_FINAL
    elif role == "human":
        flags |= MessageFlag.USER_ROLE
        if _SYSTEM_PREFIX_RE.match(text):
            flags |= MessageFlag.SYSTEM
        elif _TASK_MANAGER_RE.match(text):
            flags |= MessageFlag.TASK_MANAGER
        elif _AGENT_TO_AGENT_RE.match(text):
            flags |= MessageFlag.AGENT_TO_AGENT
        else:
            flags |= MessageFlag.PLAIN_USER

    # An inter-agent prefix can also appear on non-human transports; annotate it
    # wherever the prefix is present and it wasn't already set above.
    if not (flags & MessageFlag.AGENT_TO_AGENT) and _AGENT_TO_AGENT_RE.match(text):
        flags |= MessageFlag.AGENT_TO_AGENT

    return flags


@dataclass
class MessageEmbeddingStats:
    """Result of :func:`build_message_embeddings`."""

    total_messages: int = 0
    """Total messages in the store after the operation (== N rows)."""

    new_messages: int = 0
    """Messages embedded by this call (0 == already up to date)."""

    files_scanned: int = 0
    rebuilt: bool = False
    """True when the store was rebuilt from scratch instead of extended."""

    def __repr__(self) -> str:
        return (
            f"MessageEmbeddingStats(total={self.total_messages}, "
            f"new={self.new_messages}, files={self.files_scanned}, "
            f"rebuilt={self.rebuilt})"
        )


@dataclass
class MessageMetadataStats:
    """Result of :func:`build_message_metadata`."""

    total_messages: int = 0
    flag_counts: dict[str, int] = field(default_factory=lambda: {})

    def __repr__(self) -> str:
        return (
            f"MessageMetadataStats(total={self.total_messages}, "
            f"flags={self.flag_counts})"
        )


# ---------------------------------------------------------------------------
# Paths / manifest
# ---------------------------------------------------------------------------


def store_dir(summaries_dir: str | Path) -> Path:
    return Path(summaries_dir) / STORE_DIR_NAME


def message_embeddings_exist(summaries_dir: str | Path) -> bool:
    d = store_dir(summaries_dir)
    return (
        (d / _EMB_FILE).exists()
        and (d / _LEN_FILE).exists()
        and (d / _MANIFEST_FILE).exists()
    )


def _load_manifest(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / _MANIFEST_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def store_file_entry_counts(summaries_dir: str | Path) -> dict[str, int]:
    """Public accessor for the store's per-file parsed-entry watermark.

    Returns ``manifest["file_entry_counts"]`` — a mapping of each source
    ``*.jsonl`` absolute path to the number of *parsed* JSON entries (the same
    unit :func:`count_jsonl_lines` counts, i.e. blank/undecodable lines already
    dropped) that were consumed when the store was last built. Rows are
    append-only, so this watermark identifies exactly which prefix of each file
    produced the frozen store rows; entries beyond it were appended *after* the
    build and are not represented in the store.

    Used by the read-only query layer (:mod:`semantic_search.message_search`) to
    re-derive row source data over the SAME prefix the store was built from, so
    later-appended history (summarisation etc.) does not desynchronise the
    positional mapping. Returns ``{}`` if the manifest is absent/unreadable or
    predates watermark tracking. Exposed underscore-free to avoid a
    ``reportPrivateUsage`` cross-module import (see DESIGN_MESSAGE_EMBEDDING_STORE.md §5).
    """
    from typing import cast  # noqa: PLC0415

    raw: Any = _load_manifest(store_dir(summaries_dir)).get("file_entry_counts", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in cast("dict[Any, Any]", raw).items():
        try:
            out[str(k)] = int(v)  # pyright: ignore[reportAny]
        except (TypeError, ValueError):
            continue
    return out


def _chmod_600(path: Path) -> None:
    """Restrict a store file to owner read/write (policy: no 0666/0777)."""
    import os  # noqa: PLC0415

    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _save_manifest(d: Path, manifest: dict[str, Any]) -> None:
    p = d / _MANIFEST_FILE
    p.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _chmod_600(p)


# ---------------------------------------------------------------------------
# Load (light — numpy only, no model)
# ---------------------------------------------------------------------------


def load_message_embeddings(
    summaries_dir: str | Path, as_torch: bool = True
) -> tuple[Any, Any]:
    """Load the persisted ``(embeddings, token_lengths)`` without recomputing.

    Fast: reads two ``.npy`` files (no embedder loaded). With ``as_torch=True``
    (default) returns ``torch`` tensors; with ``as_torch=False`` returns the
    underlying ``numpy`` arrays (no torch import).

    Raises ``FileNotFoundError`` if the store does not exist — call
    :func:`build_message_embeddings` first.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    d = store_dir(summaries_dir)
    if not message_embeddings_exist(summaries_dir):
        raise FileNotFoundError(
            f"No message-embedding store at {d}. "
            "Run build_message_embeddings() first."
        )
    embs = np.load(d / _EMB_FILE)
    lens = np.load(d / _LEN_FILE)
    if not as_torch:
        return embs, lens
    import torch as _torch  # noqa: PLC0415

    torch: Any = _torch
    return torch.from_numpy(embs), torch.from_numpy(lens)


def message_metadata_exist(summaries_dir: str | Path) -> bool:
    """True if the per-message metadata tensors (indices + flags) are present."""
    d = store_dir(summaries_dir)
    return (d / _INDEX_FILE).exists() and (d / _FLAGS_FILE).exists()


def load_message_metadata(
    summaries_dir: str | Path, as_torch: bool = True
) -> tuple[Any, Any]:
    """Load ``(message_indices, flags)`` without recomputing embeddings.

    * ``message_indices`` — ``int64 (N,)``, the global 0-based ordinal of the
      source message (file-sorted, then parsed-entry order) for each embedding
      row. This is the canonical, unique, append-only-stable message id — note
      the JSONL ``message_index`` field is NOT unique, so the ordinal is used
      instead (see DESIGN_MESSAGE_EMBEDDING_STORE.md).
    * ``flags`` — ``int64 (N,)`` :class:`MessageFlag` bitmask per row.

    Raises ``FileNotFoundError`` if metadata has not been built — call
    :func:`build_message_metadata` first.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    d = store_dir(summaries_dir)
    if not message_metadata_exist(summaries_dir):
        raise FileNotFoundError(
            f"No message metadata at {d}. Run build_message_metadata() first."
        )
    idx = np.load(d / _INDEX_FILE)
    flags = np.load(d / _FLAGS_FILE)
    if not as_torch:
        return idx, flags
    import torch as _torch  # noqa: PLC0415

    torch: Any = _torch
    return torch.from_numpy(idx), torch.from_numpy(flags)


# ---------------------------------------------------------------------------
# Metadata build (indices + flags) — NO embedding recomputation
# ---------------------------------------------------------------------------


def build_message_metadata(
    summaries_dir: str | Path,
    config: IndexConfig | None = None,
) -> MessageMetadataStats:
    """Build ``message_indices.npy`` + ``flags.npy`` for an existing store.

    Does a deterministic re-pass over the same ``*.jsonl`` files in the same
    order, selecting the same non-empty messages the embedding build selected
    (via :func:`message_entries_from_file`). For each selected message it
    records its global ordinal and its :class:`MessageFlag` bitmask. **No
    embeddings are recomputed.**

    The produced arrays have exactly one row per embedding row, in the same
    order, so ``message_indices[i]`` / ``flags[i]`` describe ``embeddings[i]``.
    Verifies the row count matches the embedding store and raises if it does
    not (guards against a stale/mismatched store).
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np

    if config is None:
        config = IndexConfig()

    summaries_dir = Path(summaries_dir)
    d = store_dir(summaries_dir)
    if not message_embeddings_exist(summaries_dir):
        raise FileNotFoundError(
            f"No message-embedding store at {d}. "
            "Run build_message_embeddings() first."
        )

    files = sorted(summaries_dir.glob("*.jsonl"))
    # Flatten the RAW parsed stream across all files (sorted) so AGENT_FINAL can
    # look at the true next message — including non-embeddable ones (e.g. a
    # tool-call-only ai is followed by a tool result, so it is NOT final). The
    # embedding selection (is_embeddable_message) then picks which rows exist,
    # exactly mirroring the embedding build's order and selection.
    raw: list[dict[str, Any]] = []
    for fp in files:
        raw.extend(read_jsonl_entries(fp))

    indices: list[int] = []
    flags: list[int] = []
    ordinal = 0
    for i, entry in enumerate(raw):
        if not is_embeddable_message(entry):
            continue
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        indices.append(ordinal)
        flags.append(classify_message(entry, next_entry=nxt))
        ordinal += 1

    # Cross-check against the actual embedding rows.
    emb_rows = int(np.load(d / _EMB_FILE, mmap_mode="r").shape[0])
    if len(indices) != emb_rows:
        raise RuntimeError(
            "message metadata row count "
            f"({len(indices)}) != embedding rows ({emb_rows}); the history "
            "files changed since the embeddings were built. Re-run "
            "build_message_embeddings() first, then build_message_metadata()."
        )

    idx_arr = np.asarray(indices, dtype=np.int64)
    flags_arr = np.asarray(flags, dtype=np.int64)
    _atomic_save_npy(d / _INDEX_FILE, idx_arr)
    _atomic_save_npy(d / _FLAGS_FILE, flags_arr)

    # Record in the manifest (extend, don't clobber the embedding fields).
    manifest = _load_manifest(d)
    manifest["metadata_schema_version"] = _SCHEMA_VERSION
    manifest["has_message_indices"] = True
    manifest["has_flags"] = True
    _save_manifest(d, manifest)

    stats = MessageMetadataStats(total_messages=len(indices))
    # Per-flag tallies for the caller's summary (canonical FLAG_NAMES order).
    for name in FLAG_NAMES:
        bit = getattr(MessageFlag, name)
        stats.flag_counts[name] = int((flags_arr & bit != 0).sum())
    return stats


# ---------------------------------------------------------------------------
# Build / incremental update
# ---------------------------------------------------------------------------


def build_message_embeddings(
    summaries_dir: str | Path,
    reset: bool = False,
    embed_model: Any | None = None,
    config: IndexConfig | None = None,
    batch_size: int = 64,
) -> MessageEmbeddingStats:
    """Create or incrementally extend the per-message embedding store.

    On first run (or ``reset=True``, or a model/dim/schema mismatch) the whole
    history is embedded. Afterwards only messages appended since the last run
    are embedded and their rows appended. Returns :class:`MessageEmbeddingStats`.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np

    if config is None:
        config = IndexConfig()

    summaries_dir = Path(summaries_dir)
    d = store_dir(summaries_dir)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)

    files = sorted(summaries_dir.glob("*.jsonl"))
    stats = MessageEmbeddingStats(files_scanned=len(files))

    manifest = {} if reset else _load_manifest(d)
    prev_counts: dict[str, int] = manifest.get("file_entry_counts", {}) if manifest else {}

    # Decide whether we can extend or must rebuild.
    have_store = message_embeddings_exist(summaries_dir) and not reset
    model_name = config.embedding_model
    if have_store:
        if (
            manifest.get("schema_version") != _SCHEMA_VERSION
            or manifest.get("embedding_model") != model_name
        ):
            have_store = False  # incompatible -> rebuild
    # Detect shrunk/rewritten files -> rebuild for safety.
    if have_store:
        for fp in files:
            if count_jsonl_lines(fp) < prev_counts.get(str(fp), 0):
                have_store = False
                break

    if not have_store:
        prev_counts = {}
        stats.rebuilt = True

    # Gather only the NEW message texts per file (append-only delta).
    new_texts: list[str] = []
    new_counts: dict[str, int] = {}
    for fp in files:
        total_entries = count_jsonl_lines(fp)
        start = prev_counts.get(str(fp), 0)
        if total_entries > start:
            new_texts.extend(message_texts_from_file(fp, start_line=start))
        new_counts[str(fp)] = total_entries

    # Nothing new and store already valid -> no-op.
    if have_store and not new_texts:
        existing = np.load(d / _EMB_FILE)
        stats.total_messages = int(existing.shape[0])
        stats.new_messages = 0
        return stats

    # Embed the delta (or everything, on rebuild).
    st, _tokenize_fn, eff_max = resolve_sentence_transformer(embed_model, config)
    dim = int(st.get_sentence_embedding_dimension())
    new_embs, new_lens = embed_message_texts(
        new_texts, st, eff_max, batch_size=batch_size
    )

    if have_store:
        old_embs = np.load(d / _EMB_FILE)
        old_lens = np.load(d / _LEN_FILE)
        embs = np.concatenate([old_embs, new_embs], axis=0) if new_embs.shape[0] else old_embs
        lens = np.concatenate([old_lens, new_lens], axis=0) if new_lens.shape[0] else old_lens
    else:
        embs, lens = new_embs, new_lens

    # Persist atomically-ish (write temp then replace).
    _atomic_save_npy(d / _EMB_FILE, embs)
    _atomic_save_npy(d / _LEN_FILE, lens)
    # The embedding rows changed, so any previously-built metadata (indices /
    # flags) is now stale. Drop it so callers must rebuild it explicitly via
    # build_message_metadata() against the new rows.
    _invalidate_metadata(d)
    _save_manifest(
        d,
        {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": model_name,
            "dim": dim,
            "total_messages": int(embs.shape[0]),
            "file_entry_counts": new_counts,
        },
    )

    stats.total_messages = int(embs.shape[0])
    stats.new_messages = int(new_embs.shape[0])
    return stats


def _invalidate_metadata(d: Path) -> None:
    """Remove stale metadata tensors after the embedding rows change."""
    for name in (_INDEX_FILE, _FLAGS_FILE):
        try:
            (d / name).unlink()
        except FileNotFoundError:
            pass


def _atomic_save_npy(path: Path, arr: Any) -> None:
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    # ``np.save`` appends ``.npy`` unless the name already ends in ``.npy``;
    # use a temp name that already ends in ``.npy`` so the path is exact, then
    # atomically replace the target.
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    _chmod_600(tmp)
    tmp.replace(path)
    _chmod_600(path)
