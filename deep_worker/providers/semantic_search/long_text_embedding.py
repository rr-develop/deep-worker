"""Length-robust embeddings for arbitrary-length text and per-message history.

Two capabilities, both built on the SAME embedder the rest of
:mod:`semantic_search` uses (``HuggingFaceEmbedding(intfloat/multilingual-e5-
small)``, an L2-normalising ``SentenceTransformer`` with a hard 512-token
window):

1. :func:`embed_long_text` — embed a single text of *any* length. Text longer
   than the embedder's effective token ceiling (510 for e5: 512 minus the two
   ``<s>``/``</s>`` special tokens) is split into token-accurate windows, each
   window is embedded, and the windows are combined as a **token-count-weighted
   mean, then L2-normalised**::

       emb = normalize(w1 * emb1 + w2 * emb2 + ...)     # wi = tokens in chunk i

   So no tail is silently truncated — every token contributes to the result.

2. :func:`embed_history_messages` — one embedding per message in an agent's
   summarised history (a strict 1-to-1 correspondence, unlike the search index
   which merges/splits messages into retrieval chunks). Message text is
   extracted the same way the history indexer extracts it (role prefix +
   multi-modal text join). Returns two tensors: all embeddings ``(N, dim)`` and
   all per-message token lengths ``(N,)``.

Heavy dependencies (``numpy``, ``torch``, the embedder/SentenceTransformer) are
imported lazily *inside* the functions, so importing this module costs nothing
until a function is actually called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ._tokenizer import effective_max_tokens, embed_model_tokenizer
from .config import IndexConfig

# Heavy numerical deps (``numpy``, ``torch``) are imported lazily *inside* the
# public functions, so importing this module is cheap. Their concrete types are
# therefore not available to the strict type-checker; return values that are
# numpy/torch objects are annotated ``Any`` on purpose.

# Role prefixes — kept byte-identical to history_indexer._chunk_messages so the
# text we embed here matches what the search index was built from.
_ROLE_PREFIX = {"human": "User: ", "ai": "Assistant: ", "tool": "Tool: "}


# ---------------------------------------------------------------------------
# Embedder access
# ---------------------------------------------------------------------------


def resolve_sentence_transformer(
    embed_model: Any | None, config: IndexConfig
) -> tuple[Any, Any, int]:
    """Return ``(sentence_transformer, tokenize_fn, effective_max)``.

    Reuses the project's embedder (and its own tokenizer via
    :func:`embed_model_tokenizer`) so the vector space and the 510-token
    ceiling match the search index exactly.
    """
    # Route through the shared process-wide cache so this fifth call site
    # cannot silently rebuild the 449 MB embedder on every invocation.
    # See a related internal incident analysis (not included here).
    from ._embed_cache import resolve_embed_model  # noqa: PLC0415

    embed_model = resolve_embed_model(config, embed_model)

    tokenize_fn, model_max_length, reserve = embed_model_tokenizer(embed_model)
    eff_max = effective_max_tokens(config, model_max_length, reserve)

    st = getattr(embed_model, "_model", None)
    if st is None or not hasattr(st, "encode"):
        raise RuntimeError(
            "embed_model does not expose a SentenceTransformer with .encode(); "
            "see DESIGN_FIX_TOKENIZER_MISMATCH.md §4.1"
        )
    return st, tokenize_fn, eff_max


# ---------------------------------------------------------------------------
# Token-window splitting
# ---------------------------------------------------------------------------


def _token_windows(
    text: str, tokenizer: Any, eff_max: int
) -> list[tuple[str, int]]:
    """Split *text* into ``(window_text, token_count)`` pieces ≤ *eff_max*.

    Splits on raw token ids (no special tokens) and decodes each window back to
    text, so the windowing is token-accurate rather than character-heuristic.
    Returns a single piece for short text (the common case).
    """
    enc: Any = tokenizer(text, add_special_tokens=False)
    if isinstance(enc, dict):
        ids_any: Any = cast("dict[str, Any]", enc).get("input_ids", [])
    else:
        ids_any = getattr(enc, "input_ids", [])
    ids: list[int] = list(cast("list[int]", ids_any))
    if ids and isinstance(ids[0], list):
        ids = list(cast("list[int]", ids[0]))

    if len(ids) <= eff_max:
        return [(text, len(ids))]

    windows: list[tuple[str, int]] = []
    for start in range(0, len(ids), eff_max):
        window_ids = ids[start : start + eff_max]
        window_text = tokenizer.decode(window_ids)
        windows.append((window_text, len(window_ids)))
    return windows


# ---------------------------------------------------------------------------
# Public API 1 — arbitrary-length single text
# ---------------------------------------------------------------------------


def embed_long_text(
    text: str,
    embed_model: Any | None = None,
    config: IndexConfig | None = None,
) -> Any:
    """Embed *text* of any length as a token-weighted, normalised mean.

    Short text (≤ effective ceiling) is embedded directly. Longer text is split
    into token-accurate windows; each window is embedded and the windows are
    combined as ``normalize(Σ tokensᵢ · embᵢ)``.

    Returns a 1-D ``float32`` ``np.ndarray`` of length ``dim`` (384 for e5),
    L2-normalised. Empty/whitespace text returns a zero vector.

    Return type is annotated ``Any`` because ``numpy`` is a lazy dependency;
    the concrete runtime type is ``numpy.ndarray`` of shape ``(dim,)``.
    """
    import numpy as np  # noqa: PLC0415

    if config is None:
        config = IndexConfig()
    st, _tokenize_fn, eff_max = resolve_sentence_transformer(embed_model, config)
    tokenizer = st.tokenizer

    if not text or not text.strip():
        dim = int(st.get_sentence_embedding_dimension())
        return np.zeros(dim, dtype=np.float32)

    windows = _token_windows(text, tokenizer, eff_max)
    texts = [w for w, _ in windows]
    weights = np.asarray([n for _, n in windows], dtype=np.float32)

    embs = st.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if embs.shape[0] == 1:
        return embs[0]

    pooled = (weights[:, None] * embs).sum(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0.0:
        pooled = pooled / norm
    return pooled.astype(np.float32)


# ---------------------------------------------------------------------------
# History message extraction (mirrors history_indexer._chunk_messages)
# ---------------------------------------------------------------------------


def _message_text(entry: dict[str, Any]) -> str:
    """Extract a single message's embeddable text, role-prefixed.

    Mirrors the per-entry extraction in
    :func:`semantic_search.history_indexer._chunk_messages` so the text matches
    what the search index was built from. Returns ``""`` for empty content.
    """
    msg: Any = entry.get("message", {})
    msg_type = entry.get("message_type") or "unknown"
    data: Any = cast("dict[str, Any]", msg).get("data", {}) if isinstance(msg, dict) else {}
    content = ""
    if isinstance(data, dict):
        raw: Any = cast("dict[str, Any]", data).get("content", "")
        if isinstance(raw, list):
            parts: list[str] = []
            for c in cast("list[Any]", raw):
                if isinstance(c, dict):
                    cd = cast("dict[str, Any]", c)
                    if cd.get("type") == "text":
                        parts.append(str(cd.get("text", "")))
            content = "\n".join(parts)
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw)

    if content and content.strip():
        prefix = _ROLE_PREFIX.get(msg_type, f"{msg_type}: ")
        return prefix + content.strip()
    return ""


def message_text(entry: dict[str, Any]) -> str:
    """Public accessor for one entry's embeddable, role-prefixed text.

    Thin wrapper over the module-internal :func:`_message_text` so other modules
    in the package can extract a single message's text without a
    ``reportPrivateUsage`` (pyright strict) cross-module private import — same
    convention as the other underscore-free public helpers (see
    DESIGN_MESSAGE_EMBEDDING_STORE.md §5). Returns ``""`` for empty content.
    """
    return _message_text(entry)


def message_texts_from_file(path: Path, start_line: int = 0) -> list[str]:
    """Extract embeddable, role-prefixed texts for a JSONL file's messages.

    Reads ``path`` and returns the non-empty message texts for entries at line
    index ``>= start_line`` (0-based, append-only friendly), preserving order.
    Empty-content messages are skipped (they contribute no embedding), so the
    returned list maps 1-to-1 onto the *non-empty* new messages — the same
    notion of "message" used by :func:`embed_history_messages`.
    """
    entries = _read_jsonl(path)
    out: list[str] = []
    for entry in entries[start_line:]:
        t = _message_text(entry)
        if t:
            out.append(t)
    return out


def entry_message_type(entry: dict[str, Any]) -> str:
    """The role of a JSONL history entry (``human``/``ai``/``tool``/``system``/…)."""
    return str(entry.get("message_type") or "unknown")


def entry_has_tool_call(entry: dict[str, Any]) -> bool:
    """True if an ``ai`` entry carries a tool call.

    Tool calls appear either as ``additional_kwargs.tool_calls`` /
    top-level ``tool_calls`` (LangChain), or as ``tool_use`` blocks inside a
    list-shaped ``content`` (Anthropic block format). Both are detected.
    """
    msg: Any = entry.get("message", {})
    data: Any = cast("dict[str, Any]", msg).get("data", {}) if isinstance(msg, dict) else {}
    if not isinstance(data, dict):
        return False
    d = cast("dict[str, Any]", data)
    ak: Any = d.get("additional_kwargs", {}) or {}
    if isinstance(ak, dict) and cast("dict[str, Any]", ak).get("tool_calls"):
        return True
    if d.get("tool_calls"):
        return True
    content: Any = d.get("content", "")
    if isinstance(content, list):
        for b in cast("list[Any]", content):
            if isinstance(b, dict) and cast("dict[str, Any]", b).get("type") == "tool_use":
                return True
    return False


def message_entries_from_file(
    path: Path, start_line: int = 0
) -> list[dict[str, Any]]:
    """Return the raw JSONL entries that produce an embedding, in order.

    Mirrors :func:`message_texts_from_file` exactly (same non-empty
    ``_message_text`` selection and order) but returns the full parsed entries
    instead of just their text, so callers (e.g. the metadata/flag pass of the
    message-embedding store) can classify each embedded message by role,
    prefix, tool-calls, etc. **without** re-embedding. The i-th returned entry
    corresponds to the i-th text from :func:`message_texts_from_file`.
    """
    entries = _read_jsonl(path)
    out: list[dict[str, Any]] = []
    for entry in entries[start_line:]:
        if _message_text(entry):
            out.append(entry)
    return out


def read_jsonl_entries(path: Path) -> list[dict[str, Any]]:
    """All parseable JSONL entries of *path*, in order (empty list if unreadable).

    Public accessor over the raw parsed stream, used by callers that need raw
    adjacency (e.g. look-ahead to the next message) rather than the
    embedding-selected subset returned by :func:`message_entries_from_file`.
    """
    return _read_jsonl(path)


def is_embeddable_message(entry: dict[str, Any]) -> bool:
    """True if *entry* yields an embedding (non-empty role-prefixed text).

    This is the exact selection predicate used by the embedding build and by
    :func:`message_entries_from_file` / :func:`message_texts_from_file`.
    """
    return bool(_message_text(entry))


def count_jsonl_lines(path: Path) -> int:
    """Number of parseable JSON lines in *path* (0 if unreadable)."""
    return len(_read_jsonl(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Public API 2 — per-message embeddings + token lengths
# ---------------------------------------------------------------------------


def embed_history_messages(
    summaries_dir: str | Path,
    embed_model: Any | None = None,
    config: IndexConfig | None = None,
    batch_size: int = 64,
) -> tuple[Any, Any]:
    """Embed every message in an agent's history, 1-to-1.

    Walks all ``*.jsonl`` files under *summaries_dir* (sorted by name for
    determinism), extracts each message's text exactly like the history
    indexer, and produces one embedding per non-empty message using
    :func:`embed_long_text` semantics (token-weighted mean for messages over the
    512-token window).

    Returns ``(embeddings, token_lengths)`` as ``torch`` tensors:

    * ``embeddings`` — ``float32`` ``(N, dim)``, each row L2-normalised.
    * ``token_lengths`` — ``int64`` ``(N,)``, the embedder-tokenizer token count
      of each message (special tokens excluded, full length even when the
      message spanned multiple windows).

    ``N`` is the number of non-empty messages, in file-then-line order. All
    chunk windows across all messages are embedded in a single batched pass for
    efficiency.
    """
    import torch as _torch  # noqa: PLC0415

    torch: Any = _torch

    if config is None:
        config = IndexConfig()
    st, _tokenize_fn, eff_max = resolve_sentence_transformer(embed_model, config)

    summaries_dir = Path(summaries_dir)
    texts: list[str] = []
    for fp in sorted(summaries_dir.glob("*.jsonl")):
        for entry in _read_jsonl(fp):
            t = _message_text(entry)
            if t:
                texts.append(t)

    pooled_np, lens_np = embed_message_texts(
        texts, st, eff_max, batch_size=batch_size
    )
    return torch.from_numpy(pooled_np), torch.from_numpy(lens_np)


def embed_message_texts(
    texts: list[str],
    sentence_transformer: Any,
    eff_max: int,
    batch_size: int = 64,
) -> tuple[Any, Any]:
    """Embed a list of already-extracted message texts, 1-to-1.

    Core routine shared by :func:`embed_history_messages` and the incremental
    message-embedding store. Each text is split into token-accurate windows
    (≤ *eff_max*), all windows across all texts are embedded in a single batched
    ``SentenceTransformer.encode`` pass, and per-text vectors are formed as
    ``normalize(Σ tokensᵢ · embᵢ)`` via a vectorized scatter-add.

    Returns ``(embeddings, token_lengths)`` as **numpy** arrays:
    ``float32 (N, dim)`` (L2-normalised rows) and ``int64 (N,)``. Returns empty
    arrays when *texts* is empty.
    """
    import numpy as _np  # noqa: PLC0415
    import torch as _torch  # noqa: PLC0415

    np: Any = _np
    torch: Any = _torch

    tokenizer = sentence_transformer.tokenizer
    dim = int(sentence_transformer.get_sentence_embedding_dimension())

    window_texts: list[str] = []
    owner: list[int] = []
    win_weight: list[int] = []
    msg_token_len: list[int] = []

    for msg_idx, text in enumerate(texts):
        windows = _token_windows(text, tokenizer, eff_max)
        msg_token_len.append(sum(n for _, n in windows))
        for wtext, wn in windows:
            window_texts.append(wtext)
            owner.append(msg_idx)
            win_weight.append(wn)

    n_msgs = len(texts)
    if n_msgs == 0:
        return (
            np.zeros((0, dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    win_embs_np = sentence_transformer.encode(
        window_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    win_embs = torch.from_numpy(win_embs_np)  # (W, dim)
    owner_t = torch.tensor(owner, dtype=torch.int64)  # (W,)
    weight_t = torch.tensor(win_weight, dtype=torch.float32).unsqueeze(1)  # (W,1)

    pooled = torch.zeros((n_msgs, dim), dtype=torch.float32)
    pooled.index_add_(0, owner_t, win_embs * weight_t)
    pooled = torch.nn.functional.normalize(pooled, p=2.0, dim=1)

    return pooled.numpy(), np.asarray(msg_token_len, dtype=np.int64)
