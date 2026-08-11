"""Embedder-aligned tokenizer for the chunkers.

The chunker MUST measure chunk size with the SAME tokenizer the embedder will
use, otherwise the embedder silently truncates the tail of every chunk that
ends up above its ``model_max_length`` (XLM-RoBERTa default for the e5 family
is 512). See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.

Public surface
--------------
* :func:`embed_model_tokenizer` — extracts ``(tokenize_fn, model_max_length,
  special_tokens_reserve)`` from a resolved ``HuggingFaceEmbedding`` instance.
* :func:`effective_max_tokens` — ``min(child_chunk_size,
  model_max_length - special_tokens_reserve)``.

Both are imported by :mod:`semantic_search.history_indexer` and
:mod:`semantic_search.indexer`. The module name is underscore-prefixed
(``_tokenizer``) to mark it as a package-internal helper; the functions
themselves are public *within the package* so the strict type-checker does
not flag cross-module imports as private usage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .config import IndexConfig

TokenizeFn = Callable[[str], list[int]]

_LOUD_ERROR = (
    "embed_model has no accessible tokenizer; "
    "see DESIGN_FIX_TOKENIZER_MISMATCH.md §4.1"
)


def embed_model_tokenizer(
    embed_model: Any,
) -> tuple[TokenizeFn, int, int]:
    """Return ``(tokenize_fn, model_max_length, special_tokens_reserve)``.

    The returned ``tokenize_fn(text)`` produces a ``list[int]`` of token ids
    (the *raw* sub-word ids without special tokens, so the count reflects what
    a *fragment* contributes; special-token overhead is reported separately
    via ``special_tokens_reserve`` so the caller can reserve room for it).

    Raises
    ------
    RuntimeError
        If the embedder does not expose a usable tokenizer through the same
        attribute path it uses internally (``embed_model._model.tokenizer``).
        A silent fallback to ``tiktoken`` is **forbidden** — it would
        resurrect the very bug this module exists to close (RC-1).
    """
    inner = getattr(embed_model, "_model", None)
    if inner is None:
        raise RuntimeError(_LOUD_ERROR)
    tokenizer = getattr(inner, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(_LOUD_ERROR)
    if not callable(tokenizer):
        raise RuntimeError(_LOUD_ERROR)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int):
        raise RuntimeError(_LOUD_ERROR)
    if not hasattr(tokenizer, "num_special_tokens_to_add"):
        raise RuntimeError(_LOUD_ERROR)
    special_tokens_reserve = int(
        tokenizer.num_special_tokens_to_add(pair=False)  # type: ignore[attr-defined]
    )

    def tokenize_fn(text: str) -> list[int]:
        out: Any = tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict):
            raw: Any = cast("dict[str, Any]", out).get("input_ids", [])
        else:
            raw = getattr(out, "input_ids", [])
        ids: list[Any] = list(cast("list[Any]", raw))
        # Be tolerant of nested-list shapes (batch=1) without flattening real
        # token-id lists.
        if ids and isinstance(ids[0], list):
            ids = list(cast("list[Any]", ids[0]))
        return [int(t) for t in ids]

    return tokenize_fn, model_max_length, special_tokens_reserve


def effective_max_tokens(
    config: IndexConfig,
    model_max_length: int,
    special_tokens_reserve: int,
) -> int:
    """Per-chunk hard ceiling, in the embedder's own token units."""
    return min(
        config.child_chunk_size,
        model_max_length - special_tokens_reserve,
    )
