"""Process-wide cache for embedding models used by all semantic-search paths.

Why this module exists
----------------------

Before this file, three call sites each held (or were missing) their own
``HuggingFaceEmbedding`` cache:

* ``retriever._search`` had a module-level ``_embedding_cache`` and used it.
* ``history_retriever._ensure_embed_model`` had the same cache pattern.
* ``history_indexer._resolve_embed_and_tokenizer`` had **no cache at all** —
  every summarization built a fresh 449 MB model, retaining ~163 MB per
  build. A related internal incident analysis (not included here) identified
  this defect and recommended auditing the sibling paths, and importing
  ``_ensure_embed_model`` from ``history_retriever`` (or hoisting it into a
  shared module) so the two paths cannot drift apart again. This module is
  that shared resolver.

Contract
--------

``resolve_embed_model(config, embed_model=None)`` returns an embedding model
instance suitable for indexing OR retrieval. When ``embed_model`` is
supplied (typically in tests, e.g. ``MockEmbedding``) it is honoured
verbatim and NEVER stored in the cache: caller-supplied instances may have
arbitrary lifetimes and would poison the cache for the rest of the process.

When ``embed_model`` is ``None`` the resolver returns the cached instance
for ``config.embedding_model`` if one exists, otherwise constructs a new
``HuggingFaceEmbedding`` with ``local_files_only=True`` (strictly offline,
per ``docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md``) and stores it.
Each construction emits ONE ``INFO`` log line naming the model — that line
is a diagnostic so future incidents do not have to be reconstructed by
counting tqdm progress bars in stderr.

The cache is process-wide. Sharing one instance across calls is safe: the
model is loaded read-only from a local snapshot; the two retrievers have
been sharing an instance this way for months without incident.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import IndexConfig

_logger = logging.getLogger(__name__)

# Keyed on ``config.embedding_model`` (e.g. ``"intfloat/multilingual-e5-small"``).
# One entry per distinct model name for the life of the process.
_embedding_cache: dict[str, Any] = {}


def resolve_embed_model(
    config: IndexConfig, embed_model: Any | None = None
) -> Any:
    """Return a shared embedding model, constructing at most one per model name.

    Args:
        config: Index configuration. Only ``config.embedding_model`` is
            consulted here; other fields are the caller's concern.
        embed_model: Explicit override (tests). When not ``None`` it is
            returned as-is and NOT stored in the cache.

    Returns:
        An embedding model instance ready to hand to LlamaIndex.
    """
    if embed_model is not None:
        return embed_model

    cached = _embedding_cache.get(config.embedding_model)
    if cached is not None:
        return cached

    # Lazy import — ``llama_index.embeddings.huggingface`` pulls in
    # ``transformers`` / ``torch`` (hundreds of MB of shared libraries), so
    # we only pay for it on the first real construction.
    from llama_index.embeddings.huggingface import (  # type: ignore[import-untyped]  # noqa: PLC0415
        HuggingFaceEmbedding,
    )

    # ``local_files_only=True`` is required for the offline guarantee. Two
    # of the three former call sites already set it (retriever,
    # history_retriever); ``history_indexer`` also set it but was missing
    # the cache. Every call site now goes through here, so the invariant
    # is uniform.
    _logger.info(
        "loading embedding model %s (local_files_only=True) — first construction "
        "in this process; subsequent calls will hit the cache",
        config.embedding_model,
    )
    model = HuggingFaceEmbedding(
        model_name=config.embedding_model,
        model_kwargs={"local_files_only": True},
    )
    _embedding_cache[config.embedding_model] = model
    return model


def _reset_cache_for_tests() -> None:
    """Empty the module-level cache. Test-only.

    Production code MUST NOT call this: dropping the reference lets a
    subsequent ``resolve_embed_model`` construct a fresh 449 MB model,
    which is exactly the bug this module exists to prevent. Tests that
    need a clean slate (e.g. counting construction calls) call this in a
    fixture.
    """
    _embedding_cache.clear()
