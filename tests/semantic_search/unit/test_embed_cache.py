"""Regression tests for the shared embedding-model cache.

Pins the invariant that regressed in a related internal incident analysis
(not included here): the
write path (indexer, history_indexer, long_text_embedding) must resolve
the embedder through the SAME process-wide cache the read path
(retriever, history_retriever) has always used, so ``HuggingFaceEmbedding``
is constructed at most once per model name per process — regardless of how
many summarization / index / search cycles run.

The tests count ``HuggingFaceEmbedding.__init__`` calls by monkey-patching
the constructor at the exact import site used by
``_embed_cache.resolve_embed_model``. They are safe to run in the normal
unit-test tree: no real weights are loaded, no network is touched, no
LanceDB / llama_index index is built.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from deep_worker.providers.semantic_search import _embed_cache
from deep_worker.providers.semantic_search.config import IndexConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


from llama_index.core.embeddings.mock_embed_model import MockEmbedding


class _FakeTokenizer:
    """Minimum surface ``_tokenizer.embed_model_tokenizer`` reads back.

    ``embed_model_tokenizer`` checks four properties (RC-1 guard, see
    ``_tokenizer.py:37-67``): the tokenizer must be callable, must expose
    ``model_max_length: int``, and must have ``num_special_tokens_to_add``.
    We implement exactly those.
    """

    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # 1 id per char is enough — no test asserts on the ids themselves.
        return {"input_ids": [1] * len(text)}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:  # noqa: ARG002
        return 2  # matches XLM-R e5's [CLS] + [SEP] reserve


class _FakeInnerModel:
    """Stand-in for the ``SentenceTransformer`` behind ``HuggingFaceEmbedding``.

    ``_tokenizer.embed_model_tokenizer`` looks up ``embed_model._model.tokenizer``
    and rejects anything without a callable tokenizer at that path (RC-1
    guard — see ``DESIGN_FIX_TOKENIZER_MISMATCH.md`` §4.1). We supply
    exactly that.
    """

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


# Module-level constructor counter — Pydantic blocks class-var attributes on
# subclasses of ``BaseEmbedding``, so we keep the counter out of the class.
_INSTANCES: list["_FakeEmbedding"] = []


class _FakeEmbedding(MockEmbedding):
    """Cheap stand-in for ``HuggingFaceEmbedding``.

    Subclasses ``MockEmbedding`` so ``Settings.embed_model = <fake>`` passes
    llama_index's ``isinstance(BaseEmbedding)`` check inside the two
    indexer paths (``history_indexer._resolve_embed_and_tokenizer`` and
    ``indexer.index_repository`` both do that assignment right after
    resolution). The module-level :data:`_INSTANCES` list is what tests
    read back to prove the shared cache stopped multiple constructions.

    The private ``_model`` attribute is a plain object (not registered as a
    Pydantic field) whose ``tokenizer`` satisfies the RC-1 guard in
    :func:`embed_model_tokenizer`.
    """

    def __init__(self, *, model_name: str, model_kwargs: dict[str, Any]) -> None:
        super().__init__(embed_dim=1)
        # Pydantic v2 (which ``BaseEmbedding`` uses) blocks unknown attribute
        # sets by default, but underscore-prefixed private attributes are
        # allowed. ``embed_model._model.tokenizer`` is the exact lookup path
        # ``_tokenizer.embed_model_tokenizer`` walks.
        object.__setattr__(self, "_model", _FakeInnerModel())
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_kwargs", model_kwargs)
        _INSTANCES.append(self)


@pytest.fixture(autouse=True)
def _clean_cache_and_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate each test: reset the shared cache and the constructor counter,
    and redirect ``llama_index.embeddings.huggingface.HuggingFaceEmbedding``
    to :class:`_FakeEmbedding` so no real model is ever loaded.
    """
    _embed_cache._reset_cache_for_tests()
    _INSTANCES.clear()

    # Install a fake ``llama_index.embeddings.huggingface`` module so the
    # lazy ``from … import HuggingFaceEmbedding`` inside the resolver picks
    # up our stand-in without importing torch / transformers.
    fake_pkg = types.ModuleType("llama_index.embeddings.huggingface")
    fake_pkg.HuggingFaceEmbedding = _FakeEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "llama_index.embeddings.huggingface", fake_pkg
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSharedResolver:
    """The single-load invariant is the whole reason this module exists.

    See the related internal incident analysis (not included here).
    """

    def test_second_call_reuses_cached_instance(self) -> None:
        cfg = IndexConfig()
        first = _embed_cache.resolve_embed_model(cfg)
        second = _embed_cache.resolve_embed_model(cfg)
        assert first is second, (
            "resolve_embed_model must return the SAME instance for the same "
            "model name — otherwise the write path leaks ~163 MB per call "
            "(see the related internal incident analysis)."
        )
        assert len(_INSTANCES) == 1, (
            "HuggingFaceEmbedding must be constructed EXACTLY once, not "
            "once per call."
        )

    def test_ten_calls_construct_exactly_one_model(self) -> None:
        cfg = IndexConfig()
        for _ in range(10):
            _embed_cache.resolve_embed_model(cfg)
        assert len(_INSTANCES) == 1, (
            f"Expected 1 construction across 10 calls, got "
            f"{len(_INSTANCES)}. This is exactly the regression "
            f"that produced the 26 loads / 4.15 GiB spike in the incident."
        )

    def test_explicit_override_bypasses_cache_and_is_not_stored(self) -> None:
        cfg = IndexConfig()
        sentinel = object()
        got = _embed_cache.resolve_embed_model(cfg, embed_model=sentinel)
        assert got is sentinel, "explicit override must be returned verbatim"
        assert len(_INSTANCES) == 0, (
            "explicit override MUST NOT trigger a real construction"
        )
        # And it must NOT poison the cache for subsequent lookups.
        cached = _embed_cache.resolve_embed_model(cfg)
        assert cached is not sentinel, (
            "explicit override must not be stored in the shared cache — "
            "caller-supplied models may have arbitrary lifetimes"
        )
        assert len(_INSTANCES) == 1

    def test_construction_uses_local_files_only(self) -> None:
        """Offline guarantee (``DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md``).

        A previous incident hung an agent turn indefinitely when a stray
        HF-Hub network round-trip fired; every construction must be
        strictly offline.
        """
        _embed_cache.resolve_embed_model(IndexConfig())
        assert len(_INSTANCES) == 1
        inst = _INSTANCES[0]
        assert inst.model_kwargs.get("local_files_only") is True, (
            "resolve_embed_model must set local_files_only=True — the "
            "offline guarantee is uniform across every semantic-search "
            "entry point after the §7.1 hoist."
        )


class TestCallSitesGoThroughSharedCache:
    """The four write/read paths must all delegate to the shared resolver.

    This is the acceptance test §7.1 asks for: constructing the model
    should be counted once no matter how many entry points are exercised.
    """

    def test_history_indexer_uses_shared_cache(self) -> None:
        from deep_worker.providers.semantic_search import history_indexer

        # We can't call the full ``index_history_file`` without an on-disk
        # summaries dir, so exercise ``_resolve_embed_and_tokenizer``
        # directly — that is the exact hop that used to bypass the cache.
        cfg = IndexConfig()
        # First call — one construction.
        history_indexer._resolve_embed_and_tokenizer(None, cfg)
        # Second and third calls — no additional constructions.
        history_indexer._resolve_embed_and_tokenizer(None, cfg)
        history_indexer._resolve_embed_and_tokenizer(None, cfg)
        assert len(_INSTANCES) == 1, (
            "history_indexer._resolve_embed_and_tokenizer must reuse the "
            "shared cache; this is the exact call site the incident named."
        )

    def test_history_retriever_uses_shared_cache(self) -> None:
        from deep_worker.providers.semantic_search import history_retriever

        cfg = IndexConfig()
        history_retriever._ensure_embed_model(cfg, None)
        history_retriever._ensure_embed_model(cfg, None)
        assert len(_INSTANCES) == 1

    def test_history_indexer_and_retriever_share_one_instance(self) -> None:
        """The cross-path invariant §7.1's 'so the two paths cannot drift
        apart again' clause exists to guarantee."""
        from deep_worker.providers.semantic_search import (
            history_indexer,
            history_retriever,
        )

        cfg = IndexConfig()
        idx_model, _tok, _mx, _r = history_indexer._resolve_embed_and_tokenizer(
            None, cfg
        )
        ret_model = history_retriever._ensure_embed_model(cfg, None)
        assert idx_model is ret_model, (
            "the indexer and the retriever MUST share one embedding "
            "instance per model name — separate caches were the exact "
            "asymmetry that caused the incident."
        )
        assert len(_INSTANCES) == 1
