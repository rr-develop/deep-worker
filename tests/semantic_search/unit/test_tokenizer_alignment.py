"""Unit tests for ``semantic_search._tokenizer``.

Covers the helper that exposes the embedder's own tokenizer to the chunker,
and the ``effective_max_tokens`` ceiling that prevents silent truncation at
embed time. See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md`` §4.1, §4.2 and the
RC-1 / RC-2 sentinels in §5.1.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest


# ---- import-time env discipline ------------------------------------------
# Force strictly-offline before importing anything that may touch HF Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")


from deep_worker.providers.semantic_search._tokenizer import (  # noqa: E402
    embed_model_tokenizer,
    effective_max_tokens,
)
from deep_worker.providers.semantic_search.config import IndexConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Real e5-small probes (offline; model is already cached on the host)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_embed_model() -> Any:
    """The real HuggingFaceEmbedding(intfloat/multilingual-e5-small), loaded
    strictly offline. Same instance is reused across this module."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


class TestEmbedModelTokenizer:
    def test_returns_callable_and_512_limit_and_two_specials(
        self, real_embed_model: Any
    ) -> None:
        tokenize_fn, model_max_length, special_tokens_reserve = (
            embed_model_tokenizer(real_embed_model)
        )
        assert callable(tokenize_fn)
        ids = tokenize_fn("hello world")
        assert isinstance(ids, list)
        assert ids and all(isinstance(t, int) for t in ids)
        assert model_max_length == 512
        assert special_tokens_reserve == 2

    def test_special_tokens_reserve_matches_real_tokenizer(
        self, real_embed_model: Any
    ) -> None:
        """RC-2 sentinel: the '2' we observe for XLM-R must be what the
        tokenizer itself reports via the public HF API, not a coincidence."""
        tok = real_embed_model._model.tokenizer  # noqa: SLF001
        assert tok.num_special_tokens_to_add(pair=False) == 2


class TestTiktokenVsXlmrDisagreement:
    """Sentinels that prove the bug we are fixing exists: the two tokenizers
    must NOT agree on real input — otherwise the silent-truncation bug
    could not occur, and this whole change is unmotivated.

    We intentionally bypass ``llama_index.core.utils.get_tokenizer`` here and
    instantiate ``tiktoken`` directly: ``get_tokenizer`` consults the global
    ``Settings.tokenizer`` which other tests in this suite (correctly) set to
    the embedder's XLM-R tokenizer. Using the global would make this test
    self-referential.
    """

    @staticmethod
    def _tiktoken_count(text: str) -> int:
        import tiktoken  # type: ignore[import-untyped]

        # gpt-3.5-turbo / cl100k_base — the same model name and encoding
        # llama_index.core.utils.get_tokenizer uses by default before any
        # override.
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))

    def test_xlmr_under_counts_tiktoken_on_russian(
        self, real_embed_model: Any
    ) -> None:
        tokenize_fn, _max, _rsv = embed_model_tokenizer(real_embed_model)
        ru = (
            "Это длинная строка по-русски, проведение тестирования имеет "
            "важное значение для согласованности токенайзера и эмбеддера."
        )
        xlmr = len(tokenize_fn(ru))
        tt = self._tiktoken_count(ru)
        # XLM-R should be the smaller count on Russian (tiktoken
        # over-counts because cl100k_base splits Cyrillic into many BPE
        # pieces). At least 25% difference.
        assert xlmr < tt
        assert (tt - xlmr) / tt >= 0.25, (
            f"expected >=25% disagreement on Russian; got tt={tt}, xlmr={xlmr}"
        )

    def test_xlmr_over_counts_tiktoken_on_english(
        self, real_embed_model: Any
    ) -> None:
        tokenize_fn, _max, _rsv = embed_model_tokenizer(real_embed_model)
        en = (
            "This is an English sentence for the same proceeding tokenizer "
            "alignment test that exercises the other direction of error."
        )
        xlmr = len(tokenize_fn(en))
        tt = self._tiktoken_count(en)
        # On general English XLM-R typically produces MORE sub-word
        # tokens than tiktoken's gpt-3.5 BPE (which has long common-word
        # merges). Strict inequality: xlmr > tt.
        assert xlmr > tt


# ---------------------------------------------------------------------------
# effective_max_tokens
# ---------------------------------------------------------------------------

def _cfg(child_chunk_size: int = 1024) -> IndexConfig:
    return IndexConfig(child_chunk_size=child_chunk_size)


class TestEffectiveMaxTokens:
    def test_short_config_binds(self) -> None:
        assert effective_max_tokens(_cfg(128), 512, 2) == 128

    def test_e5_model_binds_with_reserve_applied(self) -> None:
        # IndexConfig default 1024, e5-small 512, XLM-R reserves 2.
        assert effective_max_tokens(_cfg(1024), 512, 2) == 510

    def test_8k_model_lifts_ceiling_back_to_config(self) -> None:
        assert effective_max_tokens(_cfg(1024), 8192, 2) == 1024

    def test_zero_special_tokens(self) -> None:
        assert effective_max_tokens(_cfg(1024), 512, 0) == 512


# ---------------------------------------------------------------------------
# RC-1 — no silent fallback
# ---------------------------------------------------------------------------

class TestNoSilentFallback:
    @staticmethod
    def _assert_loud(exc: pytest.ExceptionInfo[RuntimeError]) -> None:
        msg = str(exc.value)
        assert "embed_model has no accessible tokenizer" in msg
        assert "DESIGN_FIX_TOKENIZER_MISMATCH.md" in msg

    def test_raises_when_embed_model_has_no_model_attr(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(SimpleNamespace())  # no `_model`
        self._assert_loud(exc)

    def test_raises_when_inner_model_is_none(self) -> None:
        bad = SimpleNamespace(_model=None)
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(bad)
        self._assert_loud(exc)

    def test_raises_when_inner_model_has_no_tokenizer(self) -> None:
        bad = SimpleNamespace(_model=SimpleNamespace())  # _model present, no .tokenizer
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(bad)
        self._assert_loud(exc)

    def test_raises_when_tokenizer_not_callable(self) -> None:
        tok = SimpleNamespace(
            model_max_length=512,
            num_special_tokens_to_add=lambda pair=False: 2,
        )
        bad = SimpleNamespace(_model=SimpleNamespace(tokenizer=tok))
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(bad)
        self._assert_loud(exc)

    def test_raises_when_tokenizer_missing_model_max_length(self) -> None:
        def _call(text: str, **_kw: Any) -> dict[str, list[int]]:
            return {"input_ids": [1, 2, 3]}

        tok = SimpleNamespace(num_special_tokens_to_add=lambda pair=False: 2)
        tok.__call__ = _call  # type: ignore[attr-defined]
        # Make it callable via type
        callable_tok = type("Tok", (), {
            "__call__": lambda self, text, **kw: {"input_ids": [1, 2]},
            "num_special_tokens_to_add": lambda self, pair=False: 2,
            # no model_max_length
        })()
        bad = SimpleNamespace(_model=SimpleNamespace(tokenizer=callable_tok))
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(bad)
        self._assert_loud(exc)

    def test_raises_when_tokenizer_missing_num_special_tokens_to_add(self) -> None:
        callable_tok = type("Tok", (), {
            "__call__": lambda self, text, **kw: {"input_ids": [1, 2]},
            "model_max_length": 512,
            # no num_special_tokens_to_add
        })()
        bad = SimpleNamespace(_model=SimpleNamespace(tokenizer=callable_tok))
        with pytest.raises(RuntimeError) as exc:
            embed_model_tokenizer(bad)
        self._assert_loud(exc)
