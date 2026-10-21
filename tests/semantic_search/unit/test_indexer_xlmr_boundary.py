"""Phase 1.5 invariant tests for ``_merge_small_chunks`` (RC-3).

These cover the hard guarantee that **no merged node** leaving
``_merge_small_chunks`` exceeds ``effective_max`` *as measured by the
embedder's own tokenizer*, even when the upstream splitter measured the
same text with a different (tiktoken-backed) counter and let it through.

See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md`` §4.3 (Phase 1.5).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

pytest.importorskip("llama_index")  # semantic-search extra; absent in CI

from llama_index.core import Settings  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402

from deep_worker.providers.semantic_search._tokenizer import embed_model_tokenizer  # noqa: E402
from deep_worker.providers.semantic_search.indexer import _merge_small_chunks  # noqa: E402


@pytest.fixture(scope="module")
def real_embed_model() -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


@pytest.fixture(scope="module")
def xlmr_tools(real_embed_model: Any) -> tuple[Any, int, int]:
    return embed_model_tokenizer(real_embed_model)


def _set_settings_tokenizer(tokenize_fn: Any) -> None:
    # Phase 1.5 SentenceSplitter reads Settings.tokenizer for its own
    # accounting; align it with the embedder, as index_repository does.
    Settings.tokenizer = tokenize_fn  # type: ignore[assignment]


class TestPhase15Invariant:
    def test_oversized_single_node_is_split_below_ceiling(
        self, xlmr_tools: tuple[Any, int, int]
    ) -> None:
        tokenize_fn, model_max_length, reserve = xlmr_tools
        _set_settings_tokenizer(tokenize_fn)
        effective_max = min(64, model_max_length - reserve)  # small ceiling

        # Build text that is comfortably over effective_max in XLM-R units.
        # Russian inflates XLM-R counts; repeat to guarantee overflow.
        big = ("Проверка разбиения очень длинного фрагмента текста. " * 80)
        node = TextNode(text=big)

        out = _merge_small_chunks(
            [node],
            effective_max=effective_max,
            tokenize_fn=tokenize_fn,
            min_chars=1,
        )

        assert out, "oversized node must yield at least one sub-node"
        for n in out:
            assert len(tokenize_fn(n.get_content())) <= effective_max

    def test_all_outputs_respect_ceiling_for_mixed_inputs(
        self, xlmr_tools: tuple[Any, int, int]
    ) -> None:
        tokenize_fn, model_max_length, reserve = xlmr_tools
        _set_settings_tokenizer(tokenize_fn)
        effective_max = min(96, model_max_length - reserve)

        nodes = [
            TextNode(text="short one " * 3),
            TextNode(text="Длинный русский фрагмент текста для проверки. " * 60),
            TextNode(text="another short tail " * 4),
        ]

        out = _merge_small_chunks(
            nodes,
            effective_max=effective_max,
            tokenize_fn=tokenize_fn,
            min_chars=1,
        )

        assert out
        for n in out:
            assert len(tokenize_fn(n.get_content())) <= effective_max
