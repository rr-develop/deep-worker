"""§5.2 boundary tests for ``_chunk_messages`` (history path).

These use a deterministic **fake tokenizer** (one fake token id per
whitespace-separated word) plus a fake ``effective_max`` so the chunker's
size-accounting is fully predictable without loading the real XLM-R model.
The chunker only ever consults ``tokenize_fn`` for counts, so the fake is a
faithful stand-in. See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md`` §5.2.

When ``_chunk_messages`` splits an oversized group it delegates to
``SentenceSplitter``, which measures with the *global* ``Settings.tokenizer``;
these tests align ``Settings.tokenizer`` with the same fake tokenizer (exactly
as ``index_history_file`` aligns it with the embedder's tokenizer) so the
post-split ``<= effective_max`` assertions are meaningful.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from llama_index.core import Settings  # noqa: E402

from deep_worker.providers.semantic_search.history_indexer import _chunk_messages  # noqa: E402


def _fake_tokenize(text: str) -> list[int]:
    """One fake token id per whitespace-separated word."""
    return [hash(w) & 0xFFFF for w in text.split()]


@pytest.fixture(autouse=True)
def _align_settings_tokenizer() -> object:
    """Align Settings.tokenizer with the fake so the SentenceSplitter used in
    the oversized-group branch measures in the same units the test asserts on.
    Restores the previous value afterwards to avoid cross-test bleed."""
    prev = Settings.tokenizer
    Settings.tokenizer = _fake_tokenize  # type: ignore[assignment]
    yield
    Settings.tokenizer = prev  # type: ignore[assignment]


def _group(content: str, *, ts: str = "2026-01-01T00:00:00", role: str = "human") -> list[dict]:
    return [{"summarized_at": ts, "message_type": role, "message": {"data": {"content": content}}}]


def _words(n: int) -> str:
    # Distinct words so the fake tokenizer yields exactly n tokens and the
    # text is comfortably above MIN_CHARS_PER_CHUNK (50).
    return " ".join(f"word{i:04d}" for i in range(n))


class TestChunkMessagesBoundary:
    def test_group_under_effective_max_kept_whole(self) -> None:
        groups = [_group(_words(300))]
        chunks = _chunk_messages(groups, effective_max=510, tokenize_fn=_fake_tokenize)
        assert len(chunks) == 1
        _text, meta = chunks[0]
        # Whole-group path does not set chunk_index (only the split branch does).
        assert "chunk_index" not in meta

    def test_group_at_exact_effective_max_kept_whole(self) -> None:
        # role prefix "User: " adds one token ("User:"), so 509 words → 510 tokens.
        groups = [_group(_words(509))]
        chunks = _chunk_messages(groups, effective_max=510, tokenize_fn=_fake_tokenize)
        assert len(chunks) == 1
        _text, meta = chunks[0]
        assert "chunk_index" not in meta

    def test_group_above_effective_max_is_sentence_split(self) -> None:
        groups = [_group(_words(800))]
        chunks = _chunk_messages(groups, effective_max=510, tokenize_fn=_fake_tokenize)
        assert len(chunks) > 1
        for text, meta in chunks:
            assert len(_fake_tokenize(text)) <= 510
            assert "chunk_index" in meta
        # chunk_index is a contiguous 0-based sequence.
        idxs = [int(meta["chunk_index"]) for _t, meta in chunks]
        assert idxs == list(range(len(idxs)))

    def test_two_small_adjacent_groups_merge_until_effective_max(self) -> None:
        # Three groups of 200 tokens each, same timestamp so they are separate
        # groups fed in sequence. 200 + 200 = 400 fits; adding the third (600)
        # exceeds 510 → flush. Expect two chunks (400, 200), not three.
        groups = [_group(_words(200), ts=f"ts{i}") for i in range(3)]
        chunks = _chunk_messages(groups, effective_max=510, tokenize_fn=_fake_tokenize)
        assert len(chunks) == 2

    def test_subchunk_below_min_chars_dropped_in_multi_chunk_case(self) -> None:
        # An oversized group whose tail sub-chunk is < MIN_CHARS_PER_CHUNK (50)
        # must be dropped when more than one sub-node is produced. Build a body
        # that splits into a large head + a tiny tail.
        body = _words(800) + " x"
        groups = [_group(body)]
        chunks = _chunk_messages(groups, effective_max=510, tokenize_fn=_fake_tokenize)
        assert chunks
        for text, _meta in chunks:
            # Either a real chunk (>=50 chars) or, if only one sub-node existed,
            # it would be kept; multi-chunk case drops the tiny tail.
            if len(chunks) > 1:
                assert len(text) >= 50
