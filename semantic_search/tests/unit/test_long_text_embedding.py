"""Unit tests for long_text_embedding core (windowing + weighted pooling).

Uses a fake whitespace tokenizer + axis-unit-vector encoder, so no real model
is needed. Covers: token-window splitting at the ceiling, the token-weighted
normalized mean for over-ceiling text, and per-message extraction order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np

from semantic_search.long_text_embedding import (
    _message_text,
    _token_windows,
    count_jsonl_lines,
    embed_message_texts,
    message_texts_from_file,
)

_DIM = 6


class _FakeTok:
    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


class _FakeST:
    def __init__(self) -> None:
        self.tokenizer = _FakeTok()

    def get_sentence_embedding_dimension(self) -> int:
        return _DIM

    def encode(self, texts: list[str], **kw: Any) -> Any:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, abs(hash(t)) % _DIM] = 1.0
        return out


class TestTokenWindows:
    def test_short_text_single_window(self) -> None:
        tok = _FakeTok()
        wins = _token_windows("a b c d", tok, eff_max=10)
        assert len(wins) == 1
        assert wins[0][1] == 4

    def test_long_text_splits_at_ceiling(self) -> None:
        tok = _FakeTok()
        text = " ".join(["w"] * 25)
        wins = _token_windows(text, tok, eff_max=10)
        assert [n for _, n in wins] == [10, 10, 5]
        assert all(n <= 10 for _, n in wins)


class TestEmbedMessageTexts:
    def test_empty_returns_empty(self) -> None:
        st = _FakeST()
        embs, lens = embed_message_texts([], st, eff_max=10)
        assert embs.shape == (0, _DIM)
        assert lens.shape == (0,)

    def test_lengths_and_norms(self) -> None:
        st = _FakeST()
        embs, lens = embed_message_texts(["a b c", "d e"], st, eff_max=10)
        assert embs.shape == (2, _DIM)
        assert list(lens) == [3, 2]
        assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)

    def test_over_ceiling_is_weighted_normalized(self) -> None:
        st = _FakeST()
        # 25 tokens, ceiling 10 -> 3 windows; result must be a unit vector
        text = " ".join(["w"] * 25)
        embs, lens = embed_message_texts([text], st, eff_max=10)
        assert int(lens[0]) == 25
        assert embs.shape == (1, _DIM)
        assert abs(float(np.linalg.norm(embs[0])) - 1.0) < 1e-5


class TestMessageExtraction:
    def test_role_prefix_and_skip_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        rows = [
            {"summarized_at": "t", "message_type": "human", "message": {"data": {"content": "hi"}}},
            {"summarized_at": "t", "message_type": "ai", "message": {"data": {"content": ""}}},
            {"summarized_at": "t", "message_type": "tool", "message": {"data": {"content": "out"}}},
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        assert count_jsonl_lines(f) == 3
        texts = message_texts_from_file(f, start_line=0)
        # empty ai message skipped
        assert texts == ["User: hi", "Tool: out"]

    def test_start_line_offset(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        rows = [
            {"summarized_at": "t", "message_type": "human", "message": {"data": {"content": f"m{i}"}}}
            for i in range(4)
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        texts = message_texts_from_file(f, start_line=2)
        assert texts == ["User: m2", "User: m3"]

    def test_multimodal_text_join(self) -> None:
        entry = {
            "message_type": "human",
            "message": {"data": {"content": [
                {"type": "text", "text": "part1"},
                {"type": "image", "url": "x"},
                {"type": "text", "text": "part2"},
            ]}},
        }
        assert _message_text(entry) == "User: part1\npart2"
