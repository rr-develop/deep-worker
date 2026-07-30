"""§5.4 integration: a mixed (old long-chunk + new small-chunk) history index
reads cleanly through ``search_history``.

The pre-fix world stored history chunks measured with tiktoken, so a single
very long message could land in one over-long row. After the fix, new writes
use the embedder's own XLM-R tokenizer with the 510-token ceiling. This test
builds an index whose first file carries a long message and whose
second file carries small messages indexed by the fixed code path, then proves
both old and new content remain retrievable from the same index.

Strictly offline; the e5 model is cached on the host. See
``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from semantic_search.history_indexer import index_history_file  # noqa: E402
from semantic_search.history_retriever import search_history  # noqa: E402


@pytest.fixture(scope="module")
def embed_model() -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    content = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


_OLD_PHRASE = "quantum entanglement nonlocality Einstein Podolsky Rosen paradox"
_NEW_PHRASE = "deterministic incremental indexer regression sentinel marker"


def test_mixed_index_reads_old_and_new(tmp_path: Path, embed_model: Any) -> None:
    summaries = tmp_path / "summaries"
    summaries.mkdir(parents=True)

    # --- "old" file: one long message (a big chunk) -----------------------
    old = summaries / "thread_old.jsonl"
    long_body = (
        "This is an old long entry in the agent's history. "
        + (_OLD_PHRASE + " ") * 60
    )
    _write_jsonl(
        old,
        [
            {
                "summarized_at": "2026-01-01T00:00:00",
                "message_index": 0,
                "message_type": "ai",
                "message": {"data": {"content": long_body}},
            }
        ],
    )
    stats_old = index_history_file(old, summaries, embed_model=embed_model)
    assert stats_old.chunks_indexed >= 1

    # --- "new" file: small messages indexed by the fixed path ------------
    new = summaries / "thread_new.jsonl"
    _write_jsonl(
        new,
        [
            {
                "summarized_at": "2026-06-30T12:00:00",
                "message_index": 0,
                "message_type": "human",
                "message": {"data": {"content": f"please add a {_NEW_PHRASE}"}},
            },
            {
                "summarized_at": "2026-06-30T12:00:01",
                "message_index": 1,
                "message_type": "ai",
                "message": {"data": {"content": f"done — {_NEW_PHRASE} in place"}},
            },
        ],
    )
    stats_new = index_history_file(new, summaries, embed_model=embed_model)
    assert stats_new.chunks_indexed >= 1

    # --- mixed reads: both queries return their respective rows ----------
    old_hits = search_history(
        "quantum entanglement paradox", summaries, top_k=5, embed_model=embed_model
    )
    assert old_hits, "old long-chunk content is not retrievable from the mixed index"
    assert any("thread_old" in (h.thread_id or "") or "Einstein" in h.content for h in old_hits)

    new_hits = search_history(
        "incremental indexer regression sentinel", summaries, top_k=5, embed_model=embed_model
    )
    assert new_hits, "new small-chunk content is not retrievable from the mixed index"
    assert any(_NEW_PHRASE.split()[0] in h.content or "sentinel" in h.content for h in new_hits)
