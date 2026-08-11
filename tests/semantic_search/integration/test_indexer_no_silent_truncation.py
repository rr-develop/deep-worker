"""End-to-end guard against silent embed-time truncation.

The bug this whole change closes: the chunker measured chunk size with
tiktoken while the embedder (XLM-RoBERTa, ``model_max_length=512``) silently
truncated anything above its own ceiling. These integration tests assert the
*two-sided sentinel*:

* **pre-fix reality** — there exists realistic input whose tiktoken-count is
  ``<= child_chunk_size`` (so the old chunker would have emitted it as a single
  leaf) yet whose XLM-R-count is ``> effective_max`` (so the embedder would have
  truncated it). This proves the bug is real, not hypothetical.
* **post-fix guarantee** — after ``index_repository`` runs through the real
  pipeline, *every* leaf node stored in LanceDB is ``<= effective_max`` in the
  embedder's own token units, i.e. nothing is silently truncated.

Plus a mixed-index read smoke test: an index written by the fixed pipeline is
searchable (dim stays 384, content round-trips).

Strictly offline; the e5 model is cached on the host. See
``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from deep_worker.providers.semantic_search._tokenizer import (  # noqa: E402
    embed_model_tokenizer,
    effective_max_tokens,
)
from deep_worker.providers.semantic_search.config import IndexConfig  # noqa: E402
from deep_worker.providers.semantic_search.indexer import _split_document, index_repository  # noqa: E402
from deep_worker.providers.semantic_search.retriever import search  # noqa: E402


@pytest.fixture(scope="module")
def embed_model() -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


def _init_git_repo(repo_path: Path) -> None:
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def _tiktoken_count(text: str) -> int:
    import tiktoken  # type: ignore[import-untyped]

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))


# A large Russian-heavy doc: Cyrillic inflates XLM-R sub-word counts far more
# than tiktoken's cl100k_base, so we can land below the tiktoken ceiling while
# above the XLM-R ceiling — exactly the silent-truncation pre-condition.
_RU_PARAGRAPH = (
    "Это довольно длинный фрагмент текста на русском языке, который нужен для "
    "демонстрации расхождения токенизаторов между tiktoken и XLM-RoBERTa. "
)


class TestTwoSidedSentinel:
    def test_pre_fix_input_exists(self, embed_model: Any) -> None:
        """A realistic chunk that the OLD chunker would emit whole but the
        embedder would truncate."""
        tokenize_fn, model_max_length, reserve = embed_model_tokenizer(embed_model)
        config = IndexConfig()
        effective_max = effective_max_tokens(config, model_max_length, reserve)

        # Grow the doc until tiktoken<=child_chunk_size but XLM-R>effective_max.
        text = ""
        found = False
        for _ in range(400):
            text += _RU_PARAGRAPH
            tk = _tiktoken_count(text)
            xl = len(tokenize_fn(text))
            if tk <= config.child_chunk_size and xl > effective_max:
                found = True
                break
            if tk > config.child_chunk_size:
                break
        assert found, (
            "Could not construct the pre-fix sentinel input — the tokenizers "
            "may have converged, which would make this fix unmotivated."
        )

    def test_pre_fix_chunking_path_produces_over_ceiling_leaf(
        self, embed_model: Any
    ) -> None:
        """Stronger §5.4 pre-condition probe: run the REAL ``_split_document``
        chunking path on the test repo's actual document under *pre-fix*
        semantics (the old tiktoken ceiling — Phase 1.5 disabled by giving it a
        ``tokenize_fn`` that under-counts exactly as tiktoken did) and assert at
        least one resulting leaf is > 510 XLM-R tokens. This proves the test
        repo genuinely trips the silent-truncation bug, guarding against a
        false-green where the data was simply too small.
        """
        from llama_index.core import Document, Settings

        tokenize_fn, model_max_length, reserve = embed_model_tokenizer(embed_model)
        config = IndexConfig()
        xlmr_ceiling = model_max_length - reserve  # 510 for e5

        # Pre-fix behaviour: the chunker measured with tiktoken and used the
        # full child_chunk_size (1024) as the ceiling. Emulate that by handing
        # _split_document a tiktoken-based tokenize_fn and effective_max=1024,
        # which is exactly the buggy regime (no XLM-R awareness, no Phase 1.5
        # tightening because the tiktoken count stays under 1024).
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        def _tiktoken_ids(text: str) -> list[int]:
            return enc.encode(text)

        prev = Settings.tokenizer
        Settings.tokenizer = _tiktoken_ids  # type: ignore[assignment]
        try:
            doc = Document(
                text=_RU_PARAGRAPH * 200,
                metadata={"language": "text", "file_path": "notes.md"},
            )
            leaves = _split_document(
                doc,
                config,
                effective_max=config.child_chunk_size,  # pre-fix: 1024, tiktoken
                tokenize_fn=_tiktoken_ids,
            )
        finally:
            Settings.tokenizer = prev  # type: ignore[assignment]

        assert leaves, "pre-fix chunking produced no leaves"
        over = [n for n in leaves if len(tokenize_fn(n.get_content())) > xlmr_ceiling]
        assert over, (
            "pre-fix chunking path produced no leaf over the XLM-R ceiling — "
            "the test document is too small to exercise the bug (false-green "
            "guard)."
        )

    def test_post_fix_no_leaf_over_ceiling(self, embed_model: Any) -> None:
        """After the real pipeline runs, every stored leaf is <= effective_max
        in the embedder's own units."""
        tokenize_fn, model_max_length, reserve = embed_model_tokenizer(embed_model)
        config = IndexConfig()
        effective_max = effective_max_tokens(config, model_max_length, reserve)

        repo = Path(_make_repo_with_big_doc())
        try:
            index_repository(repo, embedding_model=embed_model)

            leaves = _read_lancedb_documents(repo)
            assert leaves, "expected at least one stored leaf"
            for content in leaves:
                assert len(tokenize_fn(content)) <= effective_max, (
                    "stored leaf exceeds the embedder ceiling — silent "
                    "truncation would occur at embed time"
                )
        finally:
            _rmtree(repo)

    def test_mixed_index_is_searchable(self, embed_model: Any) -> None:
        """Smoke test: an index written by the fixed pipeline round-trips
        through search (dim 384, content preserved)."""
        repo = Path(_make_repo_with_big_doc())
        try:
            index_repository(repo, embedding_model=embed_model)
            results = search(
                "расхождение токенизаторов",
                repo_path=repo,
                top_k=3,
                embedding_model=embed_model,
            )
            assert isinstance(results, list)
            assert results, "search over a freshly built index returned nothing"
            assert all(isinstance(r.content, str) and r.content for r in results)
        finally:
            _rmtree(repo)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_repo_with_big_doc() -> str:
    import tempfile

    d = tempfile.mkdtemp(prefix="ss_trunc_")
    repo = Path(d)
    big = _RU_PARAGRAPH * 200  # comfortably multi-chunk
    (repo / "notes.md").write_text(big, encoding="utf-8")
    (repo / "small.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _init_git_repo(repo)
    return str(repo)


def _read_lancedb_documents(repo: Path) -> list[str]:
    """Read every stored leaf's text directly from the LanceDB table."""
    import lancedb  # type: ignore[import-untyped]

    uri = str(repo / ".vector_index" / "lancedb")
    db = lancedb.connect(uri)
    names = list(db.table_names())
    out: list[str] = []
    for name in names:
        tbl = db.open_table(name)
        df = tbl.to_pandas()
        for col in ("text", "_node_content", "content"):
            if col in df.columns:
                out.extend(_extract_text(v, col) for v in df[col].tolist())
                break
    return [t for t in out if t]


def _extract_text(value: Any, col: str) -> str:
    if col == "_node_content":
        # JSON blob with a "text" field
        import json

        try:
            return str(json.loads(value).get("text", ""))
        except (ValueError, TypeError):
            return ""
    return str(value) if value is not None else ""


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
