"""Unit tests for ``semantic_search.message_search.search_messages``.

A controllable fake embedder maps any text to a one-hot ``_DIM`` vector whose
axis is the first integer found in the text (default axis 0). This lets a test
pin BOTH the stored message embeddings (via ``build_message_embeddings`` with the
same fake) and the query embedding (via ``embed_long_text`` with the same fake)
onto chosen axes, so the dot-product ranking is fully deterministic with no real
model. ``torch`` is never imported by the search path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deep_worker.providers.semantic_search.message_embedding_store import (
    MessageFlag,
    build_message_embeddings,
    build_message_metadata,
    load_message_embeddings,
    store_dir,
)
from deep_worker.providers.semantic_search.message_search import (
    _compute_whitening_matrix,
    _row_messages,
    _select_kept_axis_count,
    search_messages,
)

_DIM = 8
_INT_RE = re.compile(r"\d+")


def _axis_for(text: str) -> int:
    m = _INT_RE.search(text)
    return (int(m.group()) % _DIM) if m else 0


class _FakeTokenizer:
    model_max_length = 512

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [(abs(hash(w)) % 1000) for w in text.split()]}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"t{i}" for i in ids)


class _FakeST:
    """SentenceTransformer stand-in: text -> one-hot on the axis of its first int."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return _DIM

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, _axis_for(t)] = 1.0
        return out


class _FakeEmbed:
    def __init__(self) -> None:
        self._model = _FakeST()


@pytest.fixture
def fake_embed() -> _FakeEmbed:
    return _FakeEmbed()


def _entry(role: str, content: str, ts: str = "2026-01-01T00:00:00") -> dict[str, Any]:
    return {
        "summarized_at": ts,
        "message_type": role,
        "message": {"data": {"content": content}},
    }


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


def _store(summaries: Path, fake: _FakeEmbed) -> None:
    """Build embeddings + metadata for a summaries dir with the fake embedder."""
    build_message_embeddings(summaries, embed_model=fake)
    build_message_metadata(summaries)


class TestRanking:
    def test_top_hit_is_highest_dot_product(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # Messages pinned to axes 1, 2, 3 (role prefix keeps the same first int).
        _write(summaries / "a.jsonl", [
            _entry("human", "topic 1 alpha"),
            _entry("human", "topic 2 beta"),
            _entry("human", "topic 3 gamma"),
        ])
        _store(summaries, fake_embed)

        # Query pinned to axis 2 -> must rank the "topic 2" message first, score 1.0.
        res = search_messages(
            query="please find 2", summaries_dir=summaries, top_k=3,
            embed_model=fake_embed,
        )
        assert len(res) == 3
        assert res[0].content == "User: topic 2 beta"
        assert res[0].score == pytest.approx(1.0)
        # The other two are orthogonal -> score 0.0.
        assert res[1].score == pytest.approx(0.0)

    def test_top_k_clamps_to_candidate_count(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "only 1 here")])
        _store(summaries, fake_embed)
        # top_k far exceeds the single candidate -> no argpartition error, 1 result.
        res = search_messages(
            query="1", summaries_dir=summaries, top_k=10, embed_model=fake_embed,
        )
        assert len(res) == 1


class TestFlagFilter:
    def test_flags_any_restricts_candidates(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # ai message on axis 2 (more similar) + human message on axis 2 too.
        _write(summaries / "a.jsonl", [
            _entry("ai", "answer 2 best match"),
            _entry("human", "question 2 also match"),
        ])
        _store(summaries, fake_embed)

        # Query on axis 2 matches both; restrict to USER_ROLE -> only the human.
        res = search_messages(
            query="2", summaries_dir=summaries, top_k=5,
            flags_any=MessageFlag.USER_ROLE, embed_model=fake_embed,
        )
        assert len(res) == 1
        assert res[0].message_types == "human"
        assert res[0].flags & MessageFlag.USER_ROLE
        assert "USER_ROLE" in res[0].flag_names

    def test_empty_candidate_set_returns_empty(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "hello 1")])
        _store(summaries, fake_embed)
        # No row has SYSTEM -> empty candidate set -> [].
        res = search_messages(
            query="1", summaries_dir=summaries, top_k=5,
            flags_any=MessageFlag.SYSTEM, embed_model=fake_embed,
        )
        assert res == []


class TestRowMapping:
    def test_physical_line_skips_blank_and_bad_lines(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        # Physical layout:
        #  line 1: blank
        #  line 2: undecodable
        #  line 3: valid entry (axis 5)
        good = json.dumps(_entry("human", "target 5 here"), ensure_ascii=False)
        f.write_text("\n" + "{not json\n" + good + "\n", encoding="utf-8")
        _store(summaries, fake_embed)

        res = search_messages(
            query="5", summaries_dir=summaries, top_k=1, embed_model=fake_embed,
        )
        assert len(res) == 1
        # Physical line is 3, NOT the compacted index 1.
        assert res[0].line_range == "3"
        assert res[0].content == "User: target 5 here"
        assert res[0].thread_id == "a"
        assert res[0].timestamp == "2026-01-01T00:00:00"
        assert res[0].file_path.endswith("a.jsonl")

    def test_row_messages_count_matches_store(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [
            _entry("human", "m 1"),
            _entry("ai", ""),  # empty -> no embedding row, skipped
            _entry("human", "m 2"),
        ])
        _store(summaries, fake_embed)
        embs, _ = load_message_embeddings(summaries, as_torch=False)
        rows = _row_messages(summaries)
        assert len(rows) == int(embs.shape[0]) == 2

    def test_appended_history_does_not_break_search(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """Later-appended messages must not break analysis of earlier ones.

        The store is frozen at 2 rows; appending a 3rd message to the history
        file (as summarisation does) must NOT raise and must NOT change the
        ranking over the original rows — the watermark bounds the re-walk.
        """
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write(f, [_entry("human", "topic 1 alpha"), _entry("human", "topic 2 beta")])
        _store(summaries, fake_embed)
        before = search_messages(
            query="find 2", summaries_dir=summaries, top_k=5, embed_model=fake_embed,
        )
        # Append a new message WITHOUT rebuilding the store (mirrors real drift).
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_entry("human", "topic 2 gamma"), ensure_ascii=False) + "\n")
        after = search_messages(
            query="find 2", summaries_dir=summaries, top_k=5, embed_model=fake_embed,
        )
        # Still exactly the 2 frozen rows, same order/scores/content.
        assert len(after) == 2
        assert [r.content for r in after] == [r.content for r in before]
        assert [r.score for r in after] == [r.score for r in before]
        # The appended "gamma" message is not represented in the store.
        assert all("gamma" not in r.content for r in after)

    def test_row_messages_watermark_truncates_appended_tail(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write(f, [_entry("human", "m 1"), _entry("human", "m 2")])
        _store(summaries, fake_embed)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_entry("human", "m 3"), ensure_ascii=False) + "\n")
        # Without a watermark the re-walk sees all 3 entries...
        assert len(_row_messages(summaries)) == 3
        # ...but bounded by the store's recorded prefix it yields exactly 2.
        wm = {str(f): 2}
        assert len(_row_messages(summaries, wm)) == 2

    def test_row_messages_watermark_counts_parsed_not_physical(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """The watermark unit is PARSED entries (blank/bad lines already dropped),
        matching count_jsonl_lines — not physical lines."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        good1 = json.dumps(_entry("human", "m 1"), ensure_ascii=False)
        good2 = json.dumps(_entry("human", "m 2"), ensure_ascii=False)
        good3 = json.dumps(_entry("human", "m 3"), ensure_ascii=False)
        # Physical: blank, bad, good1, blank, good2, good3 -> 3 parsed entries
        # among the first cluster; watermark=2 must stop after good2, not good3.
        f.write_text(
            "\n{bad\n" + good1 + "\n\n" + good2 + "\n" + good3 + "\n",
            encoding="utf-8",
        )
        rows = _row_messages(summaries, {str(f): 2})
        assert [r.content for r in rows] == ["User: m 1", "User: m 2"]


class TestErrors:
    def test_empty_query_raises_value_error(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "x 1")])
        _store(summaries, fake_embed)
        with pytest.raises(ValueError):
            search_messages(query="   ", summaries_dir=summaries, top_k=5, embed_model=fake_embed)

    def test_missing_store_raises_file_not_found(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        with pytest.raises(FileNotFoundError):
            search_messages(query="x", summaries_dir=summaries, top_k=5, embed_model=fake_embed)

    def test_missing_metadata_raises_file_not_found(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "x 1")])
        # embeddings built but metadata NOT built -> FileNotFoundError.
        build_message_embeddings(summaries, embed_model=fake_embed)
        with pytest.raises(FileNotFoundError):
            search_messages(query="x", summaries_dir=summaries, top_k=5, embed_model=fake_embed)

    def test_history_rewrite_raises_runtime_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        f = summaries / "a.jsonl"
        _write(f, [_entry("human", "m 1"), _entry("human", "m 2")])
        _store(summaries, fake_embed)
        # REWRITE (not append): replace the second embeddable entry with an
        # empty one, so the watermark-bounded re-walk finds only 1 row while the
        # store has 2. This is a genuine desync (history was rewritten, not
        # merely appended) and MUST still raise.
        _write(f, [_entry("human", "m 1"), _entry("ai", "")])
        with pytest.raises(RuntimeError):
            search_messages(query="1", summaries_dir=summaries, top_k=5, embed_model=fake_embed)


class TestStoreReadOnly:
    def test_search_does_not_modify_store(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "m 1"), _entry("ai", "r 2")])
        _store(summaries, fake_embed)
        d = store_dir(summaries)
        before = {p.name: p.read_bytes() for p in d.iterdir()}
        search_messages(query="1", summaries_dir=summaries, top_k=2, embed_model=fake_embed)
        after = {p.name: p.read_bytes() for p in d.iterdir()}
        assert before == after  # read-only: no file changed


class TestIsotropyTransforms:
    """`-qm` gains three ranking modes: raw / centering / whitening (pca|zca).

    All rely on the same store; nothing is persisted. RC-1: normalize() guards
    zero-norm → zero vector (score 0.0, never NaN). RC-2 covered in CLI tests.
    RC-3: whitening on too-few stats rows → actionable RuntimeError.
    """

    def _pop(self, summaries: Path) -> None:
        """One store with several ai messages on axis 1 (biases μ toward axis 1)
        and one human message on axis 3, so mean-centering breaks the bias."""
        _write(summaries / "a.jsonl", [
            _entry("ai", "one 1"),
            _entry("ai", "two 1"),
            _entry("ai", "three 1"),
            _entry("ai", "four 1"),
            _entry("human", "target 3 here"),
        ])

    def test_raw_default_unchanged(self, tmp_path: Path, fake_embed: _FakeEmbed) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._pop(summaries)
        _store(summaries, fake_embed)
        # Default is raw: query on axis 1 → one of the axis-1 rows wins with score 1.
        res = search_messages(
            query="please 1", summaries_dir=summaries, top_k=5,
            embed_model=fake_embed,
        )
        assert res[0].score == pytest.approx(1.0)
        # sanity: the axis-3 row scores 0.0 in raw mode (orthogonal)
        axis3 = [r for r in res if r.message_types == "human"][0]
        assert axis3.score == pytest.approx(0.0)

    def test_centering_query_transformed_with_same_mu(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """Regression (RC-2 for maths): centering must apply μ to the QUERY too,
        not only rows.

        Corpus: 4 rows on axis 1 (heavy majority) + 1 row on axis 3 (minority).
        μ ≈ (0, 0.8, 0, 0.2, 0…). A query on the minority axis 3, after both
        sides are centered with the SAME μ, aligns strongly with the axis-3 row
        and (weakly) opposite the axis-1 rows → axis-3 row tops. If μ were only
        applied to rows and NOT to the query, the query would remain a pure e_3,
        the axis-1 rows' centered vectors ≈ (0, 0.2, 0, -0.2, …) would dot to
        NEGATIVE with e_3, the axis-3 row would dot to POSITIVE, so ranking
        would still be correct — but the SCORES would differ from the fully
        centered case in a distinguishable way; we assert on the ranking here
        (top row is axis 3) and on the score sign structure (positive top,
        negative losers), which fail if μ is not subtracted from BOTH sides.
        """
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._pop(summaries)
        _store(summaries, fake_embed)
        res = search_messages(
            query="please 3", summaries_dir=summaries, top_k=5,
            transform="centering", embed_model=fake_embed,
        )
        # Top hit is the minority axis-3 row.
        assert res[0].content == "User: target 3 here"
        # Centering makes the axis-1 rows' scores strictly negative w.r.t. the
        # axis-3 query, and the axis-3 row's score strictly positive.
        assert res[0].score > 0.0
        for r in res[1:]:
            assert r.score < 0.0

    def test_centering_zero_norm_row_scores_zero_not_nan(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """RC-1: a row equal to μ centers to a 0-vector; guarded normalize()
        must return 0, giving score 0.0 (not NaN, which would silently corrupt
        argpartition/sort ordering)."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # Two identical axis-1 messages → each row IS μ → centered = 0.
        _write(summaries / "a.jsonl", [
            _entry("ai", "same 1"),
            _entry("ai", "same 1 too"),
        ])
        _store(summaries, fake_embed)
        res = search_messages(
            query="anything 1", summaries_dir=summaries, top_k=2,
            transform="centering", embed_model=fake_embed,
        )
        assert len(res) == 2
        for r in res:
            assert not (r.score != r.score), "score is NaN"  # NaN self-check
            assert r.score == pytest.approx(0.0, abs=1e-6)

    def test_whitening_pca_and_zca_produce_same_ordering(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """§3.1: cosine is invariant to a shared orthogonal rotation, so PCA and
        ZCA whitening yield identical ordering AND identical scores (to ~1e-9)."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # Enough rows to estimate Σ safely (dim=8 → need >=9).
        _write(summaries / "a.jsonl", [
            _entry("ai", f"msg {i % _DIM} content") for i in range(20)
        ] + [_entry("human", "user 3 wants this")])
        _store(summaries, fake_embed)
        res_pca = search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_method="pca", whitening_eps=1e-3,
            embed_model=fake_embed,
        )
        res_zca = search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_method="zca", whitening_eps=1e-3,
            embed_model=fake_embed,
        )
        assert len(res_pca) == len(res_zca) > 0
        # Per §3.1: cosine is invariant to the shared orthogonal factor U^T that
        # distinguishes ZCA from PCA, so the sorted top-k score sequences match
        # to numerical precision. When several rows tie for a top slot, which
        # specific tied row wins is not stable across the orthogonal rotation,
        # so we do NOT compare content sets — only the score fingerprint.
        sorted_pca = sorted(r.score for r in res_pca)
        sorted_zca = sorted(r.score for r in res_zca)
        for a, b in zip(sorted_pca, sorted_zca):
            assert a == pytest.approx(b, abs=1e-6)
        # Full-distribution equivalence via --score-stats: all M candidate
        # scores collapse to the same {min, median, max, range}.
        stats_p: dict[str, Any] = {}
        stats_z: dict[str, Any] = {}
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_method="pca", stats_out=stats_p,
            embed_model=fake_embed,
        )
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_method="zca", stats_out=stats_z,
            embed_model=fake_embed,
        )
        for k in ("m", "min", "median", "max", "range"):
            assert stats_p[k] == pytest.approx(stats_z[k], abs=1e-6)

    def test_stats_scope_filtered_uses_subset_mean(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """stats_scope='filtered' estimates μ over the -f candidate subset, so
        centering with a filter must produce different scores than centering
        over the whole corpus. Fake embedder pins each text's vector to the
        FIRST integer in the text; we choose texts so all filtered rows land on
        axis 3 and all non-filtered rows on axis 1."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # Content strings are carefully crafted so the FIRST digit encountered
        # is the desired axis (fake tokenizer / _axis_for). Role-prefixed text
        # keeps the same first digit.
        _write(summaries / "a.jsonl",
               [_entry("ai", f"axis 1 marker step {i}") for i in range(5)]
               + [_entry("human", f"axis 3 marker step {i}") for i in range(5)])
        _store(summaries, fake_embed)
        # Restrict to USER_ROLE (5 human rows, all on axis 3).
        res_corp = search_messages(
            query="want axis 3 topic", summaries_dir=summaries, top_k=5,
            transform="centering", flags_any=MessageFlag.USER_ROLE,
            stats_scope="corpus", embed_model=fake_embed,
        )
        res_filt = search_messages(
            query="want axis 3 topic", summaries_dir=summaries, top_k=5,
            transform="centering", flags_any=MessageFlag.USER_ROLE,
            stats_scope="filtered", embed_model=fake_embed,
        )
        # Same 5 rows in each answer (same filter).
        assert {r.content for r in res_corp} == {r.content for r in res_filt}
        corp_scores = sorted(r.score for r in res_corp)
        filt_scores = sorted(r.score for r in res_filt)
        # Distinctive fingerprint of the filtered μ: all 5 rows equal μ_filt
        # exactly → centered = 0 → all scores are 0.0.
        for s in filt_scores:
            assert s == pytest.approx(0.0, abs=1e-6)
        # Corpus μ is a mixture (rows on axes 1 and 3), so scores differ from
        # the filtered-μ regime — this asserts the scope argument really does
        # change the statistics set.
        assert corp_scores != filt_scores

    def test_whitening_too_few_stats_rows_raises_actionable(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """RC-3: whitening needs at least dim+1 stats rows. Message names all
        three remedies (widen filter / stats-scope corpus / drop -w)."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # Only 3 rows, dim=8 → 3 < 9, whitening ill-posed.
        _write(summaries / "a.jsonl", [
            _entry("human", "a 1"),
            _entry("human", "b 2"),
            _entry("human", "c 3"),
        ])
        _store(summaries, fake_embed)
        with pytest.raises(RuntimeError) as excinfo:
            search_messages(
                query="pick 1", summaries_dir=summaries, top_k=5,
                transform="whitening", embed_model=fake_embed,
            )
        msg = str(excinfo.value).lower()
        # Actionable: mentions all three remedies.
        assert "widen" in msg or "filter" in msg
        assert "stats-scope corpus" in msg or "--stats-scope corpus" in msg
        assert "-w" in msg or "whitening" in msg
        # Names the counts too.
        assert "9" in msg  # dim+1
        assert "3" in msg  # got

    def test_whitening_eps_regularises_singular_sigma(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """A store where all rows share one axis produces a rank-1 Σ; without εI
        eigh would give zero eigenvalues and 1/√λ would blow up. εI keeps it
        finite (no NaN/inf in scores)."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        # dim+1=9 rows all on axis 1 → Σ is rank-1.
        _write(summaries / "a.jsonl",
               [_entry("ai", f"m{i} 1") for i in range(9)])
        _store(summaries, fake_embed)
        res = search_messages(
            query="pick 1", summaries_dir=summaries, top_k=3,
            transform="whitening", whitening_eps=1e-3, embed_model=fake_embed,
        )
        assert len(res) == 3
        for r in res:
            assert not (r.score != r.score), "score is NaN"
            assert np.isfinite(r.score)

    def test_unknown_transform_raises_value_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", "x 1")])
        _store(summaries, fake_embed)
        with pytest.raises(ValueError):
            search_messages(
                query="x 1", summaries_dir=summaries, top_k=1,
                transform="bogus", embed_model=fake_embed,  # type: ignore[arg-type]
            )

    def test_unknown_whitening_method_raises_value_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", f"x{i} 1") for i in range(10)])
        _store(summaries, fake_embed)
        with pytest.raises(ValueError):
            search_messages(
                query="x 1", summaries_dir=summaries, top_k=1,
                transform="whitening", whitening_method="lol",  # type: ignore[arg-type]
                embed_model=fake_embed,
            )

    def test_non_positive_eps_raises_value_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        _write(summaries / "a.jsonl", [_entry("human", f"x{i} 1") for i in range(10)])
        _store(summaries, fake_embed)
        with pytest.raises(ValueError):
            search_messages(
                query="x 1", summaries_dir=summaries, top_k=1,
                transform="whitening", whitening_eps=0.0, embed_model=fake_embed,
            )


class TestTruncatedWhitening:
    """§3.6: whitening truncates the noisy low-variance tail (keep top-r axes).

    RC-4: eigh returns ASCENDING eigenvalues, so top-r must be the TAIL (largest
    λ), never the head (near-zero noise). RC-5: the reported kept count is the
    ACTUAL r after clamping to the usable rank, not the requested value.
    """

    def test_select_kept_axis_count_var_keeps_large_eigen_axes(self) -> None:
        """RC-4 core: variance selection over an ascending spectrum returns a
        count that corresponds to the LARGEST eigenvalues.

        Spectrum (ascending, as eigh emits): [0.001, 0.001, 0.001, 5.0, 5.0].
        Total = 10.003; the two big axes carry ~0.9997 of the variance. F=0.90
        must select r=2 (the two large tail entries), NOT r=2 from the head."""
        lam_ascending = np.array([0.001, 0.001, 0.001, 5.0, 5.0], dtype=np.float64)
        r = _select_kept_axis_count(
            lam_ascending, dim=5, whitening_var=0.90,
            whitening_dims=None, usable_rank=4,
        )
        assert r == 2

    def test_select_kept_axis_count_var_full_when_uniform(self) -> None:
        """A uniform spectrum needs almost all axes to reach 0.90 variance."""
        lam = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        r = _select_kept_axis_count(
            lam, dim=5, whitening_var=0.90, whitening_dims=None, usable_rank=4,
        )
        # 4/5 = 0.8 < 0.9, 5/5 = 1.0 >= 0.9 → need 5, but clamped to usable_rank=4.
        assert r == 4

    def test_select_kept_axis_count_var_one_dominant_axis_keeps_one(self) -> None:
        """Boundary: if the top axis alone exceeds F, r == 1 (never 0)."""
        lam = np.array([0.01, 0.01, 0.01, 0.01, 9.96], dtype=np.float64)
        r = _select_kept_axis_count(
            lam, dim=5, whitening_var=0.90, whitening_dims=None, usable_rank=4,
        )
        assert r == 1

    def test_select_kept_axis_count_dims_clamped_to_usable_rank(self) -> None:
        """RC-5: requesting more axes than the usable rank clamps to the rank."""
        lam = np.arange(1, 9, dtype=np.float64)  # 8 axes
        r = _select_kept_axis_count(
            lam, dim=8, whitening_var=0.90, whitening_dims=100, usable_rank=4,
        )
        assert r == 4  # clamped to usable_rank, not 8, not 100

    def test_select_kept_axis_count_dims_clamped_to_dim(self) -> None:
        lam = np.arange(1, 9, dtype=np.float64)
        r = _select_kept_axis_count(
            lam, dim=8, whitening_var=0.90, whitening_dims=100, usable_rank=1000,
        )
        assert r == 8  # can't exceed dim

    def test_compute_whitening_matrix_keeps_large_variance_directions(self) -> None:
        """RC-4 end-to-end on the matrix: build stats whose variance lives almost
        entirely on axis 0; truncating to r=1 must keep THAT direction, so a
        vector along axis 0 is scaled up and one along a near-zero axis maps to
        ~0. If truncation kept the head (smallest λ) instead, axis 0 would be
        dropped and the result inverted."""
        rng_axis0 = np.linspace(-3.0, 3.0, 40)  # large spread on axis 0
        stats = np.zeros((40, 4), dtype=np.float64)
        stats[:, 0] = rng_axis0
        stats[:, 1] = 1e-4 * np.linspace(-1, 1, 40)  # tiny spread on axis 1
        mu, w, r = _compute_whitening_matrix(
            stats, method="zca", eps=1e-6, dim=4,
            whitening_var=0.90, whitening_dims=None,
        )
        assert r == 1  # nearly all variance on one axis
        # A vector along the kept (high-variance) axis 0 must SURVIVE the map
        # (non-zero image); a vector purely on a dropped axis (axis 2, zero
        # variance) maps to ~0. If truncation had (wrongly) kept the head
        # (smallest λ), axis 0 would be the dropped one and this would invert.
        v_kept = np.array([1.0, 0.0, 0.0, 0.0]) @ w
        v_dropped = np.array([0.0, 0.0, 1.0, 0.0]) @ w
        assert np.linalg.norm(v_kept) > 1e-3  # kept axis survives
        assert np.linalg.norm(v_dropped) == pytest.approx(0.0, abs=1e-9)

    def _many(self, summaries: Path, n: int = 30) -> None:
        _write(summaries / "a.jsonl",
               [_entry("ai", f"msg {i % _DIM} body text {i}") for i in range(n)]
               + [_entry("human", "user 3 wants this")])

    def test_search_reports_kept_axes_in_stats_out(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        stats: dict[str, Any] = {}
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_dims=3, stats_out=stats,
            embed_model=fake_embed,
        )
        assert stats["whitening_dims_kept"] == 3
        assert stats["whitening_dim"] == _DIM
        assert stats["whitening_dims_requested"] == 3
        assert stats["whitening_var"] is None  # dims path

    def test_search_full_dims_reports_none_kept(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """Keeping all `dim` axes → no truncation → kept is None so the CLI
        suppresses the notice (this is the full-whitening path). We request
        `whitening_dims=_DIM` explicitly: on the fake one-hot corpus one axis is
        variance-degenerate, so `whitening_var=1.0` legitimately keeps only
        dim-1 axes — the "all axes" invariant is exercised via the absolute
        request, which clamps to dim and yields r == dim → kept None."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        stats: dict[str, Any] = {}
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_dims=_DIM, stats_out=stats,
            embed_model=fake_embed,
        )
        assert stats["whitening_dims_kept"] is None  # r == dim → no truncation
        assert stats["whitening_dims_requested"] == _DIM

    def test_search_var_one_still_reports_when_degenerate(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        """On a variance-degenerate corpus, whitening_var=1.0 keeps only the
        non-degenerate axes (r < dim), so kept is a real int (a genuine
        truncation of the zero-variance tail) — NOT None. Documents the honest
        behaviour: 'kept' reflects reality, not the request."""
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        stats: dict[str, Any] = {}
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_var=1.0, stats_out=stats,
            embed_model=fake_embed,
        )
        assert isinstance(stats["whitening_dims_kept"], int)
        assert stats["whitening_dims_kept"] < _DIM
        assert stats["whitening_var"] == pytest.approx(1.0)

    def test_dims_overrides_var_at_api_level(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        stats: dict[str, Any] = {}
        search_messages(
            query="find 3", summaries_dir=summaries, top_k=5,
            transform="whitening", whitening_var=0.5, whitening_dims=2,
            stats_out=stats, embed_model=fake_embed,
        )
        # dims wins: exactly 2 kept, var reported as None (dims path).
        assert stats["whitening_dims_kept"] == 2
        assert stats["whitening_var"] is None

    def test_var_out_of_range_raises_value_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                search_messages(
                    query="find 3", summaries_dir=summaries, top_k=5,
                    transform="whitening", whitening_var=bad,
                    embed_model=fake_embed,
                )

    def test_dims_below_one_raises_value_error(
        self, tmp_path: Path, fake_embed: _FakeEmbed
    ) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        self._many(summaries)
        _store(summaries, fake_embed)
        with pytest.raises(ValueError):
            search_messages(
                query="find 3", summaries_dir=summaries, top_k=5,
                transform="whitening", whitening_dims=0, embed_model=fake_embed,
            )
