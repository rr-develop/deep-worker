"""Engine-side contracts that do not need the heavy stack to verify."""

from __future__ import annotations

import json

import pytest

from deep_worker_cli.semantic_memory.service import engine

# The pending-line contracts below call into the engine, which resolves the
# semantic-search provider and raises ProviderUnavailable when the optional
# extra is absent. Skip rather than fail: the extra is not part of a plain
# `pip install -e .`. Same convention as the rest of tests/unit_tests.
pytest.importorskip(
    "llama_index", reason="semantic-search extra is absent"
)  # semantic-search dep; skip when absent


def _make_history_index(summaries_dir, indexed_counts):
    """Build a history index sidecar the way the indexer really writes it.

    Counts are keyed by the ABSOLUTE path (``line_counts[str(jsonl_path)]``
    in ``history_indexer``), not by filename. An earlier version of this
    helper keyed by name; every lookup missed, so the engine reported all
    records pending and the tests passed against that bug because they
    shared the mistake. Keys are built here from the same paths the
    engine will resolve, so a divergence shows up as a failure.
    """
    index = summaries_dir / ".history_index"
    (index / "lancedb").mkdir(parents=True)
    (index / "docstore").mkdir(parents=True)
    (index / "docstore" / "docstore.json").write_text("{}")
    absolute = {
        str(summaries_dir / name): count for name, count in indexed_counts.items()
    }
    (index / "line_hashes.json").write_text(json.dumps(absolute))


class TestIndexPresence:
    def test_partial_project_index_is_not_an_index(self, tmp_path):
        """A bare directory is not a usable index.

        ``RepoIndex.exists()`` checks only that ``.vector_index/`` is
        there and would call a half-written index usable; the service
        commits to all three components (design §11.3).
        """
        (tmp_path / ".vector_index").mkdir()
        assert engine.project_index_exists(tmp_path) is False

    def test_complete_project_index_is_recognized(self, tmp_path):
        index = tmp_path / ".vector_index"
        (index / "lancedb").mkdir(parents=True)
        (index / "docstore").mkdir(parents=True)
        (index / "docstore" / "docstore.json").write_text("{}")
        (index / "file_hashes.json").write_text("{}")
        assert engine.project_index_exists(tmp_path) is True


class TestPendingLines:
    """The count must agree with the indexer's, exactly."""

    def test_unicode_separators_do_not_inflate_the_count(self, tmp_path):
        """JSONL separates records on ``\\n`` alone.

        ``str.splitlines()`` also breaks on U+2028/U+2029, which appear
        legitimately inside JSON string values. A record containing one
        would be counted as two, so ``pending`` would never reach zero
        and the file would be re-indexed forever — the failure the
        indexer's own ``JSONL_RECORD_SEPARATOR`` rule exists to prevent.
        """
        summaries = tmp_path
        # ensure_ascii=False keeps the separators RAW in the file. The
        # default escapes them to "\\u2028"/"\\u2029", and escaped ones
        # cannot reproduce the bug: the test would pass against a
        # splitlines() implementation and prove nothing.
        record = json.dumps({"text": "para graph break"}, ensure_ascii=False)
        assert " " in record and " " in record, "separators must stay raw"
        (summaries / "t.jsonl").write_text(record + "\n", encoding="utf-8")
        _make_history_index(summaries, {"t.jsonl": 1})

        assert engine.history_pending_lines(summaries) == 0

    def test_new_records_are_reported_as_pending(self, tmp_path):
        summaries = tmp_path
        (summaries / "t.jsonl").write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
        _make_history_index(summaries, {"t.jsonl": 1})
        assert engine.history_pending_lines(summaries) == 2

    def test_missing_sidecar_counts_everything_as_pending(self, tmp_path):
        summaries = tmp_path
        (summaries / "t.jsonl").write_text('{"a":1}\n{"a":2}\n')
        _make_history_index(summaries, {})
        assert engine.history_pending_lines(summaries) == 2


class TestListTenants:
    def test_project_roots_are_discoverable(self, tmp_path, monkeypatch):
        """Discovering an existing index is not auto-indexing.

        §16 A6 forbids creating indexes for repositories; it does not
        forbid naming the ones already on disk. Returning nothing would
        leave the "name a target explicitly" hint with no names in it.
        """
        indexed = tmp_path / "with_index"
        (indexed / ".vector_index" / "lancedb").mkdir(parents=True)
        (indexed / ".vector_index" / "docstore").mkdir(parents=True)
        (indexed / ".vector_index" / "docstore" / "docstore.json").write_text("{}")
        (indexed / ".vector_index" / "file_hashes.json").write_text("{}")
        (tmp_path / "without_index").mkdir()

        monkeypatch.setenv(engine.PROJECT_SEARCH_ROOTS_ENV, str(tmp_path))
        tenants = engine.list_tenants(project=True)

        assert tenants == [str(indexed)]

    def test_unreadable_root_is_skipped_not_fatal(self, monkeypatch, tmp_path):
        monkeypatch.setenv(engine.PROJECT_SEARCH_ROOTS_ENV, str(tmp_path / "nope"))
        assert engine.list_tenants(project=True) == []


class TestTopKValidation:
    @pytest.mark.parametrize("bad", [0, 21, -1])
    def test_out_of_range_top_k_is_rejected_before_the_model_loads(self, bad, tmp_path):
        """Rejected up front: loading a 884 MB model to then refuse the
        request would be the expensive way to say no."""
        assert "top_k must be between" in engine.search_project("q", tmp_path, bad)
        assert "top_k must be between" in engine.search_history("q", tmp_path, bad)
