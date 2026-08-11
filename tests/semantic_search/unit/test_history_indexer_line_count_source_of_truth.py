"""Single-source-of-truth line counting for incremental indexing.

Regression for the ``U+2028`` desync bug:

``_read_jsonl_lines`` split the file with ``str.splitlines()``, which breaks
not only on ``\\n`` but on ~8 Unicode line boundaries (``\\r``, ``\\v``,
``\\f``, ``U+2028``, ``U+2029`` ...). When a JSONL record's *content* held a
``U+2028``, that single valid record was split into fragments, ``json.loads``
dropped the halves, and the indexer under-counted records.

Meanwhile the startup cheap-check counted physical ``\\n`` lines. The two
counts diverged permanently for such files, so the cheap-check always believed
there were "new" lines and ``_pending_chunks_for_file`` re-indexed the WHOLE
file on every start.

The fix:
  * one shared cheap counter — :func:`count_jsonl_lines` — that counts JSONL
    records by physical ``\\n`` lines (NO JSON parsing), and
  * ``_read_jsonl_lines`` splits on ``\\n`` only, so ``U+2028`` inside a value
    no longer shatters a record, and
  * the incremental comparison is a no-op at equality.
"""

from __future__ import annotations

import json
from pathlib import Path

from deep_worker.providers.semantic_search.history_indexer import (
    _read_jsonl_lines,
    count_jsonl_lines,
)

# U+2028 LINE SEPARATOR embedded inside a JSON *string value*: valid JSON on a
# single physical line, but str.splitlines() treats it as a line break.
U2028 = chr(0x2028)


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    # ensure_ascii=False is REQUIRED: with the default (ensure_ascii=True),
    # json.dumps escapes U+2028 to the 6-char literal "\\u2028", which
    # str.splitlines() does NOT break on — the test would pass even against the
    # buggy splitlines() code (false-green). Real summaries are written with
    # ensure_ascii=False, so the raw U+2028 lands in the file. Writing it raw
    # here is what makes this a genuine RED-without-fix regression guard.
    path.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n",
        encoding="utf-8",
    )


class TestCountJsonlLines:
    def test_counts_physical_newline_records(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        _write_jsonl(f, [{"i": 0}, {"i": 1}, {"i": 2}])
        assert count_jsonl_lines(f) == 3

    def test_u2028_in_value_does_not_inflate_count(self, tmp_path: Path) -> None:
        """A record containing U+2028 is still ONE record."""
        f = tmp_path / "h.jsonl"
        _write_jsonl(
            f,
            [
                {"i": 0, "text": f"before{U2028}after{U2028}end"},
                {"i": 1, "text": "plain"},
            ],
        )
        assert count_jsonl_lines(f) == 2

    def test_stable_across_clean_and_dirty(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.jsonl"
        dirty = tmp_path / "dirty.jsonl"
        _write_jsonl(clean, [{"i": 0, "t": "x"}, {"i": 1, "t": "y"}])
        _write_jsonl(dirty, [{"i": 0, "t": f"x{U2028}{U2028}"}, {"i": 1, "t": "y"}])
        assert count_jsonl_lines(clean) == count_jsonl_lines(dirty) == 2

    def test_ignores_blank_trailing_line(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        f.write_text('{"i": 0}\n{"i": 1}\n', encoding="utf-8")
        assert count_jsonl_lines(f) == 2

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        f.write_text('{"i": 0}\n{"i": 1}', encoding="utf-8")
        assert count_jsonl_lines(f) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert count_jsonl_lines(f) == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        assert count_jsonl_lines(tmp_path / "nope.jsonl") == 0

    def test_agrees_with_read_jsonl_lines_on_valid_file(self, tmp_path: Path) -> None:
        """The cheap counter and the parser MUST agree on a file whose records
        are all valid JSON — even when values contain U+2028."""
        f = tmp_path / "h.jsonl"
        _write_jsonl(
            f,
            [
                {"i": 0, "t": f"a{U2028}b"},
                {"i": 1, "t": "c"},
                {"i": 2, "t": f"d{U2028}e{U2028}f"},
            ],
        )
        assert count_jsonl_lines(f) == len(_read_jsonl_lines(f)) == 3


class TestReadJsonlLinesU2028:
    def test_u2028_does_not_shatter_a_record(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        _write_jsonl(
            f,
            [
                {"i": 0, "text": f"before{U2028}after"},
                {"i": 1, "text": "plain"},
            ],
        )
        records = _read_jsonl_lines(f)
        assert len(records) == 2
        assert records[0]["i"] == 0
        assert records[0]["text"] == f"before{U2028}after"
        assert records[1]["i"] == 1

    def test_still_skips_genuinely_bad_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "h.jsonl"
        f.write_text(
            '{"a": 1}\nnot json\n{"b": 2}\n',
            encoding="utf-8",
        )
        records = _read_jsonl_lines(f)
        assert len(records) == 2
