"""Unit tests for deep_worker_cli.summaries — the file-based offloaded
conversation-history inventory & pruning helpers.

Design: deep_worker/docs/DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md

Layout under test:
    $HOME/.deep_worker/<agent_name>/summaries/<thread_id>.md
    each file contains one or more "## Summarized at <iso-ts>\n\n<body>\n\n"
    sections, appended chronologically.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


from deep_worker_cli.storage_paths import deep_worker_home
# The functions under test (do not exist yet — these are the RED pin).
from deep_worker_cli.summaries import (
    SummaryFileInfo,
    get_summaries_dir,
    iter_summary_files,
    parse_summary_headers,
    prune_summaries_before,
)


def _write_summary(path: Path, timestamps: list[str], body_per_ts: str = "body") -> None:
    """Create a summary .md file with N sections at the given timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"## Summarized at {ts}\n\n{body_per_ts}\n\n" for ts in timestamps]
    path.write_text("".join(parts))


class TestGetSummariesDir:
    def test_returns_home_deep_worker_agent_summaries(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Post-v6: agent summaries live under ``agents/<name>/summaries/``
        # (DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §3).
        assert get_summaries_dir("foo") == deep_worker_home(tmp_path) / "agents" / "foo" / "summaries"

    def test_does_not_create_directory_on_its_own(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        d = get_summaries_dir("foo")
        # The function is pure — does NOT materialize the dir. Callers do.
        assert not d.exists()


class TestIterSummaryFiles:
    def test_empty_home_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert list(iter_summary_files()) == []

    def test_lists_one_agent_one_thread(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "agent_1" / "summaries" / "agent_1.md"
        _write_summary(f, ["2026-04-22T15:40:18.122420+00:00"])
        items = list(iter_summary_files())
        assert len(items) == 1
        info = items[0]
        assert isinstance(info, SummaryFileInfo)
        assert info.agent == "agent_1"
        assert info.thread == "agent_1"
        assert info.path == f
        assert info.events == 1
        assert info.size > 0
        assert info.latest_ts == "2026-04-22T15:40:18.122420+00:00"

    def test_lists_many_agents_and_threads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_summary(
            deep_worker_home(tmp_path) / "a1" / "summaries" / "a1.md",
            ["2026-04-01T00:00:00+00:00"],
        )
        _write_summary(
            deep_worker_home(tmp_path) / "a1" / "summaries" / "a1:tm.md",
            ["2026-04-02T00:00:00+00:00", "2026-04-03T00:00:00+00:00"],
        )
        _write_summary(
            deep_worker_home(tmp_path) / "b2" / "summaries" / "b2.md",
            ["2026-04-04T00:00:00+00:00"],
        )
        items = sorted(list(iter_summary_files()), key=lambda i: (i.agent, i.thread))
        assert [(i.agent, i.thread, i.events) for i in items] == [
            ("a1", "a1", 1),
            ("a1", "a1:tm", 2),
            ("b2", "b2", 1),
        ]
        tm = next(i for i in items if i.thread == "a1:tm")
        assert tm.latest_ts == "2026-04-03T00:00:00+00:00"

    def test_filter_by_agent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_summary(deep_worker_home(tmp_path) / "a1" / "summaries" / "a1.md", ["2026-04-01T00:00:00+00:00"])
        _write_summary(deep_worker_home(tmp_path) / "b2" / "summaries" / "b2.md", ["2026-04-02T00:00:00+00:00"])
        items = list(iter_summary_files(agent_name="a1"))
        assert len(items) == 1
        assert items[0].agent == "a1"

    def test_silently_skips_non_md_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        d = deep_worker_home(tmp_path) / "a" / "summaries"
        d.mkdir(parents=True)
        (d / "a.md").write_text("## Summarized at 2026-04-01T00:00:00+00:00\n\nbody\n\n")
        (d / "stray.txt").write_text("noise")
        (d / "README").write_text("ignored")
        items = list(iter_summary_files())
        assert len(items) == 1
        assert items[0].thread == "a"

    def test_file_with_no_headers_reports_zero_events_none_latest(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        f.parent.mkdir(parents=True)
        f.write_text("# not-a-summary\n\ncorrupted\n")
        items = list(iter_summary_files())
        assert len(items) == 1
        assert items[0].events == 0
        assert items[0].latest_ts is None


class TestParseSummaryHeaders:
    def test_empty_text(self):
        assert parse_summary_headers("") == []

    def test_single_header(self):
        txt = "## Summarized at 2026-04-01T00:00:00+00:00\n\nbody\n\n"
        assert parse_summary_headers(txt) == ["2026-04-01T00:00:00+00:00"]

    def test_many_headers_order_preserved(self):
        txt = (
            "## Summarized at 2026-04-01T00:00:00+00:00\n\na\n\n"
            "## Summarized at 2026-04-02T00:00:00+00:00\n\nb\n\n"
            "## Summarized at 2026-04-03T00:00:00+00:00\n\nc\n\n"
        )
        assert parse_summary_headers(txt) == [
            "2026-04-01T00:00:00+00:00",
            "2026-04-02T00:00:00+00:00",
            "2026-04-03T00:00:00+00:00",
        ]

    def test_ignores_non_header_lines(self):
        txt = "random\n## Summarized at 2026-04-01T00:00:00+00:00\n\nbody\n\n## Other\n"
        assert parse_summary_headers(txt) == ["2026-04-01T00:00:00+00:00"]


class TestPruneSummariesBefore:
    def test_dry_run_does_not_write(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _write_summary(f, ["2026-01-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00"])
        original = f.read_text()
        cutoff = datetime(2026, 3, 1, tzinfo=UTC)
        result = prune_summaries_before(cutoff, dry_run=True)
        # state untouched
        assert f.read_text() == original
        # result reports what would happen
        assert result["files_scanned"] >= 1
        assert result["sections_to_drop"] >= 1

    def test_deletes_file_when_all_sections_older(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _write_summary(f, ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"])
        cutoff = datetime(2026, 3, 1, tzinfo=UTC)
        result = prune_summaries_before(cutoff, dry_run=False)
        assert not f.exists()
        assert result["files_deleted"] == 1

    def test_rewrites_file_keeping_only_newer_sections(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _write_summary(
            f,
            [
                "2026-01-01T00:00:00+00:00",
                "2026-02-01T00:00:00+00:00",
                "2026-05-01T00:00:00+00:00",
                "2026-06-01T00:00:00+00:00",
            ],
            body_per_ts="BODY",
        )
        cutoff = datetime(2026, 3, 1, tzinfo=UTC)
        f.chmod(0o644)
        result = prune_summaries_before(cutoff, dry_run=False)
        assert f.exists(), "file kept because some sections are newer"
        assert oct(f.stat().st_mode & 0o777) == "0o600"
        remaining = f.read_text()
        assert "2026-01-01T00:00:00+00:00" not in remaining
        assert "2026-02-01T00:00:00+00:00" not in remaining
        assert "2026-05-01T00:00:00+00:00" in remaining
        assert "2026-06-01T00:00:00+00:00" in remaining
        # result bookkeeping
        assert result["files_rewritten"] == 1
        assert result["sections_deleted"] == 2

    def test_no_op_when_all_sections_newer(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _write_summary(f, ["2026-05-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"])
        original = f.read_text()
        cutoff = datetime(2026, 3, 1, tzinfo=UTC)
        result = prune_summaries_before(cutoff, dry_run=False)
        assert f.read_text() == original
        assert result["files_rewritten"] == 0
        assert result["files_deleted"] == 0

    def test_atomic_rewrite_via_tmp_then_replace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = deep_worker_home(tmp_path) / "a" / "summaries" / "a.md"
        _write_summary(
            f,
            ["2026-01-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00"],
        )
        seen_replace: list[tuple[str, str]] = []
        real_replace = __import__("os").replace
        import os as _os
        def _spy_replace(src, dst):
            seen_replace.append((str(src), str(dst)))
            return real_replace(src, dst)
        monkeypatch.setattr(_os, "replace", _spy_replace)
        cutoff = datetime(2026, 3, 1, tzinfo=UTC)
        prune_summaries_before(cutoff, dry_run=False)
        # Exactly one rewrite → one os.replace call
        assert len(seen_replace) == 1
        src, dst = seen_replace[0]
        assert src.endswith(".md.tmp")
        assert dst == str(f)
