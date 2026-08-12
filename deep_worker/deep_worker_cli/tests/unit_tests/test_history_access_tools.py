"""Unit tests for history access tools (history_range, history_at, history_slice).

Tests cover:
- Missing/empty files
- Normal files with markdown sections
- Edge cases (boundary timestamps, out-of-range indices, negative indices)
- Middleware registration
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from deep_worker_cli.history_access_tool import history_at, history_range, history_slice
from deep_worker_cli.middleware.history_access import HistoryAccessMiddleware


from deep_worker_cli.storage_paths import deep_worker_home
# ── Helpers ──────────────────────────────────────────────────────────────

def _make_section(timestamp: str, body: str) -> str:
    """Create a single markdown section as the middleware writes it."""
    return f"## Summarized at {timestamp}\n\n{body}\n"


def _make_md(sections: str) -> str:
    """Combine sections into a full summary file."""
    return sections.strip() + "\n"


def _write_summary(base: Path, agent: str, thread: str, content: str) -> Path:
    """Write a summary .md file under deep_worker_home(base)/<agent>/summaries/<thread>.md."""
    d = deep_worker_home(base) / agent / "summaries"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    p = d / f"{thread}.md"
    p.write_text(content, encoding="utf-8")
    os.chmod(p, 0o600)
    return p


def _make_runtime(
    assistant_id: str,
    thread_id: str,
    home: Path,
) -> MagicMock:
    """Create a mock runtime with state set by HistoryAccessMiddleware.before_agent()."""
    runtime = MagicMock()
    type(runtime).state = PropertyMock(
        return_value={
            "history_access_assistant_id": assistant_id,
            "history_access_home": str(home),
        }
    )
    type(runtime).config = PropertyMock(
        return_value={
            "configurable": {"thread_id": thread_id},
        }
    )
    return runtime


SECTIONS_4 = (
    _make_section("2026-01-15T10:30:00+00:00", "First message hello")
    + _make_section("2026-02-10T12:00:00+00:00", "Second message world")
    + _make_section("2026-03-20T14:00:00+00:00", "Third message foo")
    + _make_section("2026-04-22T15:40:18+00:00", "Fourth message bar")
)


# ── Test history_range ───────────────────────────────────────────────────

class TestHistoryRange:
    """Tests for history_range tool."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """No file at all returns earliest=None, latest=None, count=0."""
        runtime = _make_runtime("missing_agent", "th1", tmp_path)
        result = json.loads(history_range.func(runtime=runtime))
        assert result == {"earliest": None, "latest": None, "count": 0}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns earliest=None, latest=None, count=0."""
        _write_summary(tmp_path, "agent1", "thread1", "")
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_range.func(runtime=runtime))
        assert result == {"earliest": None, "latest": None, "count": 0}

    def test_no_runtime(self, tmp_path: Path) -> None:
        """No runtime returns empty result (can't resolve path)."""
        result = json.loads(history_range.func(runtime=None))
        assert result == {"earliest": None, "latest": None, "count": 0}

    def test_single_section(self, tmp_path: Path) -> None:
        """Single section returns its timestamp as both earliest and latest."""
        content = _make_md(
            _make_section("2026-01-15T10:30:00+00:00", "Hello world")
        )
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_range.func(runtime=runtime))
        assert result["earliest"] == "2026-01-15T10:30:00+00:00"
        assert result["latest"] == "2026-01-15T10:30:00+00:00"
        assert result["count"] == 1

    def test_multiple_sections(self, tmp_path: Path) -> None:
        """Multiple sections returns correct range and count."""
        content = _make_md(
            _make_section("2026-01-15T10:30:00+00:00", "First")
            + _make_section("2026-03-20T14:00:00+00:00", "Second")
            + _make_section("2026-04-22T15:40:18+00:00", "Third")
        )
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_range.func(runtime=runtime))
        assert result["earliest"] == "2026-01-15T10:30:00+00:00"
        assert result["latest"] == "2026-04-22T15:40:18+00:00"
        assert result["count"] == 3

    def test_file_without_headers(self, tmp_path: Path) -> None:
        """File with no ## Summarized at headers returns empty range."""
        _write_summary(tmp_path, "agent1", "thread1", "some plain text\nno headers\n")
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_range.func(runtime=runtime))
        assert result == {"earliest": None, "latest": None, "count": 0}


# ── Test history_at ─────────────────────────────────────────────────────

class TestHistoryAt:
    """Tests for history_at tool."""

    SECTIONS = SECTIONS_4  # 4 sections A(Jan), B(Feb), C(Mar), D(Apr)

    def test_missing_file(self, tmp_path: Path) -> None:
        """No file returns empty result."""
        runtime = _make_runtime("no_agent", "th1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2026-03-20T14:00:00+00:00",
            before=1,
            at_or_after=2,
            runtime=runtime,
        ))
        assert result["messages"] == []
        assert result["indices"]["start"] == 0

    def test_timestamp_in_middle(self, tmp_path: Path) -> None:
        """Timestamp that matches a section header returns correct surrounding."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2026-03-20T14:00:00+00:00",
            before=1,
            at_or_after=2,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 3  # B, C, D
        assert result["indices"]["start"] == 1
        assert result["indices"]["end"] == 4

    def test_timestamp_before_all(self, tmp_path: Path) -> None:
        """Timestamp before all sections returns only at_or_after."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2025-01-01T00:00:00+00:00",
            before=1,
            at_or_after=2,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 2  # A, B (first two)
        assert result["indices"]["start"] == 0
        assert result["indices"]["end"] == 2

    def test_timestamp_after_all(self, tmp_path: Path) -> None:
        """Timestamp after all sections returns only before entries."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2027-01-01T00:00:00+00:00",
            before=2,
            at_or_after=2,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 2  # C, D (last two before)
        assert result["indices"]["start"] == 2
        assert result["indices"]["end"] == 4

    def test_before_zero(self, tmp_path: Path) -> None:
        """before=0 returns only at_or_after entries."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2026-03-20T14:00:00+00:00",
            before=0,
            at_or_after=1,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 1  # C only
        assert result["indices"]["start"] == 2
        assert result["indices"]["end"] == 3

    def test_at_or_after_zero(self, tmp_path: Path) -> None:
        """at_or_after=0 returns only before entries."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2026-03-20T14:00:00+00:00",
            before=2,
            at_or_after=0,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 2  # A, B
        assert result["indices"]["start"] == 0
        assert result["indices"]["end"] == 2

    def test_malformed_timestamp(self, tmp_path: Path) -> None:
        """Malformed timestamp returns empty result."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="not-a-timestamp",
            before=1,
            at_or_after=1,
            runtime=runtime,
        ))
        assert result["messages"] == []

    def test_no_runtime(self, tmp_path: Path) -> None:
        """No runtime returns empty result."""
        result = json.loads(history_at.func(
            timestamp="2026-03-20T14:00:00+00:00",
            before=1,
            at_or_after=1,
            runtime=None,
        ))
        assert result["messages"] == []

    def test_timezone_z_suffix(self, tmp_path: Path) -> None:
        """Timestamp with Z suffix is handled correctly."""
        content = _make_md(
            _make_section("2026-03-20T14:00:00Z", "Z section")
        )
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_at.func(
            timestamp="2026-03-20T13:59:00Z",
            before=0,
            at_or_after=1,
            runtime=runtime,
        ))
        assert len(result["messages"]) == 1
        assert result["messages"][0]["summarized_at"] == "2026-03-20T14:00:00Z"


# ── Test history_slice ───────────────────────────────────────────────────

class TestHistorySlice:
    """Tests for history_slice tool."""

    SECTIONS = SECTIONS_4  # 4 sections: A(index 0), B(1), C(2), D(3)

    def test_missing_file(self, tmp_path: Path) -> None:
        """No file returns empty result."""
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=0, end=5, runtime=runtime))
        assert result["messages"] == []

    def test_normal_slice(self, tmp_path: Path) -> None:
        """Normal slice returns correct messages and indices."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=1, end=3, runtime=runtime))
        assert len(result["messages"]) == 2
        assert result["indices"]["start"] == 1
        assert result["indices"]["end"] == 3

    def test_start_zero(self, tmp_path: Path) -> None:
        """start=0 returns from the beginning."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=0, end=2, runtime=runtime))
        assert len(result["messages"]) == 2
        assert result["indices"]["start"] == 0
        assert result["indices"]["end"] == 2

    def test_end_beyond_count(self, tmp_path: Path) -> None:
        """end > count clamps to count and returns what's available."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=3, end=999, runtime=runtime))
        assert len(result["messages"]) == 1  # Only D
        assert result["indices"]["start"] == 3
        assert result["indices"]["end"] == 4

    def test_start_beyond_count(self, tmp_path: Path) -> None:
        """start >= count returns empty."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=10, end=20, runtime=runtime))
        assert result["messages"] == []

    def test_end_less_than_start(self, tmp_path: Path) -> None:
        """end <= start returns empty."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=3, end=1, runtime=runtime))
        assert result["messages"] == []

    def test_negative_start(self, tmp_path: Path) -> None:
        """Negative start clamped to 0."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=-5, end=2, runtime=runtime))
        assert len(result["messages"]) == 2
        assert result["indices"]["start"] == 0
        assert result["indices"]["end"] == 2

    def test_negative_end(self, tmp_path: Path) -> None:
        """Negative end clamped to 0 (result is empty because end <= start)."""
        content = _make_md(self.SECTIONS)
        _write_summary(tmp_path, "agent1", "thread1", content)
        runtime = _make_runtime("agent1", "thread1", tmp_path)
        result = json.loads(history_slice.func(start=0, end=-1, runtime=runtime))
        assert result["messages"] == []

    def test_no_runtime(self, tmp_path: Path) -> None:
        """No runtime returns empty result."""
        result = json.loads(history_slice.func(start=0, end=5, runtime=None))
        assert result["messages"] == []


# ── Test HistoryAccessMiddleware ─────────────────────────────────────────

class TestHistoryAccessMiddleware:
    """Tests for HistoryAccessMiddleware itself."""

    def test_tools_registered(self) -> None:
        """Middleware registers all 3 tools."""
        mw = HistoryAccessMiddleware(assistant_id="test_agent")
        tool_names = {t.name for t in mw.tools}
        assert tool_names == {"history_range", "history_at", "history_slice"}

    def test_before_agent_sets_state(self) -> None:
        """before_agent returns correct state dict."""
        mw = HistoryAccessMiddleware(assistant_id="test_agent")
        state = mw.before_agent(state={}, runtime=None)
        assert state is not None
        assert state["history_access_assistant_id"] == "test_agent"
        assert "history_access_home" in state
