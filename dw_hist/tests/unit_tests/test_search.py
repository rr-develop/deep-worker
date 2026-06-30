"""Tests for the formatting helpers in dw_hist.search."""

from __future__ import annotations

from dataclasses import dataclass

from dw_hist import search


@dataclass
class _R:
    content: str
    score: float
    thread_id: str
    line_range: str
    timestamp: str
    message_types: str
    file_path: str


def test_format_text_matches_history_memory_tool_shape() -> None:
    """Parity test (RC-F): byte-shape must mirror history_memory_tool output."""
    results = [
        _R("hello world", 0.876, "t1", "10-15",
           "2026-01-01T00:00:00", "human,ai", "f1.jsonl"),
    ]
    out = search.format_text(query="q", results=results)

    assert out.startswith("History Memory: 'q'\n")
    assert "--- Result 1 (score=0.876) ---" in out
    assert "Thread: t1" in out
    assert "Lines: 10-15" in out
    assert "When: 2026-01-01T00:00:00" in out
    assert "Types: human,ai" in out
    assert "hello world" in out


def test_format_text_empty_results() -> None:
    out = search.format_text(query="missing", results=[])
    assert "missing" in out
    assert "no matching results" in out.lower()


def test_format_json_shape() -> None:
    results = [
        _R("c", 0.5, "t", "1-2", "now", "human", "f.jsonl"),
        _R("c2", 0.4, "t2", "3-4", "later", "ai", "f2.jsonl"),
    ]
    payload = search.format_json(agent="alpha", query="q", results=results)
    assert payload["agent"] == "alpha"
    assert payload["query"] == "q"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["score"] == 0.5
    assert payload["results"][1]["thread_id"] == "t2"
