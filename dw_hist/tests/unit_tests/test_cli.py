"""Tests for dw_hist.cli argument parsing & exit-code routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from dw_hist import cli


# -------- argparse / exit-code routing -----------------------------------


def _run(argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
    """Invoke cli.main with capsys-like capture; return (code, stdout, stderr)."""
    import io
    import contextlib

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv, **kwargs)
    return code, out.getvalue(), err.getvalue()


def test_list_with_empty_home_exits_0_no_stdout(tmp_path: Path) -> None:
    code, stdout, _ = _run(["--list"], home=tmp_path)
    assert code == 0
    assert stdout == ""


def test_list_with_two_agents_text_output(tmp_path: Path) -> None:
    for name in ("alpha", "bravo"):
        (tmp_path / Path("~/.deep_worker").name / name / "summaries" / ".history_index" / "lancedb").mkdir(
            parents=True
        )
    code, stdout, _ = _run(["--list"], home=tmp_path)
    assert code == 0
    assert stdout.splitlines() == ["alpha", "bravo"]


def test_list_json_output(tmp_path: Path) -> None:
    (tmp_path / Path("~/.deep_worker").name / "solo" / "summaries" / ".history_index" / "lancedb").mkdir(
        parents=True
    )
    code, stdout, _ = _run(["--list", "--format", "json"], home=tmp_path)
    assert code == 0
    payload = json.loads(stdout)
    assert payload == {
        "agents": [
            {
                "name": "solo",
                "summaries_dir": str(tmp_path / Path("~/.deep_worker").name / "solo" / "summaries"),
            }
        ]
    }


def test_list_with_agent_name_is_usage_error(tmp_path: Path) -> None:
    code, _, stderr = _run(["--list", "alpha"], home=tmp_path)
    assert code == 2
    assert "cannot combine" in stderr.lower() or "conflict" in stderr.lower()


def test_agent_without_query_is_usage_error(tmp_path: Path) -> None:
    code, _, stderr = _run(["alpha"], home=tmp_path)
    assert code == 2
    assert "-q" in stderr or "query" in stderr.lower()


def test_query_without_agent_is_usage_error(tmp_path: Path) -> None:
    code, _, stderr = _run(["-q", "anything"], home=tmp_path)
    assert code == 2
    assert "agent" in stderr.lower()


def test_unknown_agent_exits_2(tmp_path: Path) -> None:
    code, _, stderr = _run(["ghost", "-q", "anything"], home=tmp_path)
    assert code == 2
    assert "ghost" in stderr


def test_agent_without_index_exits_3(tmp_path: Path) -> None:
    (tmp_path / Path("~/.deep_worker").name / "novice" / "summaries").mkdir(parents=True)
    # No .history_index/lancedb/
    code, _, stderr = _run(["novice", "-q", "anything"], home=tmp_path)
    assert code == 3
    assert "no history index" in stderr.lower()


@pytest.mark.parametrize("bad_k", ["0", "-1", "21", "100"])
def test_invalid_top_k_exits_2_before_any_search(tmp_path: Path, bad_k: str) -> None:
    (tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries" / ".history_index" / "lancedb").mkdir(
        parents=True
    )
    called = []

    def _stub(*args: Any, **kwargs: Any) -> list:  # would be called only on success
        called.append((args, kwargs))
        return []

    code, _, stderr = _run(["alpha", "-q", "x", "-k", bad_k], home=tmp_path, search_func=_stub)
    assert code == 2
    assert "top-k" in stderr.lower() or "top_k" in stderr.lower()
    assert called == [], "search_func must not be invoked when --top-k is invalid"


# -------- happy path: stubbed search returning fixed results -------------


def test_search_calls_search_history_and_formats_text(tmp_path: Path) -> None:
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_search(query: str, summaries_dir: str | Path, top_k: int = 5) -> list:
        captured["query"] = query
        captured["summaries_dir"] = str(summaries_dir)
        captured["top_k"] = top_k
        from dataclasses import dataclass

        @dataclass
        class R:
            content: str
            score: float
            thread_id: str
            line_range: str
            timestamp: str
            message_types: str
            file_path: str

        return [
            R("first hit text", 0.876, "t1", "10-15", "2026-01-01T00:00:00", "human,ai", "f1.jsonl"),
            R("second hit text", 0.654, "t2", "20-25", "2026-01-02T00:00:00", "ai", "f2.jsonl"),
        ]

    code, stdout, _ = _run(
        ["alpha", "-q", "what was decided", "-k", "2"],
        home=tmp_path,
        search_func=fake_search,
    )

    assert code == 0
    assert captured == {
        "query": "what was decided",
        "summaries_dir": str(summaries),
        "top_k": 2,
    }
    # score must appear, full precision >= 3 decimals as in history_memory_tool
    assert "score=0.876" in stdout
    assert "score=0.654" in stdout
    assert "first hit text" in stdout
    assert "second hit text" in stdout
    assert "Thread: t1" in stdout


def test_search_json_output(tmp_path: Path) -> None:
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    def fake_search(query: str, summaries_dir: str | Path, top_k: int = 5) -> list:
        from dataclasses import dataclass

        @dataclass
        class R:
            content: str
            score: float
            thread_id: str
            line_range: str
            timestamp: str
            message_types: str
            file_path: str

        return [R("only", 0.5, "t", "1-2", "now", "human", "f.jsonl")]

    code, stdout, _ = _run(
        ["alpha", "-q", "x", "--format", "json"],
        home=tmp_path,
        search_func=fake_search,
    )
    assert code == 0
    payload = json.loads(stdout)
    assert payload["query"] == "x"
    assert payload["agent"] == "alpha"
    assert len(payload["results"]) == 1
    r0 = payload["results"][0]
    assert r0["content"] == "only"
    assert r0["score"] == 0.5
    assert r0["thread_id"] == "t"


def test_search_isolates_stdout_noise_from_json(tmp_path: Path) -> None:
    """A dependency printing to stdout during search (e.g. llama_index's
    "LLM is explicitly disabled. Using MockLLM." banner) must not corrupt
    dw-hist's own stdout — JSON output must stay parseable; the noise goes to
    stderr instead."""
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    def noisy_search(query: str, summaries_dir: str | Path, top_k: int = 5) -> list:
        # Emulate a library writing to the *real* process stdout.
        print("LLM is explicitly disabled. Using MockLLM.")
        from dataclasses import dataclass

        @dataclass
        class R:
            content: str
            score: float
            thread_id: str
            line_range: str
            timestamp: str
            message_types: str
            file_path: str

        return [R("body", 0.5, "t", "1-2", "now", "human", "f.jsonl")]

    code, stdout, stderr = _run(
        ["alpha", "-q", "x", "--format", "json"],
        home=tmp_path,
        search_func=noisy_search,
    )
    assert code == 0
    # stdout must be pure JSON despite the banner the search printed.
    payload = json.loads(stdout)
    assert payload["results"][0]["content"] == "body"
    assert "MockLLM" not in stdout
    # The banner was redirected to stderr.
    assert "MockLLM" in stderr


def test_search_zero_results_exit_0(tmp_path: Path) -> None:
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    def fake_search(*a: Any, **kw: Any) -> list:
        return []

    code, stdout, _ = _run(
        ["alpha", "-q", "nothing"], home=tmp_path, search_func=fake_search
    )
    assert code == 0
    assert "no matching results" in stdout.lower()


def test_search_runtime_error_exits_4(tmp_path: Path) -> None:
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    def fake_search(*a: Any, **kw: Any) -> list:
        raise RuntimeError("lancedb went sideways")

    code, _, stderr = _run(
        ["alpha", "-q", "x"], home=tmp_path, search_func=fake_search
    )
    assert code == 4
    assert "search error" in stderr.lower()
    assert "RuntimeError" in stderr
    assert "lancedb went sideways" in stderr


def test_search_file_not_found_during_search_maps_to_exit_3(tmp_path: Path) -> None:
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    def fake_search(*a: Any, **kw: Any) -> list:
        raise FileNotFoundError("index race")

    code, _, stderr = _run(["alpha", "-q", "x"], home=tmp_path, search_func=fake_search)
    assert code == 3
