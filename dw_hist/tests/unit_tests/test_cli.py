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


def _make_noisy_search() -> Any:
    """Build a fake search_func that splashes the same kind of noise on BOTH
    stdout and stderr that the real semantic_search + transformers + llama_index
    stack produces on the first index load (see DESIGN §5.2)."""
    import sys as _sys

    def noisy_search(query: str, summaries_dir: str | Path, top_k: int = 5) -> list:
        # llama_index's bare print() banner — goes to stdout.
        print("LLM is explicitly disabled. Using MockLLM.")
        # transformers / HF unauth warning — goes to stderr.
        print("Warning: You are sending unauthenticated requests to the HF Hub.",
              file=_sys.stderr)
        # tqdm-style progress bar — goes to stderr.
        print("Loading weights: 100%|####| 199/199 [00:00<00:00, 3000it/s]",
              file=_sys.stderr)
        # BertModel LOAD REPORT — goes to stderr.
        print("BertModel LOAD REPORT from: intfloat/multilingual-e5-small",
              file=_sys.stderr)
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

    return noisy_search


def test_search_silences_both_streams_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode: third-party noise on BOTH stdout and stderr (HF warning,
    tqdm progress bar, BertModel load report, llama_index MockLLM banner) must
    NOT leak through. stdout carries only dw-hist's formatted output and is
    JSON-parseable; stderr stays empty."""
    monkeypatch.delenv("DW_HIST_VERBOSE", raising=False)
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    code, stdout, stderr = _run(
        ["alpha", "-q", "x", "--format", "json"],
        home=tmp_path,
        search_func=_make_noisy_search(),
    )
    assert code == 0
    # stdout is pure JSON.
    payload = json.loads(stdout)
    assert payload["results"][0]["content"] == "body"
    # No noise leaked into either of OUR streams.
    for needle in (
        "MockLLM",
        "HF Hub",
        "Loading weights",
        "BertModel LOAD REPORT",
    ):
        assert needle not in stdout, f"{needle!r} leaked to stdout"
        assert needle not in stderr, f"{needle!r} leaked to stderr"


def test_search_verbose_lets_noise_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --verbose the operator opts in to seeing the third-party noise on
    stderr (useful for debugging cold-cache / model-load issues). stdout is
    still our formatted output."""
    # Make sure the env-var that --verbose sets does not leak out of this test.
    monkeypatch.delenv("DW_HIST_VERBOSE", raising=False)
    summaries = tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries"
    (summaries / ".history_index" / "lancedb").mkdir(parents=True)

    code, stdout, stderr = _run(
        ["alpha", "-q", "x", "--verbose"],
        home=tmp_path,
        search_func=_make_noisy_search(),
    )
    assert code == 0
    assert "Thread: t" in stdout
    # Library noise is visible — stderr-bound noise stays on stderr;
    # stdout-bound noise (the MockLLM banner) lands on stdout because
    # --verbose disables the redirect.
    assert "HF Hub" in stderr
    assert "Loading weights" in stderr
    assert "BertModel LOAD REPORT" in stderr
    assert "MockLLM" in stdout


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


# -------- -qm / --flag (per-message embedding search) --------------------


def _make_message_store(home: Path, agent: str = "alpha") -> Path:
    """Create the minimal per-message store layout the -qm pre-check requires."""
    summaries = home / Path("~/.deep_worker").name / agent / "summaries"
    store = summaries / ".message_embeddings"
    store.mkdir(parents=True)
    (store / "embeddings.npy").write_bytes(b"\x00")
    (store / "flags.npy").write_bytes(b"\x00")
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    return summaries


def _fake_msg_result(**over: Any) -> Any:
    from dataclasses import dataclass, field

    @dataclass
    class R:
        content: str = "msg body"
        score: float = 0.812
        thread_id: str = "alpha"
        line_range: str = "1187"
        timestamp: str = "2026-01-01T00:00:00"
        message_types: str = "human"
        file_path: str = "alpha.jsonl"
        flags: int = 3
        flag_names: list[str] = field(default_factory=lambda: ["USER_ROLE", "TASK_MANAGER"])

    return R(**over)


def test_q_and_qm_mutually_exclusive_exit_2(tmp_path: Path) -> None:
    code, _, stderr = _run(["alpha", "-q", "a", "-qm", "b"], home=tmp_path)
    assert code == 2
    assert "mutually exclusive" in stderr.lower()


def test_flag_without_qm_exit_2(tmp_path: Path) -> None:
    code, _, stderr = _run(["alpha", "-q", "a", "-f", "AGENT"], home=tmp_path)
    assert code == 2
    assert "-f" in stderr or "flag" in stderr.lower()


def test_qm_without_agent_exit_2(tmp_path: Path) -> None:
    code, _, stderr = _run(["-qm", "anything"], home=tmp_path)
    assert code == 2
    assert "agent" in stderr.lower()


def test_qm_empty_query_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: list[Any] = []

    def stub(*a: Any, **kw: Any) -> list:
        captured.append(kw)
        return []

    code, _, stderr = _run(
        ["alpha", "-qm", "   "], home=tmp_path, search_messages_func=stub
    )
    assert code == 2
    assert "non-empty" in stderr.lower()
    assert captured == [], "search must not run for an empty query"


def test_qm_unknown_flag_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-f", "NOT_A_FLAG"], home=tmp_path,
        search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "unknown flag" in stderr.lower()


def test_qm_without_store_exit_3(tmp_path: Path) -> None:
    # Agent root exists but no .message_embeddings store.
    (tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries").mkdir(parents=True)
    code, _, stderr = _run(
        ["alpha", "-qm", "x"], home=tmp_path, search_messages_func=lambda **kw: []
    )
    assert code == 3
    assert "per-message embedding store" in stderr.lower()


def test_qm_invalid_top_k_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-k", "0"], home=tmp_path,
        search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "top-k" in stderr.lower() or "top_k" in stderr.lower()


def test_qm_happy_path_text_includes_flags(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: dict[str, Any] = {}

    def stub(**kw: Any) -> list:
        # Accept any extra kwargs (transform/whitening_*/stats_scope/stats_out)
        # the CLI forwards for the isotropy modes; capture only what we assert on.
        captured.update(
            query=kw["query"], summaries_dir=str(kw["summaries_dir"]),
            top_k=kw["top_k"], flags_any=kw["flags_any"],
        )
        return [_fake_msg_result()]

    code, stdout, _ = _run(
        ["alpha", "-qm", "deadlock", "-k", "3", "-f", "USER_ROLE", "-f", "task_manager"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert captured["query"] == "deadlock"
    assert captured["top_k"] == 3
    # USER_ROLE (1) | TASK_MANAGER (2) == 3; case-insensitive resolution.
    assert captured["flags_any"] == 3
    assert "score=0.812" in stdout
    assert "Flags: USER_ROLE,TASK_MANAGER" in stdout
    assert "msg body" in stdout


def test_qm_json_includes_flags(tmp_path: Path) -> None:
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        return [_fake_msg_result()]

    code, stdout, _ = _run(
        ["alpha", "-qm", "x", "--format", "json"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    payload = json.loads(stdout)
    r0 = payload["results"][0]
    assert r0["flags"] == 3
    assert r0["flag_names"] == ["USER_ROLE", "TASK_MANAGER"]
    assert r0["score"] == 0.812


def test_qm_no_flag_filter_passes_zero_mask(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: dict[str, Any] = {}

    def stub(**kw: Any) -> list:
        captured.update(kw)
        return []

    code, _, _ = _run(
        ["alpha", "-qm", "x"], home=tmp_path, search_messages_func=stub
    )
    assert code == 0
    assert captured["flags_any"] == 0


def test_qm_runtime_error_exit_4(tmp_path: Path) -> None:
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        raise RuntimeError("store drift")

    code, _, stderr = _run(
        ["alpha", "-qm", "x"], home=tmp_path, search_messages_func=stub
    )
    assert code == 4
    assert "store drift" in stderr


# ============================================================================
# -qm isotropy transforms: -c / -w / --whitening-method / --whitening-eps /
# --stats-scope / --score-stats  (see docs/DESIGN_DW_HIST_QM_ISOTROPY.md §4)
# ============================================================================


def test_isotropy_c_and_w_mutually_exclusive(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-c", "-w"],
        home=tmp_path, search_messages_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "mutually exclusive" in stderr.lower()
    assert called == [], "search_messages must not be invoked on usage error"


def test_isotropy_c_without_qm_exit_2(tmp_path: Path) -> None:
    (tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries").mkdir(parents=True)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-q", "anything", "-c"],
        home=tmp_path,
        search_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "-qm" in stderr


def test_isotropy_w_without_qm_exit_2(tmp_path: Path) -> None:
    (tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries").mkdir(parents=True)
    code, _, stderr = _run(
        ["alpha", "-q", "x", "-w"], home=tmp_path,
        search_func=lambda **kw: [],
    )
    assert code == 2
    assert "-qm" in stderr


def test_whitening_method_without_w_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "--whitening-method", "pca"],
        home=tmp_path, search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "whitening-method" in stderr.lower() or "-w" in stderr


def test_whitening_eps_without_w_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "--whitening-eps", "1e-2"],
        home=tmp_path, search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "whitening-eps" in stderr.lower() or "-w" in stderr


def test_stats_scope_without_c_or_w_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "--stats-scope", "filtered"],
        home=tmp_path, search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "stats-scope" in stderr.lower()


def test_whitening_eps_non_positive_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "--whitening-eps", "0"],
        home=tmp_path,
        search_messages_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "> 0" in stderr or "positive" in stderr.lower()
    assert called == []


def test_score_stats_without_qm_exit_2(tmp_path: Path) -> None:
    (tmp_path / Path("~/.deep_worker").name / "alpha" / "summaries").mkdir(parents=True)
    code, _, stderr = _run(
        ["alpha", "-q", "x", "--score-stats"], home=tmp_path,
        search_func=lambda **kw: [],
    )
    assert code == 2
    assert "score-stats" in stderr.lower() or "-qm" in stderr


def test_centering_forwards_transform_param(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: dict[str, Any] = {}

    def stub(**kw: Any) -> list:
        captured.update(kw)
        return [_fake_msg_result()]

    code, _, _ = _run(
        ["alpha", "-qm", "x", "-c"], home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert captured["transform"] == "centering"
    # Defaults for the other isotropy args are forwarded too.
    assert captured["stats_scope"] == "corpus"


def test_whitening_forwards_method_eps_scope(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: dict[str, Any] = {}

    def stub(**kw: Any) -> list:
        captured.update(kw)
        return [_fake_msg_result()]

    code, _, _ = _run(
        ["alpha", "-qm", "x", "-w",
         "--whitening-method", "pca",
         "--whitening-eps", "5e-3",
         "--stats-scope", "filtered"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert captured["transform"] == "whitening"
    assert captured["whitening_method"] == "pca"
    assert captured["whitening_eps"] == pytest.approx(5e-3)
    assert captured["stats_scope"] == "filtered"


def test_raw_default_forwards_transform_raw(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    captured: dict[str, Any] = {}

    def stub(**kw: Any) -> list:
        captured.update(kw)
        return [_fake_msg_result()]

    code, _, _ = _run(
        ["alpha", "-qm", "x"], home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert captured["transform"] == "raw"


def test_score_stats_prints_to_stderr_and_stdout_stays_json(tmp_path: Path) -> None:
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        so = kw["stats_out"]
        assert so is not None, "CLI must allocate stats_out when --score-stats"
        so.update(m=2492, min=-0.167, median=0.014, max=0.265,
                  range=0.432, transform="centering")
        return [_fake_msg_result()]

    code, stdout, stderr = _run(
        ["alpha", "-qm", "x", "-c", "--score-stats", "--format", "json"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    # stdout is pure JSON
    payload = json.loads(stdout)
    assert payload["results"][0]["score"] == 0.812
    # stderr carries the one-line stats
    assert "score stats" in stderr.lower()
    assert "centering" in stderr
    assert "M=2492" in stderr
    assert "range=0.432" in stderr


def test_score_stats_without_flag_leaves_stats_out_none(tmp_path: Path) -> None:
    """When --score-stats is NOT given, stats_out is None (no diagnostic
    computation happens)."""
    _make_message_store(tmp_path)
    seen_stats_out: list[Any] = []

    def stub(**kw: Any) -> list:
        seen_stats_out.append(kw.get("stats_out"))
        return [_fake_msg_result()]

    code, _, stderr = _run(
        ["alpha", "-qm", "x"], home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert seen_stats_out == [None]
    assert "score stats" not in stderr.lower()


# --- Truncated whitening: -wv / -wd (see DESIGN §3.6) ------------------------


def test_wv_and_wd_mutually_exclusive_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wv", "0.9", "-wd", "128"],
        home=tmp_path,
        search_messages_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "mutually exclusive" in stderr.lower()
    assert called == []


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.2"])
def test_wv_out_of_range_exit_2(tmp_path: Path, bad: str) -> None:
    _make_message_store(tmp_path)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wv", bad],
        home=tmp_path,
        search_messages_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "whitening-var" in stderr.lower()
    assert called == []


def test_wd_below_one_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    called: list[Any] = []
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wd", "0"],
        home=tmp_path,
        search_messages_func=lambda **kw: called.append(kw) or [],
    )
    assert code == 2
    assert "whitening-dims" in stderr.lower()
    assert called == []


def test_wv_without_w_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-wv", "0.9"],
        home=tmp_path, search_messages_func=lambda **kw: [],
    )
    assert code == 2
    assert "whitening-var" in stderr.lower() or "-w" in stderr


def test_wd_without_qm_exit_2(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    code, _, stderr = _run(
        ["alpha", "-q", "x", "-w", "-wd", "128"],
        home=tmp_path, search_func=lambda **kw: [],
    )
    assert code == 2


def test_wv_forwards_and_alias_matches_long_form(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    cap_alias: dict[str, Any] = {}
    cap_long: dict[str, Any] = {}

    def stub_alias(**kw: Any) -> list:
        cap_alias.update(kw)
        return [_fake_msg_result()]

    def stub_long(**kw: Any) -> list:
        cap_long.update(kw)
        return [_fake_msg_result()]

    code_a, _, _ = _run(
        ["alpha", "-qm", "x", "-w", "-wv", "0.95"],
        home=tmp_path, search_messages_func=stub_alias,
    )
    code_l, _, _ = _run(
        ["alpha", "-qm", "x", "-w", "--whitening-var", "0.95"],
        home=tmp_path, search_messages_func=stub_long,
    )
    assert code_a == code_l == 0
    assert cap_alias["whitening_var"] == pytest.approx(0.95)
    assert cap_long["whitening_var"] == pytest.approx(0.95)
    assert cap_alias["transform"] == "whitening"


def test_wd_forwards_and_alias_matches_long_form(tmp_path: Path) -> None:
    _make_message_store(tmp_path)
    cap_alias: dict[str, Any] = {}
    cap_long: dict[str, Any] = {}

    code_a, _, _ = _run(
        ["alpha", "-qm", "x", "-w", "-wd", "128"],
        home=tmp_path,
        search_messages_func=lambda **kw: cap_alias.update(kw) or [_fake_msg_result()],
    )
    code_l, _, _ = _run(
        ["alpha", "-qm", "x", "-w", "--whitening-dims", "128"],
        home=tmp_path,
        search_messages_func=lambda **kw: cap_long.update(kw) or [_fake_msg_result()],
    )
    assert code_a == code_l == 0
    assert cap_alias["whitening_dims"] == 128
    assert cap_long["whitening_dims"] == 128


def test_kept_axis_notice_shown_without_score_stats(tmp_path: Path) -> None:
    """RC-5: when whitening truncates, the kept-axis notice prints on stderr even
    without --score-stats, and stdout stays JSON-parseable."""
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        so = kw["stats_out"]
        assert so is not None, "CLI must allocate stats_out for whitening"
        so.update(
            whitening_dim=384, whitening_dims_kept=141,
            whitening_var=0.90, whitening_dims_requested=None,
        )
        return [_fake_msg_result()]

    code, stdout, stderr = _run(
        ["alpha", "-qm", "x", "-w", "--format", "json"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    json.loads(stdout)  # stdout is pure JSON
    assert "kept 141 of 384 axes" in stderr
    assert "variance target 0.90" in stderr
    assert "score stats" not in stderr.lower()  # not requested


def test_kept_axis_notice_reports_requested_for_dims(tmp_path: Path) -> None:
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        kw["stats_out"].update(
            whitening_dim=384, whitening_dims_kept=128,
            whitening_var=None, whitening_dims_requested=128,
        )
        return [_fake_msg_result()]

    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wd", "128"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert "kept 128 of 384 axes" in stderr
    assert "requested 128" in stderr


def test_kept_axis_notice_flags_clamp_when_requested_differs(tmp_path: Path) -> None:
    """RC-5: if the requested dims were clamped (kept < requested), the notice
    says so instead of falsely echoing the request."""
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        kw["stats_out"].update(
            whitening_dim=384, whitening_dims_kept=40,
            whitening_var=None, whitening_dims_requested=300,
        )
        return [_fake_msg_result()]

    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wd", "300", "--stats-scope", "filtered"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert "kept 40 of 384 axes" in stderr
    assert "clamped" in stderr.lower()


def test_full_whitening_suppresses_kept_notice(tmp_path: Path) -> None:
    """When no truncation happened (kept None), the notice must NOT print."""
    _make_message_store(tmp_path)

    def stub(**kw: Any) -> list:
        kw["stats_out"].update(
            whitening_dim=384, whitening_dims_kept=None,
            whitening_var=1.0, whitening_dims_requested=None,
        )
        return [_fake_msg_result()]

    code, _, stderr = _run(
        ["alpha", "-qm", "x", "-w", "-wv", "1.0"],
        home=tmp_path, search_messages_func=stub,
    )
    assert code == 0
    assert "kept" not in stderr.lower()
