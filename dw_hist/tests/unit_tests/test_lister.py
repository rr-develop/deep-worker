"""Tests for dw_hist.lister.discover_agents."""

from __future__ import annotations

from pathlib import Path

from dw_hist.lister import discover_agents


from deep_worker_cli.storage_paths import deep_worker_home
def _make_agent(home: Path, name: str, *, with_index: bool, empty_index_dir: bool = False) -> None:
    """Create ``~/.deep_worker/<name>/summaries/[.history_index/lancedb/]``."""
    summaries = deep_worker_home(home) / name / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    if with_index:
        (summaries / ".history_index" / "lancedb").mkdir(parents=True, exist_ok=True)
    elif empty_index_dir:
        (summaries / ".history_index").mkdir(parents=True, exist_ok=True)


def test_discover_returns_only_agents_with_real_index(tmp_path: Path) -> None:
    _make_agent(tmp_path, "alpha", with_index=True)
    _make_agent(tmp_path, "bravo", with_index=False)               # no index at all
    _make_agent(tmp_path, "charlie", with_index=False, empty_index_dir=True)  # dir but no lancedb
    _make_agent(tmp_path, "delta", with_index=True)

    found = discover_agents(home=tmp_path)

    assert found == ["alpha", "delta"], (
        "discover_agents must return only agents whose summaries dir contains "
        f".history_index/lancedb/; got {found!r}"
    )


def test_discover_empty_home_returns_empty_list(tmp_path: Path) -> None:
    assert discover_agents(home=tmp_path) == []


def test_discover_skips_non_directory_entries(tmp_path: Path) -> None:
    (deep_worker_home(tmp_path)).mkdir()
    (deep_worker_home(tmp_path) / "stray_file").write_text("ignore me")
    _make_agent(tmp_path, "real", with_index=True)
    assert discover_agents(home=tmp_path) == ["real"]


def test_discover_is_sorted_stably(tmp_path: Path) -> None:
    for name in ("zeta", "alpha", "mu", "beta"):
        _make_agent(tmp_path, name, with_index=True)
    assert discover_agents(home=tmp_path) == ["alpha", "beta", "mu", "zeta"]
