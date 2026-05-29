"""Wiring-level tests for the fix to where conversation-history files land.

Design: deep_worker/docs/DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md §3.6.

These tests do NOT spin up a full agent — they verify the static code
contract of the wiring changes so regressions are caught without needing
a real LLM/RPyC stack.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_create_deep_agent_accepts_summarization_history_prefix():
    """create_deep_agent must accept an opt-in summarization_history_prefix kwarg."""
    from deepagents.graph import create_deep_agent
    sig = inspect.signature(create_deep_agent)
    assert "summarization_history_prefix" in sig.parameters, (
        "create_deep_agent must accept summarization_history_prefix as a keyword."
    )
    # Default must be None (opt-in — do not break callers that don't pass it).
    assert sig.parameters["summarization_history_prefix"].default is None


def test_summarizationmiddleware_accepts_history_path_prefix():
    """Sanity: the underlying middleware already accepts history_path_prefix."""
    from deepagents.middleware.summarization import SummarizationMiddleware
    sig = inspect.signature(SummarizationMiddleware)
    assert "history_path_prefix" in sig.parameters


def test_agent_py_does_not_route_conversation_history_to_tempdir():
    """deep_worker_cli/agent.py must NOT register a /conversation_history/ route.

    The source is read as text — we only need a static guarantee. We check
    that ACTIVE code (route declaration + tempdir construction) is gone.
    Comments that mention the old path for historical context are allowed.
    """
    src = Path("deep_worker/deep_worker_cli/deep_worker_cli/agent.py").read_text()
    # The route declaration is a string KEY in a `routes={...}` dict. It
    # sits on its own line with a comma. A pure-comment mention (starting
    # with '#') is not a route declaration.
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comments are OK
        if '"/conversation_history/"' in stripped:
            raise AssertionError(
                "agent.py still declares an active /conversation_history/ "
                f"route: {line!r}"
            )
    # The tempdir factory must be gone as active code (comments OK).
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "deepagents_conversation_history_" in stripped:
            raise AssertionError(
                "agent.py still creates a tempdir prefix "
                f"deepagents_conversation_history_: {line!r}"
            )


def test_agent_py_passes_per_agent_summaries_prefix_into_create_deep_agent():
    """The wiring must pass the summaries directory into create_deep_agent."""
    src = Path("deep_worker/deep_worker_cli/deep_worker_cli/agent.py").read_text()
    assert "summarization_history_prefix=" in src, (
        "agent.py must forward a per-agent summaries prefix to create_deep_agent."
    )
    # The prefix must be derived from the blessed per-agent data layout
    # (<agent-data-dir>/<agent>/summaries).
    assert "summaries" in src


def test_ensure_summaries_dir_enforces_private_permissions(tmp_path, monkeypatch):
    """Summaries directory is per-agent storage and must be owner-only."""
    monkeypatch.setenv("HOME", str(tmp_path))
    summaries_dir = tmp_path / Path("~/.deep_worker").name / "test_v3" / "summaries"
    summaries_dir.mkdir(parents=True, mode=0o777)
    summaries_dir.chmod(0o777)

    from deep_worker_cli.summaries import ensure_summaries_dir

    result = ensure_summaries_dir("test_v3")

    assert result == summaries_dir
    assert oct((tmp_path / Path("~/.deep_worker").name / "test_v3").stat().st_mode & 0o777) == "0o700"
    assert oct(summaries_dir.stat().st_mode & 0o777) == "0o700"


def test_large_results_route_unchanged():
    """Negative control — we didn't accidentally nuke the other route."""
    src = Path("deep_worker/deep_worker_cli/deep_worker_cli/agent.py").read_text()
    assert '"/large_tool_results/"' in src
    assert "deepagents_large_results_" in src
