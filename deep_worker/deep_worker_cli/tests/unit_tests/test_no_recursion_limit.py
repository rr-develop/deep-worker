"""Tests to ensure NO recursion_limit is set on agent graphs.

Recursion limits are STRICTLY FORBIDDEN. Agents must be able to perform
unlimited tool calls without artificial caps. Any limit causes agents to
crash with GraphRecursionError on long-running tasks.
"""

import re
from pathlib import Path

import pytest

_CLI_ROOT = Path(__file__).resolve().parents[2]


def _find_active_recursion_limit(source: str) -> list[str]:
    """Find lines that actively SET recursion_limit (not comments/docstrings).

    Returns list of offending lines.
    """
    offending = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Check for actual assignment/dict usage of recursion_limit
        if "recursion_limit" in stripped and ("=" in stripped or ":" in stripped):
            # e.g. {"recursion_limit": 1000} or config = {"recursion_limit": ...}
            # But NOT: # recursion_limit is forbidden
            if not stripped.startswith("#"):
                offending.append(f"  line {i}: {stripped}")
    return offending


class TestNoRecursionLimit:
    """Ensure recursion_limit is never actively set in agent configuration."""

    def test_config_py_has_no_recursion_limit(self):
        """config.py must NOT set recursion_limit."""
        source = (_CLI_ROOT / "deep_worker_cli" / "config.py").read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "config.py must NOT set recursion_limit. "
            "Agents must run without any step limits.\n"
            + "\n".join(offending)
        )

    def test_agent_py_has_no_recursion_limit(self):
        """agent.py must NOT set recursion_limit on the graph."""
        source = (_CLI_ROOT / "deep_worker_cli" / "agent.py").read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "agent.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )

    def test_agent_process_py_has_no_recursion_limit(self):
        """agent_process.py must NOT set recursion_limit."""
        source = (
            _CLI_ROOT / "deep_worker_cli" / "process" / "agent_process.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "agent_process.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )

    def test_deepagents_graph_py_has_no_recursion_limit(self):
        """deepagents/graph.py must NOT set recursion_limit."""
        source = (
            _CLI_ROOT.parent / "deepagents" / "deepagents" / "graph.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "deepagents/graph.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )
