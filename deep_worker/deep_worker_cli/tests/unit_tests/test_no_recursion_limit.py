"""Tests to ensure NO recursion_limit is set on agent graphs.

Recursion limits are STRICTLY FORBIDDEN. Agents must be able to perform
unlimited tool calls without artificial caps. Any limit causes agents to
crash with GraphRecursionError on long-running tasks.
"""

import re
from pathlib import Path

import pytest


def _find_active_recursion_limit(source: str) -> list[str]:
    """Find lines that actively SET recursion_limit (not comments/docstrings).

    The unlimited sentinel ``sys.maxsize`` (and any UPPER_SNAKE_CASE
    constant that resolves to ``sys.maxsize``) is allowed: it is the
    canonical way to opt out of LangGraph's step counter, see
    ``deepagents/middleware/subagents.py`` for the rationale and the
    paired tests in
    ``deepagents/tests/unit_tests/middleware/test_subagents_no_recursion_limit.py``.

    Returns list of offending lines.
    """
    offending = []
    # We only flag lines where ``recursion_limit`` appears as the
    # KEY of an assignment or dict-literal entry — i.e. the LHS:
    #   "recursion_limit": 25     # dict-literal
    #   recursion_limit=25        # kwarg
    # Reading the value (e.g. ``x = config["recursion_limit"]`` or
    # the runtime guard ``if config["recursion_limit"] < ...``) is
    # NOT a "set" — it's a read, and reads are fine. The earlier
    # version of this helper false-positived on the runtime guard
    # in subagents.py.
    set_pattern = re.compile(
        r"""
        (?:
            ['"]recursion_limit['"]\s*:    # dict-literal key
            |
            \brecursion_limit\s*=          # kwarg / assignment
        )
        """,
        re.VERBOSE,
    )
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if not set_pattern.search(stripped):
            continue
        # Allow the unlimited-sentinel forms:
        #   "recursion_limit": sys.maxsize
        #   "recursion_limit": _SOME_UNLIMITED_CONST
        # (any UPPER_SNAKE_CASE / leading-underscore identifier
        # whose name carries "UNLIMITED" or "MAXSIZE").
        if "sys.maxsize" in stripped:
            continue
        allow_const = re.search(
            r"recursion_limit['\"]?\s*[:=]\s*"
            r"_?[A-Z][A-Z0-9_]*(?:UNLIMITED|MAXSIZE)[A-Z0-9_]*",
            stripped,
        )
        if allow_const is not None:
            continue
        offending.append(f"  line {i}: {stripped}")
    return offending


class TestNoRecursionLimit:
    """Ensure recursion_limit is never actively set in agent configuration."""

    def test_config_py_has_no_recursion_limit(self):
        """config.py must NOT set recursion_limit."""
        source = Path(
            "deep_worker/deep_worker_cli/deep_worker_cli/config.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "config.py must NOT set recursion_limit. "
            "Agents must run without any step limits.\n"
            + "\n".join(offending)
        )

    def test_agent_py_has_no_recursion_limit(self):
        """agent.py must NOT set recursion_limit on the graph."""
        source = Path(
            "deep_worker/deep_worker_cli/deep_worker_cli/agent.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "agent.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )

    def test_agent_process_py_has_no_recursion_limit(self):
        """agent_process.py must NOT set recursion_limit."""
        source = Path(
            "deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "agent_process.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )

    def test_deepagents_graph_py_has_no_recursion_limit(self):
        """deepagents/graph.py must NOT set recursion_limit."""
        source = Path(
            "deep_worker/deepagents/deepagents/graph.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "deepagents/graph.py must NOT set recursion_limit.\n" + "\n".join(offending)
        )

    def test_deepagents_subagents_py_has_no_bounded_recursion_limit(self):
        """deepagents/middleware/subagents.py must NOT set
        recursion_limit to a bounded literal — only the unlimited
        sentinel (``sys.maxsize`` or a UPPER_SNAKE_CASE alias) is
        allowed.

        Regression guard for the GraphRecursionError that crashed
        demo_agent on 2026-04-25 when a content-audit subagent hit
        the langchain_core default of 25 step limit. See
        ``deepagents/tests/unit_tests/middleware/test_subagents_no_recursion_limit.py``
        for the runtime-level invariant; this test pins the source
        invariant from the deep_worker_cli side."""
        source = Path(
            "deep_worker/deepagents/deepagents/middleware/subagents.py"
        ).read_text()
        offending = _find_active_recursion_limit(source)
        assert not offending, (
            "deepagents/middleware/subagents.py must NOT set "
            "recursion_limit to a bounded literal — use sys.maxsize "
            "as the unlimited sentinel.\n" + "\n".join(offending)
        )
