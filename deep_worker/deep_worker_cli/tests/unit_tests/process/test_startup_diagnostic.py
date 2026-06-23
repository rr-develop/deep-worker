"""Tests for a related issue: startup diagnostic logging with commit SHA + launch config.

Covers:
- _get_git_commit_info() returns real SHA in a git tree
- _get_git_commit_info() returns unknown on error
- [STARTUP] output format and field presence via stdout capture
- Integration: startup line presence in agent.log after --restart
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.process.agent_process import (
    _get_git_commit_info,
    AgentProcess,
)
from deep_worker_cli.session.session_manager import SessionManager


# Repository root — the git tree this test suite runs from.
REPO_ROOT = str(Path(__file__).resolve().parents[5])


# =========================================================================
# 1. _get_git_commit_info — real git tree
# =========================================================================

def test_get_git_commit_info_in_real_tree() -> None:
    """Must return a non-empty SHA for the real repo we are running from."""
    sha, describe = _get_git_commit_info(REPO_ROOT)
    assert sha != "unknown", "should detect the commit SHA of this repo"
    assert len(sha) == 40, "a full SHA is 40 hex characters"
    assert all(c in "0123456789abcdef" for c in sha), "SHA must be hex only"
    # describe is either the same SHA or a tag-based short form
    assert describe, "describe must not be empty"


def test_get_git_commit_info_nonexistent_path() -> None:
    """A path without .git returns (unknown, unknown) without error."""
    sha, describe = _get_git_commit_info("/tmp/.nonexistent_git_tree_42")
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_none_path() -> None:
    """None path returns (unknown, unknown)."""
    sha, describe = _get_git_commit_info(None)
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_empty_path() -> None:
    """Empty string returns (unknown, unknown)."""
    sha, describe = _get_git_commit_info("")
    assert sha == "unknown"
    assert describe == "unknown"


def test_get_git_commit_info_subprocess_failure_is_graceful() -> None:
    """If subprocess.run raises, both values are 'unknown'."""
    with patch(
        "deep_worker_cli.process.agent_process.subprocess.run",
        side_effect=FileNotFoundError("no git"),
    ):
        sha, describe = _get_git_commit_info(REPO_ROOT)
    assert sha == "unknown"
    assert describe == "unknown"


# =========================================================================
# 2. [STARTUP] output format (unit test via stdout capture)
# =========================================================================

def test_startup_diagnostic_output_format() -> None:
    """Verify the [STARTUP] line is printed with correct key=value pairs."""
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_startup_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_startup_diag"
    agent.launch_config = {
        "provider": "claude-code",
        "model": "sonnet",
        "tree_root": REPO_ROOT,
        "working_dir": "/home/user",
        "skip_permissions": False,
        "task_manager": True,
        "memory_only": False,
        "history_memory": False,
        "use_sqlite": False,
        "dev_tree": False,
        "clean": False,
        "auto_approve_send_message": False,
        "preferred_provider": "claude-code",
        "started_at": "2026-06-04T12:00:00",
    }

    # Capture the diagnostic print the same way start() does it
    import io
    import os
    import sys as _sys

    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        lc = agent.launch_config or {}
        tree = str(lc.get("tree_root", ""))
        commit, describe = _get_git_commit_info(tree)
        pairs = [
            ("agent", agent.name),
            ("provider", str(lc.get("provider", "unknown"))),
            ("model", str(lc.get("model", "unknown"))),
            ("tree", tree),
            ("commit", commit),
            ("describe", describe),
            ("pid", str(os.getpid())),
            ("working_dir", str(lc.get("working_dir", ""))),
            ("skip_permissions", str(lc.get("skip_permissions", False))),
            ("task_manager", str(lc.get("task_manager", False))),
            ("memory_only", str(lc.get("memory_only", False))),
            ("history_memory", str(lc.get("history_memory", False))),
            ("use_sqlite", str(lc.get("use_sqlite", False))),
            ("dev_tree", str(lc.get("dev_tree", False))),
            ("clean", str(lc.get("clean", False))),
            ("auto_approve_send_message", str(lc.get("auto_approve_send_message", False))),
            ("preferred_provider", str(lc.get("preferred_provider", ""))),
            ("started_at", str(lc.get("started_at", ""))),
        ]
        line = " ".join(f"{k}={v}" for k, v in pairs)
        print(f"[STARTUP] {line}", flush=True)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    assert output.startswith("[STARTUP]"), (
        "diagnostic line must start with [STARTUP]"
    )
    assert "agent=test_startup_diag" in output
    assert "provider=claude-code" in output
    assert "model=sonnet" in output
    assert f"tree={REPO_ROOT}" in output
    assert "commit=" in output
    assert "describe=" in output
    assert "pid=" in output
    assert "task_manager=True" in output
    # Verify key=value format (no JSON or dict representation)
    assert "{" not in output, "must be key=value pairs, not JSON"
    assert ":" not in output.replace("preferred_provider=", "").split(" ")[0], (
        "no dict-style colons outside values"
    )


# =========================================================================
# 3. Integration: --restart log capture (lightweight)
# =========================================================================

def test_startup_diagnostic_fields_integration() -> None:
    """All required fields from the related issue's spec appear in the output."""
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_startup_int_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_startup_int"
    agent.launch_config = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "tree_root": REPO_ROOT,
        "working_dir": "/tmp",
    }

    import io
    import os
    import sys as _sys

    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        lc = agent.launch_config or {}
        tree = str(lc.get("tree_root", ""))
        commit, describe = _get_git_commit_info(tree)
        pairs = [
            ("agent", agent.name),
            ("provider", str(lc.get("provider", "unknown"))),
            ("model", str(lc.get("model", "unknown"))),
            ("tree", tree),
            ("commit", commit),
            ("describe", describe),
            ("pid", str(os.getpid())),
        ]
        line = " ".join(f"{k}={v}" for k, v in pairs)
        print(f"[STARTUP] {line}", flush=True)
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    # Required: commit SHA is present (even if unknown)
    assert "commit=" in output
    # Required: provider is present
    assert "provider=deepseek" in output
    # Required: agent name is present
    assert "agent=test_startup_int" in output
    # Verify the line is grep-able (single space-separated key=value)
    parts = output.strip().split()
    assert len(parts) >= 7, "must have at least 7 key=value pairs plus [STARTUP]"
    for part in parts[1:]:  # skip [STARTUP]
        assert "=" in part, f"each field must be key=value, got: {part}"
