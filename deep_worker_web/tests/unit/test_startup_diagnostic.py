"""Tests for a related issue: startup diagnostic logging with commit SHA + config.

Covers:
- get_git_commit_info() returns real SHA in a git tree
- get_git_commit_info() returns unknown on error
- [STARTUP] log line is emitted during lifespan with all required fields
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_git_commit_info


# =========================================================================
# 1. get_git_commit_info — real git tree
# =========================================================================

def test_git_commit_info_in_real_tree() -> None:
    """Must return a non-empty SHA for the real repo."""
    sha, describe = get_git_commit_info("/path/to/tasks")
    assert sha != "unknown", "should detect the commit SHA of this repo"
    assert len(sha) == 40, "a full SHA is 40 hex characters"
    assert all(c in "0123456789abcdef" for c in sha), "SHA must be hex only"
    assert describe, "describe must not be empty"


def test_git_commit_info_nonexistent_path() -> None:
    """A path without .git returns (unknown, unknown) without error."""
    sha, describe = get_git_commit_info("/tmp/.nonexistent_git_tree_42")
    assert sha == "unknown"
    assert describe == "unknown"


def test_git_commit_info_none_path() -> None:
    """None path returns (unknown, unknown)."""
    sha, describe = get_git_commit_info(None)
    assert sha == "unknown"
    assert describe == "unknown"


def test_git_commit_info_empty_path() -> None:
    """Empty string returns (unknown, unknown)."""
    sha, describe = get_git_commit_info("")
    assert sha == "unknown"
    assert describe == "unknown"


def test_git_commit_info_subprocess_failure_is_graceful() -> None:
    """If subprocess.run raises, both values are 'unknown'."""
    with patch(
        "app.config.subprocess.run",
        side_effect=FileNotFoundError("no git"),
    ):
        sha, describe = get_git_commit_info("/path/to/tasks")
    assert sha == "unknown"
    assert describe == "unknown"


# =========================================================================
# 2. [STARTUP] diagnostic in lifespan logs
# =========================================================================

def test_startup_diagnostic_appears_in_lifespan_logs(caplog) -> None:
    """The [STARTUP] line must be emitted during FastAPI lifespan startup."""
    from app.main import app

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(app) as client:
            # Trigger lifespan by making a request
            resp = client.get("/health")

    assert resp.status_code == 200

    # Find the [STARTUP] record
    startup_records = [
        rec.getMessage() for rec in caplog.records
        if rec.getMessage().startswith("[STARTUP]")
    ]
    assert startup_records, (
        "lifespan must emit a [STARTUP] diagnostic record; "
        "all records: " + "; ".join(rec.getMessage() for rec in caplog.records)
    )

    line = startup_records[0]
    assert "component=deep_worker_web" in line, (
        f"missing component=deep_worker_web in: {line}"
    )
    assert "commit=" in line, f"missing commit= in: {line}"
    assert "describe=" in line, f"missing describe= in: {line}"
    assert "tree=" in line, f"missing tree= in: {line}"
    assert "pid=" in line, f"missing pid= in: {line}"
    assert "host=127.0.0.1" in line, f"missing host= in: {line}"
    assert "port=" in line, f"missing port= in: {line}"
    assert "debug=" in line, f"missing debug= in: {line}"
    assert "reload=" in line, f"missing reload= in: {line}"
    # Verify key=value format (no JSON)
    assert "{" not in line, f"must be key=value pairs, not JSON: {line}"


def test_startup_diagnostic_includes_env_vars(caplog) -> None:
    """If DEEP_WORKER_WEB_* env vars are set, they appear in the record."""
    import os

    from app.main import app

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 200

    startup_records = [
        rec.getMessage() for rec in caplog.records
        if rec.getMessage().startswith("[STARTUP]")
    ]
    assert startup_records

    line = startup_records[0]
    # The environment from the test runner may include DEEP_WORKER_WEB_*
    # variables; if any are set they must appear in the output.
    for key in os.environ:
        if key.startswith("DEEP_WORKER_"):
            assert key in line, (
                f"env var {key} must appear in the startup diagnostic: {line}"
            )


def test_startup_diagnostic_key_value_format(caplog) -> None:
    """All fields must be space-separated key=value pairs."""
    from app.main import app

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 200

    startup_records = [
        rec.getMessage() for rec in caplog.records
        if rec.getMessage().startswith("[STARTUP]")
    ]
    assert startup_records

    line = startup_records[0]
    # Strip the [STARTUP] prefix, then verify remaining parts
    prefix = "[STARTUP] "
    assert line.startswith(prefix)
    rest = line[len(prefix):]
    parts = rest.split()
    assert len(parts) >= 8, (
        f"must have at least 8 key=value pairs after [STARTUP], "
        f"got {len(parts)}: {rest}"
    )
    for part in parts:
        assert "=" in part, (
            f"each field must be key=value, got: {part!r} in {line}"
        )
