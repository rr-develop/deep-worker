"""Integration tests for execute tool retry mechanism with real subprocesses.

These tests use REAL subprocesses (echo, false) because the semantics under
test — that the retry loop doesn't break normal execution — are only
meaningful with real process creation. See test_execute_retry.py for the
mocked tests that verify retry-on-OSError logic directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIShellBackend


@pytest.mark.asyncio
async def test_fast_command_succeeds_normally() -> None:
    """A quick command on the first attempt succeeds as before.

    Regression: the retry loop must not interfere with successful execution.
    """
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.execute("echo hello-world", timeout=10)
    assert "hello-world" in result.output
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_failing_command_returns_exit_code_1() -> None:
    """A command that fails (non-zero exit) still returns exit_code=1.

    The retry loop only catches OSError; a failed command exit is not an
    exception and must propagate normally.
    """
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.execute("false", timeout=10)
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_combined_stdout_stderr() -> None:
    """stderr is prefixed with [stderr] in the combined output."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    result = await backend.execute(
        'bash -c "echo out; echo err >&2"', timeout=10
    )
    assert "out" in result.output
    assert "[stderr] err" in result.output
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_multiple_fast_commands_sequential() -> None:
    """Multiple sequential commands all work after the retry loop change."""
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    for i in range(5):
        result = await backend.execute(f"echo seq-{i}", timeout=10)
        assert f"seq-{i}" in result.output
        assert result.exit_code == 0
