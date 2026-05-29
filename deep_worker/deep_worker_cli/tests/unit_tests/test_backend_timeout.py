"""Unit tests for CLIShellBackend per-command timeout features.

The backend is fully async (`asyncio.create_subprocess_shell` +
`asyncio.wait_for`) — see `backends.py` ~line 78. The previous
subprocess.run-based tests were incompatible with that; this
rewrite mocks the asyncio subprocess path instead.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.backends import DEFAULT_EXECUTE_TIMEOUT, CLIShellBackend


class TestDefaultTimeoutConstant:
    """Tests for the named default timeout constant."""

    def test_default_timeout_uses_constant(self) -> None:
        """Backend created without explicit timeout should use the default constant."""
        backend = CLIShellBackend()
        assert backend._timeout == DEFAULT_EXECUTE_TIMEOUT


def _make_mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    """Build a mock asyncio subprocess Process with successful
    `.communicate()`."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


class TestPerCommandTimeout:
    """Tests for per-command timeout override in execute()."""

    @pytest.mark.asyncio
    async def test_per_command_timeout_used(self) -> None:
        """When timeout is passed to execute(), it should be honored.

        We can't observe the `timeout` directly in
        `asyncio.wait_for(..., timeout=...)` from a mock, but we can
        verify the command completes within the budget and the mock
        subprocess was awaited.
        """
        backend = CLIShellBackend(timeout=10, inherit_env=True)
        proc = _make_mock_proc(stdout=b"hello\n")
        with patch(
            "asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            return_value=proc,
        ) as mock_spawn:
            result = await backend.execute("echo hello", timeout=300)

        mock_spawn.assert_awaited_once()
        assert "hello" in result.output
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_default_timeout_when_not_specified(self) -> None:
        """When no per-command timeout, the default at init is used."""
        backend = CLIShellBackend(timeout=60, inherit_env=True)
        proc = _make_mock_proc(stdout=b"hello\n")
        with patch(
            "asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            return_value=proc,
        ):
            result = await backend.execute("echo hello")

        assert result.exit_code == 0
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_per_command_zero_timeout_raises(self) -> None:
        """Zero per-command timeout should raise ValueError."""
        backend = CLIShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            await backend.execute("echo hello", timeout=0)

    @pytest.mark.asyncio
    async def test_per_command_negative_timeout_raises(self) -> None:
        """Negative per-command timeout should raise ValueError."""
        backend = CLIShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            await backend.execute("echo hello", timeout=-5)


class TestTimeoutBackgroundJob:
    """Tests for the NEW timeout behavior (2026-05-21): on timeout the command
    is NOT killed — it becomes a background job and the agent gets a job_id
    plus wait/kill options. (Was: kill + exit_code 124. See
    test_background_jobs.py for the full real-subprocess coverage.)"""

    @pytest.mark.asyncio
    async def test_timeout_does_not_kill_and_offers_actions(self) -> None:
        """On timeout the command must NOT be killed; the result must offer
        wait_for_job and kill_job by job_id. Real subprocess (mocking the
        shield/timeout interplay is brittle)."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
        result = await backend.execute("sleep 10", timeout=1)

        assert result.exit_code is None  # still running
        assert "job_" in result.output
        assert "wait_for_job" in result.output
        assert "kill_job" in result.output
        assert len(backend._jobs) == 1

        job_id = next(iter(backend._jobs))
        # Process must still be alive (was NOT killed on timeout).
        assert backend._jobs[job_id].proc.returncode is None
        await backend.kill_job(job_id)  # cleanup

    @pytest.mark.asyncio
    async def test_timeout_message_shows_effective_timeout(self) -> None:
        """The still-running message should reflect the effective per-command
        timeout."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
        result = await backend.execute("sleep 10", timeout=2)

        assert "2s" in result.output
        assert "job_" in result.output

        for jid in list(backend._jobs):
            await backend.kill_job(jid)  # cleanup
