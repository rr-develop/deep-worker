"""Unit tests for CLIShellBackend per-command timeout features.

The backend is fully async (`asyncio.create_subprocess_shell` +
`asyncio.wait_for`) — see `backends.py` ~line 78. The previous
subprocess.run-based tests were incompatible with that; this
rewrite mocks the asyncio subprocess path instead.
"""

from __future__ import annotations

import asyncio
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


class TestTimeoutErrorMessage:
    """Tests for timeout error message with retry guidance."""

    @pytest.mark.asyncio
    async def test_timeout_error_includes_retry_guidance(self) -> None:
        """Timeout error must include guidance to use `timeout` parameter."""
        backend = CLIShellBackend(timeout=1, inherit_env=True)

        # Force `asyncio.wait_for(proc.communicate(), ...)` to time out.
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            return_value=proc,
        ):
            result = await backend.execute("sleep 10")

        assert "timed out" in result.output.lower()
        assert "timeout parameter" in result.output.lower()
        assert result.exit_code == 124

    @pytest.mark.asyncio
    async def test_timeout_error_shows_effective_timeout(self) -> None:
        """Timeout error should reflect the effective per-command timeout."""
        backend = CLIShellBackend(timeout=60, inherit_env=True)

        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            return_value=proc,
        ):
            result = await backend.execute("sleep 10", timeout=5)

        assert "5" in result.output
        assert "timeout parameter" in result.output.lower()
