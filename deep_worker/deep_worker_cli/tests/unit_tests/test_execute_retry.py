"""Unit tests for execute tool retry-on-OSError behaviour.

TDD for issue #c4a3f69a964547c5adeb93456c55a4db:
"Execute tool fails with Connection refused".

Design: DESIGN_FIX_EXECUTE_TOOL_CONNECTION_REFUSED_RETRY.md

CLIShellBackend.execute() wraps subprocess spawn in a (1, 2) retry
loop that only retries OSError (incl. ConnectionRefusedError);
other exceptions pass through immediately.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.backends import CLIShellBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    """Build a mock asyncio subprocess Process with happy-path communicate()."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecuteRetryOnOSError:
    """Retry behaviour for transient OSError during subprocess creation."""

    @pytest.mark.asyncio
    async def test_first_oserror_then_success(self) -> None:
        """First call raises OSError → retry → second call succeeds.

        Expected: ExecuteResponse with exit_code=0, correct output.
        """
        backend = CLIShellBackend(timeout=30, inherit_env=True)
        success_proc = _make_mock_proc(stdout=b"retried OK\n")

        # First raise OSError, then succeed
        mock_spawn = AsyncMock(
            side_effect=[
                OSError("Connection refused"),
                success_proc,
            ]
        )

        with (
            patch("asyncio.create_subprocess_shell", mock_spawn),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await backend.execute("echo hello")

        assert mock_spawn.await_count == 2
        mock_sleep.assert_awaited_once_with(0.5)
        assert result.exit_code == 0
        assert "retried OK" in result.output

    @pytest.mark.asyncio
    async def test_both_oserror_returns_error(self) -> None:
        """Both attempts raise OSError → error response, not exception.

        Expected: ExecuteResponse with exit_code=1 and error message.
        """
        backend = CLIShellBackend(timeout=30, inherit_env=True)

        mock_spawn = AsyncMock(
            side_effect=OSError("Connection refused"),
        )

        with (
            patch("asyncio.create_subprocess_shell", mock_spawn),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await backend.execute("echo hello")

        assert mock_spawn.await_count == 2
        mock_sleep.assert_awaited_once_with(0.5)
        assert result.exit_code == 1
        assert "Error executing command" in result.output
        assert "OSError" in result.output

    @pytest.mark.asyncio
    async def test_value_error_does_not_retry(self) -> None:
        """ValueError (non-OSError) should NOT trigger a retry.

        Expected: create_subprocess_shell called exactly once.
        """
        backend = CLIShellBackend(timeout=30, inherit_env=True)

        mock_spawn = AsyncMock(
            side_effect=ValueError("invalid arg"),
        )

        with (
            patch("asyncio.create_subprocess_shell", mock_spawn),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await backend.execute("echo hello")

        assert mock_spawn.await_count == 1
        mock_sleep.assert_not_awaited()
        assert result.exit_code == 1
        assert "Error executing command" in result.output
        assert "ValueError" in result.output

    @pytest.mark.asyncio
    async def test_logger_warning_on_retry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logger.warning should be called on the OSError retry path."""
        backend = CLIShellBackend(timeout=30, inherit_env=True)
        success_proc = _make_mock_proc(stdout=b"ok\n")

        mock_spawn = AsyncMock(
            side_effect=[
                OSError("Connection refused"),
                success_proc,
            ]
        )

        with (
            patch("asyncio.create_subprocess_shell", mock_spawn),
            patch("asyncio.sleep", AsyncMock()),
            caplog.at_level(logging.WARNING, logger="deep_worker_cli.backends"),
        ):
            await backend.execute("echo hello")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "retrying" in caplog.text
        assert "OSError" in caplog.text
        assert "Connection refused" in caplog.text

    @pytest.mark.asyncio
    async def test_successful_first_attempt_no_retry(self) -> None:
        """Happy path — no OSError → no retry, no sleep, no warning.

        Expected: create_subprocess_shell called exactly once.
        """
        backend = CLIShellBackend(timeout=30, inherit_env=True)
        proc = _make_mock_proc(stdout=b"hello world\n")

        with (
            patch("asyncio.create_subprocess_shell", AsyncMock(return_value=proc)) as mock_spawn,
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await backend.execute("echo hello")

        mock_spawn.assert_awaited_once()
        mock_sleep.assert_not_awaited()
        assert result.exit_code == 0
        assert "hello world" in result.output
