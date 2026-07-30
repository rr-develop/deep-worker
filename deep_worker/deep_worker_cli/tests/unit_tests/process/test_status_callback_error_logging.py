from __future__ import annotations

import re
from uuid import UUID
from unittest.mock import AsyncMock

import pytest

from deep_worker_cli.process.status_callback import StatusTrackingCallback
from deep_worker_cli.types import AgentStatus


@pytest.mark.asyncio
async def test_tool_error_log_includes_utc_timestamp_and_run_ids(capsys: pytest.CaptureFixture[str]) -> None:
    """Post-mortem stderr line format MUST NOT change (grep patterns depend on it).

    The status transition changes per the new fatal/recoverable split
    (see the fatal/recoverable tests below); this test only asserts
    the log line contract.
    """
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    parent_run_id = UUID("22222222-2222-2222-2222-222222222222")

    await callback.on_tool_error(
        RuntimeError("tool exploded"),
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    stderr = capsys.readouterr().err
    assert re.search(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00\] \[TOOL_ERROR\]", stderr)
    assert "error=tool exploded" in stderr
    assert f"run_id={run_id}" in stderr
    assert f"parent_run_id={parent_run_id}" in stderr


# ---------------------------------------------------------------------------
# No-Agent-Error-State Policy — recoverable vs fatal classification.
# See docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md §3.1.
# ---------------------------------------------------------------------------


class _StubValidationError(Exception):
    """Stand-in for pydantic.ValidationError — matched by class name via
    the ``is_fatal_exception`` registry, which does NOT list it → recoverable."""


@pytest.mark.asyncio
async def test_recoverable_pydantic_error_keeps_running_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tool-arg validation failure MUST NOT flip the agent to ERROR."""
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("11111111-1111-1111-1111-111111111111")

    await callback.on_tool_error(
        _StubValidationError("1 validation error for read_file"),
        run_id=run_id,
    )

    # Recoverable → return to RUNNING so the model can process the ToolMessage.
    session_manager.set_status.assert_awaited_once_with(AgentStatus.RUNNING)
    # MUST NOT surface an agent-level error message for recoverable failures.
    session_manager.set_error.assert_not_called()


@pytest.mark.asyncio
async def test_recoverable_generic_exception_keeps_running_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare RuntimeError from a sloppy tool MUST NOT flip the agent to ERROR."""
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("11111111-1111-1111-1111-111111111111")

    await callback.on_tool_error(RuntimeError("boom"), run_id=run_id)

    session_manager.set_status.assert_awaited_once_with(AgentStatus.RUNNING)
    session_manager.set_error.assert_not_called()


# Names come from FATAL_EXCEPTION_TYPES in deep_worker_cli/exceptions.py.
# is_fatal_exception matches by __name__, so a locally-defined subclass with the
# same name is classified as fatal — that is the whole point of the by-name
# registry (kept provider-agnostic).
_FATAL_CLASS_NAMES = [
    "ClaudeUsageLimitError",
    "CodexUsageLimitError",
    "GrokUsageLimitError",
    "CheckpointCorruptedError",
    "ProviderAuthFatalError",
    "AgentNotInitialisedError",
    "HistoryOffloadError",
]


@pytest.mark.parametrize("fatal_name", _FATAL_CLASS_NAMES)
@pytest.mark.asyncio
async def test_fatal_exception_transitions_to_error(
    fatal_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """For truly fatal exceptions the callback MUST persist an error message."""
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("11111111-1111-1111-1111-111111111111")

    # Build a fresh exception class whose __name__ matches the fatal registry.
    exc_cls = type(fatal_name, (RuntimeError,), {})
    exc = exc_cls("fatal detail")

    await callback.on_tool_error(exc, run_id=run_id)

    # Fatal path uses set_error (records error_message + flips to ERROR),
    # NOT bare set_status(ERROR) — matching agent_process._handle_invoke.
    session_manager.set_error.assert_awaited_once_with("fatal detail")
    session_manager.set_status.assert_not_called()


@pytest.mark.asyncio
async def test_stderr_line_still_emitted_for_recoverable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The [TOOL_ERROR] line MUST be emitted regardless of classification."""
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    await callback.on_tool_error(RuntimeError("recoverable"), run_id=run_id)

    stderr = capsys.readouterr().err
    assert "[TOOL_ERROR]" in stderr
    assert "error=recoverable" in stderr
    assert f"run_id={run_id}" in stderr


@pytest.mark.asyncio
async def test_stderr_line_still_emitted_for_fatal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same guarantee for fatal exceptions (post-mortem parity)."""
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    exc_cls = type("ClaudeUsageLimitError", (RuntimeError,), {})
    await callback.on_tool_error(exc_cls("usage limit reached"), run_id=run_id)

    stderr = capsys.readouterr().err
    assert "[TOOL_ERROR]" in stderr
    assert "error=usage limit reached" in stderr
    assert f"run_id={run_id}" in stderr


@pytest.mark.asyncio
async def test_no_session_manager_is_safe() -> None:
    """When session_manager is None the callback MUST still log and return."""
    callback = StatusTrackingCallback(session_manager=None)
    run_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    # Must not raise.
    await callback.on_tool_error(RuntimeError("no manager"), run_id=run_id)
