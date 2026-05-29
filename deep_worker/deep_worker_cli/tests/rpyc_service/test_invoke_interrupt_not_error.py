"""Regression: a user-initiated interrupt must NOT surface as an error.

User report (2026-05-20): pressing Interrupt in the web UI stops the
agent correctly, but ALSO pops a side notification and a "Last Error"
showing ``asyncio.exceptions.CancelledError`` with a Remote Traceback.
The user: "I don't expect an interrupt call to produce any errors."

ROOT CAUSE (confirmed by reproducing against a live test agent):
``AgentProcess._handle_invoke`` correctly treats the interrupt's
``CancelledError`` as a NORMAL state transition (sets INTERRUPTED,
broadcasts ``agent_interrupted``) but then RE-RAISES it
(agent_process.py:1859). ``CancelledError`` is a ``BaseException``, so
``exposed_invoke_agent``'s ``except Exception`` does NOT catch it — it
escapes the RPyC handler, crosses the RPyC boundary as a remote
exception, and the web side's ``invoke_agent`` ``error_callback``
broadcasts it as ``agent_error`` → toast + Last Error panel.

Fix: ``exposed_invoke_agent`` catches ``CancelledError`` and returns a
structured ``{"success": False, "interrupted": True, ...}`` result
instead of letting it propagate. Interrupt is a normal user action,
not an error — the web caller gets a clean result, not a remote
exception, so no error is shown.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


@pytest.mark.asyncio
async def test_invoke_returns_structured_result_on_interrupt_not_raises() -> None:
    """When ``_handle_invoke`` raises ``CancelledError`` (user interrupt),
    ``exposed_invoke_agent`` must RETURN a structured result, never let
    the CancelledError escape to the RPyC caller."""
    service = AgentService()
    agent = MagicMock()
    # The interrupt cancels the worker task → _handle_invoke raises.
    agent._handle_invoke = AsyncMock(side_effect=asyncio.CancelledError())
    service._agent_process = agent
    service._notify_subscribers = AsyncMock()

    # MUST NOT raise CancelledError out of the RPyC handler.
    result_str = await service.exposed_invoke_agent("do something", stream=False)

    result = json.loads(result_str)
    assert result.get("success") is False
    assert result.get("interrupted") is True, (
        "interrupt must be reported as a clean 'interrupted' result, not "
        "an error — so the web UI shows no error popup/Last Error "
        "(2026-05-20 interrupt-as-error bug)"
    )


@pytest.mark.asyncio
async def test_invoke_interrupt_result_has_no_error_field() -> None:
    """The interrupted result must NOT carry an ``error`` field — the
    web on_error path keys off exceptions / error text; a clean
    interrupted result must not look like an error."""
    service = AgentService()
    agent = MagicMock()
    agent._handle_invoke = AsyncMock(side_effect=asyncio.CancelledError())
    service._agent_process = agent
    service._notify_subscribers = AsyncMock()

    result = json.loads(await service.exposed_invoke_agent("x", stream=False))

    # No "error" text that the UI would render as a failure.
    assert not result.get("error"), (
        f"interrupted result must not carry an error string; got {result}"
    )


@pytest.mark.asyncio
async def test_real_exception_still_reported_as_error() -> None:
    """A genuine (non-cancellation) exception must STILL be reported as
    an error — the fix must not swallow real failures."""
    service = AgentService()
    agent = MagicMock()
    agent._handle_invoke = AsyncMock(side_effect=RuntimeError("real boom"))
    service._agent_process = agent
    service._notify_subscribers = AsyncMock()

    result = json.loads(await service.exposed_invoke_agent("x", stream=False))

    assert result.get("success") is False
    assert not result.get("interrupted")
    assert "real boom" in str(result.get("error", "")), (
        "a real exception must surface as an error (not be misclassified "
        "as an interrupt)"
    )
