"""Regression: a Claude usage/session-limit hit must surface as ERROR with a
clear, user-facing message — not a silent idle and not the raw
``Agent invocation failed: ...`` wrapper.

Incident (observed live, 2026-05-21): when the account hit its usage cap, the
Claude CLI returned a degenerate "You've hit your session limit · resets
<time>" response. The provider now raises ``ClaudeUsageLimitError``;
``_handle_invoke`` must catch it (by class name, to stay provider-agnostic),
set ``status=ERROR``, and return a clean ``Claude usage limit reached — …``
message so the web UI's persistent error panel shows the cause.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class ClaudeUsageLimitError(RuntimeError):
    """Stand-in matching the provider exception's class NAME. The handler
    keys on ``type(e).__name__`` so codex agents never import the claude
    module."""


class CodexUsageLimitError(Exception):
    """Stand-in for the codex provider's usage-limit (real one subclasses
    ``CodexProtocolError``). The invoke-handler must treat it symmetrically to
    Claude: ERROR status + clean message + failover. See
    a related internal incident analysis (not included here)."""


class GrokUsageLimitError(RuntimeError):
    """Stand-in matching ``grok_provider.limit_errors.GrokUsageLimitError``."""


def _make_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_session_limit_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_session_limit"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    return agent


@pytest.mark.asyncio
async def test_session_limit_sets_error_with_clean_message() -> None:
    agent = _make_agent()

    limit = "You've hit your session limit · resets 5:40pm (Europe/Amsterdam)"
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=ClaudeUsageLimitError(limit))
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR, (
        "a usage-limit hit must surface as ERROR, not a silent idle "
        "(observed live, 2026-05-21)"
    )
    # Clean, user-facing message — not the raw "Agent invocation failed:".
    assert result["error"].startswith("Claude usage limit reached")
    assert "5:40pm" in result["error"]
    assert "Agent invocation failed" not in result["error"]
    # The same clean message is persisted for the web UI error panel.
    assert agent.session_manager.error_message is not None
    assert "usage limit" in agent.session_manager.error_message.lower()


@pytest.mark.asyncio
async def test_codex_session_limit_sets_error_and_schedules_failover() -> None:
    """End-to-end for the 2026-06-17 incident: a CODEX usage-limit reaching the
    invoke-handler must set ERROR (status no longer the false ``running``),
    return a clean ``Codex usage limit reached — …`` message, AND trigger
    provider failover (codex → next in the chain). Symmetric to Claude."""
    agent = _make_agent()
    agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

    limit = (
        "Codex stream error: You've hit your usage limit ... "
        "try again at Jun 18th, 2026 12:39 AM."
    )
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=CodexUsageLimitError(limit))
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    # No longer stuck on a false ``running`` — surfaced as ERROR.
    assert agent.session_manager.status == AgentStatus.ERROR
    assert result["error"].startswith("Codex usage limit reached")
    assert "Agent invocation failed" not in result["error"]
    assert agent.session_manager.error_message is not None
    assert "usage limit" in agent.session_manager.error_message.lower()
    # Failover was triggered for codex (→ grok per the chain).
    agent._schedule_failover.assert_awaited_once()
    assert agent._schedule_failover.await_args.args[0] == "codex"


@pytest.mark.asyncio
async def test_grok_session_limit_sets_error_and_schedules_failover() -> None:
    """Grok usage-limit must set ERROR and schedule failover to deepseek."""
    agent = _make_agent()
    agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

    limit = "Grok usage limit reached — You've hit your usage limit."
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=GrokUsageLimitError(limit))
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR
    assert result["error"].startswith("Grok usage limit reached")
    agent._schedule_failover.assert_awaited_once()
    assert agent._schedule_failover.await_args.args[0] == "grok"


@pytest.mark.asyncio
async def test_session_limit_error_message_reaches_notify_callback() -> None:
    """The TUI/web subscribers must receive the limit as an error event."""
    agent = _make_agent()

    limit = "You've hit your session limit · resets 11:10pm (Europe/Amsterdam)"
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=ClaudeUsageLimitError(limit))
    agent.agent = fake_graph

    events: list[dict] = []

    async def capture(msg: dict) -> None:
        events.append(msg)

    await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=capture,
        approval_callback=None,
    )

    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, "limit must be broadcast as an error event"
    assert "usage limit" in error_events[-1]["content"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
