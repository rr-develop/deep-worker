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
from types import SimpleNamespace
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
    """Stand-in matching ``grok_provider.limit_errors.GrokUsageLimitError``.

    LEGACY shape. Since 2026-08-24 grok runs on the shared ACP client and its
    cap raises ``AcpUsageLimitError`` instead — see
    ``test_acp_usage_limit_fails_over_to_the_CONFIGURED_provider`` below, which
    drives the path grok actually takes now. This class is kept because the
    branch that handles it is still reachable from a stale record.
    """


class AcpUsageLimitError(RuntimeError):
    """Stand-in matching ``providers.acp.limit_errors.AcpUsageLimitError``.

    Raised by the SHARED ACP client, so it names the transport rather than the
    account: claude-code, grok and any future ACP provider all raise this one
    class. The handler must therefore resolve the failover target from
    ``config.provider``.
    """


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_provider", "expected_failover_arg"),
    [("grok", "grok"), ("claude-code", "claude-code")],
)
async def test_acp_usage_limit_fails_over_to_the_CONFIGURED_provider(
    configured_provider: str, expected_failover_arg: str
) -> None:
    """An ACP cap must be attributed to the account, not to the transport.

    ``AcpUsageLimitError`` is raised by the SHARED ACP client, so its class
    name identifies the transport and says nothing about WHICH provider hit the
    cap. The handler used to hardcode ``_schedule_failover("claude-code")``,
    which was fine only while ACP had a single provider.

    When grok moved onto the shared client (2026-08-24) it stopped raising
    ``GrokUsageLimitError``, so a grok cap would have:
      1. recorded the failure against claude-code — poisoning the availability
         of a provider that never failed, and
      2. failed over along ``failover_candidates("claude-code")`` =
         [codex, grok, deepseek] — possibly straight back to grok, the
         provider that had just capped — instead of
         ``failover_candidates("grok")`` = [deepseek].

    The pre-existing grok test could not catch this: it raises a LOCAL
    ``GrokUsageLimitError`` stand-in, a class production no longer produces, so
    it stayed green while the real path routed to claude-code.
    """
    agent = _make_agent()
    agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]
    agent.config = SimpleNamespace(provider=configured_provider)

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        side_effect=AcpUsageLimitError("You have hit your usage limit")
    )
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR
    agent._schedule_failover.assert_awaited_once()
    assert agent._schedule_failover.await_args.args[0] == expected_failover_arg


@pytest.mark.asyncio
async def test_acp_usage_limit_without_config_still_attributes_claude_code() -> None:
    """No config → keep the historical target rather than crash or skip failover."""
    agent = _make_agent()
    agent._schedule_failover = AsyncMock()  # type: ignore[method-assign]

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        side_effect=AcpUsageLimitError("You have hit your usage limit")
    )
    agent.agent = fake_graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "do work", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    agent._schedule_failover.assert_awaited_once()
    assert agent._schedule_failover.await_args.args[0] == "claude-code"


def test_grok_cap_routes_to_deepseek_not_back_to_grok(tmp_path) -> None:
    """The consequence the routing exists to produce, pinned on the real chain.

    Asserting the ``_schedule_failover`` ARGUMENT proves attribution; this
    proves the argument actually yields a different, safe destination — and
    that the buggy target would have offered grok itself as a candidate.

    ``failover_candidates`` returns steps rather than names since chains became
    operator config; the property under test is unchanged.
    """
    from deep_worker_cli.runtime.failover_chains import get_chain
    from deep_worker_cli.runtime.provider_chain import failover_candidates

    chain = get_chain(home=tmp_path)

    def providers(after: str) -> list[str]:
        return [
            step.provider
            for step in failover_candidates(after, chain=chain, home=tmp_path)
        ]

    assert providers("grok") == ["deepseek"]
    assert "grok" not in providers("grok")
    assert "grok" in providers("claude-code")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
