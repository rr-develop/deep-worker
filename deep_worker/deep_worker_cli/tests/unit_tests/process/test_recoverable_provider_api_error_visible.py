"""Regression: a RECOVERABLE provider/API error must be surfaced to the UI as a
visible, persistent ERROR — not disguised as a benign ``type="system"`` "please
type continue" note (which the web renders as nothing durable) and not a toast
that self-dismisses.

Past incident: grok returned HTTP 402 nested in the JSON-RPC
error. Even before the failover-detection fix, the *class* of bug is that a
provider API error reaching the recoverable path in ``_handle_invoke`` notified
subscribers with ``type="system"`` and appended a "type continue" system
message — the actual API error was neither shown as an error toast nor pinned
in the persistent error panel. POLICIES.md §9 forbids silent/fleeting API
errors: recoverable ones stay IDLE but MUST emit ``type="error"`` carrying the
real API text.

We drive the REAL ``_handle_invoke`` with a graph whose ``ainvoke`` raises a
provider-API error class (matched by name via ``PROVIDER_API_ERROR_TYPES``) and
assert the notification the subscribers receive.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class GrokAcpClientError(RuntimeError):
    """Stand-in matching ``grok_provider.acp_client.GrokAcpClientError`` by
    class NAME (the handler keys on ``type(e).__name__`` so it stays
    provider-agnostic)."""


class CodexProtocolError(RuntimeError):
    """Stand-in matching ``codex_provider.exec_runner.CodexProtocolError``."""


class SomeInternalHiccup(RuntimeError):
    """A NON-provider recoverable error — must keep the old benign
    ``type="system"`` behaviour (not every recoverable error is an API error)."""


def _make_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_provider_api_err_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_provider_api_err"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    agent.thread_id = "default"
    # Avoid the aupdate_state history append touching a real checkpointer.
    agent._append_recoverable_error_to_history = AsyncMock()  # type: ignore[method-assign]
    return agent


async def _invoke_capturing(
    agent: AgentProcess,
) -> tuple[list[dict], dict]:
    events: list[dict] = []

    async def capture(msg: dict) -> None:
        events.append(msg)

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "2+2", "stream": False},
        connection=None,
        notify_callback=capture,
        approval_callback=None,
    )
    return events, result


@pytest.mark.asyncio
async def test_recoverable_grok_api_error_emits_error_event_not_system() -> None:
    agent = _make_agent()
    api_text = (
        "session/prompt failed: API error (status 502 Bad Gateway): "
        "upstream connection reset"
    )
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=GrokAcpClientError(api_text))
    agent.agent = fake_graph

    events, result = await _invoke_capturing(agent)

    # The agent stays alive (recoverable) — NOT ERROR.
    assert agent.session_manager.status == AgentStatus.IDLE

    # It MUST be broadcast as a visible ERROR event carrying the real API text.
    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, (
        "a recoverable provider/API error must be broadcast as type='error' "
        "(persistent UI surface), not type='system' (POLICIES.md §9)"
    )
    assert "502" in error_events[-1]["content"] or "Bad Gateway" in (
        error_events[-1]["content"]
    )
    # And it must NOT be disguised as a benign system 'type continue' note.
    system_events = [e for e in events if e.get("type") == "system"]
    assert not system_events, (
        "a provider/API error must not be hidden behind a benign system note"
    )
    # The return value must ALSO carry the real API text (not the generic
    # wrapper), so a programmatic consumer of result['error'] sees the cause
    # (review nit).
    assert result["success"] is False
    assert "502" in result["error"] or "Bad Gateway" in result["error"], (
        f"result['error'] must carry the real API cause; got: {result['error']!r}"
    )


@pytest.mark.asyncio
async def test_recoverable_codex_api_error_emits_error_event() -> None:
    agent = _make_agent()
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        side_effect=CodexProtocolError("codex stream error: malformed frame")
    )
    agent.agent = fake_graph

    events, result = await _invoke_capturing(agent)
    assert agent.session_manager.status == AgentStatus.IDLE
    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, "codex protocol error must surface as type='error'"
    assert "codex" in error_events[-1]["content"].lower()
    assert "codex" in result["error"].lower(), (
        "result['error'] must carry the real codex API cause"
    )


@pytest.mark.asyncio
async def test_non_provider_recoverable_error_stays_system() -> None:
    """Not every recoverable error is an API error. A generic internal hiccup
    keeps the benign ``type="system"`` 'type continue' behaviour so we don't
    cry wolf on the persistent error panel."""
    agent = _make_agent()
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=SomeInternalHiccup("boom"))
    agent.agent = fake_graph

    events, result = await _invoke_capturing(agent)
    assert agent.session_manager.status == AgentStatus.IDLE
    assert result["success"] is False
    system_events = [e for e in events if e.get("type") == "system"]
    assert system_events, (
        "a generic recoverable error should remain a benign system note"
    )
    error_events = [e for e in events if e.get("type") == "error"]
    assert not error_events, (
        "a generic (non-API) recoverable error must not raise a persistent "
        "error surface"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
