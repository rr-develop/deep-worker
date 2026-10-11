"""TDD for the codex usage-event callback (DESIGN_AGENT_TOKEN_STATS.md §5.2).

``CodexChatModel`` reports token-stats through the SAME event callback
contract as claude-code-sdk:

* ``ainvoke`` start → ``cb(event="invocation_start")`` (no record).
* ``TurnCompletedEvent`` (terminal, per-turn usage, branch A) →
  ``cb(event="terminal", status="ok", usage=..., session_id=<codex uuid>,
      turn_id=<last AgentMessageItem.id>)``.
* ``TurnFailedEvent`` (terminal failure) → ``cb(event="terminal",
  status="failed", usage=None, ...)``.
* Tool early-return (turn not over) → no terminal event.
* Best-effort: callback exceptions are logged, never break the turn —
  except ``asyncio.CancelledError`` which propagates.

Subprocess and proxy MCP listener are mocked (same harness as
``test_chat_model.py``).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_pkg_root = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402

if str(_pkg_root) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.codex_provider import CodexChatModel  # noqa: E402
from deep_worker.providers.codex_provider.client_state import get_global_client_state  # noqa: E402
from deep_worker.providers.codex_provider.exec_runner import (  # noqa: E402
    AgentMessageItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from tests.codex_provider.test_chat_model import _FakeExecRunner  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_global_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    get_global_client_state()._states.clear()  # type: ignore[attr-defined]


def _make_model(runner: _FakeExecRunner) -> CodexChatModel:
    m = CodexChatModel()
    m._exec_runner = runner  # type: ignore[attr-defined]  # injection for tests
    m._proxy_mcp = MagicMock()
    m._proxy_mcp.start = AsyncMock()
    m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
    m._proxy_mcp.tool_names = []
    m._proxy_mcp.server_name = "deepworker_test"
    m._proxy_started = True  # type: ignore[attr-defined]  # skip real codex mcp add
    return m


class _RecordingCallback:
    """Records every event the chat model emits (§5.2 contract)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


@pytest.mark.asyncio
class TestCodexUsageEventCallback:
    async def test_installs_and_emits_invocation_start_then_terminal_ok(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-1"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="HELLO")),
            TurnCompletedEvent(
                usage={"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 4}
            ),
        ])
        m = _make_model(runner)
        cb = _RecordingCallback()
        m.set_usage_event_callback(cb)

        result = await m.ainvoke(
            [HumanMessage(content="say HELLO")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert "HELLO" in result.content

        assert len(cb.events) == 2
        assert cb.events[0]["event"] == "invocation_start"
        term = cb.events[1]
        assert term["event"] == "terminal"
        assert term["status"] == "ok"
        assert term["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_input_tokens": 4,
        }
        assert term["session_id"] == "codex-uuid-1"
        assert term["turn_id"] == "i0"
        assert term["model"] == ""

    async def test_turn_completed_without_agent_message_has_null_turn_id(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-2"),
            TurnCompletedEvent(usage={"input_tokens": 3, "output_tokens": 1}),
        ])
        m = _make_model(runner)
        cb = _RecordingCallback()
        m.set_usage_event_callback(cb)

        await m.ainvoke(
            [HumanMessage(content="hi")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        term = cb.events[-1]
        assert term["event"] == "terminal"
        assert term["status"] == "ok"
        assert term["session_id"] == "codex-uuid-2"
        assert term["turn_id"] is None

    async def test_turn_failed_emits_terminal_failed_then_raises(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-3"),
            ItemCompletedEvent(item=AgentMessageItem(id="i9", text="partial")),
            TurnFailedEvent(error_message="session not found"),
        ])
        m = _make_model(runner)
        cb = _RecordingCallback()
        m.set_usage_event_callback(cb)

        with pytest.raises(RuntimeError, match="Codex turn failed"):
            await m.ainvoke(
                [HumanMessage(content="hi")],
                config={"configurable": {"thread_id": "agent_x"}},
            )
        assert len(cb.events) == 2
        assert cb.events[0]["event"] == "invocation_start"
        term = cb.events[1]
        assert term["event"] == "terminal"
        assert term["status"] == "failed"
        assert term["usage"] is None
        assert term["session_id"] == "codex-uuid-3"
        assert term["turn_id"] == "i9"

    async def test_tool_early_return_emits_no_terminal_event(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-4"),
            ItemStartedEvent(
                item=McpToolCallItem(
                    id="t1",
                    server="deepworker_test",
                    tool="execute",
                    arguments={"cmd": "ls"},
                    status="in_progress",
                )
            ),
        ])
        m = _make_model(runner)
        m._proxy_mcp.wait_for_pending_calls = AsyncMock(
            return_value=[{"id": "t1", "name": "execute", "args": {"cmd": "ls"}}]
        )
        cb = _RecordingCallback()
        m.set_usage_event_callback(cb)

        result = await m.ainvoke(
            [HumanMessage(content="run ls")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        # Turn is NOT over — no terminal event, only the invocation start.
        assert [e["event"] for e in cb.events] == ["invocation_start"]

    async def test_callback_exception_does_not_break_turn(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-5"),
            ItemCompletedEvent(item=AgentMessageItem(id="i5", text="OK")),
            TurnCompletedEvent(usage={"input_tokens": 1, "output_tokens": 1}),
        ])
        m = _make_model(runner)

        async def boom(**kwargs: Any) -> None:
            raise RuntimeError("stats recorder exploded")

        m.set_usage_event_callback(boom)
        result = await m.ainvoke(
            [HumanMessage(content="hi")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert "OK" in result.content

    async def test_callback_cancelled_error_propagates(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-6"),
            ItemCompletedEvent(item=AgentMessageItem(id="i6", text="OK")),
            TurnCompletedEvent(usage={"input_tokens": 1, "output_tokens": 1}),
        ])
        m = _make_model(runner)

        async def cancel(**kwargs: Any) -> None:
            raise asyncio.CancelledError

        m.set_usage_event_callback(cancel)
        with pytest.raises(asyncio.CancelledError):
            await m.ainvoke(
                [HumanMessage(content="hi")],
                config={"configurable": {"thread_id": "agent_x"}},
            )

    async def test_no_callback_is_noop(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-7"),
            ItemCompletedEvent(item=AgentMessageItem(id="i7", text="OK")),
            TurnCompletedEvent(usage={"input_tokens": 1, "output_tokens": 1}),
        ])
        m = _make_model(runner)
        result = await m.ainvoke(
            [HumanMessage(content="hi")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
