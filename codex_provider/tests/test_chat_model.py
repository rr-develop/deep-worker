"""Tests for ``codex_provider.chat_model.CodexChatModel``.

Covered behaviours:

* ``__init__`` constructs cleanly; Pydantic JSON-Schema /  model_dump
  overrides exclude callable fields (LangGraph pickles models during
  checkpointing — must not crash on ``custom_get_token_ids``).
* ``_llm_type`` returns ``"codex"``.
* ``_generate`` raises ``NotImplementedError`` (async-only provider).
* ``bind_tools`` extracts metadata from BaseTool / StructuredTool.
* ``ainvoke`` requires explicit ``thread_id`` (no fallback).
* ``ainvoke`` on the first turn captures Codex's session UUID, on the
  next turn passes it to ``codex exec resume <uuid>``.
* ``ainvoke`` mirrors only the tail ``messages[sent_count:]`` per turn.
* On ``sent_count > len(messages)`` the Codex session UUID is dropped
  (summarization). The ``agent_thread_id`` key is never mutated.
* ``ToolMessage`` resolves a Future on the proxy MCP server; it is
  NOT serialized into the prompt.

Subprocess and proxy MCP listener are mocked; the live integration test
lives in ``test_integration.py``.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider import CodexChatModel  # noqa: E402
from codex_provider.client_state import get_global_client_state  # noqa: E402
from codex_provider.exec_runner import (  # noqa: E402
    AgentMessageItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadStartedEvent,
    TurnCompletedEvent,
)


@pytest.fixture(autouse=True)
def _isolate_global_state() -> None:
    """Each test starts with a clean process-global state dict."""
    get_global_client_state()._states.clear()  # type: ignore[attr-defined]


class TestConstruction:
    def test_minimal_construction(self) -> None:
        m = CodexChatModel()
        assert m._llm_type == "codex"

    def test_construction_with_model(self) -> None:
        m = CodexChatModel(model="gpt-5.1")
        assert m._llm_type == "codex"

    def test_pydantic_model_dump_excludes_callables(self) -> None:
        """LangGraph pickles models — ``custom_get_token_ids`` is callable
        and must not blow up JSON-schema generation."""
        m = CodexChatModel(model="gpt-5.1")
        dumped = m.model_dump()
        assert "custom_get_token_ids" not in dumped
        assert "callbacks" not in dumped
        assert "rate_limiter" not in dumped

    def test_generate_is_async_only(self) -> None:
        m = CodexChatModel()
        with pytest.raises(NotImplementedError):
            m._generate([HumanMessage(content="hi")])

    def test_working_dir_settable_via_setattr(self) -> None:
        """agent_process.py:447-448 does setattr(model, '_working_dir', ...).
        This must work."""
        m = CodexChatModel()
        m._working_dir = "/tmp/agent_workdir"
        assert m._working_dir == "/tmp/agent_workdir"


class TestBindTools:
    def test_bind_tools_extracts_metadata_from_decorated_tool(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        m = CodexChatModel()
        bound = m.bind_tools([add])
        # bind_tools returns a model (LangChain pattern). Our impl returns self.
        assert bound._tools_metadata
        meta = bound._tools_metadata[0]
        assert meta["name"] == "add"
        assert "Add" in meta["description"]
        assert meta["input_schema"]["type"] == "object"
        assert set(meta["input_schema"]["properties"].keys()) >= {"a", "b"}

    def test_bind_tools_skips_non_basetool(self) -> None:
        m = CodexChatModel()
        bound = m.bind_tools([42, "not a tool"])  # type: ignore[arg-type]
        assert bound._tools_metadata == []


@pytest.mark.asyncio
class TestAinvokeContract:
    async def test_requires_thread_id_in_config(self) -> None:
        m = CodexChatModel()
        with pytest.raises(ValueError, match="thread_id"):
            await m.ainvoke([HumanMessage(content="hi")])

    async def test_requires_thread_id_not_just_configurable(self) -> None:
        m = CodexChatModel()
        with pytest.raises(ValueError, match="thread_id"):
            await m.ainvoke(
                [HumanMessage(content="hi")],
                config={"configurable": {}},
            )


# ---------------------------------------------------------------------------
# Mocked exec_runner — drives the event stream for ainvoke tests.
# ---------------------------------------------------------------------------
class _FakeExecHandle:
    """Stand-in for ExecHandle with a scripted event stream."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    async def _gen(self):
        for ev in self._events:
            yield ev

    @property
    def events(self):
        return self._gen()

    async def aclose(self) -> None:
        self.closed = True


class _FakeExecRunner:
    """Captures spawn arguments and returns scripted ExecHandles."""

    def __init__(self) -> None:
        self.spawn_calls: list[Any] = []
        self.next_events: list[list[Any]] = []

    def queue_events(self, events: list[Any]) -> None:
        self.next_events.append(events)

    async def spawn(self, args: Any) -> Any:
        self.spawn_calls.append(args)
        events = self.next_events.pop(0) if self.next_events else []
        return _FakeExecHandle(events)


@pytest.mark.asyncio
class TestAinvokeHappyPath:
    async def test_first_turn_captures_codex_session_uuid(self) -> None:
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-1"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="HELLO")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner  # injection for tests
        # No bind_tools — proxy can be a no-op stub.
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True  # skip real codex mcp add in tests

        result = await m.ainvoke(
            [HumanMessage(content="say HELLO")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert "HELLO" in result.content

        state = get_global_client_state().get("agent_x")
        assert state is not None
        assert state.codex_session_uuid == "codex-uuid-1"
        assert state.sent_count == 1  # one HumanMessage was mirrored

    async def test_second_turn_resumes_with_cached_uuid(self) -> None:
        runner = _FakeExecRunner()
        # Pre-seed state as if turn 1 already happened
        get_global_client_state().get_or_create("agent_x").codex_session_uuid = "uuid-A"
        get_global_client_state().get_or_create("agent_x").sent_count = 2

        runner.queue_events([
            ItemCompletedEvent(item=AgentMessageItem(id="i1", text="hi again")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        # Turn 2: history has 3 messages, sent_count is 2 → delta = 1
        await m.ainvoke(
            [
                HumanMessage(content="msg1"),
                AIMessage(content="reply1"),
                HumanMessage(content="msg2"),  # the new one
            ],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert len(runner.spawn_calls) == 1
        args = runner.spawn_calls[0]
        # Should resume the cached uuid
        assert args.session_uuid == "uuid-A"
        # Prompt should contain msg2 but not msg1 (incremental mirror)
        assert "msg2" in args.prompt
        assert "msg1" not in args.prompt

    async def test_summarization_drops_codex_session_uuid(self) -> None:
        """When LangGraph shrinks history (sent_count > len(messages)),
        the provider drops the Codex session UUID and starts fresh.
        Crucially: agent_thread_id itself is NOT touched."""
        runner = _FakeExecRunner()
        get_global_client_state().get_or_create("agent_x").codex_session_uuid = "old-uuid"
        get_global_client_state().get_or_create("agent_x").sent_count = 500

        runner.queue_events([
            ThreadStartedEvent(thread_id="new-uuid"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="ok")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        # History shrank to 5 messages — way below sent_count=500
        await m.ainvoke(
            [HumanMessage(content="post-summary") for _ in range(5)],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        # First spawn was called WITHOUT session_uuid (fresh session)
        assert runner.spawn_calls[0].session_uuid is None
        # Slot still exists under the same key — only contents reset
        state = get_global_client_state().get("agent_x")
        assert state is not None
        # New session captured, sent_count updated
        assert state.codex_session_uuid == "new-uuid"
        assert state.sent_count == 5


@pytest.mark.asyncio
class TestAinvokeToolCalls:
    async def test_tool_call_breaks_with_pending_calls(self) -> None:
        """When Codex emits an mcp_tool_call item, ainvoke returns
        AIMessage(tool_calls=[...]) and DOES NOT continue consuming the
        stream. The exec stays alive for the next ainvoke."""
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            # item.started for mcp_tool_call is what triggers the break
            ItemStartedEvent(item=McpToolCallItem(
                id="i0",
                server="deepworker_999",
                tool="read_file",
                arguments={"path": "/tmp/x"},
                status="in_progress",
            )),
            # extra events the loop should NOT process — proves we broke
            ItemCompletedEvent(item=AgentMessageItem(id="i1", text="unreachable")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner

        proxy = MagicMock()
        proxy.start = AsyncMock()
        proxy.has_pending_calls = MagicMock(return_value=True)
        proxy.get_pending_calls = MagicMock(return_value=[
            {"id": "read_file_1_xxx", "name": "read_file", "args": {"path": "/tmp/x"}},
        ])
        proxy.tool_names = ["read_file"]
        proxy.server_name = "deepworker_999"
        m._proxy_mcp = proxy
        m._proxy_started = True

        result = await m.ainvoke(
            [HumanMessage(content="read /tmp/x")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        assert result.tool_calls[0]["name"] == "read_file"
        # Text content should be empty or not contain the unreachable msg
        assert "unreachable" not in (result.content or "")

    async def test_toolmessage_resolves_future_not_in_prompt(self) -> None:
        """A ToolMessage in the delta must call resolve_call and NOT
        be serialized into the prompt sent to Codex."""
        runner = _FakeExecRunner()
        # Pre-state: a turn previously emitted a tool_call and we have
        # a current_exec alive.
        state = get_global_client_state().get_or_create("agent_x")
        state.codex_session_uuid = "uuid-A"
        state.sent_count = 2
        # Fake current exec with no further events — the resolve_call
        # path is what we're checking, not the final answer here.
        state.current_exec = _FakeExecHandle([
            ItemCompletedEvent(item=AgentMessageItem(id="iN", text="done")),
            TurnCompletedEvent(),
        ])

        proxy = MagicMock()
        proxy.start = AsyncMock()
        proxy.has_pending_calls = MagicMock(return_value=False)
        proxy.get_pending_calls = MagicMock(return_value=[])
        proxy.tool_names = []
        proxy.server_name = "deepworker_999"
        proxy.resolve_call = MagicMock()

        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = proxy
        m._proxy_started = True

        messages = [
            HumanMessage(content="read /tmp/x"),
            AIMessage(content="", tool_calls=[{
                "id": "tc-1", "name": "read_file", "args": {"path": "/tmp/x"}
            }]),
            ToolMessage(
                content="file contents here",
                tool_call_id="tc-1",
                name="read_file",
            ),
        ]
        await m.ainvoke(
            messages,
            config={"configurable": {"thread_id": "agent_x"}},
        )

        # resolve_call was invoked with the tool_call_id
        assert proxy.resolve_call.called
        call_args = proxy.resolve_call.call_args
        assert call_args.args[0] == "tc-1"
        # The MCPResult dict contains a text block with the tool output
        result_arg = call_args.args[1]
        assert any(b.get("text") == "file contents here"
                   for b in result_arg.get("content", []))
        # No new exec was spawned — we kept consuming the existing one
        assert len(runner.spawn_calls) == 0
