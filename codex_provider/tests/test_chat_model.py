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

import json
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
from langchain_core.tools import tool

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider import CodexChatModel  # noqa: E402
from codex_provider.chat_model import resolve_model_from_session_log  # noqa: E402
from codex_provider.client_state import get_global_client_state  # noqa: E402
from codex_provider.exec_runner import (  # noqa: E402
    AgentMessageItem,
    GenericItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadStartedEvent,
    TurnCompletedEvent,
)


@pytest.fixture(autouse=True)
def isolate_global_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each test starts with a clean process-global state dict AND an
    isolated ``Path.home()``.

    The home isolation matters since the restart-persistence fix
    (DESIGN_HISTORY_RESTORE_ON_RESTART.md): ``CodexChatModel.ainvoke`` now
    persists ``sent_count`` + ``codex_session_uuid`` to the per-agent state
    file (``codex_session_state.json``) under the agent's data directory, and
    ``CodexClientState.get_or_create`` lazily seeds from it. Without pointing
    ``Path.home()`` at a fresh tmp dir per test, these tests would write into
    the real home and a leaked file would seed a non-zero ``sent_count`` /
    stale uuid into an unrelated test (observed: test_client_state fresh-state
    assertions failing).
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
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


class TestResolvedModelFromSessionLog:
    def test_extracts_turn_context_model(self, tmp_path: Path) -> None:
        thread_id = "019e6100-e176-7c01-ae9b-8995d28a7ebd"
        session_dir = tmp_path / "sessions" / "2026" / "05" / "25"
        session_dir.mkdir(parents=True)
        session_file = session_dir / f"rollout-2026-05-25T23-18-28-{thread_id}.jsonl"
        session_file.write_text(
            "not json\n"
            + json.dumps({"type": "session_meta", "payload": {"id": thread_id}})
            + "\n"
            + json.dumps({"type": "turn_context", "payload": {"model": "gpt-live-default"}})
            + "\n"
        )

        assert resolve_model_from_session_log(thread_id, codex_home=tmp_path) == "gpt-live-default"

    def test_missing_session_file_returns_none(self, tmp_path: Path) -> None:
        assert resolve_model_from_session_log("missing", codex_home=tmp_path) is None

    def test_ignores_turn_context_without_string_model(self, tmp_path: Path) -> None:
        thread_id = "uuid-no-model"
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        (session_dir / f"rollout-{thread_id}.jsonl").write_text(
            json.dumps({"type": "turn_context", "payload": {"model": None}}) + "\n"
        )

        assert resolve_model_from_session_log(thread_id, codex_home=tmp_path) is None


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
    """Stand-in for ExecHandle.

    Mirrors the real ExecHandle's contract for the unit-test surface:
    ``ainvoke_queue`` + ``native_futures``. We run the REAL
    ``event_pump`` against a scripted async generator so tests
    exercise the same routing logic as production.
    """

    def __init__(self, events: list[Any]) -> None:
        import asyncio as _asyncio
        from codex_provider.event_pump import event_pump

        async def _gen():
            for ev in events:
                yield ev

        self._scripted_gen = _gen()
        self.events = self._scripted_gen
        self.ainvoke_queue: _asyncio.Queue[Any] = _asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, _asyncio.Future[str]] = {}
        self.closed = False
        # Mirror the real ExecHandle: the pump records a fatal stream
        # error here so ainvoke can re-raise it on END_OF_STREAM.
        self.pump_error: BaseException | None = None
        # The pump runs the same routing the production pump does.
        self.pump_task: _asyncio.Task[None] = _asyncio.create_task(
            event_pump(self, proxy_mcp_server_name="deepworker_test"),
            name="fake-codex-event-pump",
        )

    async def aclose(self) -> None:
        import asyncio as _asyncio
        self.closed = True
        if not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except (_asyncio.CancelledError, BaseException):
                pass


class _FakeExecRunner:
    """Captures spawn arguments and returns scripted ExecHandles."""

    def __init__(self) -> None:
        self.spawn_calls: list[Any] = []
        self.next_events: list[list[Any]] = []

    def queue_events(self, events: list[Any]) -> None:
        self.next_events.append(events)

    async def spawn(
        self,
        args: Any,
        *,
        proxy_mcp_server_name: str | None = None,
    ) -> Any:
        # Mirror real spawn signature; we ignore proxy_mcp_server_name
        # in the fake because every script uses "deepworker_test".
        _ = proxy_mcp_server_name
        self.spawn_calls.append(args)
        events = self.next_events.pop(0) if self.next_events else []
        return _FakeExecHandle(events)


@pytest.mark.asyncio
class TestAinvokeHappyPath:
    async def test_first_turn_reports_resolved_default_model(self, tmp_path: Path) -> None:
        runner = _FakeExecRunner()
        thread_id = "codex-uuid-model"
        session_dir = tmp_path / ".codex" / "sessions"
        session_dir.mkdir(parents=True)
        (session_dir / f"rollout-{thread_id}.jsonl").write_text(
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-dynamic-default"}}) + "\n"
        )
        runner.queue_events([
            ThreadStartedEvent(thread_id=thread_id),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="HELLO")),
            TurnCompletedEvent(),
        ])
        seen: list[str] = []

        async def on_model(model: str) -> None:
            seen.append(model)

        m = CodexChatModel()
        m.set_resolved_model_callback(on_model)
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        await m.ainvoke(
            [HumanMessage(content="say HELLO")],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        assert m.resolved_model_name == "gpt-dynamic-default"
        assert seen == ["gpt-dynamic-default"]
        assert m._model_name is None

    async def test_resolved_model_callback_invoked_once_for_same_value(self, tmp_path: Path) -> None:
        runner = _FakeExecRunner()
        thread_id = "codex-uuid-once"
        session_dir = tmp_path / ".codex" / "sessions"
        session_dir.mkdir(parents=True)
        (session_dir / f"rollout-{thread_id}.jsonl").write_text(
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-once"}}) + "\n"
        )
        runner.queue_events([
            ThreadStartedEvent(thread_id=thread_id),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="HELLO")),
            TurnCompletedEvent(),
        ])
        runner.queue_events([
            ThreadStartedEvent(thread_id=thread_id),
            ItemCompletedEvent(item=AgentMessageItem(id="i1", text="HELLO2")),
            TurnCompletedEvent(),
        ])
        seen: list[str] = []

        m = CodexChatModel()
        m.set_resolved_model_callback(lambda model: seen.append(model))
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        await m.ainvoke(
            [HumanMessage(content="one")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        state = get_global_client_state().get_or_create("agent_x")
        state.codex_session_uuid = None
        state.sent_count = 0
        await m.ainvoke(
            [HumanMessage(content="two")],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        assert seen == ["gpt-once"]

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
    async def test_tool_call_breaks_with_pending_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        proxy.has_pending_calls = MagicMock(return_value=False)
        proxy.get_pending_calls = MagicMock(return_value=[])
        proxy.wait_for_pending_calls = AsyncMock(return_value=[
            {"id": "read_file_1_xxx", "name": "read_file", "args": {"path": "/tmp/x"}},
        ])
        proxy.tool_names = ["read_file"]
        proxy.server_name = "deepworker_999"
        m._proxy_mcp = proxy
        m._proxy_started = True

        async def _sleep_is_polling(_delay: float) -> None:
            raise AssertionError("polling sleep is forbidden in CodexChatModel")

        monkeypatch.setattr("codex_provider.chat_model.asyncio.sleep", _sleep_is_polling)

        result = await m.ainvoke(
            [HumanMessage(content="read /tmp/x")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert result.tool_calls
        assert result.tool_calls[0]["name"] == "read_file"
        proxy.wait_for_pending_calls.assert_awaited_once_with(
            tool_name="read_file",
            arguments={"path": "/tmp/x"},
            timeout=30.0,
        )
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

    async def test_system_prompt_extracted_as_plain_text_for_string_content(
        self,
    ) -> None:
        """SystemMessage with plain string content must reach
        ExecArgs.base_instructions verbatim (no escaping, no repr).
        Regression baseline: string case was already correct."""
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="ok")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        await m.ainvoke(
            [
                SystemMessage(content="You are deep-worker. Use proxy tools only."),
                HumanMessage(content="hi"),
            ],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        assert len(runner.spawn_calls) == 1
        bi = runner.spawn_calls[0].base_instructions
        assert isinstance(bi, str)
        assert "You are deep-worker. Use proxy tools only." in bi

    async def test_system_prompt_flattens_content_blocks_to_plain_text(
        self,
    ) -> None:
        """SystemMessage with content_blocks (the shape produced by
        deepagents middleware that calls request.override(system_message=
        SystemMessage(content_blocks=[...]))) must be flattened to plain
        text — NOT serialised as Python ``repr(list)``.

        BEFORE the fix this is what reached AGENTS.override.md:

            [{'type': 'text', 'text': 'You are an AI assistant ...'},
             {'type': 'text', 'text': '... Project Knowledge Memory ...'}]

        Codex still parses it as a long literal token soup but the model
        can no longer reliably follow the instructions, and any
        non-text block (image, tool_use) would land in the file as raw
        Python repr — a giant footgun.
        """
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="ok")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        system_msg = SystemMessage(content=[
            {"type": "text", "text": "You are deep-worker."},
            {"type": "text", "text": "Use proxy tools only."},
        ])
        await m.ainvoke(
            [system_msg, HumanMessage(content="hi")],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        bi = runner.spawn_calls[0].base_instructions
        assert isinstance(bi, str)
        # The Python repr-style markers must NEVER appear.
        assert "{'type': 'text'" not in bi
        assert "[{" not in bi
        # The actual texts must be present and ordered.
        assert "You are deep-worker." in bi
        assert "Use proxy tools only." in bi
        idx1 = bi.index("You are deep-worker.")
        idx2 = bi.index("Use proxy tools only.")
        assert idx1 < idx2, "text blocks must appear in their original order"

    async def test_system_prompt_skips_non_text_blocks_in_content(self) -> None:
        """Non-text blocks (images, tool_use, anything else) must be
        skipped silently when collapsing content_blocks to a base
        instructions string. AGENTS.override.md is a plain markdown
        file — there is no place for an image_url block there.
        """
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="ok")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        system_msg = SystemMessage(content=[
            {"type": "text", "text": "PART_ONE"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text", "text": "PART_TWO"},
        ])
        await m.ainvoke(
            [system_msg, HumanMessage(content="hi")],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        bi = runner.spawn_calls[0].base_instructions
        assert isinstance(bi, str)
        assert "PART_ONE" in bi
        assert "PART_TWO" in bi
        # No raw image-block leakage.
        assert "image_url" not in bi
        assert "http://x" not in bi

    async def test_native_item_started_emits_real_tool_call_to_langgraph(
        self,
    ) -> None:
        """When Codex emits ``item.started`` for a native item, the
        provider must return an ``AIMessage`` carrying a tool_call
        with Codex's REAL native tool name, the REAL item.id, and the
        REAL arguments. No synthetic prefix, no surrogate name.

        This replaces the old ``test_native_codex_tool_item_is_rejected``
        — we no longer reject native tool events, we record them as
        passthrough tool calls. The async tool body in
        ``codex_provider.native_tools`` parks on a Future that the
        event pump resolves at ``item.completed``.
        """
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            ItemStartedEvent(item=GenericItem(
                id="item_0",
                type="command_execution",
                payload={
                    "command": "/bin/bash -lc 'ls /tmp'",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            )),
        ])

        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.server_name = "deepworker_999"
        m._proxy_started = True

        # NO RuntimeError. We must get back an AIMessage with a
        # tool_call describing the native command_execution.
        result = await m.ainvoke(
            [HumanMessage(content="run ls /tmp")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)
        assert result.tool_calls, "ainvoke must surface a tool_call"
        call = result.tool_calls[0]
        assert call["name"] == "command_execution", (
            "tool_call name MUST be the Codex native name verbatim — "
            "no synthetic prefix"
        )
        assert call["id"] == "item_0", (
            "tool_call id MUST be Codex's own item.id"
        )
        assert call["args"]["command"] == "/bin/bash -lc 'ls /tmp'"

    async def test_native_toolmessage_does_not_call_proxy_resolve_call(
        self,
    ) -> None:
        """When LangGraph hands us back a ToolMessage produced by the
        native passthrough path (marked via
        ``additional_kwargs['codex_native']=True``), the provider must
        NOT call ``_proxy_mcp.resolve_call`` for it — Codex was never
        blocked on our HTTP handler for that tool_call. It also must
        NOT spawn a fresh ``codex exec``: the existing one is still
        alive and waiting for events on its ainvoke_queue.
        """
        runner = _FakeExecRunner()
        # Pre-seed state so it looks like turn 1 already created a
        # native pending call. We pre-fill state.current_exec with a
        # live (but scripted) handle that emits the next event:
        # turn.completed → ainvoke returns a final AIMessage.
        seeded_handle = _FakeExecHandle([
            ItemCompletedEvent(item=AgentMessageItem(id="i1", text="done")),
            TurnCompletedEvent(),
        ])
        state = get_global_client_state().get_or_create("agent_x")
        state.current_exec = seeded_handle  # type: ignore[assignment]
        state.codex_session_uuid = "uuid-A"
        state.sent_count = 1  # one HumanMessage already mirrored

        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.resolve_call = MagicMock()
        m._proxy_mcp.server_name = "deepworker_999"
        m._proxy_started = True

        # Pass back the native ToolMessage carrying the codex_native
        # marker. The original HumanMessage stays in the history so
        # sent_count math lines up.
        native_tm = ToolMessage(
            content="file1\nfile2",
            tool_call_id="item_0",
            name="command_execution",
            additional_kwargs={"codex_native": True},
        )
        await m.ainvoke(
            [HumanMessage(content="run ls /tmp"), native_tm],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        # Crucial: resolve_call NEVER called for a native ToolMessage
        assert not m._proxy_mcp.resolve_call.called, (
            "resolve_call must NOT be invoked for native passthrough "
            "ToolMessages — Codex never paused for them"
        )
        # And no fresh exec was spawned — we kept reading the existing
        # handle's queue.
        assert len(runner.spawn_calls) == 0, (
            "native ToolMessage resume must NOT spawn a new codex exec"
        )

    async def test_no_runtime_error_for_native_command_execution(
        self,
    ) -> None:
        """End-to-end native-only turn: ``item.started`` →
        ``item.completed`` → eventually ``turn.completed``. The
        provider must not raise the legacy RuntimeError that used to
        kill the turn under the old defense-in-depth scheme."""
        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-A"),
            ItemStartedEvent(item=GenericItem(
                id="item_0",
                type="command_execution",
                payload={
                    "command": "/bin/bash -lc 'ls /tmp'",
                    "status": "in_progress",
                },
            )),
        ])

        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.server_name = "deepworker_999"
        m._proxy_started = True

        # Must NOT raise. Whatever it returns is fine; we just assert
        # we got an AIMessage rather than an exception.
        result = await m.ainvoke(
            [HumanMessage(content="run ls /tmp")],
            config={"configurable": {"thread_id": "agent_x"}},
        )
        assert isinstance(result, AIMessage)


class _RaisingExecHandle:
    """ExecHandle stand-in whose event stream raises mid-stream.

    Runs the REAL ``event_pump`` against a generator that yields an
    optional prefix then raises ``exc`` — mirroring ``_iter_events``
    hitting a Codex API rejection (e.g. unsupported model → HTTP 400).
    The real pump records the error on ``pump_error`` and enqueues
    END_OF_STREAM, exactly like production.
    """

    def __init__(self, exc: BaseException, prefix: list[Any] | None = None) -> None:
        import asyncio as _asyncio
        from codex_provider.event_pump import event_pump

        async def _gen():
            for ev in (prefix or []):
                yield ev
            raise exc

        self.events = _gen()
        self.ainvoke_queue: _asyncio.Queue[Any] = _asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, _asyncio.Future[str]] = {}
        self.closed = False
        self.pump_error: BaseException | None = None
        self.pump_task: _asyncio.Task[None] = _asyncio.create_task(
            event_pump(self, proxy_mcp_server_name="deepworker_test"),
            name="raising-codex-event-pump",
        )

    async def aclose(self) -> None:
        import asyncio as _asyncio
        self.closed = True
        if not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except (_asyncio.CancelledError, BaseException):
                pass


class _RaisingExecRunner:
    def __init__(self, exc: BaseException, prefix: list[Any] | None = None) -> None:
        self._exc = exc
        self._prefix = prefix
        self.spawn_calls: list[Any] = []

    async def spawn(self, args: Any, *, proxy_mcp_server_name: str | None = None) -> Any:
        _ = proxy_mcp_server_name
        self.spawn_calls.append(args)
        return _RaisingExecHandle(self._exc, self._prefix)


@pytest.mark.asyncio
class TestAinvokeSurfacesPumpError:
    """A fatal codex stream error must propagate OUT of ainvoke, not be
    swallowed into an empty AIMessage (observed 2026-05-20)."""

    async def test_codex_protocol_error_propagates(self) -> None:
        from codex_provider.exec_runner import CodexProtocolError

        boom = CodexProtocolError(
            'Codex stream error: {"status":400,"error":{"message":'
            '"The \'unknown\' model is not supported"}}'
        )
        runner = _RaisingExecRunner(boom)
        m = CodexChatModel()
        m._exec_runner = runner  # type: ignore[attr-defined]
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        with pytest.raises(CodexProtocolError, match="not supported"):
            await m.ainvoke(
                [HumanMessage(content="hi")],
                config={"configurable": {"thread_id": "agent_err"}},
            )

    async def test_error_after_thread_started_still_propagates(self) -> None:
        """Even if some events arrive before the error, ainvoke must
        still raise rather than return a partial/empty result."""
        from codex_provider.exec_runner import CodexProtocolError

        boom = CodexProtocolError("Codex stream error: mid-stream boom")
        runner = _RaisingExecRunner(
            boom, prefix=[ThreadStartedEvent(thread_id="uuid-z")]
        )
        m = CodexChatModel()
        m._exec_runner = runner  # type: ignore[attr-defined]
        m._proxy_mcp = MagicMock()
        m._proxy_mcp.start = AsyncMock()
        m._proxy_mcp.has_pending_calls = MagicMock(return_value=False)
        m._proxy_mcp.tool_names = []
        m._proxy_mcp.server_name = "deepworker_test"
        m._proxy_started = True

        with pytest.raises(CodexProtocolError, match="boom"):
            await m.ainvoke(
                [HumanMessage(content="hi")],
                config={"configurable": {"thread_id": "agent_err2"}},
            )
