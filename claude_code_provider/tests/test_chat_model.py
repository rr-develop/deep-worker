"""Tests for ClaudeCodeChatModel - LangChain wrapper around Claude Code SDK."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool as langchain_tool
from pydantic import BaseModel, Field

from claude_code_provider.chat_model import ClaudeCodeChatModel


# Test tool for binding — StructuredTool (via @tool decorator)
@langchain_tool
def test_calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@langchain_tool
def test_reader(path: str) -> str:
    """Read a file."""
    return f"Content of {path}"


# Test tool — manual BaseTool subclass (like WebSearchDelegationTool).
# Has _arun() but NO .coroutine/.func attributes.
class _CustomSearchInput(BaseModel):
    query: str = Field(description="Search query")


class CustomSearchTool(BaseTool):
    """Manual BaseTool subclass for testing executor extraction."""

    name: str = "custom_search"
    description: str = "Search the web"
    args_schema: type[BaseModel] = _CustomSearchInput  # type: ignore[assignment]

    async def _arun(self, query: str) -> str:
        return f"Results for: {query}"

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async")


class TestClaudeCodeChatModelInit:
    """Tests for ClaudeCodeChatModel initialization."""

    def test_init_with_base_options(self) -> None:
        """Should initialize with base options."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        model = ClaudeCodeChatModel(base_options=options)

        assert model.client_manager is not None
        assert model.tools_metadata == []

    def test_init_without_options(self) -> None:
        """Should initialize without options."""
        model = ClaudeCodeChatModel(base_options=None)

        assert model.client_manager is not None
        assert model.tools_metadata == []


class TestClaudeCodeChatModelBindTools:
    """Tests for bind_tools functionality."""

    def test_bind_tools_extracts_metadata(self) -> None:
        """Should extract metadata from LangChain tools."""
        model = ClaudeCodeChatModel(base_options=None)

        bound_model = model.bind_tools([test_calculator, test_reader])

        # Should have extracted metadata
        assert len(bound_model.tools_metadata) == 2

        # Check first tool
        assert bound_model.tools_metadata[0]["name"] == "test_calculator"
        assert "Add" in bound_model.tools_metadata[0]["description"]
        assert "input_schema" in bound_model.tools_metadata[0]

    def test_bind_tools_extracts_metadata_for_all_tool_types(self) -> None:
        """bind_tools must extract metadata for ALL BaseTool subclasses.

        Both StructuredTool (@tool decorator) and manual BaseTool subclasses
        (like WebSearchDelegationTool) must have their metadata extracted.
        Tools execute via LangGraph ToolNode, not MCP handler (proxy architecture).
        """
        model = ClaudeCodeChatModel(base_options=None)
        custom_tool = CustomSearchTool()

        bound_model = model.bind_tools([test_calculator, custom_tool])

        # Both tools must have metadata
        names = [m["name"] for m in bound_model.tools_metadata]
        assert "test_calculator" in names
        assert "custom_search" in names

    def test_bind_tools_returns_new_instance(self) -> None:
        """Should return new instance (LangChain pattern)."""
        model = ClaudeCodeChatModel(base_options=None)

        bound_model = model.bind_tools([test_calculator])

        # Should be different instance
        assert bound_model is not model
        # Original should not have tools
        assert model.tools_metadata == []
        # Bound should have tools
        assert len(bound_model.tools_metadata) == 1


class TestClaudeCodeChatModelSimpleInvoke:
    """Tests for simple invocation without tool calls."""

    @pytest.mark.asyncio
    async def test_ainvoke_simple_message(self) -> None:
        """Should handle simple message without tools."""
        from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
        from claude_agent_sdk import ResultMessage, TextBlock

        model = ClaudeCodeChatModel(base_options=None)

        # Mock client manager
        mock_client = Mock()
        mock_client.query = AsyncMock()

        # Mock response - simple text
        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Hello! How can I help?")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Hello")], config={"configurable": {"thread_id": "test"}}
            )

        # Should return AIMessage
        assert isinstance(result, AIMessage)
        assert "Hello! How can I help?" in result.content

    @pytest.mark.asyncio
    async def test_ainvoke_uses_thread_id(self) -> None:
        """Should use thread_id from config."""
        model = ClaudeCodeChatModel(base_options=None)

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            return
            yield  # Make it a generator

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        get_or_create_called = False
        thread_id_used = None

        async def mock_get_or_create(thread_id: str, tools_metadata: list[Any], **kwargs: Any) -> tuple[Mock, Mock]:
            nonlocal get_or_create_called, thread_id_used
            get_or_create_called = True
            thread_id_used = thread_id
            return mock_client, mock_proxy

        with patch.object(model.client_manager, "get_or_create_client", side_effect=mock_get_or_create):
            await model.ainvoke(
                [HumanMessage(content="Test")], config={"configurable": {"thread_id": "session-123"}}
            )

        assert get_or_create_called
        assert thread_id_used == "session-123"

    @pytest.mark.asyncio
    async def test_ainvoke_raises_without_thread_id(self) -> None:
        """ainvoke MUST raise ValueError when thread_id is not provided.

        ClaudeCodeChatModel is stateful — each thread_id maps to a separate
        Claude SDK session. Implicit defaults are forbidden to prevent
        session state corruption between main agent and subagents.
        """
        model = ClaudeCodeChatModel(base_options=None)

        with pytest.raises(ValueError, match="requires explicit thread_id"):
            await model.ainvoke([HumanMessage(content="Test")])


class TestClaudeCodeChatModelToolCalls:
    """Tests for tool call handling."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_tool_call(self) -> None:
        """Should return AIMessage with tool_calls when pending."""
        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            return
            yield

        mock_client.receive_response = mock_receive

        # Mock pending call
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=True)
        mock_proxy.get_pending_calls = Mock(
            return_value=[{"id": "call_1", "name": "test_calculator", "args": {"a": 5, "b": 3}}]
        )

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Calculate 5+3")],
                config={"configurable": {"thread_id": "test_chat"}},
            )

        # Should return AIMessage with tool_calls
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_calculator"
        assert result.tool_calls[0]["args"] == {"a": 5, "b": 3}
        assert result.tool_calls[0]["id"] == "call_1"

    @pytest.mark.asyncio
    async def test_tooluseblock_waits_event_driven_without_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ToolUseBlock → pending call bridge must be event-driven
        (POLICIES.md §2). After a related internal incident analysis
        (not included here) the wait is keyed
        by the SAME ``tool_use_id`` that flows on both sides — the
        chat_model parks on ``wait_for_pending_call_by_id(block.id)``
        and the handler wakes it the moment the matching pending
        registers. No polling, no by-name match, no fallback.
        """
        from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
        from claude_agent_sdk import ToolUseBlock

        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_1",
                        # Real CLI shape: our MCP tools are exposed ONLY as
                        # ``mcp__deepworker__<tool>``. A bare name is not a
                        # proxy call (classification is prefix-only).
                        name="mcp__deepworker__test_calculator",
                        input={"a": 5, "b": 3},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        _state = {"registered": False}

        def _has_pending_calls() -> bool:
            return _state["registered"]

        def _get_pending_calls() -> list[dict[str, Any]]:
            return [
                {"id": "toolu_1", "name": "test_calculator", "args": {"a": 5, "b": 3}}
            ] if _state["registered"] else []

        async def _wait_for_pending_call_by_id(claude_id: str) -> dict[str, Any]:
            assert claude_id == "toolu_1"  # MUST be the SAME id, not synthetic
            _state["registered"] = True
            return {"id": "toolu_1", "name": "test_calculator",
                    "args": {"a": 5, "b": 3}}

        mock_proxy.has_pending_calls = Mock(side_effect=_has_pending_calls)
        mock_proxy.get_pending_calls = Mock(side_effect=_get_pending_calls)
        mock_proxy.wait_for_pending_call_by_id = AsyncMock(
            side_effect=_wait_for_pending_call_by_id
        )

        async def _sleep_is_polling(_delay: float) -> None:
            raise AssertionError("polling sleep is forbidden in ClaudeCodeChatModel")

        monkeypatch.setattr("claude_code_provider.chat_model.asyncio.sleep", _sleep_is_polling)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Calculate 5+3")],
                config={"configurable": {"thread_id": "test_chat_event_driven"}},
            )

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_calculator"
        # The pending entry's id MUST be the SAME id Claude emitted.
        assert result.tool_calls[0]["id"] == "toolu_1"
        # The new contract: wait by id, no timeout.
        mock_proxy.wait_for_pending_call_by_id.assert_awaited_once_with(
            "toolu_1"
        )

    @pytest.mark.asyncio
    # NOTE: test_visible_block_args_diverge_from_control_channel_call was
    # REMOVED by the unified tool_use_id change. With the unified tool_use_id, visible
    # and control sides share the SAME id, so "visible args ≠ control args"
    # is no longer an identity problem. See POLICIES.md §7.

    @pytest.mark.asyncio
    async def test_single_assistant_message_multiple_proxy_tooluseblocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Claude CLI dispatches MCP control_requests SEQUENTIALLY: even when
        an AssistantMessage carries multiple proxy ToolUseBlocks, only the
        first handler is invoked and only ONE pending call exists. ainvoke
        must surface that one to LangGraph and exit — the next ainvoke
        iteration picks up the next pending call after resolve_call.

        Trying to wait for ALL handlers to register would deadlock, because
        Claude does not send the next control_request until it receives the
        previous tool_result. This was the freeze observed during
        development.
        """
        from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
        from claude_agent_sdk import ToolUseBlock

        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator, test_reader])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            # Both ToolUseBlocks visible in one AssistantMessage, mirroring
            # real Claude payload for parallel tool calls.
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_calc",
                        name="mcp__deepworker__test_calculator",
                        input={"a": 5, "b": 3},
                    ),
                    ToolUseBlock(
                        id="toolu_read",
                        name="mcp__deepworker__test_reader",
                        input={"path": "/tmp/file.txt"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        # Only the FIRST handler ever registers (Claude CLI sequential
        # dispatch). The second one will only register on the NEXT ainvoke
        # iteration after the first is resolve_call'd — out of scope here.
        # Critically, the pending key is the Claude tool_use_id, not a
        # synthetic counter.
        first_pending = [
            {"id": "toolu_calc", "name": "test_calculator", "args": {"a": 5, "b": 3}},
        ]
        _state = {"registered": False}
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(
            side_effect=lambda: _state["registered"]
        )
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: first_pending if _state["registered"] else []
        )

        async def _wait_for_pending_call_by_id(claude_id: str) -> dict[str, Any]:
            # chat_model parks on the id of the FIRST visible block. The
            # SDK's sequential dispatch guarantees that the matching
            # handler will register with the same id; the second block's
            # handler does not register until the first is resolved.
            assert claude_id == "toolu_calc"
            _state["registered"] = True
            return first_pending[0]

        mock_proxy.wait_for_pending_call_by_id = AsyncMock(
            side_effect=_wait_for_pending_call_by_id
        )

        async def _sleep_is_polling(_delay: float) -> None:
            raise AssertionError("polling sleep is forbidden in ClaudeCodeChatModel")

        monkeypatch.setattr("claude_code_provider.chat_model.asyncio.sleep", _sleep_is_polling)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Calculate and read")],
                config={"configurable": {"thread_id": "test_chat_event_batch"}},
            )

        # Critical: must return ONLY the first registered pending call, NOT
        # block waiting for both handlers (the second only registers on the
        # next ainvoke iteration after the first is resolve_call'd). And
        # the id is the SAME Claude tool_use_id, on both sides.
        assert isinstance(result, AIMessage)
        assert [tc["name"] for tc in result.tool_calls] == ["test_calculator"]
        assert result.tool_calls[0]["id"] == "toolu_calc"
        mock_proxy.wait_for_pending_call_by_id.assert_awaited_once_with(
            "toolu_calc"
        )

    @pytest.mark.asyncio
    async def test_ainvoke_multiple_tool_calls(self) -> None:
        """Should handle multiple simultaneous tool calls."""
        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator, test_reader])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            return
            yield

        mock_client.receive_response = mock_receive

        # Mock multiple pending calls
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=True)
        mock_proxy.get_pending_calls = Mock(
            return_value=[
                {"id": "call_1", "name": "test_calculator", "args": {"a": 5, "b": 3}},
                {"id": "call_2", "name": "test_reader", "args": {"path": "/tmp/file.txt"}},
            ]
        )

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="Do multiple things")],
                config={"configurable": {"thread_id": "test_chat"}},
            )

        # Should return both tool calls
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 2


class TestClaudeCodeChatModelToolResults:
    """Tests for handling ToolMessage (tool results)."""

    @pytest.mark.asyncio
    async def test_ainvoke_resolves_tool_message(self) -> None:
        """Should resolve Future when ToolMessage is provided."""
        from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
        from claude_agent_sdk import ResultMessage, TextBlock

        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        # After resolving, Claude returns final answer
        async def mock_receive():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="The answer is 8")], model="claude-sonnet-4-5-20250929"
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="test",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)
        mock_proxy.resolve_call = Mock()

        # Conversation with tool call
        messages = [
            HumanMessage(content="Calculate 5+3"),
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "test_calculator", "args": {"a": 5, "b": 3}}]),
            ToolMessage(content="8", tool_call_id="call_1"),
        ]

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                messages,
                config={"configurable": {"thread_id": "test_chat"}},
            )

        # ⚠️ PROXY ARCHITECTURE: ToolMessage resolved via resolve_call, NOT query
        # This returns tool result to Claude via MCP Future proxy.
        mock_proxy.resolve_call.assert_called_once_with("call_1", {
            "content": [{"type": "text", "text": "8"}],
        })

        # Should return final answer
        assert isinstance(result, AIMessage)
        assert "8" in result.content


class TestClaudeCodeChatModelMessageConversion:
    """Tests for message conversion between LangChain and Claude formats."""

    @pytest.mark.asyncio
    async def test_convert_human_message(self) -> None:
        """Should convert HumanMessage to Claude format."""
        model = ClaudeCodeChatModel(base_options=None)

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            return
            yield

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            await model.ainvoke(
                [HumanMessage(content="Test message")],
                config={"configurable": {"thread_id": "test_chat"}},
            )

        # Check that query was called with proper format
        mock_client.query.assert_called_once()
        # The first argument should be the messages
        query_call = mock_client.query.call_args
        assert query_call is not None

    @pytest.mark.asyncio
    async def test_skip_ai_message_with_tool_calls(self) -> None:
        """Should not re-send AIMessage with tool_calls."""
        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_calculator])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            return
            yield

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)
        mock_proxy.resolve_call = Mock()

        # Messages including AI message with tool_calls
        messages = [
            HumanMessage(content="Calculate 5+3"),
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "test_calculator", "args": {"a": 5, "b": 3}}]),
            ToolMessage(content="8", tool_call_id="call_1"),
        ]

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            await model.ainvoke(
                messages,
                config={"configurable": {"thread_id": "test_chat"}},
            )

        # ⚠️ PROXY: ToolMessage resolved via resolve_call, NOT query
        mock_proxy.resolve_call.assert_called_once()
        # query called only for HumanMessage, NOT for ToolMessage


# ─── REMOVED (POLICIES.md §7) ──────────────────────────────────────────────
#
# The class ``TestClaudeCodeChatModelControlChannelRecovery`` was deleted
# from this file. It pinned the very behaviour the root-cause fix REMOVES
# — namely:
#
#   * ``test_no_cleanup_when_first_attempt_succeeds`` asserted that
#     ``wait_for_any_pending_call`` was awaited with ``timeout=5.0``
#     and that, on success, no cleanup happened. The new contract has
#     no timeout on this wait at all (it is event-driven; the SDK's
#     sequential ``tools/call`` dispatch guarantees delivery).
#
#   * ``test_timeout_returns_visible_tool_call_fallback`` asserted that
#     ``ainvoke`` synthesised a tool_call from the visible block and
#     handed it to LangGraph on timeout. That fallback is the EXACT
#     source of the "two-three copies of one block" symptom — see
#     a related internal incident analysis (not included here).
#
# In addition, ``test_visible_block_args_diverge_from_control_channel_call``
# (formerly above) was deleted: with the unified ``tool_use_id`` the
# question of "visible args ≠ control args" cannot affect identity —
# the SAME id maps to the SAME pending entry, period.
#
# Reviewers: do NOT reintroduce any of these tests in any form. See
# POLICIES.md §7. The new contract is exercised by
# ``test_proxy_mcp_server.py`` (handler keyed by Claude id, no phantom
# timeout) and ``test_unified_tool_use_id_and_no_phantom_timeout.py``
# (parts A+B+C+D end-to-end).


# (end of removed-tests block — see comment above)
