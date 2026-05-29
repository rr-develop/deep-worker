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
        """ToolUseBlock → pending call bridge must use wait_for_any_pending_call,
        not asyncio.sleep polling. The visible block is only a signal; the
        authoritative call is read from get_pending_calls() afterwards."""
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
                        name="test_calculator",
                        input={"a": 5, "b": 3},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        # has_pending_calls returns False before the wait, True after it
        # returns — mimicking the real proxy where the handler task
        # registers the pending call only after we break out of
        # receive_response.
        _state = {"registered": False}

        def _has_pending_calls() -> bool:
            return _state["registered"]

        def _get_pending_calls() -> list[dict[str, Any]]:
            return [
                {"id": "call_1", "name": "test_calculator", "args": {"a": 5, "b": 3}}
            ] if _state["registered"] else []

        async def _wait_for_any_pending_call(*, timeout: float) -> None:
            _state["registered"] = True

        mock_proxy.has_pending_calls = Mock(side_effect=_has_pending_calls)
        mock_proxy.get_pending_calls = Mock(side_effect=_get_pending_calls)
        mock_proxy.wait_for_any_pending_call = AsyncMock(
            side_effect=_wait_for_any_pending_call
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
        mock_proxy.wait_for_any_pending_call.assert_awaited_once_with(timeout=30.0)

    @pytest.mark.asyncio
    async def test_visible_block_args_diverge_from_control_channel_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The visible message-stream ToolUseBlock and the actual MCP
        control-channel call are the SAME logical call, but Claude can stream
        slightly different (partial/earlier) argument JSON in the assistant
        message than what is finalized and delivered to the MCP handler.

        Live evidence (2026-05-26):
          visible  grep(pattern='unsubscribe_messages')
          pending  grep(pattern='unsubscribe')
          visible  read_file(..., limit=30)
          pending  read_file(..., limit=40)

        The control channel is authoritative — its pending call (keyed by the
        handler's call_id) is what gets executed and later resolve_call'd. The
        visible block is only a SIGNAL that a tool is coming. ainvoke must wait
        for a handler to register, then surface the real pending call. It must
        NOT deadlock waiting for a pending call whose args exactly equal the
        visible block's args.
        """
        from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
        from claude_agent_sdk import ToolUseBlock

        model = ClaudeCodeChatModel(base_options=None)
        model = model.bind_tools([test_reader])

        mock_client = Mock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            # Visible block carries the streamed (earlier) args limit=30.
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_read",
                        name="mcp__deepworker__test_reader",
                        input={"path": "/tmp/x.py", "limit": 30},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        # Control channel delivers the FINAL args limit=40 — same logical call,
        # different argument value than the visible block.
        control_channel_pending = [
            {
                "id": "test_reader_1_42",
                "name": "test_reader",
                "args": {"path": "/tmp/x.py", "limit": 40},
            },
        ]
        _state = {"registered": False}
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(
            side_effect=lambda: _state["registered"]
        )
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: control_channel_pending if _state["registered"] else []
        )

        async def _wait_for_any_pending_call(*, timeout: float) -> None:
            _state["registered"] = True

        mock_proxy.wait_for_any_pending_call = AsyncMock(
            side_effect=_wait_for_any_pending_call
        )

        async def _sleep_is_polling(_delay: float) -> None:
            raise AssertionError("polling sleep is forbidden in ClaudeCodeChatModel")

        monkeypatch.setattr("claude_code_provider.chat_model.asyncio.sleep", _sleep_is_polling)

        with patch.object(
            model.client_manager, "get_or_create_client", return_value=(mock_client, mock_proxy)
        ):
            result = await model.ainvoke(
                [HumanMessage(content="read the file")],
                config={"configurable": {"thread_id": "test_chat_divergent_args"}},
            )

        # Must NOT deadlock; must surface the authoritative control-channel call
        # (limit=40) — NOT the visible block's args (limit=30), and NOT error.
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_reader"
        assert result.tool_calls[0]["args"] == {"path": "/tmp/x.py", "limit": 40}
        assert result.tool_calls[0]["id"] == "test_reader_1_42"
        mock_proxy.wait_for_any_pending_call.assert_awaited_once_with(timeout=30.0)

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
        previous tool_result. This was the freeze observed in live testing.
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
                        name="test_calculator",
                        input={"a": 5, "b": 3},
                    ),
                    ToolUseBlock(
                        id="toolu_read",
                        name="test_reader",
                        input={"path": "/tmp/file.txt"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        # Only the FIRST handler ever registers (Claude CLI sequential
        # dispatch). The second one will only register on the NEXT ainvoke
        # iteration after the first is resolve_call'd — out of scope here.
        first_pending = [
            {"id": "call_calc", "name": "test_calculator", "args": {"a": 5, "b": 3}},
        ]
        _state = {"registered": False}
        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(
            side_effect=lambda: _state["registered"]
        )
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: first_pending if _state["registered"] else []
        )

        async def _wait_for_any_pending_call(*, timeout: float) -> None:
            # Only the FIRST handler registers (sequential control dispatch).
            _state["registered"] = True

        mock_proxy.wait_for_any_pending_call = AsyncMock(
            side_effect=_wait_for_any_pending_call
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
        # next ainvoke iteration after the first is resolve_call'd).
        assert isinstance(result, AIMessage)
        assert [tc["name"] for tc in result.tool_calls] == ["test_calculator"]
        mock_proxy.wait_for_any_pending_call.assert_awaited_once_with(timeout=30.0)

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


class TestClaudeCodeChatModelControlChannelRecovery:
    """Recovery behaviour when matching pending-call registration times out.

    Claude CLI's control channel can hang mid-turn: a ``ToolUseBlock``
    arrives on the message stream, but the matching ``control_request``
    is never delivered. We observed this on a live agent on
    2026-05-19: 90 s of nothing, no 429, no rate-limit, no API errors,
    Python side healthy — the Claude CLI subprocess itself was wedged.

    Retrying in-place against the same wedged subprocess is futile (an
    earlier 3 × 30 s retry policy reproduced the same hang each time).
    Contract: a single 30 s wait; on timeout, kill the stuck client via
    ``ClientManager.cleanup_client`` and raise. The next ``ainvoke``
    spawns a fresh Claude CLI subprocess and resumes the same session
    via the rotation tracking in ``ClientManager``.
    """

    @pytest.mark.asyncio
    async def test_no_cleanup_when_first_attempt_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: the first ``wait_for_any_pending_call`` succeeds
        in milliseconds (typical case). ``cleanup_client`` MUST NOT
        be invoked."""
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
                        name="test_calculator",
                        input={"a": 1, "b": 2},
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
            return (
                [{"id": "c1", "name": "test_calculator", "args": {"a": 1, "b": 2}}]
                if _state["registered"]
                else []
            )

        async def _wait(*, timeout: float) -> None:
            _state["registered"] = True

        mock_proxy.has_pending_calls = Mock(side_effect=_has_pending_calls)
        mock_proxy.get_pending_calls = Mock(side_effect=_get_pending_calls)
        mock_proxy.wait_for_any_pending_call = AsyncMock(side_effect=_wait)

        cleanup_mock = AsyncMock()

        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ), patch.object(
            model.client_manager,
            "cleanup_client",
            cleanup_mock,
        ):
            await model.ainvoke(
                [HumanMessage(content="test")],
                config={"configurable": {"thread_id": "test_recover_happy"}},
            )

        # Exactly one wait, no cleanup.
        assert mock_proxy.wait_for_any_pending_call.await_count == 1
        cleanup_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_returns_visible_tool_call_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio as _asyncio
        from langchain_core.messages import AIMessage
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
                        name="test_calculator",
                        input={"a": 1, "b": 2},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive

        mock_proxy = Mock()
        mock_proxy.has_pending_calls = Mock(return_value=False)
        mock_proxy.get_pending_calls = Mock(return_value=[])

        async def _wait_always_times_out(*, timeout: float) -> None:
            raise _asyncio.TimeoutError

        mock_proxy.wait_for_any_pending_call = AsyncMock(
            side_effect=_wait_always_times_out
        )

        cleanup_mock = AsyncMock()
        thread_id = "test_recover_fallback"

        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ), patch.object(
            model.client_manager,
            "cleanup_client",
            cleanup_mock,
        ):
            result = await model.ainvoke(
                [HumanMessage(content="test")],
                config={"configurable": {"thread_id": thread_id}},
            )

        assert isinstance(result, AIMessage)
        assert result.tool_calls == [
            {"name": "test_calculator", "args": {"a": 1, "b": 2}, "id": "toolu_1", "type": "tool_call"}
        ]
        assert mock_proxy.wait_for_any_pending_call.await_count == 1
        cleanup_mock.assert_not_awaited()
