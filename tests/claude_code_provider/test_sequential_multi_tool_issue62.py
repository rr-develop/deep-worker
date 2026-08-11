"""Regression test for Issue #62: sequential multi-tool hang in receive_response.

When Claude emits multiple ``ToolUseBlock`` values in a single
``AssistantMessage``, the Claude CLI dispatches the corresponding MCP
``control_request`` messages **sequentially** — only after the previous
tool's result is delivered does the next control_request arrive.

The bug: after the first tool is resolved via ``resolve_call``, the second
``ainvoke`` enters ``receive_response()``, whose ``__anext__()`` hangs
because the ``AssistantMessage`` with all ToolUseBlocks was already consumed
in the first ``ainvoke``. Claude has no more stream events to yield and the
agent deadlocks (RPyC quarantine after ~109 seconds).

The fix: before entering ``receive_response()`` when ``tool_resolved=True``,
check ``proxy_mcp.has_pending_calls()``. If pending calls exist (or appear
within a brief async wait), skip ``receive_response()`` entirely and return
the pending tool calls directly.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
from claude_agent_sdk import ResultMessage, TextBlock, ToolUseBlock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

_claude_sdk_path = Path(__file__).parent.parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel  # noqa: E402


@langchain_tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@langchain_tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@pytest.mark.asyncio
async def test_two_sequential_tools_does_not_hang() -> None:
    """Two ToolUseBlocks in one AssistantMessage, dispatched sequentially.

    Scenario:
      1. First ainvoke consumes AssistantMessage with [ToolUseBlock#1,
         ToolUseBlock#2]; handler #1 registers => returns tool_call #1.
      2. LangGraph runs tool #1 => ToolMessage.
      3. Second ainvoke: resolve_call(call_1).  Handler #2 registers
         (sequential dispatch).  The fix detects the pending call and
         skips receive_response() => returns tool_call #2.
      4. Third ainvoke: resolve_call(call_2), enters receive_response()
         normally, Claude sends final answer.

    Pre-fix: step 3 hangs in receive_response().__anext__() because the
    AssistantMessage was already consumed and the stream has no more events.
    """
    model = ClaudeCodeChatModel(base_options=None).bind_tools(
        [add_numbers, multiply_numbers]
    )

    mock_client = Mock()
    mock_client.query = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()

    call_count = {"receive": 0}

    # ---- Step 1: first ainvoke ----
    async def mock_receive_first() -> Any:
        call_count["receive"] += 1
        yield ClaudeAssistantMessage(
            content=[
                ToolUseBlock(id="call_1", name="add_numbers", input={"a": 5, "b": 3}),
                ToolUseBlock(id="call_2", name="multiply_numbers", input={"a": 4, "b": 6}),
            ],
            model="claude-opus-4",
        )

    mock_client.receive_response = mock_receive_first

    mock_proxy = Mock()
    mock_proxy.has_pending_calls = Mock(return_value=True)
    mock_proxy.get_pending_calls = Mock(
        return_value=[{"id": "call_1", "name": "add_numbers", "args": {"a": 5, "b": 3}}]
    )
    mock_proxy.resolve_call = Mock()
    mock_proxy.wait_for_any_pending_call = AsyncMock()

    with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ):
            result1 = await model.ainvoke(
                [HumanMessage(content="Calculate 5+3 and 4*6")],
                config={"configurable": {"thread_id": "t-seq-62"}},
            )

    assert isinstance(result1, AIMessage), f"Expected AIMessage, got {type(result1)}"
    assert len(result1.tool_calls) == 1
    assert result1.tool_calls[0]["name"] == "add_numbers"
    assert result1.tool_calls[0]["args"] == {"a": 5, "b": 3}

    # ---- Step 2: second ainvoke ----
    # receive_response is mocked to hang (simulates idle Claude stream).
    # The fix must detect the pending call BEFORE entering receive_response.
    async def mock_receive_hangs() -> Any:
        """Async generator that blocks forever on first __anext__()."""
        call_count["receive"] += 1
        await asyncio.Event().wait()
        yield  # Never reached

    mock_client.receive_response = mock_receive_hangs
    # Handler #2 already registered (after resolve_call triggered sequential dispatch)
    mock_proxy.has_pending_calls = Mock(return_value=True)
    mock_proxy.get_pending_calls = Mock(
        return_value=[{"id": "call_2", "name": "multiply_numbers", "args": {"a": 4, "b": 6}}]
    )

    model.client_manager.clients["t-seq-62"] = (mock_client, mock_proxy)

    with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ):
            result2 = await model.ainvoke(
                [
                    HumanMessage(content="Calculate 5+3 and 4*6"),
                    result1,
                    ToolMessage(content="8.0", tool_call_id="call_1"),
                ],
                config={"configurable": {"thread_id": "t-seq-62"}},
            )

    # Must NOT hang: fix skips receive_response and returns tool_call #2
    assert isinstance(result2, AIMessage), f"Expected AIMessage, got {type(result2)}"
    assert len(result2.tool_calls) == 1
    assert result2.tool_calls[0]["name"] == "multiply_numbers"
    assert result2.tool_calls[0]["args"] == {"a": 4, "b": 6}
    # Verify receive_response was NOT called in step 2 (the hang mock)
    assert call_count["receive"] == 1, (
        f"Expected receive_response called only once (step 1), "
        f"got {call_count['receive']} calls"
    )

    # ---- Step 3: resolve tool #2, get final answer ----
    mock_proxy.reset_mock()
    mock_proxy.has_pending_calls = Mock(return_value=False)
    mock_proxy.get_pending_calls = Mock(return_value=[])
    mock_proxy.resolve_call = Mock()

    async def mock_receive_final() -> Any:
        call_count["receive"] += 1
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="The sum is 8 and product is 24.")],
            model="claude-opus-4",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=2,
            session_id="t-seq-62-final",
        )

    mock_client.receive_response = mock_receive_final

    with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ):
            result3 = await model.ainvoke(
                [
                    HumanMessage(content="Calculate 5+3 and 4*6"),
                    result1,
                    ToolMessage(content="8.0", tool_call_id="call_1"),
                    result2,
                    ToolMessage(content="24.0", tool_call_id="call_2"),
                ],
                config={"configurable": {"thread_id": "t-seq-62"}},
            )

    assert isinstance(result3, AIMessage)
    assert len(result3.tool_calls) == 0
    assert "8" in result3.content or "24" in result3.content
    assert call_count["receive"] == 2, (
        f"Expected receive_response called twice (step 1 + step 3), "
        f"got {call_count['receive']} calls"
    )


@pytest.mark.asyncio
async def test_single_tool_fallback_to_receive_response() -> None:
    """When only ONE tool exists and no more pending calls after resolve,
    the fix must NOT skip receive_response -- the final answer comes from
    the stream normally."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([add_numbers])

    mock_client = Mock()
    mock_client.query = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()

    async def mock_receive_first() -> Any:
        yield ClaudeAssistantMessage(
            content=[ToolUseBlock(id="call_1", name="add_numbers", input={"a": 10, "b": 20})],
            model="claude-opus-4",
        )

    mock_client.receive_response = mock_receive_first

    mock_proxy = Mock()
    mock_proxy.has_pending_calls = Mock(return_value=True)
    mock_proxy.get_pending_calls = Mock(
        return_value=[{"id": "call_1", "name": "add_numbers", "args": {"a": 10, "b": 20}}]
    )
    mock_proxy.resolve_call = Mock()
    mock_proxy.wait_for_any_pending_call = AsyncMock()

    with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ):
            result1 = await model.ainvoke(
                [HumanMessage(content="Sum 10+20")],
                config={"configurable": {"thread_id": "t-fall-62"}},
            )

    assert isinstance(result1, AIMessage)
    assert len(result1.tool_calls) == 1
    assert result1.tool_calls[0]["name"] == "add_numbers"

    # ---- Second ainvoke: no more tools, get final answer ----
    async def mock_receive_second() -> Any:
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="The sum is 30.")],
            model="claude-opus-4",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="t-fall-62",
        )

    mock_client.receive_response = mock_receive_second
    # No more pending calls -- fix must fall through to receive_response
    mock_proxy.has_pending_calls = Mock(return_value=False)
    mock_proxy.get_pending_calls = Mock(return_value=[])
    mock_proxy.wait_for_any_pending_call = AsyncMock(return_value=None)

    model.client_manager.clients["t-fall-62"] = (mock_client, mock_proxy)

    with patch("claude_code_provider.client_manager.ClaudeSDKClient", return_value=mock_client):
        with patch.object(
            model.client_manager,
            "get_or_create_client",
            return_value=(mock_client, mock_proxy),
        ):
            result2 = await model.ainvoke(
                [
                    HumanMessage(content="Sum 10+20"),
                    result1,
                    ToolMessage(content="30.0", tool_call_id="call_1"),
                ],
                config={"configurable": {"thread_id": "t-fall-62"}},
            )

    # Must return text-only response (no more tools)
    assert isinstance(result2, AIMessage)
    assert len(result2.tool_calls) == 0
    assert "30" in result2.content
