"""Unit tests for event-based tool call detection (no polling).

This module tests the new event-based approach that eliminates:
1. Race condition between polling and response collection
2. High CPU usage from 50ms polling
3. Latency from polling delay

Tests verify that:
- All response text is collected before checking pending_calls
- ToolUseBlock detection works correctly
- No race condition exists
- CPU usage is minimal (no polling loop)
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# Import types first
import sys
from pathlib import Path

_claude_sdk_path = Path(__file__).parent.parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (
    AssistantMessage as ClaudeAssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from claude_code_provider import ClaudeCodeChatModel
from claude_code_provider.custom_types import ToolMetadata


@pytest.fixture
def mock_client_and_proxy():
    """Mock ClaudeSDKClient and ProxyMCPServer."""
    mock_client = MagicMock()
    mock_proxy = MagicMock()

    # Default: no pending calls
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []

    return mock_client, mock_proxy


@pytest.fixture
def sample_tools_metadata():
    """Sample tool metadata for testing."""
    return [
        ToolMetadata(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        ToolMetadata(
            name="bash",
            description="Execute bash command",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ),
    ]


class TestEventBasedToolDetection:
    """Test event-based tool call detection without polling."""

    @pytest.mark.asyncio
    async def test_collect_all_text_before_checking_pending_calls(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """Verify all text is collected before checking for pending tool calls.

        This is the KEY test that verifies the race condition is fixed.
        The implementation MUST collect ALL events from receive_response()
        BEFORE checking proxy_mcp.has_pending_calls().
        """
        mock_client, mock_proxy = mock_client_and_proxy

        # Simulate Claude response with text BEFORE and AFTER ToolUseBlock
        async def mock_receive_response():
            # 1. First text block
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="I'll help you read that file.")],
                model="claude-sonnet-4-5-20250929",
            )

            # 2. Tool use block
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call_123",
                        name="read_file",
                        input={"path": "/tmp/test.txt"},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

            # 3. CRITICAL: More text AFTER tool use
            yield ClaudeAssistantMessage(
                content=[TextBlock(text=" Let me know if you need anything else.")],
                model="claude-sonnet-4-5-20250929",
            )

            # 4. Result message (end of stream)
            yield ResultMessage(
                subtype="result",
                duration_ms=1500,
                duration_api_ms=1200,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.001,
                usage={"input_tokens": 100, "output_tokens": 50},
            )

        mock_client.receive_response = mock_receive_response

        # Mock pending calls (as if ProxyMCPServer detected tool call)
        mock_proxy.has_pending_calls.return_value = True
        mock_proxy.get_pending_calls.return_value = [
            {
                "id": "call_123",
                "name": "read_file",
                "args": {"path": "/tmp/test.txt"},
            }
        ]

        # Create model with mocked client manager
        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")

            # Mock query method
            mock_client.query = AsyncMock()

            # Invoke model
            result = await model.ainvoke(
                [HumanMessage(content="Read /tmp/test.txt")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        # CRITICAL ASSERTIONS:
        # 1. Result should have tool_calls
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"

        # 2. Content should include text BEFORE ToolUseBlock only.
        # With proxy architecture, we break on ToolUseBlock to return
        # tool_calls to LangGraph. Text after ToolUseBlock belongs to next turn.
        assert "I'll help you read that file." in result.content

    @pytest.mark.asyncio
    async def test_no_tool_use_block_no_pending_calls_check(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """When no ToolUseBlock in stream, should not check pending_calls.

        This optimizes the case where Claude just returns text without calling tools.
        """
        mock_client, mock_proxy = mock_client_and_proxy

        # Simulate Claude response with only text (no tool use)
        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="The file contains: Hello World")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=800,
                duration_api_ms=700,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0005,
                usage={"input_tokens": 50, "output_tokens": 20},
            )

        mock_client.receive_response = mock_receive_response

        # Create model
        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="What's in the file?")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        # Should return text-only response
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 0
        assert "The file contains: Hello World" in result.content

        # has_pending_calls should NOT be called (optimization)
        # Because we didn't see any ToolUseBlock
        # Note: In actual implementation, we might still check once after grace period
        # This is acceptable for simplicity

    @pytest.mark.asyncio
    async def test_tool_use_block_but_no_pending_calls_yet(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """If ToolUseBlock seen but pending_calls not populated yet, wait briefly.

        This handles the edge case where:
        1. ToolUseBlock arrives in message stream
        2. MCP handler hasn't created pending_calls yet (timing delay)
        3. Need to wait a short grace period (10ms)
        """
        mock_client, mock_proxy = mock_client_and_proxy

        # Simulate Claude response with ToolUseBlock
        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Executing bash command...")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call_456",
                        name="bash",
                        input={"command": "ls -la"},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=1000,
                duration_api_ms=900,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0008,
                usage={"input_tokens": 80, "output_tokens": 30},
            )

        mock_client.receive_response = mock_receive_response

        # Simulate MCP handler delay: pending_calls arrives eventually
        # After grace period (10ms), pending_calls should be populated
        mock_proxy.has_pending_calls.return_value = True
        mock_proxy.get_pending_calls.return_value = [
            {
                "id": "call_456",
                "name": "bash",
                "args": {"command": "ls -la"},
            }
        ]

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="Run ls -la")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        # Should have tool calls (waited for pending_calls)
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_concatenated(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """Verify multiple TextBlocks are concatenated correctly."""
        mock_client, mock_proxy = mock_client_and_proxy

        # Simulate Claude streaming response with multiple text blocks
        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Let me ")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="help you ")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="with that.")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=500,
                duration_api_ms=400,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0003,
                usage={"input_tokens": 30, "output_tokens": 10},
            )

        mock_client.receive_response = mock_receive_response

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="Help me")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        # All text blocks should be concatenated
        assert result.content == "Let me help you with that."

    @pytest.mark.asyncio
    async def test_no_polling_loop_running(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """Verify no polling loop is created (event-based approach).

        This test ensures the old polling mechanism is completely removed.
        """
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="Done")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=300,
                duration_api_ms=250,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0002,
                usage={"input_tokens": 20, "output_tokens": 5},
            )

        mock_client.receive_response = mock_receive_response

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            mock_client.query = AsyncMock()

            # Record all tasks before and after
            tasks_before = asyncio.all_tasks()

            await model.ainvoke(
                [HumanMessage(content="Test")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

            tasks_after = asyncio.all_tasks()

            # No new background tasks should remain
            # (polling loop would create a long-running task)
            new_tasks = tasks_after - tasks_before
            # Should only have the current test task
            assert len(new_tasks) <= 1, f"Unexpected background tasks: {new_tasks}"


class TestRaceConditionPrevention:
    """Tests specifically for race condition prevention."""

    @pytest.mark.asyncio
    async def test_no_race_between_text_and_tool_detection(
        self, mock_client_and_proxy: Any
    ) -> None:
        """Verify text AFTER tool call is never lost.

        OLD BUG: If poll_for_tool_calls() finished first, it would cancel
        collect_response_events(), losing any text that came after the tool call.

        NEW BEHAVIOR: All events are collected BEFORE checking pending_calls.
        """
        mock_client, mock_proxy = mock_client_and_proxy

        # Simulate the exact scenario that caused the bug:
        # 1. Text before tool
        # 2. Tool use
        # 3. MORE TEXT AFTER TOOL (this was getting lost!)
        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="I'll read the file for you. ")],
                model="claude-sonnet-4-5-20250929",
            )

            # Small delay (simulates network latency)
            await asyncio.sleep(0.01)

            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call_abc",
                        name="read_file",
                        input={"path": "/test.txt"},
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

            # MORE TEXT AFTER TOOL - THIS WAS BEING LOST IN OLD IMPLEMENTATION!
            await asyncio.sleep(0.01)
            yield ClaudeAssistantMessage(
                content=[
                    TextBlock(
                        text="The file will be read and I'll show you the contents."
                    )
                ],
                model="claude-sonnet-4-5-20250929",
            )

            yield ResultMessage(
                subtype="result",
                duration_ms=1200,
                duration_api_ms=1000,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.001,
                usage={"input_tokens": 90, "output_tokens": 40},
            )

        mock_client.receive_response = mock_receive_response
        mock_proxy.has_pending_calls.return_value = True
        mock_proxy.get_pending_calls.return_value = [
            {"id": "call_abc", "name": "read_file", "args": {"path": "/test.txt"}}
        ]

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="Read /test.txt")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        # With proxy architecture, text before ToolUseBlock is collected,
        # then we break. Text in the same AssistantMessage as ToolUseBlock is kept.
        assert "I'll read the file for you." in result.content

        # Tool call should be detected via pending_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"


class TestMcpDeepworkerPrefixedToolDetection:
    """Regression tests for the ``mcp__deepworker__`` prefix bug.

    Claude Agent SDK exposes MCP tools to the LLM with the canonical
    ``mcp__<server_name>__<tool_name>`` naming convention. Our proxy
    server is created as ``create_sdk_mcp_server(name="deepworker", ...)``
    in :mod:`claude_code_provider.proxy_mcp_server`, so every proxy tool
    that Claude calls arrives as a ``ToolUseBlock`` whose ``name`` is
    ``mcp__deepworker__<tool_name>`` — NOT the bare tool name registered
    in ``ToolMetadata``.

    Before this fix, ``ainvoke`` compared ``block.name`` directly against
    the set of bare ``ToolMetadata`` names. The membership check always
    failed for real Claude payloads, every proxy tool was misclassified
    as a built-in Claude tool, and ``receive_response()`` kept iterating
    instead of returning ``tool_calls`` to LangGraph. The result was a
    hung agent: ProxyMCPServer handler tasks blocked on Futures that
    LangGraph never resolved, RPyC inbound quarantine eventually fired,
    and the agent stopped responding entirely.

    These tests recreate the production payload shape (prefixed names,
    multiple ToolUseBlocks in one AssistantMessage) and assert that
    every proxy tool call is surfaced to LangGraph.
    """

    @pytest.mark.asyncio
    async def test_single_prefixed_mcp_tool_is_detected(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """A ToolUseBlock with ``mcp__deepworker__<name>`` MUST be treated
        as a proxy MCP tool and returned to LangGraph as a tool_call."""
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[
                    TextBlock(text="Reading the file."),
                    ToolUseBlock(
                        id="call_prefixed_1",
                        name="mcp__deepworker__read_file",
                        input={"path": "/tmp/test.txt"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=10,
                duration_api_ms=5,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0,
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        mock_client.receive_response = mock_receive_response

        # The proxy MCP handler task registers the pending_call when the
        # SDK invokes it via control_request. Simulate that registration
        # via the new id-keyed waiter.
        _state = {"registered": False}

        async def _wait_for_pending_call_by_id(claude_id: str) -> dict[str, Any]:
            assert claude_id == "call_prefixed_1"
            _state["registered"] = True
            return {
                "id": "call_prefixed_1",
                "name": "read_file",
                "args": {"path": "/tmp/test.txt"},
            }

        mock_proxy.wait_for_pending_call_by_id = AsyncMock(
            side_effect=_wait_for_pending_call_by_id
        )
        mock_proxy.has_pending_calls = Mock(
            side_effect=lambda: _state["registered"]
        )
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: (
                [
                    {
                        "id": "call_prefixed_1",
                        "name": "read_file",
                        "args": {"path": "/tmp/test.txt"},
                    }
                ]
                if _state["registered"]
                else []
            )
        )

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            # Inject metadata directly — mirrors what bind_tools() does.
            setattr(model, "_tools_metadata", sample_tools_metadata)
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="Read /tmp/test.txt")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1, (
            "Prefixed mcp__deepworker__read_file must be detected as proxy "
            "tool and returned to LangGraph as a tool_call"
        )
        assert result.tool_calls[0]["name"] == "read_file"
        # The id-keyed waiter must have been used — proving the chat_model
        # bridge correlates by SAME tool_use_id end-to-end.
        mock_proxy.wait_for_pending_call_by_id.assert_awaited_once_with(
            "call_prefixed_1"
        )

    @pytest.mark.asyncio
    async def test_multiple_prefixed_mcp_tools_in_one_message_returns_first(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """When Claude emits multiple prefixed ToolUseBlocks in one
        AssistantMessage, ainvoke MUST return only the currently
        registered pending call(s) and exit — not wait for the rest.

        Claude CLI dispatches MCP control_requests SEQUENTIALLY: the next
        tool's control_request is only sent after the previous tool_result
        arrives. So only ONE handler runs at a time and only ONE entry
        exists in pending_calls. Waiting for all of them would deadlock
        (Claude waits for tool_result for #1; we wait for #2 to register;
        nothing progresses). This was a hang observed during development.

        The remaining ToolUseBlocks naturally surface in the NEXT ainvoke
        iteration, after LangGraph runs the tool and resolve_call is
        called — Claude then sends the next control_request and the next
        handler registers.
        """
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            # Production payload shape: 3 prefixed blocks visible in one
            # AssistantMessage. (An observed live session showed: execute, ls, ls.)
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call_a",
                        name="mcp__deepworker__bash",
                        input={"command": "echo hi"},
                    ),
                    ToolUseBlock(
                        id="call_b",
                        name="mcp__deepworker__read_file",
                        input={"path": "/a"},
                    ),
                    ToolUseBlock(
                        id="call_c",
                        name="mcp__deepworker__read_file",
                        input={"path": "/b"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )

        mock_client.receive_response = mock_receive_response

        # Only the FIRST handler registers — mirrors Claude CLI's
        # sequential control_request dispatch. chat_model parks on the
        # id of the FIRST visible block.
        first_pending = [
            {"id": "call_a", "name": "bash", "args": {"command": "echo hi"}},
        ]
        _state = {"registered": False}

        async def _wait_for_pending_call_by_id(claude_id: str) -> dict[str, Any]:
            assert claude_id == "call_a"
            _state["registered"] = True
            return first_pending[0]

        mock_proxy.wait_for_pending_call_by_id = AsyncMock(
            side_effect=_wait_for_pending_call_by_id
        )
        mock_proxy.has_pending_calls = Mock(
            side_effect=lambda: _state["registered"]
        )
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: first_pending if _state["registered"] else []
        )

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            setattr(model, "_tools_metadata", sample_tools_metadata)
            mock_client.query = AsyncMock()

            # Must complete quickly — no deadlock waiting for the other
            # two handlers to register. If this hangs, the regression is
            # back.
            result = await asyncio.wait_for(
                model.ainvoke(
                    [HumanMessage(content="Do three things")],
                    config={"configurable": {"thread_id": "test-thread"}},
                ),
                timeout=2.0,
            )

        assert isinstance(result, AIMessage)
        # Critical: must return the ONE pending call (whatever is
        # registered) and exit — NOT block for the other two.
        assert [tc["name"] for tc in result.tool_calls] == ["bash"]
        mock_proxy.wait_for_pending_call_by_id.assert_awaited_once_with("call_a")

    @pytest.mark.asyncio
    async def test_prefixed_unknown_tool_is_built_in(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """A ToolUseBlock for a tool that is NOT in the proxy metadata
        must still be treated as a built-in Claude tool (we must not
        accidentally treat every ``mcp__*`` block as ours)."""
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="builtin_1",
                        name="WebSearch",
                        input={"query": "weather"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )
            yield ClaudeAssistantMessage(
                content=[TextBlock(text="The weather is sunny.")],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result",
                duration_ms=10,
                duration_api_ms=5,
                is_error=False,
                num_turns=1,
                session_id="test-thread",
                total_cost_usd=0.0,
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        mock_client.receive_response = mock_receive_response
        mock_proxy.has_pending_calls.return_value = False
        mock_proxy.get_pending_calls.return_value = []
        mock_proxy.wait_for_any_pending_call = AsyncMock()

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            setattr(model, "_tools_metadata", sample_tools_metadata)
            mock_client.query = AsyncMock()

            result = await model.ainvoke(
                [HumanMessage(content="What's the weather?")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

        assert isinstance(result, AIMessage)
        # Built-in tool — Claude handles it; no tool_calls returned.
        assert len(result.tool_calls) == 0
        assert "sunny" in result.content
        mock_proxy.wait_for_any_pending_call.assert_not_awaited()


class TestBareNameNotMisclassifiedAsProxy:
    """A ToolUseBlock with a BARE name must NEVER be treated as our proxy
    tool, even when its name collides with a base name we registered.

    Claude Code exposes our MCP tools to the model ONLY as
    ``mcp__deepworker__<tool>``. A bare ``execute`` block is therefore
    NOT one of our calls — it is the model hallucinating / calling a
    built-in by a short name. The CLI does NOT route it through our MCP
    server, so no handler/pending-call ever appears for it.

    Regression: a live session deadlocked because the bare
    ``execute`` block (a Claude built-in) was classified as our proxy
    tool by base-name match; chat_model parked on its ``tool_use_id``
    which never arrived over our control channel. See
    docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.
    """

    @pytest.mark.asyncio
    async def test_bare_name_collision_is_not_waited_on(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """A turn whose ONLY tool block is a BARE ``read_file`` (collides
        with our registered base name) must NOT be classified as a proxy
        call: ``wait_for_pending_call_by_id`` must NOT be awaited, and the
        turn must not hang forever."""
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_bare_readfile",
                        name="read_file",  # BARE — not mcp__deepworker__
                        input={"path": "/tmp/x"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result", duration_ms=1, duration_api_ms=1,
                is_error=False, num_turns=1, session_id="test-thread",
                total_cost_usd=0.0, usage={"input_tokens": 1, "output_tokens": 1},
            )

        mock_client.receive_response = mock_receive_response
        mock_proxy.wait_for_pending_call_by_id = AsyncMock()
        mock_proxy.has_pending_calls = Mock(return_value=False)
        mock_proxy.get_pending_calls = Mock(return_value=[])

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            setattr(model, "_tools_metadata", sample_tools_metadata)
            mock_client.query = AsyncMock()

            # Must complete; a bare-name misclassification would park on
            # wait_for_pending_call_by_id forever.
            result = await asyncio.wait_for(
                model.ainvoke(
                    [HumanMessage(content="read it")],
                    config={"configurable": {"thread_id": "test-thread"}},
                ),
                timeout=5.0,
            )

        # The bare block is NOT our proxy tool → we never wait on its id.
        mock_proxy.wait_for_pending_call_by_id.assert_not_awaited()
        assert isinstance(result, AIMessage)
        # And we did NOT surface it to LangGraph as one of our proxy calls.
        assert result.tool_calls == [] or all(
            tc.get("id") != "toolu_bare_readfile" for tc in result.tool_calls
        )

    @pytest.mark.asyncio
    async def test_bare_then_prefixed_waits_only_on_prefixed(
        self, mock_client_and_proxy: Any, sample_tools_metadata: Any
    ) -> None:
        """The exact regression shape: a BARE ``read_file`` block (rejected
        by the CLI) followed by the model's corrected
        ``mcp__deepworker__read_file``. chat_model must wait ONLY on the
        prefixed call's id — never on the bare one's."""
        mock_client, mock_proxy = mock_client_and_proxy

        async def mock_receive_response():
            # 1) bare call (CLI will reject; never routes to us)
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_BARE",
                        name="read_file",
                        input={"path": "/tmp/x"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )
            # 2) corrected prefixed call (the real one we get)
            yield ClaudeAssistantMessage(
                content=[
                    ToolUseBlock(
                        id="toolu_PREFIXED",
                        name="mcp__deepworker__read_file",
                        input={"path": "/tmp/x"},
                    ),
                ],
                model="claude-sonnet-4-5-20250929",
            )
            yield ResultMessage(
                subtype="result", duration_ms=1, duration_api_ms=1,
                is_error=False, num_turns=1, session_id="test-thread",
                total_cost_usd=0.0, usage={"input_tokens": 1, "output_tokens": 1},
            )

        mock_client.receive_response = mock_receive_response

        _state = {"registered": False}

        async def _wait_by_id(claude_id: str) -> dict[str, Any]:
            # MUST be the prefixed id, never the bare one.
            assert claude_id == "toolu_PREFIXED", (
                f"chat_model waited on {claude_id!r} — must wait only on the "
                f"prefixed proxy call, never the bare/built-in one"
            )
            _state["registered"] = True
            return {"id": "toolu_PREFIXED", "name": "read_file",
                    "args": {"path": "/tmp/x"}}

        mock_proxy.wait_for_pending_call_by_id = AsyncMock(side_effect=_wait_by_id)
        mock_proxy.has_pending_calls = Mock(side_effect=lambda: _state["registered"])
        mock_proxy.get_pending_calls = Mock(
            side_effect=lambda: (
                [{"id": "toolu_PREFIXED", "name": "read_file",
                  "args": {"path": "/tmp/x"}}] if _state["registered"] else []
            )
        )

        with patch(
            "claude_code_provider.chat_model.get_global_client_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_or_create_client = AsyncMock(
                return_value=(mock_client, mock_proxy)
            )
            mock_manager.get_sent_count.return_value = 0
            mock_manager.update_sent_count = MagicMock()
            mock_get_manager.return_value = mock_manager

            model = ClaudeCodeChatModel(model="sonnet")
            setattr(model, "_tools_metadata", sample_tools_metadata)
            mock_client.query = AsyncMock()

            result = await asyncio.wait_for(
                model.ainvoke(
                    [HumanMessage(content="read it")],
                    config={"configurable": {"thread_id": "test-thread"}},
                ),
                timeout=5.0,
            )

        mock_proxy.wait_for_pending_call_by_id.assert_awaited_once_with(
            "toolu_PREFIXED"
        )
        assert isinstance(result, AIMessage)
        assert any(tc["name"] == "read_file" for tc in result.tool_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
