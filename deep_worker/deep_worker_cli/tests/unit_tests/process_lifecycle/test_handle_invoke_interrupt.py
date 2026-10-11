"""Tests for _handle_invoke interrupt handling in non-streaming mode.

TDD: Verifies that when ainvoke returns an interrupt, _handle_invoke:
1. Does NOT re-send the same HumanMessage (no cancel→retry loop)
2. Calls the approval_callback to get a decision
3. Resumes via Command(resume=...) with the decision
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess


def _make_agent_process():
    """Create a minimal AgentProcess mock for testing _handle_invoke."""
    process = AgentProcess.__new__(AgentProcess)
    process.name = "test-agent"
    process.is_running = False
    process.current_worker_task = None
    process._last_broadcasted_count = {}

    # Mock session_manager
    process.session_manager = MagicMock()
    process.session_manager.set_status = AsyncMock()
    process.session_manager.set_error = AsyncMock()
    # A starting turn clears the previous turn's recoverable-error marker
    # (docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §4.4).
    process.session_manager.clear_error = AsyncMock()
    # ...and advances the activity counter, alongside the RUNNING
    # transition above.
    process.session_manager.increment_message_count = AsyncMock()

    # No TaskManager
    process.task_manager = None

    # Mock config
    process.config = MagicMock()
    process.config.working_dir = "/tmp"

    return process


class TestHandleInvokeInterrupt:
    """Tests for interrupt handling in the non-streaming ainvoke loop."""

    @pytest.mark.asyncio
    async def test_interrupt_calls_approval_callback(self):
        """When ainvoke returns an interrupt, approval_callback must be called."""
        process = _make_agent_process()

        # Mock Interrupt object
        mock_interrupt = MagicMock()
        mock_interrupt.id = "interrupt-1"
        mock_interrupt.value = {
            "action_requests": [
                {"name": "web_search", "args": {"query": "test"}, "description": "Web search: test"}
            ],
        }

        # First ainvoke returns interrupt, second (resume) returns final response
        from langchain_core.messages import AIMessage, HumanMessage

        interrupt_result = {
            "messages": [
                HumanMessage(content="search for test"),
                AIMessage(content="", tool_calls=[{"name": "web_search", "id": "tc1", "args": {"query": "test"}}]),
            ],
            "__interrupt__": (mock_interrupt,),
        }

        final_result = {
            "messages": [
                HumanMessage(content="search for test"),
                AIMessage(content="", tool_calls=[{"name": "web_search", "id": "tc1", "args": {"query": "test"}}]),
                AIMessage(content="Here are the results"),
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(side_effect=[interrupt_result, final_result])
        process.agent = mock_agent

        # Approval callback — auto-approve
        approval_callback = AsyncMock(return_value={
            mock_interrupt.id: {"decisions": [{"type": "approve"}]}
        })

        result = await process._handle_invoke(
            {"command": "invoke", "message": "search for test", "stream": False},
            connection=None,
            notify_callback=AsyncMock(),
            approval_callback=approval_callback,
        )

        # Approval callback was called with interrupt data
        assert approval_callback.called
        call_args = approval_callback.call_args[0][0]  # first positional arg
        assert call_args[0].id == "interrupt-1"

        # Agent was resumed with Command(resume=...), NOT with same HumanMessage
        assert mock_agent.ainvoke.call_count == 2
        second_call_input = mock_agent.ainvoke.call_args_list[1][0][0]
        # Should be a Command, not a dict with messages
        from langgraph.types import Command
        assert isinstance(second_call_input, Command)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_interrupt_without_callback_auto_approves(self):
        """When no approval_callback is provided, interrupts should auto-approve."""
        process = _make_agent_process()

        mock_interrupt = MagicMock()
        mock_interrupt.id = "interrupt-1"
        mock_interrupt.value = {
            "action_requests": [
                {"name": "web_search", "args": {"query": "test"}, "description": "Web search: test"}
            ],
        }

        from langchain_core.messages import AIMessage, HumanMessage

        interrupt_result = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="", tool_calls=[{"name": "web_search", "id": "tc1", "args": {"query": "test"}}]),
            ],
            "__interrupt__": (mock_interrupt,),
        }

        final_result = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="Done"),
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(side_effect=[interrupt_result, final_result])
        process.agent = mock_agent

        result = await process._handle_invoke(
            {"command": "invoke", "message": "test", "stream": False},
            connection=None,
            notify_callback=AsyncMock(),
            # No approval_callback
        )

        # Should auto-approve and resume
        assert mock_agent.ainvoke.call_count == 2
        from langgraph.types import Command
        second_call_input = mock_agent.ainvoke.call_args_list[1][0][0]
        assert isinstance(second_call_input, Command)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_duplicate_human_messages_on_interrupt(self):
        """ainvoke must NOT be called with the same HumanMessage on retry."""
        process = _make_agent_process()

        mock_interrupt = MagicMock()
        mock_interrupt.id = "int-1"
        mock_interrupt.value = {
            "action_requests": [{"name": "execute", "args": {"command": "ls"}}],
        }

        from langchain_core.messages import AIMessage, HumanMessage

        interrupt_result = {
            "messages": [
                HumanMessage(content="list files"),
                AIMessage(content="", tool_calls=[{"name": "execute", "id": "tc1", "args": {"command": "ls"}}]),
            ],
            "__interrupt__": (mock_interrupt,),
        }

        final_result = {
            "messages": [
                HumanMessage(content="list files"),
                AIMessage(content="file1.txt\nfile2.txt"),
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(side_effect=[interrupt_result, final_result])
        process.agent = mock_agent

        await process._handle_invoke(
            {"command": "invoke", "message": "list files", "stream": False},
            connection=None,
            notify_callback=AsyncMock(),
        )

        # First call has HumanMessage
        first_input = mock_agent.ainvoke.call_args_list[0][0][0]
        assert "messages" in first_input

        # Second call must be Command(resume=...), NOT {"messages": [...]}
        second_input = mock_agent.ainvoke.call_args_list[1][0][0]
        from langgraph.types import Command
        assert isinstance(second_input, Command)
        assert not isinstance(second_input, dict)
