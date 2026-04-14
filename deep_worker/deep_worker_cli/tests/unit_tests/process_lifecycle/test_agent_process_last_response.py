"""Tests for AgentProcess._handle_get_last_response().

TDD: These tests define the expected behavior of the new method
that returns only the last assistant (AIMessage) response.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock


def _make_human_message(content: str, msg_id: str = "h1"):
    """Create a mock HumanMessage."""
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=content, id=msg_id)


def _make_ai_message(content, msg_id: str = "a1", tool_calls=None):
    """Create a mock AIMessage."""
    from langchain_core.messages import AIMessage
    kwargs = {"content": content, "id": msg_id}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    return AIMessage(**kwargs)


def _make_tool_message(content: str, tool_call_id: str = "tc1", name: str = "bash", msg_id: str = "t1"):
    """Create a mock ToolMessage."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name, id=msg_id)


def _make_agent_process_with_messages(messages):
    """Create an AgentProcess mock with given messages in state."""
    from deep_worker_cli.process.agent_process import AgentProcess

    process = AgentProcess.__new__(AgentProcess)
    process.name = "test-agent"

    mock_agent = AsyncMock()
    mock_state = MagicMock()
    mock_state.values = {"messages": messages}
    mock_agent.aget_state = AsyncMock(return_value=mock_state)
    process.agent = mock_agent

    return process


class TestHandleGetLastResponse:
    """Tests for _handle_get_last_response method."""

    @pytest.mark.asyncio
    async def test_returns_last_ai_message(self):
        """Basic case: last message is an AIMessage."""
        messages = [
            _make_human_message("3+4"),
            _make_ai_message("7", msg_id="a1"),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"] is not None
        assert result["last_response"]["type"] == "AIMessage"
        assert result["last_response"]["content"] == "7"
        assert result["last_response"]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_skips_tool_messages_at_end(self):
        """Last AIMessage is found even if ToolMessages follow."""
        messages = [
            _make_human_message("do something"),
            _make_ai_message("I'll run a command", msg_id="a1",
                             tool_calls=[{"id": "tc1", "name": "bash", "args": {"cmd": "ls"}}]),
            _make_tool_message("file1.txt\nfile2.txt", tool_call_id="tc1"),
            _make_ai_message("Here are the files: file1.txt, file2.txt", msg_id="a2"),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"]["content"] == "Here are the files: file1.txt, file2.txt"
        assert result["last_response"]["id"] == "a2"

    @pytest.mark.asyncio
    async def test_skips_ai_message_with_only_tool_calls(self):
        """AIMessage that has tool_calls but empty text content should be skipped."""
        messages = [
            _make_human_message("list files"),
            _make_ai_message("Let me check", msg_id="a1"),
            _make_ai_message("", msg_id="a2",
                             tool_calls=[{"id": "tc1", "name": "bash", "args": {"cmd": "ls"}}]),
            _make_tool_message("file1.txt", tool_call_id="tc1"),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"]["content"] == "Let me check"
        assert result["last_response"]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_short_response(self):
        """Short responses like '7' must not be filtered out."""
        messages = [
            _make_human_message("what is 1+1?"),
            _make_ai_message("2", msg_id="a1"),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"]["content"] == "2"

    @pytest.mark.asyncio
    async def test_no_ai_messages(self):
        """Returns None when there are no AIMessages in history."""
        messages = [
            _make_human_message("hello"),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"] is None

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """Returns None when history is empty."""
        process = _make_agent_process_with_messages([])

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"] is None

    @pytest.mark.asyncio
    async def test_no_state(self):
        """Returns None when agent has no state (no checkpoint)."""
        from deep_worker_cli.process.agent_process import AgentProcess

        process = AgentProcess.__new__(AgentProcess)
        process.name = "test-agent"

        mock_agent = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {}
        mock_agent.aget_state = AsyncMock(return_value=mock_state)
        process.agent = mock_agent

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"] is None

    @pytest.mark.asyncio
    async def test_agent_not_initialized(self):
        """Returns error when agent is None."""
        from deep_worker_cli.process.agent_process import AgentProcess

        process = AgentProcess.__new__(AgentProcess)
        process.name = "test-agent"
        process.agent = None

        result = await process._handle_get_last_response()

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_content_blocks_list(self):
        """AIMessage with content as list of blocks is handled."""
        messages = [
            _make_human_message("explain"),
            _make_ai_message(
                [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}],
                msg_id="a1",
            ),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_includes_tool_calls_in_response(self):
        """If last AIMessage has both content and tool_calls, both are returned."""
        messages = [
            _make_human_message("search for X"),
            _make_ai_message(
                "I'll search for that",
                msg_id="a1",
                tool_calls=[{"id": "tc1", "name": "web_search", "args": {"q": "X"}}],
            ),
        ]
        process = _make_agent_process_with_messages(messages)

        result = await process._handle_get_last_response()

        assert result["success"] is True
        assert result["last_response"]["content"] == "I'll search for that"
        assert "tool_calls" in result["last_response"]
        assert result["last_response"]["tool_calls"][0]["name"] == "web_search"
