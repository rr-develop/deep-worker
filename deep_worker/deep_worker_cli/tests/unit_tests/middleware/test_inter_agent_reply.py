"""Unit tests for InterAgentReplyMiddleware.

This middleware detects incoming inter-agent messages and injects
cognitive hints based on the ``reply_back`` flag.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from deep_worker_cli.middleware.inter_agent_reply import (
    INTER_AGENT_PATTERN,
    InterAgentReplyMiddleware,
    _find_system_message,
    _is_inter_agent_message,
)


# ===========================================================================
# Pattern matching
# ===========================================================================


class TestInterAgentPattern:
    def test_matches_valid_inter_agent_message(self) -> None:
        match = INTER_AGENT_PATTERN.match("alpha -> beta :> hello")
        assert match is not None
        assert match.group(1) == "alpha"
        assert match.group(2) == "beta"

    def test_matches_with_spaces_in_message(self) -> None:
        match = INTER_AGENT_PATTERN.match("agent1 -> agent2 :> 2+2")
        assert match is not None
        assert match.group(1) == "agent1"
        assert match.group(2) == "agent2"

    def test_rejects_normal_user_message(self) -> None:
        assert INTER_AGENT_PATTERN.match("hello") is None

    def test_rejects_task_manager_prefix(self) -> None:
        assert INTER_AGENT_PATTERN.match("TaskManager:> hello agent") is None


class TestIsInterAgentMessage:
    def test_detects_message_addressed_to_us(self) -> None:
        msg = HumanMessage(content="alpha -> beta :> hello")
        assert _is_inter_agent_message(msg, "beta") is True

    def test_rejects_message_for_other_agent(self) -> None:
        msg = HumanMessage(content="alpha -> gamma :> hello")
        assert _is_inter_agent_message(msg, "beta") is False

    def test_rejects_normal_message(self) -> None:
        msg = HumanMessage(content="hello user")
        assert _is_inter_agent_message(msg, "beta") is False

    def test_handles_non_string_content(self) -> None:
        """_is_inter_agent_message uses getattr + isinstance, so it can
        handle objects with non-string content without crashing."""
        obj = type("FakeMsg", (), {"content": 123})()
        assert _is_inter_agent_message(obj, "beta") is False  # type: ignore[arg-type]


# ===========================================================================
# _find_system_message
# ===========================================================================


class TestFindSystemMessage:
    def test_finds_first_system_message(self) -> None:
        msgs = [
            SystemMessage(content="you are a bot"),
            HumanMessage(content="hi"),
        ]
        found = _find_system_message(msgs)
        assert found is not None
        assert found.content == "you are a bot"

    def test_returns_none_when_no_system_message(self) -> None:
        msgs = [HumanMessage(content="hi")]
        assert _find_system_message(msgs) is None

    def test_returns_none_for_empty_list(self) -> None:
        assert _find_system_message([]) is None


# ===========================================================================
# InterAgentReplyMiddleware
# ===========================================================================


class TestInterAgentReplyMiddlewareDetection:
    """Middleware correctly detects inter-agent messages and injects hints."""

    def test_passthrough_for_empty_state(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        result = mw.before_model({})
        assert result is None

    def test_passthrough_for_no_messages(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        result = mw.before_model({"messages": []})
        assert result is None

    def test_passthrough_for_non_inter_agent_message(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        state = {"messages": [HumanMessage(content="hello user")]}
        result = mw.before_model(state)
        assert result is None

    def test_passthrough_for_message_to_other_agent(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        state = {"messages": [HumanMessage(content="alpha -> gamma :> hello")]}
        result = mw.before_model(state)
        assert result is None

    def test_injects_reply_back_hint_with_system_message(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content="alpha -> beta :> please compute 2+2",
                id="msg-1",
                additional_kwargs={"reply_back": True},
            ),
        ]
        result = mw.before_model({"messages": msgs})
        assert result is not None
        assert "messages" in result
        updated_msgs = result["messages"]
        # System message should contain the hint
        sys_msg = updated_msgs[0]
        assert isinstance(sys_msg, SystemMessage)
        # Strengthened hint (conditional MUST): the reply itself is
        # optional, but IF the agent replies to a peer it MUST use the
        # send_message_to_agent tool, framed as a CRITICAL SYSTEM
        # REQUIREMENT. The conditional ("IF you need to send back") is
        # preserved on purpose — an unconditional "you MUST reply" used to
        # trigger Claude's prompt-injection defenses and produce a
        # plain-text "I'm waiting" filler instead of a tool call
        # (a related internal incident analysis (not included here)).
        assert "send_message_to_agent" in sys_msg.content
        assert "alpha" in sys_msg.content
        # Conditional trigger preserved.
        assert "IF you need to send" in sys_msg.content
        # Imperative on the CHANNEL (not on replying at all).
        assert "you MUST use" in sys_msg.content
        # Framed as a critical system requirement.
        assert "CRITICAL SYSTEM REQUIREMENT" in sys_msg.content

    def test_injects_reply_back_false_hint(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content="alpha -> beta :> tell the user hello",
                id="msg-2",
                additional_kwargs={"reply_back": False},
            ),
        ]
        result = mw.before_model({"messages": msgs})
        assert result is not None
        sys_msg = result["messages"][0]
        assert isinstance(sys_msg, SystemMessage)
        # reply_back=False hint must steer the model to the USER rather
        # than back to the peer; it must NOT carry the reply-via-tool
        # phrasing of the reply_back=True branch.
        assert "reply_back=False" in sys_msg.content
        assert "to the user" in sys_msg.content
        assert "If you need to send" not in sys_msg.content
        assert "IF you need to send" not in sys_msg.content
        assert "CRITICAL SYSTEM REQUIREMENT" not in sys_msg.content
        assert "IF you need to send" not in sys_msg.content
        assert "CRITICAL SYSTEM REQUIREMENT" not in sys_msg.content
        assert "IF you need to send" not in sys_msg.content
        assert "CRITICAL SYSTEM REQUIREMENT" not in sys_msg.content

    def test_injects_hint_only_once_per_message_id(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        incoming = HumanMessage(
            content="alpha -> beta :> hello",
            id="msg-3",
            additional_kwargs={"reply_back": True},
        )
        msgs = [SystemMessage(content="You are a bot."), incoming]

        # First call: should inject
        result1 = mw.before_model({"messages": msgs})
        assert result1 is not None
        assert "send_message_to_agent" in result1["messages"][0].content
        assert "IF you need to send" in result1["messages"][0].content
        assert "you MUST use" in result1["messages"][0].content
        assert "CRITICAL SYSTEM REQUIREMENT" in result1["messages"][0].content

        # Second call: msg-3 already tracked → should passthrough
        result2 = mw.before_model({"messages": msgs})
        assert result2 is None

    def test_default_reply_back_when_not_in_additional_kwargs(self) -> None:
        """reply_back defaults to True when additional_kwargs lacks it."""
        mw = InterAgentReplyMiddleware(name="beta")
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content="alpha -> beta :> hello",
                id="msg-4",
            ),
        ]
        result = mw.before_model({"messages": msgs})
        assert result is not None
        sys_msg = result["messages"][0]
        assert isinstance(sys_msg, SystemMessage)
        assert "send_message_to_agent" in sys_msg.content
        assert "IF you need to send" in sys_msg.content
        assert "you MUST use" in sys_msg.content
        assert "CRITICAL SYSTEM REQUIREMENT" in sys_msg.content

    def test_creates_system_message_when_missing(self) -> None:
        """If no SystemMessage exists, middleware should create one with
        the cognitive hint rather than passthrough."""
        mw = InterAgentReplyMiddleware(name="beta")
        msgs = [
            HumanMessage(
                content="alpha -> beta :> hello",
                id="msg-5",
                additional_kwargs={"reply_back": True},
            ),
        ]
        result = mw.before_model({"messages": msgs})
        assert result is not None, "Middleware should not passthrough when no SystemMessage exists"
        new_msgs = result.get("messages", [])
        assert len(new_msgs) == 2
        assert isinstance(new_msgs[0], SystemMessage), (
            f"First message should be a new SystemMessage, got {type(new_msgs[0])}"
        )
        assert "send_message_to_agent" in new_msgs[0].content
        assert "IF you need to send" in new_msgs[0].content
        assert "you MUST use" in new_msgs[0].content
        assert "CRITICAL SYSTEM REQUIREMENT" in new_msgs[0].content

    def test_handles_message_without_id(self) -> None:
        """Messages without an 'id' should still trigger hint injection
        once (tracked via fallback key)."""
        mw = InterAgentReplyMiddleware(name="beta")
        msgs = [
            SystemMessage(content="You are a bot."),
            HumanMessage(
                content="alpha -> beta :> hello",
                additional_kwargs={"reply_back": True},
            ),
        ]
        # First call: should inject
        result1 = mw.before_model({"messages": msgs})
        assert result1 is not None

        # Second call with same state: should not re-inject
        result2 = mw.before_model({"messages": msgs})
        assert result2 is None


class TestInterAgentReplyMiddlewareContract:
    """AgentMiddleware contract compliance."""

    def test_is_agent_middleware(self) -> None:
        from langchain.agents.middleware.types import AgentMiddleware

        mw = InterAgentReplyMiddleware(name="beta")
        assert isinstance(mw, AgentMiddleware)

    def test_exposes_no_extra_tools(self) -> None:
        mw = InterAgentReplyMiddleware(name="beta")
        assert mw.tools == []

    def test_requires_name_constructor_argument(self) -> None:
        mw = InterAgentReplyMiddleware(name="test-agent")
        assert mw._name == "test-agent"
