"""Graph-level proof that a registered turn-ending tool ends the turn
without a second model call.

See design document:
    deep_worker_cli/docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md
"""

from __future__ import annotations

from langchain.agents.factory import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from deep_worker_cli.middleware.tool_completion import (
    ToolCompletionMiddleware,
    TurnEndingTool,
)

SENTINEL = "SECOND MODEL CALL HAPPENED"


class _CountingFakeModel(FakeMessagesListChatModel):
    """Fake chat model that tolerates bind_tools and counts invocations."""

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


@tool
def submit_fragments(payload: str) -> dict:
    """Submit fragments and close the document."""
    return {"document_closed": True, "receipt": "r-1"}


@tool
def echoing_tool(payload: str) -> dict:
    """A turn-ending tool that produces a user-visible echo."""
    return {"success": True, "delivered_content": "the visible answer"}


def _build(tool_obj, spec: TurnEndingTool):
    first = AIMessage(
        content="",
        tool_calls=[{"name": tool_obj.name, "args": {"payload": "x"}, "id": "call-1"}],
    )
    model = _CountingFakeModel(responses=[first, AIMessage(content=SENTINEL)])
    agent = create_agent(
        model=model,
        tools=[tool_obj],
        middleware=[ToolCompletionMiddleware([spec])],
    )
    return agent


def _contents(messages) -> list[str]:
    out = []
    for m in messages:
        c = getattr(m, "content", "")
        out.append(c if isinstance(c, str) else str(c))
    return out


class TestZeroSecondModelCall:
    def test_jump_only_mode_ends_turn_without_second_call(self):
        """echo_field=None: turn ends, no synthetic message, no 2nd model call."""
        agent = _build(
            submit_fragments,
            TurnEndingTool(
                tool_name="submit_fragments",
                is_success=lambda d: d.get("document_closed") is True,
                echo_field=None,
            ),
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        messages = result["messages"]

        # The queued second response was never consumed.
        assert SENTINEL not in _contents(messages)
        # Turn ends on the tool result; no empty AIMessage was manufactured.
        assert isinstance(messages[-1], ToolMessage)
        assert not [
            m for m in messages
            if isinstance(m, AIMessage) and not m.tool_calls and not str(m.content).strip()
        ]

    def test_echo_mode_ends_turn_without_second_call(self):
        """echo_field set: the echo is injected and no 2nd model call happens."""
        agent = _build(
            echoing_tool,
            TurnEndingTool(tool_name="echoing_tool"),
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        messages = result["messages"]

        assert SENTINEL not in _contents(messages)
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content == "the visible answer"

    def test_unregistered_tool_does_not_end_turn(self):
        """Control: without registration the model IS called again."""
        agent = _build(
            submit_fragments,
            TurnEndingTool(tool_name="some_other_tool", echo_field=None),
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert SENTINEL in _contents(result["messages"])

    def test_state_remains_reinvocable_after_jump_only(self):
        """A turn ended on a ToolMessage tail leaves valid, resumable state."""
        agent = _build(
            submit_fragments,
            TurnEndingTool(
                tool_name="submit_fragments",
                is_success=lambda d: d.get("document_closed") is True,
                echo_field=None,
            ),
        )
        first = agent.invoke({"messages": [HumanMessage(content="go")]})
        follow_up = list(first["messages"]) + [HumanMessage(content="continue")]
        second = agent.invoke({"messages": follow_up})
        # The model was reachable on the next turn.
        assert SENTINEL in _contents(second["messages"])
