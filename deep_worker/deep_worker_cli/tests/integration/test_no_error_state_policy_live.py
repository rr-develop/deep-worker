"""Integration test for the No-Agent-Error-State Policy (no mocks).

Strategy: build a *real* langchain agent graph (no fakes for the graph,
no fakes for the middleware) with a single tool that raises, run it with
a deterministic in-memory chat model that emits exactly one tool-call,
then assert that:

  1. The graph completes — no exception escapes.
  2. The resulting message stream contains a ``ToolMessage`` with
     ``status="error"`` describing the tool failure (produced by our
     :class:`SafeToolCallMiddleware`).
  3. A second turn started after the error proceeds normally — i.e. the
     conversation is fully recoverable, which is exactly the
     "agent must stay IDLE so the user can type continue" requirement.

The chat model here is a small in-process ``BaseChatModel`` subclass
purely so the test is hermetic; the agent / tool / middleware code under
test is the real production code.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from deep_worker_cli.middleware.safe_tool_call import SafeToolCallMiddleware


class _ScriptedChatModel(BaseChatModel):
    """Tiny in-process chat model.

    First turn: emit one tool-call to ``always_fails``.
    Subsequent turns: emit a plain text response so the graph can finish.
    """

    @property
    def _llm_type(self) -> str:  # pragma: no cover — required by BaseChatModel
        return "scripted-for-test"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # The agent factory calls bind_tools to register the tool set.
        # Our scripted model doesn't actually need them — it emits
        # pre-baked tool-calls — but we must accept the call.
        return self

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # The agent factory calls bind_tools to register the tool set.
        # Our scripted model doesn't actually need them — it emits
        # pre-baked tool-calls — but we must accept the call.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        has_tool_msg = any(isinstance(m, ToolMessage) for m in messages)
        if not has_tool_msg:
            ai = AIMessage(
                content="",
                tool_calls=[{
                    "name": "always_fails",
                    "args": {"x": 1},
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )
        else:
            ai = AIMessage(content="I saw the tool error and stopped.")
        return ChatResult(generations=[ChatGeneration(message=ai)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


@tool
def always_fails(x: int) -> str:
    """A tool that always raises — used to verify the safety net."""
    raise RuntimeError(f"deliberate failure for x={x}")


@pytest.mark.asyncio
async def test_tool_exception_is_converted_to_tool_message_status_error():
    """Real graph + real SafeToolCallMiddleware + raising tool.  The
    final message stream must contain a ``ToolMessage(status="error")``
    and the graph must finish without raising."""

    agent = create_agent(
        model=_ScriptedChatModel(),
        tools=[always_fails],
        middleware=[SafeToolCallMiddleware()],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="run the tool please")]},
        {"configurable": {"thread_id": "test-policy-thread"}},
    )

    msgs = result["messages"]
    # Find the tool message — it must exist, must be status="error", must
    # carry the original RuntimeError text.
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected at least one ToolMessage in the result"
    assert any(getattr(m, "status", None) == "error" for m in tool_msgs), (
        "expected at least one ToolMessage with status='error' "
        f"(got statuses: {[getattr(m, 'status', None) for m in tool_msgs]})"
    )
    error_msg = next(m for m in tool_msgs if getattr(m, "status", None) == "error")
    assert "RuntimeError" in error_msg.content
    assert "deliberate failure" in error_msg.content

    # The model must have produced a final AIMessage AFTER seeing the
    # error — proving the conversation is fully recoverable.
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    final_ai = ai_msgs[-1]
    assert "stopped" in (final_ai.content or "")


@pytest.mark.asyncio
async def test_recovery_continues_in_a_second_turn():
    """Second user turn after a tool failure must succeed — modelling
    the user typing "continue" after a recoverable error."""
    from langgraph.checkpoint.memory import MemorySaver

    agent = create_agent(
        model=_ScriptedChatModel(),
        tools=[always_fails],
        middleware=[SafeToolCallMiddleware()],
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "test-policy-continue"}}

    await agent.ainvoke(
        {"messages": [HumanMessage(content="please call the broken tool")]},
        cfg,
    )

    # Second turn — the user says "continue" and the graph keeps going.
    second = await agent.ainvoke(
        {"messages": [HumanMessage(content="continue")]},
        cfg,
    )
    msgs = second["messages"]
    assert any(isinstance(m, AIMessage) for m in msgs)
    # We did not raise; the recoverable-by-default semantics held.
