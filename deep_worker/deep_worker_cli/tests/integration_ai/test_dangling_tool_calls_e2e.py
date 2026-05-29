"""End-to-end (no-mock) regression test for Bug B — re-invocation of a
dangling tool-call carried over from a previous, interrupted session.

Design doc:
    deep_worker_cli/docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md

What this test proves
---------------------
It builds a REAL LangGraph agent (``langchain.agents.create_agent`` — the same
primitive ``create_cli_agent`` delegates to via ``create_deep_agent``) with:

* a REAL ``DanglingToolCallMiddleware`` (the unit under test — NOT mocked),
* a REAL tool whose body records whether it ever executed,
* a REAL ``InMemorySaver`` checkpointer.

It then seeds the checkpoint with a history whose tail is an ``AIMessage``
carrying a poison tool-call id (``apply_patch_136_…`` — the proxy-style id seen
in a live agent crash) with NO matching ``ToolMessage`` — exactly the
state a session that was killed mid-tool leaves behind. A NEW ``HumanMessage``
is then sent (a fresh super-step, like ``agent_process`` does on every turn).

Assertions:
1. The run does NOT raise / does NOT error out.
2. The dangling tool-call was CLOSED by a synthetic ``ToolMessage`` carrying the
   warning markers — i.e. it is no longer dangling.
3. The poison tool body was NEVER executed (no uncontrolled re-invocation).

Nothing here is mocked: the graph, the middleware, the checkpointer and the
tool are all real. No model API / network / CLI is used — the model is a real
deterministic ``GenericFakeChatModel`` from langchain_core acting only as a
turn driver (it emits a plain assistant message, no new tool calls).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deep_worker_cli.middleware.dangling_tool_calls import (
    DanglingToolCallMiddleware,
)


class _ToolAwareFakeModel(GenericFakeChatModel):
    """Real ``GenericFakeChatModel`` that also supports ``bind_tools``.

    ``create_agent`` calls ``model.bind_tools(tools)`` when tools are present;
    the base fake raises ``NotImplementedError``. We override it to return self
    (the bound tool set does not affect this test — the scripted message has no
    tool calls). This is a real model implementation acting as a deterministic
    turn driver, not a mock of any unit under test.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "Any":  # noqa: ANN401
        return self

# The poison id shape observed in a live agent crash:
# f"{tool_name}_{counter}_{id(args)}" — a proxy-MCP id persisted in history.
_POISON_ID = "apply_patch_136_136716975835008"
_TOOL_NAME = "apply_patch"


@pytest.mark.asyncio
async def test_dangling_tool_call_is_closed_not_reinvoked() -> None:
    # --- a REAL tool that records whether its body ever ran ----------------
    executed: list[str] = []

    @tool(_TOOL_NAME)
    def apply_patch(patch: str) -> str:
        """Apply a patch (test double of the native passthrough)."""
        executed.append(patch)  # if this ever runs, the bug is NOT fixed
        return "patched"

    # --- a REAL deterministic model that asks for NO further tools ----------
    # GenericFakeChatModel is a real BaseChatModel implementation (not a Mock):
    # it returns the scripted AIMessage when the graph calls the model node.
    model = _ToolAwareFakeModel(
        messages=iter([AIMessage(content="Acknowledged — continuing.")])
    )

    # --- a REAL graph + REAL middleware + REAL checkpointer -----------------
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[apply_patch],
        middleware=[DanglingToolCallMiddleware()],
        checkpointer=checkpointer,
    )

    thread_id = "agent_repro"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # --- seed the checkpoint with an interrupted prior session --------------
    # The tail is an AIMessage with a tool_call that has NO matching
    # ToolMessage — the exact poison state left by a killed mid-tool session.
    poison_ai = AIMessage(
        content="",
        tool_calls=[
            {"id": _POISON_ID, "name": _TOOL_NAME, "args": {"patch": "X"}, "type": "tool_call"}
        ],
    )
    await agent.aupdate_state(
        config,
        {"messages": [HumanMessage(content="please patch the file"), poison_ai]},
    )

    # --- fresh turn: a NEW HumanMessage (like agent_process does) -----------
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="continue, what now?")]},
        config,
    )

    messages = result["messages"]

    # 1. The poison tool body was NEVER executed (no uncontrolled re-invoke).
    assert executed == [], (
        "the dangling tool call was RE-EXECUTED — Bug B is not fixed; "
        f"recorded executions: {executed!r}"
    )

    # 2. The dangling call was closed by a synthetic warning ToolMessage.
    closers = [
        m
        for m in messages
        if isinstance(m, ToolMessage)
        and m.tool_call_id == _POISON_ID
        and m.additional_kwargs.get("dangling_tool_call") is True
    ]
    assert len(closers) == 1, (
        "expected exactly one synthetic closer ToolMessage for the dangling "
        f"call; got {len(closers)}"
    )
    closer = closers[0]
    assert closer.status == "error"
    assert closer.additional_kwargs.get("warning_kind") == "outcome_unknown"
    assert closer.name == _TOOL_NAME

    # 3. No dangling tool-calls remain (every AIMessage tool_call now resolved).
    resolved_ids = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls or []:
                assert call["id"] in resolved_ids, (
                    f"tool_call {call['id']} is still dangling after the turn"
                )

    # 4. The turn completed normally — the model produced its assistant reply.
    assert any(
        isinstance(m, AIMessage) and "continuing" in (m.content or "").lower()
        for m in messages
    )
