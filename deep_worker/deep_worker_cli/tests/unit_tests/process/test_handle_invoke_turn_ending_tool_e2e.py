"""Live end-to-end repro of the agent_a 2026-08-22 defect.

Unlike ``test_handle_invoke_status_on_empty_response.py`` — which
stubs ``agent.agent.ainvoke`` with a ``MagicMock`` and hands
``_handle_invoke`` a hand-built message list — this module wires a
**real LangGraph agent** built by ``create_agent`` with a **real**
``ToolCompletionMiddleware`` into a real ``AgentProcess._handle_invoke``.

Nothing about the turn-ending shape is asserted by construction here:
the ``jump_to="end"`` decision, the absence of a trailing ``AIMessage``
and the ``ToolMessage.status`` value are all produced by the production
middleware + LangGraph runtime. Only the chat model is fake (there is
no way to make a real provider deterministic in a unit test), and it is
the same ``FakeMessagesListChatModel`` the middleware's own graph tests
use.

Scenario, verbatim from agent_a's report (``rg_extractor`` running
the ``the code-graph server`` MCP server):

    the model calls ``code_graph__submit_fragments``; the tool returns
    ``{"success": true, "document_closed": true}``; that tool is
    registered as a ``TurnEndingTool`` with ``echo_field=None``, so
    ``ToolCompletionMiddleware`` jumps to ``end`` WITHOUT injecting a
    synthetic assistant message. ``state["messages"]`` therefore ends
    with the ``ToolMessage`` itself.

Before the fix, ``_handle_invoke``'s non-TaskManager standard branch
had no exit for that tail: a ``ToolMessage`` has ``role='tool'`` and a
non-empty JSON ``content``, so it matched no break condition, the loop
re-invoked the graph until ``max_iterations``, and the caller got
``success=False`` — either "Agent exceeded maximum iterations ..." or
"Agent did not produce a response" — for a turn that had in fact
succeeded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain.agents.factory import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deep_worker_cli.middleware.tool_completion import (
    ToolCompletionMiddleware,
    TurnEndingTool,
)
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus

# If the model is ever asked for a second response, this string lands in
# the transcript — proof the turn did NOT end where it should have.
SENTINEL = "SECOND MODEL CALL HAPPENED"


class _ToolBindingFakeModel(FakeMessagesListChatModel):
    """Fake chat model that tolerates ``bind_tools``.

    ``create_agent`` binds tools to the model; the stock fake rejects it.
    Same shim the middleware graph tests use.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


@tool
def code_graph__submit_fragments(payload: str) -> dict:
    """Submit extracted fragments and close the document session."""
    return {"success": True, "document_closed": True, "receipt": "r-1"}


def _make_agent_with_real_graph() -> AgentProcess:
    """Bare ``AgentProcess`` (no ``__init__``) whose ``.agent`` is a REAL
    compiled LangGraph agent with the REAL turn-ending middleware.
    """
    agent = AgentProcess.__new__(AgentProcess)

    tmp = tempfile.mkdtemp(prefix="dwtest_turn_ending_e2e_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))

    agent.name = "test_turn_ending_e2e"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"

    # The model emits ONLY the tool call — no text. This is the exact
    # shape agent_a observed: the model delegates the whole turn to
    # the submit_* tool. The second response exists solely to prove it
    # is never consumed.
    model = _ToolBindingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "code_graph__submit_fragments",
                        "args": {"payload": "fragments"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content=SENTINEL),
        ]
    )

    spec = TurnEndingTool(
        tool_name="code_graph__submit_fragments",
        is_success=lambda d: d.get("document_closed") is True,
        echo_field=None,
    )

    agent.agent = create_agent(
        model=model,
        tools=[code_graph__submit_fragments],
        middleware=[ToolCompletionMiddleware([spec])],
    )

    return agent


@pytest.mark.asyncio
async def test_turn_ending_tool_tail_returns_success_through_real_graph() -> None:
    """RED before the fix, GREEN after.

    Real ``create_agent`` graph + real ``ToolCompletionMiddleware`` +
    real ``_handle_invoke``. The only fake is the chat model.
    """
    agent = _make_agent_with_real_graph()

    broadcast: list[dict] = []

    async def notify(payload: dict) -> None:
        broadcast.append(payload)

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "extract fragments", "stream": False},
        connection=None,
        notify_callback=notify,
        approval_callback=None,
    )

    assert result["success"] is True, (
        "a real turn-ending tool closing the turn through the real "
        "middleware must reach the caller as success; got "
        f"{result!r}. This is the agent_a 2026-08-22 defect: the "
        "operator saw 'Agent did not produce a response' for a turn "
        "whose submit_fragments call had actually succeeded"
    )
    assert agent.session_manager.status == AgentStatus.IDLE

    # No ghost assistant message: ToolCompletionMiddleware deliberately
    # refuses to manufacture one for echo_field=None, and _handle_invoke
    # must not manufacture one either.
    assistant_broadcasts = [
        p for p in broadcast if p.get("type") == "assistant"
    ]
    assert not assistant_broadcasts, (
        "an empty 'assistant' broadcast is exactly the ghost message the "
        f"middleware refuses to inject; got {assistant_broadcasts!r}"
    )


@pytest.mark.asyncio
async def test_turn_ending_tool_tail_breaks_the_loop_on_first_iteration() -> None:
    """The evaluation loop must break on iteration 1, not spin to the cap.

    Pre-fix, a ``ToolMessage`` tail matched no break condition:
    ``role == 'tool'`` (so not the assistant-text break), no interrupt,
    no tool_calls, and non-empty JSON content (so not the empty-content
    break). ``_handle_invoke`` therefore re-evaluated the SAME result 20
    times and exited via "Agent exceeded maximum iterations".

    This test observes that spin directly: the loop's per-iteration log
    line is emitted once per iteration, so its count IS the iteration
    count. Pre-fix it is 20; post-fix it is 1.
    """
    agent = _make_agent_with_real_graph()

    import logging as _logging

    records: list[_logging.LogRecord] = []

    class _Collector(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    logger = _logging.getLogger("deep_worker_cli.process.agent_process")
    handler = _Collector()
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(_logging.INFO)
    try:
        result = await agent._handle_invoke(
            {
                "command": "invoke",
                "message": "extract fragments",
                "stream": False,
            },
            connection=None,
            notify_callback=None,
            approval_callback=None,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    iteration_lines = [
        r.getMessage()
        for r in records
        if r.getMessage().startswith("[INVOKE] Iteration ")
    ]
    assert len(iteration_lines) == 1, (
        "the turn-ending ToolMessage tail must terminate the evaluation "
        f"loop on the first iteration; it ran {len(iteration_lines)} "
        "iterations, re-evaluating the identical result until the "
        f"max_iterations cap. Result: {result!r}"
    )

    assert not any(
        "Max iterations reached" in r.getMessage() for r in records
    ), "the loop hit its iteration cap on an already-completed turn"
