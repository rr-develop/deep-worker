"""Close dangling tool-calls carried over from a previous (interrupted) session.

See design document:
    deep_worker_cli/docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md

Problem (Bug B)
---------------
When an agent process is killed mid-turn (e.g. a provider CLI subprocess hung
and was terminated), the LangGraph checkpoint can be left with an ``AIMessage``
whose ``tool_calls`` have NO matching ``ToolMessage``. On the next turn after
restart the agent is invoked with a fresh ``HumanMessage``; LangGraph appends it
and re-enters the graph, and the unresolved tool-call would be routed into
``ToolNode`` — RE-EXECUTING the tool. Re-running a tool automatically has
uncontrolled side effects and must never happen implicitly. (Observed crashing
codex agents into ``status=error`` forever, since the stale call's future no
longer exists.)

Fix (Bug B)
-----------
A ``before_model`` middleware. ``langchain.agents.create_agent`` makes the first
``before_model`` middleware the graph ``entry_node`` (``START -> entry_node``,
and tools loop back to it), so this hook runs BEFORE the model and BEFORE any
tool execution on every fresh super-step. We scan the message history for
dangling tool-calls and CLOSE each with a synthetic ``ToolMessage`` so the tool
is never re-invoked and the history is valid for the model API.

The closer is a WARNING (user decision Q1): we do not know whether the tool ran
(the prior session was likely killed during the call). ``ToolMessage.status``
only allows ``"success"``/``"error"``; we encode the warning as
``status="error"`` (so it is never mistaken for a real result) plus a
machine-readable marker in ``additional_kwargs`` and an explicit
outcome-unknown message, leaving the decision to the agent.

This middleware is provider-agnostic, event-driven, idempotent (once the closers
become real ``ToolMessage``s they resolve the calls, so a later pass is a
no-op), and adds no polling and no threads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage

from deepagents.middleware.message_timestamps import ensure_message_created_at_after

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Machine-readable marker keys placed on the synthetic closer's
# ``additional_kwargs`` so downstream code / clients can distinguish a
# "previous-session, outcome-unknown" closer from a genuine tool error.
DANGLING_MARKER_KEY = "dangling_tool_call"
WARNING_KIND_KEY = "warning_kind"
WARNING_KIND_VALUE = "outcome_unknown"


def _closer_content(tool_name: str) -> str:
    """Honest, human-readable WARNING text for a dangling-call closer."""
    return (
        f"[deep-worker] WARNING: this tool call ('{tool_name}') was issued in a "
        "previous session that was interrupted (the agent process was likely "
        "killed) while the tool was being invoked. Its outcome is UNKNOWN — it "
        "may have completed, partially run, or not run at all — and it was NOT "
        "re-executed on restart (re-running tools automatically is unsafe). A "
        "new session has now started. Decide for yourself whether to inspect "
        "any resulting state and/or call the tool again."
    )


class DanglingToolCallMiddleware(AgentMiddleware):
    """Close tool-calls left unresolved by a previous interrupted session.

    Must be registered as the FIRST ``before_model`` middleware so it is the
    graph ``entry_node`` and normalises the state before any other middleware,
    the model, or ``ToolNode`` sees it.
    """

    # Explicit empty list — this middleware registers no tools of its own.
    # Declared at class level so ``middleware.tools`` is always iterable even
    # before ``__init__`` runs (some langgraph code introspects this).
    tools: list = []

    def before_model(
        self,
        state: AgentState,
        runtime: "Runtime[Any] | None" = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        # Defensive: state might be an unusual shape in tests; use .get().
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None

        # 1. Collect tool_call_ids that already have a ToolMessage (resolved).
        resolved_ids: set[str] = {
            m.tool_call_id
            for m in messages
            if isinstance(m, ToolMessage) and m.tool_call_id is not None
        }

        # 2. Find dangling tool-calls: an AIMessage tool_call whose id has no
        #    matching ToolMessage. Preserve first-seen order; de-dup ids.
        closers: list[ToolMessage] = []
        seen_dangling: set[str] = set()
        for m in messages:
            if not isinstance(m, AIMessage):
                continue
            for call in m.tool_calls or []:
                call_id = call.get("id")
                if not isinstance(call_id, str):
                    continue
                if call_id in resolved_ids or call_id in seen_dangling:
                    continue
                seen_dangling.add(call_id)
                tool_name = call.get("name") or "unknown"
                closer = ToolMessage(
                    content=_closer_content(str(tool_name)),
                    tool_call_id=call_id,
                    name=str(tool_name),
                    status="error",
                    additional_kwargs={
                        DANGLING_MARKER_KEY: True,
                        WARNING_KIND_KEY: WARNING_KIND_VALUE,
                    },
                )
                ensure_message_created_at_after(messages + closers, closer)
                closers.append(closer)

        # 3. Nothing dangling -> normal flow (zero overhead on the common path).
        if not closers:
            return None

        logger.info(
            "DanglingToolCallMiddleware: closing %d unresolved tool-call(s) "
            "from a previous session (ids=%s) — tools were NOT re-executed.",
            len(closers),
            [c.tool_call_id for c in closers],
        )
        # Append the closers; do NOT jump — let the model run with valid history.
        return {"messages": closers}
