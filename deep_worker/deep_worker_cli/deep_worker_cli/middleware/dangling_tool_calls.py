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
sit immediately after their AIMessage they resolve the calls, so a later pass
is a no-op), and adds no polling and no threads. Closers are applied via
``Overwrite`` so they are adjacent to the AIMessage — appending them would
leave a System/Human restart-notice in between and DeepSeek would 400.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Overwrite

from deepagents.middleware.message_timestamps import ensure_message_created_at_after

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Machine-readable marker keys placed on the synthetic closer's
# ``additional_kwargs`` so downstream code / clients can distinguish a
# "previous-session, outcome-unknown" closer from a genuine tool error.
DANGLING_MARKER_KEY = "deep_worker_dangling_tool_call"
WARNING_KIND_KEY = "deep_worker_warning"
WARNING_KIND_VALUE = "outcome_unknown"


def _make_warning_closer(call_id: str, tool_name: str, prior: list) -> ToolMessage:
    closer = ToolMessage(
        content=_closer_content(tool_name),
        tool_call_id=call_id,
        name=tool_name,
        status="error",
        additional_kwargs={
            DANGLING_MARKER_KEY: True,
            WARNING_KIND_KEY: WARNING_KIND_VALUE,
        },
    )
    ensure_message_created_at_after(prior, closer)
    return closer


def _adjacent_tool_ids(messages: list, ai_index: int) -> set[str]:
    ids: set[str] = set()
    j = ai_index + 1
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        tid = messages[j].tool_call_id
        if tid:
            ids.add(tid)
        j += 1
    return ids


def _rewrite_adjacent_closers(messages: list) -> tuple[list, list[str]]:
    """Place a ToolMessage immediately after each dangling AIMessage.tool_calls.

    Returns (rewritten_list, ids_acted_on). ids_acted_on is empty iff the
    history was already OpenAI-adjacent (caller should return None).
    Existing later ToolMessages are moved, not copied.
    """
    later_by_id: dict[str, ToolMessage] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        adjacent = _adjacent_tool_ids(messages, i)
        for call in msg.tool_calls or []:
            cid = call.get("id")
            if not isinstance(cid, str) or cid in adjacent:
                continue
            later = next(
                (
                    m
                    for m in messages[i + 1 :]
                    if isinstance(m, ToolMessage) and m.tool_call_id == cid
                ),
                None,
            )
            if later is not None:
                later_by_id[cid] = later

    rewritten: list = []
    acted: list[str] = []
    consumed: set[int] = set()
    for i, msg in enumerate(messages):
        if id(msg) in consumed:
            continue
        rewritten.append(msg)
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        adjacent = _adjacent_tool_ids(messages, i)
        for call in msg.tool_calls or []:
            cid = call.get("id")
            if not isinstance(cid, str) or cid in adjacent:
                continue
            moved = later_by_id.get(cid)
            if moved is not None:
                rewritten.append(moved)
                consumed.add(id(moved))
            else:
                rewritten.append(
                    _make_warning_closer(cid, str(call.get("name") or "unknown"), rewritten)
                )
            acted.append(cid)
    return rewritten, acted


def _drop_orphaned_tool_messages(messages: list) -> tuple[list, list[str]]:
    """Drop ``ToolMessage`` objects whose ``AIMessage.tool_calls`` is gone.

    The MIRROR of ``_rewrite_adjacent_closers``: that one handles an
    ``AIMessage`` whose ``ToolMessage`` never arrived, this one handles a
    ``ToolMessage`` whose ``AIMessage`` no longer exists. Both shapes are
    rejected by the provider, but only the first had a fix.

    How an orphan is produced (repo_graph ingest, 2026-09-23): a
    ``--sum-query`` run wipes the whole message channel via
    ``RemoveMessage(REMOVE_ALL_MESSAGES)`` (agent_process.py:2529). The wipe
    is atomic over what it can see, so it removes the ``AIMessage`` and its
    ``ToolMessage`` together — but a tool-call still IN FLIGHT when the wipe
    lands resolves afterwards, and ``add_messages`` appends its
    ``ToolMessage`` onto the freshly-installed summary state. Its parent
    ``AIMessage`` is already gone, so the very next request carries a
    ``ToolMessage`` with nothing to answer, and the provider returns:

        400 — Messages with role 'tool' must be a response to a preceding
        message with 'tool_calls'

    Dropping (not closing) is the only honest repair: a closer needs a
    ``tool_call`` to answer, and here there is none — the request it belonged
    to was deliberately erased. The tool's OUTPUT is not lost silently: the
    wipe path offloads the pre-wipe history to ``<agent>.jsonl`` first, and
    the drop is logged with the ids.

    Returns ``(rewritten, dropped_ids)``; ``dropped_ids`` is empty when the
    history holds no orphans (caller then returns ``None``).
    """
    live_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                cid = call.get("id")
                if isinstance(cid, str):
                    live_call_ids.add(cid)

    rewritten: list = []
    dropped: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id not in live_call_ids:
            dropped.append(str(msg.tool_call_id))
            continue
        rewritten.append(msg)
    return rewritten, dropped


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

        # Orphan sweep FIRST: a ToolMessage with no surviving parent cannot be
        # paired by the closer pass below (that pass only ever walks
        # AIMessage.tool_calls, so an orphan is invisible to it) and would
        # reach the provider as a 400. Running it first also means the closer
        # pass sees an already-consistent list.
        messages, dropped = _drop_orphaned_tool_messages(messages)
        if dropped:
            logger.warning(
                "DanglingToolCallMiddleware: dropping %d orphaned ToolMessage(s) "
                "(ids=%s) whose AIMessage.tool_calls no longer exists — most "
                "likely an in-flight tool resolved after a --sum-query history "
                "wipe. Their content was offloaded to the agent's JSONL history "
                "by the wipe path; dropping here keeps the request valid "
                "instead of failing with a provider 400.",
                len(dropped),
                dropped,
            )

        rewritten, moved_or_new = _rewrite_adjacent_closers(messages)
        if not moved_or_new:
            # Orphans alone still require an Overwrite — the caller's state
            # must lose them, not merely this local list.
            if dropped:
                return {"messages": Overwrite(messages)}
            return None

        logger.info(
            "DanglingToolCallMiddleware: closing %d unresolved tool-call(s) "
            "from a previous session (ids=%s) — tools were NOT re-executed. "
            "Closers placed immediately after their AIMessage (OpenAI adjacency).",
            len(moved_or_new),
            moved_or_new,
        )
        # Overwrite, do not append: add_messages would put closers at the
        # end, after restart NOTICE / continuation, which DeepSeek rejects.
        return {"messages": Overwrite(rewritten)}
