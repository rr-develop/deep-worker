"""Middleware to patch dangling tool calls in the messages history.

See ``deep_worker/docs/DESIGN_CONTEXT_DELTA_CACHE.md`` §4.6. Older
revisions of this middleware returned a full ``Overwrite(messages)`` on
every invocation even when no patching was needed. That defeated
delta-checkpointing (every turn rewrote the entire messages channel
instead of appending) and forced every downstream cache to invalidate.

The current behaviour is:

    * A tool_call is closed only if a ToolMessage with that id sits in
      the contiguous ToolMessage run immediately after its AIMessage
      (OpenAI / DeepSeek adjacency). A later non-adjacent ToolMessage
      does not count — it is moved next to the AIMessage.
    * If the order changes, return
      ``{"messages": Overwrite(patched_messages)}``.
    * Otherwise return ``None`` — the channel is untouched and the
      checkpointer can store an append-delta for a new human/AI turn.
    * High-water mark is armed only after a clean (None) scan, so an
      unapplied Overwrite cannot skip the next pass on the same list.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


def _cancelled_closer(tool_call: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=(
            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
            "cancelled - another message came in before it could be completed."
        ),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )


def _adjacent_tool_ids(messages: list[Any], ai_index: int) -> set[str]:
    """tool_call_ids of the contiguous ToolMessage run after messages[ai_index]."""
    ids: set[str] = set()
    j = ai_index + 1
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        tid = messages[j].tool_call_id
        if tid:
            ids.add(tid)
        j += 1
    return ids


def _rewrite_openai_adjacent(
    messages: list[Any],
) -> tuple[list[Any], bool]:
    """Return (rewritten, changed) so every AIMessage.tool_calls is adjacent.

    Existing later ToolMessages with a needed id are *moved* next to the
    AIMessage (not copied) so the trailing duplicate cannot poison OpenAI.
    Missing ids get a synthetic cancelled closer.
    """
    needed_later: dict[str, ToolMessage] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        adjacent = _adjacent_tool_ids(messages, i)
        for call in msg.tool_calls:
            cid = call.get("id")
            if not cid or cid in adjacent:
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
                needed_later[cid] = later

    patched: list[Any] = []
    changed = False
    consumed: set[int] = set()
    for i, msg in enumerate(messages):
        if id(msg) in consumed:
            changed = True
            continue
        patched.append(msg)
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        adjacent = _adjacent_tool_ids(messages, i)
        for call in msg.tool_calls:
            cid = call.get("id")
            if not cid or cid in adjacent:
                continue
            moved = needed_later.get(cid)
            if moved is not None:
                patched.append(moved)
                consumed.add(id(moved))
            else:
                patched.append(_cancelled_closer(call))
            changed = True
    return patched, changed


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def __init__(self) -> None:
        super().__init__()
        # High-water mark: the length of the last fully-scanned
        # messages list. If the current list has grown and its prefix
        # is the same, we only scan the tail.
        self._scanned_len: int = 0

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage.

        Returns ``None`` when the history needs no patching — callers
        MUST NOT rewrite the messages channel when nothing is dangling
        (see module docstring for why).
        """
        messages = state["messages"]
        if not messages:
            self._scanned_len = 0
            return None

        # Fast-path: last scan returned None (history was already adjacent)
        # and length is unchanged → nothing to do. Do NOT arm this skip
        # after an Overwrite: if the rewrite was not applied, the next
        # call must still patch the same poisoned list.
        if self._scanned_len == len(messages):
            return None

        patched_messages, any_patched = _rewrite_openai_adjacent(messages)

        if not any_patched:
            # Channel untouched — let delta checkpointing append cleanly.
            # Only a clean scan may arm the high-water mark.
            self._scanned_len = len(messages)
            return None
        return {"messages": Overwrite(patched_messages)}
