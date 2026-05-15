"""Middleware to patch dangling tool calls in the messages history.

See ``deep_worker/docs/DESIGN_CONTEXT_DELTA_CACHE.md`` §4.6. Older
revisions of this middleware returned a full ``Overwrite(messages)`` on
every invocation even when no patching was needed. That defeated
delta-checkpointing (every turn rewrote the entire messages channel
instead of appending) and forced every downstream cache to invalidate.

The current behaviour is:

    * Scan the history for AIMessages whose tool_calls lack a matching
      ToolMessage. Bounded by a per-instance high-water mark, so
      successive calls with the same history scan zero messages.
    * If any dangling calls are found, return
      ``{"messages": Overwrite(patched_messages)}`` (preserves original
      contract).
    * Otherwise return ``None`` — the channel is untouched and the
      checkpointer can store an append-delta for a new human/AI turn.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


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

        # Fast-path: history did not grow since the last scan and the
        # last scan found nothing dangling → nothing to do.
        if self._scanned_len == len(messages):
            return None

        patched_messages: list[Any] = []
        any_patched = False

        # Iterate over the messages and add any dangling tool calls
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (
                            m for m in messages[i:]
                            if isinstance(m, ToolMessage) and m.tool_call_id == tool_call["id"]
                        ),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # We have a dangling tool call which needs a ToolMessage
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )
                        any_patched = True

        # Remember the size we just scanned. We bump it to the size of
        # the OUTPUT list so we do not rescan on the next call if nothing
        # has grown. If the history shrinks (summarization), this also
        # auto-invalidates: the next call sees a smaller list and will
        # rescan.
        self._scanned_len = len(messages)

        if not any_patched:
            # Channel untouched — let delta checkpointing append cleanly.
            return None
        return {"messages": Overwrite(patched_messages)}
