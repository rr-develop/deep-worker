"""Enforce a tool selection at the boundary the model reads tools from.

Tools reach the model from two directions: the list assembled in
``create_cli_agent``, and the middleware that contribute their own
(``ProjectMemoryMiddleware``, ``SubAgentMiddleware``,
``FilesystemMiddleware``, the history middlewares, …). Filtering the
assembled list alone would leave the middleware tools in place and report
"all tools off" while the agent could still call half of them.

``wrap_model_call`` sees the union, after every middleware has had its
say, so that is where the selection is applied. ``FilesystemMiddleware``
already drops ``execute`` this way when the backend cannot run it, so the
mechanism is the established one rather than a new invention.

Ordering matters: this middleware must run LAST so that tools added by
middleware appended after it are still subject to the filter.

Withholding a schema is NOT enforcement. ``ToolNode`` resolves a call
against its OWN registry — the full tool list handed to the graph — not
against the filtered ``ModelRequest``, so a model that names a tool from
memory (or from a hint that survived elsewhere in the prompt) still
reaches the real implementation. Observed live on 2026-08-30: an agent
launched with ``--tools '*,project_memory'`` logged "1 of 23 tools
offered to the model" and then successfully ran ``ls`` twice. The
selection therefore has to be applied a SECOND time, at dispatch, where
``wrap_tool_call`` sees the call that ToolNode is about to execute.

Running last also makes the prompt honest. ``wrap_model_call`` wrappers
nest outer-to-inner, so by the time this one runs every other middleware
has already appended its own advertisement — "You have access to an
``execute`` tool", "This project has a semantic search index", and so on
— to ``request.system_message``. Removing a tool's schema while leaving
that text in place would tell the model it has tools it cannot call,
which corrupts the very measurement the feature exists to support. Since
each middleware appends its text as a SEPARATE content block (see
``append_to_system_message``), the blocks belonging to filtered-out tools
can be identified and dropped here, at the one point that sees the whole
assembled prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ToolCallRequest,
)
from deepagents.middleware._utils import append_to_system_message

from deep_worker_cli.tool_selection import (
    ToolSelection,
    describe_selection,
)

logger = logging.getLogger(__name__)


def _refusal_text(name: str) -> str:
    """What the model is told when it names a tool outside the selection.

    Both refusal paths — the pre-dispatch cut and the dispatch barrier —
    report the same thing, so a denied call reads identically whether it
    was removed before ``ToolNode`` saw it or refused on the way in.
    """
    return (
        f"Tool '{name}' is disabled for this agent by its tool "
        f"selection and was NOT executed. Do not retry it and do "
        f"not look for another route to the same capability — "
        f"the restriction is deliberate. Continue with the tools "
        f"you do have, or tell the user what you cannot do."
    )


# Prompt sections contributed by tool-owning middleware, keyed by a stable
# marker from the section's own heading and mapped to the tools the section
# advertises. A section is dropped when NONE of its tools survive the
# selection; it is kept verbatim when at least one does, because the
# surviving tool still needs its instructions.
#
# Matching is by heading text rather than by asking each middleware,
# because the middleware do not expose which block they appended — the
# blocks are anonymous by the time they reach this wrapper. A heading that
# drifts makes a section un-droppable (the hint survives, the schema is
# still gone), which is why ``test_tool_selection_middleware`` pins every
# marker below against the real constants it comes from.
_PROMPT_SECTION_OWNERS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "## Filesystem Tools",
        frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"}),
    ),
    ("## Execute Tool", frozenset({"execute"})),
    ("## `task` (subagent spawner)", frozenset({"task"})),
    # Contributed by upstream ``TodoListMiddleware``. Without an owner here
    # the section is "unowned text" and survives the filter: an agent whose
    # selection drops ``write_todos`` still reads "You have access to the
    # write_todos tool", calls it, is refused, and reads the same
    # instruction again on the next iteration — a loop that ends only at
    # the per-turn cap. Observed live 2026-09-04 (agent ``test_lsp``,
    # ``--tools 'none,+serena_*'``, MaxIterationsExceeded).
    ("## `write_todos`", frozenset({"write_todos"})),
    ("### Project Knowledge Memory", frozenset({"project_memory"})),
    ("### Conversation History Memory", frozenset({"history_memory"})),
    # Skills are read THROUGH the filesystem tools; with those gone the
    # skills catalogue is unreachable and advertising it is a lie.
    ("## Skills System", frozenset({"read_file"})),
)


class ToolSelectionMiddleware(AgentMiddleware):
    """Remove every tool the selection does not permit.

    The empty selection is not merely a no-op by arithmetic — the
    middleware is not installed at all in that case (see
    ``create_cli_agent``), so the default path keeps its exact previous
    shape.
    """

    tools: list = []

    # Read by ``create_deep_agent`` to decide which middleware follow the
    # agent into its subagents. A subagent is a separate graph with its own
    # FilesystemMiddleware, so without this the delegation path hands back
    # every tool the selection removed — verified: the denied tool's body
    # ran inside the subagent while the main graph refused the same call.
    #
    # Declared as a plain attribute rather than resolved by an import so
    # ``deepagents`` (which sits below this package) does not have to know
    # this class exists.
    applies_to_subagents: bool = True

    def __init__(
        self, selection: ToolSelection, *, native_caveat: str | None = None
    ) -> None:
        super().__init__()
        self._selection = selection
        self._native_caveat = native_caveat
        self._logged = False

    @staticmethod
    def _kept_names(kept: list[Any]) -> set[str]:
        return {
            str(getattr(tool, "name", ""))
            for tool in kept
            if getattr(tool, "name", "")
        }

    def _strip_dead_sections(
        self, system_message: SystemMessage | None, kept_names: set[str]
    ) -> SystemMessage | None:
        """Drop prompt sections whose tools the selection removed.

        Returns the message unchanged when nothing needs dropping, so an
        unaffected prompt is not rebuilt (and not subtly reshaped) for
        nothing.
        """
        if system_message is None:
            return None

        blocks = list(system_message.content_blocks)
        surviving: list[Any] = []
        dropped = 0
        for block in blocks:
            text = block.get("text", "") if isinstance(block, dict) else ""
            owners = self._section_owners(text)
            if owners is not None and not (owners & kept_names):
                dropped += 1
                continue
            surviving.append(block)

        if not dropped:
            return system_message
        return SystemMessage(content=surviving)

    @staticmethod
    def _section_owners(text: str) -> frozenset[str] | None:
        """Return the tools a prompt block advertises, or None if unowned.

        Unowned text (the agent's own instructions, the user's prompt
        additions) is never touched — only sections a tool-owning
        middleware contributed are candidates for removal.
        """
        stripped = text.lstrip()
        for marker, owners in _PROMPT_SECTION_OWNERS:
            if stripped.startswith(marker):
                return owners
        return None

    def _instruction(self, kept: list[Any]) -> str:
        names = sorted(self._kept_names(kept))
        enabled = ", ".join(names) if names else "none"
        lines = [
            "### Active Tool Selection",
            "",
            f"Enabled deep-worker tools: {enabled}.",
            "Do not use or claim access to any deep-worker tool not listed above.",
        ]
        if self._native_caveat:
            lines += [
                "",
                self._native_caveat,
                "For provider-native tools this is a preference, not enforcement:",
                "do not use native tools outside the selection.",
            ]
        return "\n".join(lines)

    def _filter(self, request: ModelRequest) -> ModelRequest:
        kept = self._selection.apply(request.tools)
        if not self._logged:
            self._logged = True
            logger.info(
                "tool selection (%s): %d of %d tools offered to the model",
                describe_selection(self._selection),
                len(kept),
                len(request.tools),
            )
        pruned = self._strip_dead_sections(
            request.system_message, self._kept_names(kept)
        )
        return request.override(
            tools=kept,
            system_message=append_to_system_message(
                pruned, self._instruction(kept)
            ),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        return await handler(self._filter(request))

    # ---- pre-dispatch cut ------------------------------------------------
    #
    # ``HumanInTheLoopMiddleware`` does its work in an ``after_model`` GRAPH
    # NODE, not in ``wrap_tool_call``. That node runs as soon as the model
    # returns — strictly before the tool-call wrapper chain the barrier
    # below lives in — so no amount of middleware reordering can put the
    # barrier in front of it. Observed: a denied ``ls`` raised an approval
    # interrupt asking the user to authorise a tool the agent is not
    # allowed to have; only after approval would the barrier refuse it.
    #
    # Asking permission for something already forbidden is not a safety
    # prompt, it is a nonsensical question — and a user who approves is
    # entitled to expect the call to happen. So denied calls are stripped
    # from the AIMessage here, in our own ``after_model``, and answered
    # with the same refusal the barrier would have produced. HITL then
    # never sees them.

    def _prune_tool_calls(self, state: Any) -> dict[str, Any] | None:
        messages = (state or {}).get("messages") or []
        if not messages:
            return None
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        kept, refusals = [], []
        for call in last_ai.tool_calls:
            name = call.get("name") or ""
            if not name or self._selection.permits(str(name)):
                kept.append(call)
                continue
            logger.warning(
                "tool selection (%s): removed disallowed tool call %r "
                "before dispatch",
                describe_selection(self._selection),
                name,
            )
            refusals.append(
                ToolMessage(
                    tool_call_id=call.get("id") or "unknown",
                    name=str(name),
                    status="error",
                    content=_refusal_text(str(name)),
                )
            )

        if not refusals:
            return None
        # Rewriting the AIMessage in place (same id) replaces it in the
        # checkpoint rather than appending a second copy.
        revised = last_ai.model_copy(update={"tool_calls": kept})
        return {"messages": [revised, *refusals]}

    def after_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        del runtime
        return self._prune_tool_calls(state)

    async def aafter_model(
        self, state: Any, runtime: Any = None
    ) -> dict[str, Any] | None:
        del runtime
        return self._prune_tool_calls(state)

    # ---- dispatch barrier ------------------------------------------------
    #
    # The model can still NAME a tool whose schema it was never offered:
    # from its own memory, from a prompt hint we failed to strip, or from
    # a native built-in that shares our tool's name. ToolNode would then
    # run the real implementation, because it resolves names against the
    # graph's full registry rather than against the filtered request. The
    # refusal below is what makes the selection a limit instead of a hint.

    @staticmethod
    def _refusal(request: ToolCallRequest, name: str) -> ToolMessage:
        call = getattr(request, "tool_call", None) or {}
        return ToolMessage(
            tool_call_id=call.get("id") or "unknown",
            name=name,
            status="error",
            content=_refusal_text(name),
        )

    def _blocked(self, request: ToolCallRequest) -> ToolMessage | None:
        """The refusal to return instead of dispatching, or None to allow."""
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name") or getattr(
            getattr(request, "tool", None), "name", ""
        )
        # A nameless call cannot be matched against the selection. Let it
        # through rather than guess: ToolNode rejects it on its own, and
        # inventing a refusal here would misreport WHY it failed.
        if not name or self._selection.permits(str(name)):
            return None
        logger.warning(
            "tool selection (%s): refused disallowed tool call %r",
            describe_selection(self._selection),
            name,
        )
        return self._refusal(request, str(name))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        return self._blocked(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        blocked = self._blocked(request)
        if blocked is not None:
            return blocked
        return await handler(request)
