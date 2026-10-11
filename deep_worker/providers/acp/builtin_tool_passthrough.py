"""LangGraph BaseTool objects that record an ACP CLI's BUILT-IN tools into
LangGraph history — the MANDATORY main channel for built-ins (channel B).

ACP CLIs (Grok, Claude Code, …) run their local built-ins (``Bash``/``Shell``,
``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``, ``WebSearch``, …) LOCALLY:
they cannot be told not to, and they do not ask permission for most. So
prevention is impossible — but the ACP ``tool_call`` / ``tool_call_update``
notifications give us the toolCallId, name, args (``rawInput``) and result
(``content`` / metadata), which is everything needed to reconstruct the SAME
``AIMessage(tool_calls=[...])`` + ``ToolMessage`` pair LangGraph emits for a
deep-worker MCP tool.

Mechanism (mirror ``codex_provider/native_tools.py``):
  1. ``AcpChatModel`` binds these tools (one per known built-in + a generic
     wildcard) ONLY when the profile lists built-ins.
  2. ``ainvoke`` surfaces a CLI ``tool_call`` for one of these names as
     ``AIMessage.tool_calls`` and registers an ``asyncio.Future`` on
     ``ThreadState.builtin_passthrough_futures[tool_call_id]``.
  3. LangGraph's ToolNode invokes the tool body, which AWAITS that Future
     (pure event-driven; no polling). The work already happened inside the
     CLI — there is nothing for us to execute.
  4. The reader resolves the Future on ``tool_call_update`` status in
     {completed, failed, cancelled} via :func:`resolve_builtin_future`.
  5. The body returns the rendered text → ``ToolMessage`` carrying
     ``additional_kwargs`` markers so the next ``ainvoke`` knows to SKIP the
     proxy-MCP resolve for this id.

The args schema is OPEN (``extra="allow"``) because we forward the CLI's
``rawInput`` verbatim — the CLI validated it before executing.
"""
from __future__ import annotations

import logging
from typing import Any

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, ConfigDict, Field

from .client_state import get_global_client_state
from .custom_types import CALL_ID_MAP_MAX_ENTRIES

logger = logging.getLogger(__name__)

# additional_kwargs markers the chat_model writes on the resulting
# ToolMessage. ainvoke keys off ``builtin_marker`` to skip
# proxy_mcp.resolve_call. ``meta_key`` carries the CLI metadata.
BUILTIN_MARKER = "acp_builtin"
META_KEY = "acp_builtin_meta"

# The plan recorder. NOT a CLI built-in: no ACP CLI ships a tool by this name.
# ``session/update`` carries ``kind == "plan"`` as its OWN update kind, and the
# shared layer synthesises an ``update_plan`` tool_call from it so the plan
# lands in history (see ``chat_model._plan_to_tool_call``). Because the shared
# layer mints the name for EVERY ACP provider, the tool recording it must be
# bound for every ACP provider too — not declared per-profile in
# ``known_builtins``, which would both be a lie (it is our name, not the CLI's)
# and leave the next provider broken until someone remembered to add it.
#
# A ``plan`` update is a COMPLETED FACT, not the start of work: the CLI sends
# the entries and never follows up with ``tool_call_update status=completed``.
# So this tool must NOT park on a Future the way a real built-in does — nothing
# would ever resolve it and the body would hang forever, which is worse than the
# fabricated error it replaces. ``chat_model._preresolve_plan_result`` stores the
# payload in ``ThreadState.builtin_passthrough_results`` at emission time and the
# body pops it before parking, so the call returns immediately.
PLAN_TOOL_NAME = "update_plan"


class _OpenArgs(BaseModel):
    """An open argument schema — accept and forward any keys the CLI sends in
    ``rawInput`` verbatim. Validation happens on the CLI's side.

    ``tool_call_id`` is declared as an InjectedToolCallId so LangGraph's
    ToolNode fills it from the surfaced tool_call (it is NOT something the
    model supplies). ``extra="allow"`` lets the rest of the rawInput flow
    through untouched."""

    model_config = ConfigDict(extra="allow")

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(default="")


def _thread_id_from_config(config: RunnableConfig | None) -> str | None:
    cfg = (config or {}).get("configurable") or {}
    tid = cfg.get("thread_id")
    return tid if isinstance(tid, str) else None


async def _await_builtin_future(
    config: RunnableConfig | None, tool_call_id: str
) -> dict[str, Any]:
    """Park-and-await the Future the reader registered for ``tool_call_id``.

    The Future lives on ``ThreadState.builtin_passthrough_futures`` for the
    thread named by the runnable config. Pure event-driven wait — resolved by
    :func:`resolve_builtin_future`. Pops the entry on completion so the dict
    cannot grow unbounded.
    """
    thread_id = _thread_id_from_config(config)
    if thread_id is None:
        raise RuntimeError(
            "ACP built-in passthrough invoked without thread_id in runnable "
            "config — cannot locate the parked future."
        )
    state = get_global_client_state().get(thread_id)
    if state is None:
        raise RuntimeError(
            f"No ACP ThreadState for thread_id={thread_id!r}; the built-in "
            f"passthrough body cannot resolve."
        )
    # An already-known result (a plan: a completed fact with no follow-up
    # ``tool_call_update``) is stored as a plain dict, NOT a Future. Creating a
    # Future requires an event loop, and ``_apply_update`` is sync by contract
    # and legitimately runs with no loop; a Future made on one loop would also
    # break when awaited on another. A dict has neither problem and needs no
    # await at all.
    ready = state.builtin_passthrough_results.pop(tool_call_id, None)
    if ready is not None:
        return ready
    fut = state.builtin_passthrough_futures.get(tool_call_id)
    if fut is None:
        raise RuntimeError(
            f"No pending ACP built-in registered for id={tool_call_id!r}. "
            f"Invariant violation: ainvoke must register the Future on the "
            f"tool_call event before LangGraph routes into ToolNode."
        )
    try:
        return await fut
    finally:
        state.builtin_passthrough_futures.pop(tool_call_id, None)


def _tool_call_id_from_input(input: Any) -> str:
    if isinstance(input, dict):
        tcid = input.get("id")
        if isinstance(tcid, str) and tcid:
            return tcid
        args = input.get("args")
        if isinstance(args, dict):
            nested = args.get("tool_call_id")
            if isinstance(nested, str) and nested:
                return nested
    return ""


def _payload_to_tool_message(
    payload: dict[str, Any], *, tool_call_id: str, tool_name: str
) -> ToolMessage:
    text = str(payload.get("text", ""))
    if payload.get("is_error") and text:
        text = f"Error: {text}"
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return ToolMessage(
        content=text,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error" if payload.get("is_error") else "success",
        additional_kwargs={
            BUILTIN_MARKER: True,
            META_KEY: meta,
        },
    )


class _BuiltinPassthroughTool(BaseTool):
    """A LangChain tool that mirrors ONE CLI built-in for history.

    The body does not execute anything — the CLI already ran the tool. It
    only awaits the reader-resolved Future keyed by ``tool_call_id``.
    """

    args_schema: type[BaseModel] = _OpenArgs

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError(
            "ACP built-in passthrough is async-only; use ainvoke()."
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ToolMessage:
        tool_call_id = _tool_call_id_from_input(input)
        if not tool_call_id and isinstance(kwargs.get("tool_call_id"), str):
            tool_call_id = kwargs["tool_call_id"]
        if not tool_call_id:
            raise RuntimeError(
                f"ACP built-in {self.name!r} invoked without a tool_call_id."
            )
        payload = await _await_builtin_future(config, tool_call_id)
        if not isinstance(payload, dict):
            payload = {"text": str(payload)}
        return _payload_to_tool_message(
            payload, tool_call_id=tool_call_id, tool_name=self.name
        )

    async def _arun(
        self,
        config: RunnableConfig = None,  # type: ignore[assignment]
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        tool_call_id = kwargs.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise RuntimeError(
                f"ACP built-in {self.name!r} invoked without a tool_call_id."
            )
        payload = await _await_builtin_future(config, tool_call_id)
        if isinstance(payload, dict):
            return str(payload.get("text", ""))
        return str(payload)


def _make_tool(name: str) -> _BuiltinPassthroughTool:
    return _BuiltinPassthroughTool(
        name=name,
        description=(
            f"CLI built-in {name!r}. Recorded into deep-worker history; "
            f"executed locally by the CLI. Prefer the equivalent deepworker "
            f"MCP tool when one exists."
        ),
    )


def is_builtin_passthrough_tool(tool: Any) -> bool:
    """``True`` iff ``tool`` is one of our channel-B passthrough tools.

    Used by ``AcpChatModel.bind_tools`` to EXCLUDE passthrough tools from the
    metadata published to ``proxy_mcp_server`` (they are bound for LangGraph's
    ToolNode only — exposing them as MCP tools would let the CLI dispatch its
    native built-in through the MCP channel and park a Future on a server
    that will never resolve it).
    """
    return isinstance(tool, _BuiltinPassthroughTool)


def build_passthrough_tools(
    known_builtins: frozenset[str],
    generic_tool_name: str,
) -> tuple[BaseTool, ...]:
    """Build one passthrough tool per known built-in, plus the generic
    wildcard. The MCP dispatch tool is deliberately excluded — it dispatches
    our proxy MCP tools (channel A), never a local built-in."""
    tools: list[BaseTool] = [_make_tool(name) for name in sorted(known_builtins)]
    tools.append(
        _BuiltinPassthroughTool(
            name=generic_tool_name,
            description=(
                "Fallback recorder for a CLI built-in not known at design "
                "time. Records the call into deep-worker history."
            ),
        )
    )
    # The plan recorder, bound for EVERY ACP provider because the name is
    # minted by the shared layer, not by any one CLI (see PLAN_TOOL_NAME).
    # Unconditional on purpose: keying it off a per-profile declaration would
    # leave the next provider fabricating "update_plan is not a valid tool"
    # until someone remembered to declare our own synthetic name.
    tools.append(
        _BuiltinPassthroughTool(
            name=PLAN_TOOL_NAME,
            description=(
                "Recorder for the CLI's plan update. Records the plan into "
                "deep-worker history; the plan itself is already final."
            ),
        )
    )
    return tuple(tools)


def build_passthrough_tools_for_profile(profile: Any) -> tuple[BaseTool, ...]:
    """Passthrough tools for ``profile``, or ``()`` if it declares no built-ins.

    The CAPABILITY-driven entry point: callers pass the profile they already
    hold and get whatever that CLI needs, with no provider name anywhere. A
    provider that runs no local built-ins (``known_builtins`` empty) gets an
    empty tuple and binds nothing.

    BOUNDARY: an empty ``known_builtins`` yields
    ``()`` here — including no plan recorder, because this wrapper returns
    before reaching the unconditional one in :func:`build_passthrough_tools`.
    No such provider exists today, so the gap is unreachable; an ACP CLI that
    emits a plan while declaring no other built-ins would fabricate
    "update_plan is not a valid tool" again and require revisiting this.

    Why this exists. The wiring used to be keyed on
    ``provider == "grok"``, so claude-code — whose profile declares
    ``{Bash, Read, Write, Edit, Glob, Grep, ...}`` and whose chat model
    faithfully surfaces those names as ``AIMessage.tool_calls`` — had NO
    matching ``BaseTool`` bound. LangGraph's ToolNode then took the
    unknown-name branch (``langgraph/prebuilt/tool_node.py:_validate_tool_call``)
    and fabricated ``ToolMessage(status="error", "Error: Bash is not a valid
    tool, try one of [...]")`` WITHOUT executing anything. The CLI had already
    run the command successfully, so the checkpoint recorded a refusal on top
    of real work: 3439 such records in that one agent (2396 of them ``Bash``),
    and the genuine output was never persisted at all. Every downstream reader
    — ``history_*`` tools, summarization, the web UI — saw failure where work
    had succeeded.

    Keying on a capability instead of a name is the same lesson this repo
    already recorded for permission allowlists in
    ``claude_code_provider/profile.py``: "the lesson is not 'keep the list
    fresh' but 'do not keep a list'". A new ACP provider now gets correct
    history by declaring ``known_builtins``, with no edit here.

    ``generic_builtin_name`` falls back to the SAME derivation as
    ``AcpChatModel._generic_builtin_name`` so the recorder the model surfaces
    and the tool bound for it always agree on one name — a mismatch would
    reintroduce the exact fabricated-error defect for unknown built-ins.
    """
    # Duck-typed on purpose: the caller hands us whatever its model carries,
    # which may be a real AcpProfile, langchain's upstream ``ModelProfile``
    # (a plain dict with no ``known_builtins``), or something else entirely.
    # Only genuine strings may become tool names — a non-string would reach
    # BaseTool.name and raise a pydantic ValidationError at bind time, turning
    # a provider that simply has no built-ins into a hard startup failure.
    raw_known = getattr(profile, "known_builtins", None)
    known = frozenset(
        name for name in raw_known if isinstance(name, str) and name
    ) if isinstance(raw_known, (set, frozenset, list, tuple)) else frozenset()
    if not known:
        return ()

    provider = getattr(profile, "provider", "")
    provider = provider if isinstance(provider, str) else ""
    generic = getattr(profile, "generic_builtin_name", "")
    if not isinstance(generic, str) or not generic:
        generic = f"{provider}Builtin".replace("-", "_").replace("_", "")
    return build_passthrough_tools(known, generic)


def resolve_builtin_future(
    client_state: Any,
    agent_thread_id: str,
    tool_call_id: str,
    payload: dict[str, Any],
) -> None:
    """Resolve the parked built-in Future for ``tool_call_id``.

    Called by the ACP reader on ``tool_call_update`` status in
    {completed, failed, cancelled}. ``payload`` is
    ``{"text": str, "is_error"?: bool, "metadata"?: dict, "meta"?: dict}``.
    Idempotent / no-op for an unknown thread or id (e.g. a late update after
    teardown).
    """
    state = client_state.get(agent_thread_id)
    if state is None:
        return
    fut = state.builtin_passthrough_futures.get(tool_call_id)
    if fut is None or fut.done():
        return
    fut.set_result(payload)
