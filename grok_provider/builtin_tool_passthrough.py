"""LangGraph BaseTool objects that record grok's BUILT-IN tools into
LangGraph history — the MANDATORY main channel for built-ins (channel B).

Grok 0.2.54 in ``grok agent stdio`` runs its built-ins (``Shell``,
``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``, ``WebSearch``,
``Task``, ``Plan``, ...) LOCALLY: it cannot be told not to, and it does
not ask permission (Q1 live-check 2026-06-17, see
docs/RESOLUTION_Q1_DISALLOWED_TOOLS.md). So prevention is impossible — but
the ACP ``tool_call`` / ``tool_call_update`` notifications give us the
toolCallId, name, args (``rawInput``) and result (``content`` / metadata),
which is everything needed to reconstruct the SAME
``AIMessage(tool_calls=[...])`` + ``ToolMessage`` pair LangGraph emits for
a deep-worker MCP tool.

Mechanism (mirror ``codex_provider/native_tools.py``):
  1. ``GrokChatModel`` binds these tools (one per known built-in + a
     generic wildcard) ONLY when ``provider == "grok"``.
  2. ``ainvoke`` surfaces a grok ``tool_call`` for one of these names as
     ``AIMessage.tool_calls`` and registers an ``asyncio.Future`` on
     ``ThreadState.builtin_passthrough_futures[tool_call_id]``.
  3. LangGraph's ToolNode invokes the tool body, which AWAITS that Future
     (pure event-driven; no polling). The work already happened inside
     grok — there is nothing for us to execute.
  4. The reader resolves the Future on ``tool_call_update`` status in
     {completed, failed, cancelled} via :func:`resolve_builtin_future`.
  5. The body returns the rendered text → ``ToolMessage`` carrying
     ``additional_kwargs={"grok_builtin": True, "grok_meta": {...}}`` so
     the next ``ainvoke`` knows to SKIP the proxy-MCP resolve for this id.

The args schema is OPEN (``extra="allow"``) because we forward grok's
``rawInput`` verbatim — grok validated it before executing. See DESIGN
§11A.4-§11A.6.
"""
from __future__ import annotations

import logging
from typing import Any

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, ConfigDict, Field

from grok_provider.acp_types import KNOWN_GROK_BUILTINS
from grok_provider.client_state import get_global_client_state

logger = logging.getLogger(__name__)

_GENERIC_TOOL_NAME = "GrokBuiltin"

# additional_kwargs marker the chat_model writes on the resulting
# ToolMessage. ainvoke keys off it to skip proxy_mcp.resolve_call.
GROK_BUILTIN_MARKER = "grok_builtin"
GROK_META_KEY = "grok_meta"


class _OpenArgs(BaseModel):
    """An open argument schema — accept and forward any keys grok sends in
    ``rawInput`` verbatim. Validation happens on grok's side.

    ``tool_call_id`` is declared as an InjectedToolCallId so LangGraph's
    ToolNode fills it from the surfaced tool_call (it is NOT something the
    model supplies). ``extra="allow"`` lets the rest of grok's rawInput
    flow through untouched."""

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
    thread named by the runnable config. Pure event-driven wait — resolved
    by :func:`resolve_builtin_future`. Pops the entry on completion so the
    dict cannot grow unbounded.
    """
    thread_id = _thread_id_from_config(config)
    if thread_id is None:
        raise RuntimeError(
            "grok built-in passthrough invoked without thread_id in runnable "
            "config — cannot locate the parked future."
        )
    state = get_global_client_state().get(thread_id)
    if state is None:
        raise RuntimeError(
            f"No grok ThreadState for thread_id={thread_id!r}; the built-in "
            f"passthrough body cannot resolve."
        )
    fut = state.builtin_passthrough_futures.get(tool_call_id)
    if fut is None:
        raise RuntimeError(
            f"No pending grok built-in registered for id={tool_call_id!r}. "
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
    grok_meta = payload.get("grok_meta")
    if not isinstance(grok_meta, dict):
        grok_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return ToolMessage(
        content=text,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error" if payload.get("is_error") else "success",
        additional_kwargs={
            GROK_BUILTIN_MARKER: True,
            GROK_META_KEY: grok_meta,
        },
    )


class _BuiltinPassthroughTool(BaseTool):
    """A LangChain tool that mirrors ONE grok built-in for history.

    The body does not execute anything — grok already ran the tool. It
    only awaits the reader-resolved Future keyed by ``tool_call_id``.
    """

    args_schema: type[BaseModel] = _OpenArgs

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError(
            "grok built-in passthrough is async-only; use ainvoke()."
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
                f"grok built-in {self.name!r} invoked without a tool_call_id."
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
                f"grok built-in {self.name!r} invoked without a tool_call_id."
            )
        payload = await _await_builtin_future(config, tool_call_id)
        if isinstance(payload, dict):
            return str(payload.get("text", ""))
        return str(payload)


def _make_tool(name: str) -> _BuiltinPassthroughTool:
    return _BuiltinPassthroughTool(
        name=name,
        description=(
            f"Grok built-in {name!r}. Recorded into deep-worker history; "
            f"executed locally by grok. Prefer the equivalent deepworker MCP "
            f"tool when one exists."
        ),
    )


# A single shared generic wildcard tool instance for unknown built-ins.
_GENERIC_TOOL = _BuiltinPassthroughTool(
    name=_GENERIC_TOOL_NAME,
    description=(
        "Fallback recorder for a grok built-in not known at design time. "
        "Records the call into deep-worker history."
    ),
)


def get_generic_passthrough_tool() -> _BuiltinPassthroughTool:
    """Return the wildcard passthrough tool for unknown grok built-ins."""
    return _GENERIC_TOOL


def is_grok_builtin_passthrough_tool(tool: Any) -> bool:
    """``True`` iff ``tool`` is one of our channel-B passthrough tools.

    Used by ``GrokChatModel.bind_tools`` to EXCLUDE passthrough tools from
    the metadata published to ``proxy_mcp_server`` (they are bound for
    LangGraph's ToolNode only — exposing them as MCP tools would let grok
    dispatch its native built-in through CallMcpTool and park a Future on
    a server that will never resolve it).
    """
    return isinstance(tool, _BuiltinPassthroughTool)


def build_passthrough_tools() -> tuple[BaseTool, ...]:
    """Build one passthrough tool per known grok built-in, plus the generic
    wildcard. CallMcpTool is deliberately excluded — it dispatches our proxy
    MCP tools (channel A), never a local built-in."""
    tools: list[BaseTool] = [_make_tool(name) for name in sorted(KNOWN_GROK_BUILTINS)]
    tools.append(_GENERIC_TOOL)
    return tuple(tools)


# ---------------------------------------------------------------------------
# Reader-side resolution helper.
# ---------------------------------------------------------------------------
def resolve_builtin_future(
    client_state: Any,
    agent_thread_id: str,
    tool_call_id: str,
    payload: dict[str, Any],
) -> None:
    """Resolve the parked built-in Future for ``tool_call_id``.

    Called by the ACP reader on ``tool_call_update`` status in
    {completed, failed, cancelled}. ``payload`` is
    ``{"text": str, "is_error"?: bool, "metadata"?: dict, "grok_meta"?: dict}``. Idempotent /
    no-op for an unknown thread or id (e.g. a late update after teardown).
    """
    state = client_state.get(agent_thread_id)
    if state is None:
        return
    fut = state.builtin_passthrough_futures.get(tool_call_id)
    if fut is None or fut.done():
        return
    fut.set_result(payload)
