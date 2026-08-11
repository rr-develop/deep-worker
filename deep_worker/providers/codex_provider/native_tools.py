"""LangGraph BaseTool objects that surface Codex's native tools.

When Codex emits ``item.started`` for one of its built-in tools
(``command_execution``, ``apply_patch``, etc.), ``CodexChatModel.ainvoke``
returns an ``AIMessage`` whose ``tool_calls`` reference one of the
tools defined here. LangGraph's ToolNode then invokes the matching
tool body. The body parks on an ``asyncio.Future`` that lives on the
current ``ExecHandle``; the event pump resolves the Future the moment
Codex emits ``item.completed``. No polling — pure event-driven wait.

The result string the body returns becomes ``ToolMessage.content``,
exactly as for any other LangGraph tool. The history a user sees is a
real Codex tool call: ``tool_call.name="command_execution"``,
``tool_call.id`` matches Codex's ``item.id``, the ToolMessage carries
the actual command output — no synthetic prefix, no "fake-tool" marker.

These tools are wired into the bound tool set by
``deep_worker_cli/agent.py`` ONLY when ``config.provider == "codex"``
— they are an internal mechanism of this provider, not a deep-worker
tool concept.

See ``docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md`` for the full design.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from .client_state import get_global_client_state


# ---------------------------------------------------------------------------
# The future-lookup helper — used by every tool body.
# ---------------------------------------------------------------------------
def _native_future_for(
    config: RunnableConfig, tool_call_id: str
) -> asyncio.Future[str]:
    """Find the ``asyncio.Future`` the event pump parked for this tool
    call, raising a descriptive RuntimeError if it is missing.

    The Future lives on the current ``ExecHandle.native_futures``
    dict. We reach it through the per-thread state singleton; the
    LangGraph runnable config carries the ``thread_id`` we use as the
    state key.
    """
    cfg_configurable: dict[str, Any] = config.get("configurable") or {}
    thread_id = cfg_configurable.get("thread_id")
    if not isinstance(thread_id, str):
        raise RuntimeError(
            "Codex native tool body invoked without thread_id in "
            "runnable config — cannot locate the parked future."
        )

    state = get_global_client_state().get(thread_id)
    if state is None or state.current_exec is None:
        raise RuntimeError(
            f"No live Codex ExecHandle for thread_id={thread_id!r}; "
            f"the native tool body cannot resolve."
        )
    handle = state.current_exec
    native_futures_obj = getattr(handle, "native_futures", {})
    native_futures: dict[str, asyncio.Future[str]] = (
        native_futures_obj if isinstance(native_futures_obj, dict) else {}
    )
    fut = native_futures.get(tool_call_id)
    if fut is None:
        raise RuntimeError(
            f"No pending native call registered for id={tool_call_id!r}. "
            f"This is an invariant violation: the event pump must "
            f"create the Future at item.started before LangGraph routes "
            f"into ToolNode."
        )
    return fut


async def _await_and_pop_future(
    config: RunnableConfig, tool_call_id: str
) -> str:
    """Pure event-driven wait. Resolved by the pump's
    ``set_result`` / ``set_exception`` on the matching
    ``item.completed`` event. No sleep, no polling.

    Pops the entry on completion so the dict does not grow unbounded
    across a long session.
    """
    fut = _native_future_for(config, tool_call_id)
    try:
        return await fut
    finally:
        # Remove regardless of success/error; the call is over.
        cfg_configurable: dict[str, Any] = config.get("configurable") or {}
        thread_id = cfg_configurable.get("thread_id")
        if isinstance(thread_id, str):
            state = get_global_client_state().get(thread_id)
            if state is not None and state.current_exec is not None:
                native_futures = getattr(
                    state.current_exec, "native_futures", None
                )
                if isinstance(native_futures, dict):
                    native_futures.pop(tool_call_id, None)


# ---------------------------------------------------------------------------
# One BaseTool per native Codex item type.
#
# Tool NAMES are exactly the Codex ``item.type`` values; that is what
# makes LangGraph history read as a real Codex tool call rather than as
# any kind of synthetic surrogate. Tool signatures roughly mirror the
# ``_native_args`` projection — they exist to populate the JSON schema
# LangGraph uses internally; the arguments are not actually consumed
# by the body (the body only awaits the Future).
# ---------------------------------------------------------------------------
@tool
async def command_execution(
    command: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native shell command executor."""
    _ = command
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def apply_patch(
    patch: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native unified-diff patcher."""
    _ = patch
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def file_change(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native file-change result envelope."""
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def web_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native web search."""
    _ = query
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def tool_search(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native tool-discovery."""
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def custom_tool_call(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native custom-tool wrapper."""
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def function_call(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native function-call envelope."""
    return await _await_and_pop_future(config, tool_call_id)


@tool
async def function_call_output(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> str:
    """Codex native function-call output envelope."""
    return await _await_and_pop_future(config, tool_call_id)


# The bound tool set. Append to deep-worker's `web_tools` when
# `config.provider == "codex"`. Order mirrors NATIVE_TOOL_ITEM_TYPES
# in event_pump.py.
NATIVE_TOOLS: tuple[BaseTool, ...] = (
    command_execution,
    apply_patch,
    file_change,
    web_search,
    tool_search,
    custom_tool_call,
    function_call,
    function_call_output,
)
