"""Background event pump for ``codex exec --experimental-json``.

⚠️ ARCHITECTURAL INVARIANT
═══════════════════════════════════════════════════════════════════════════════
The pump is the SINGLE reader of ``ExecHandle.events`` (the JSONL async
generator). No other code path reads from it. Every event is routed to
exactly one destination:

- ``ExecHandle.ainvoke_queue`` (asyncio.Queue) — consumed by
  ``CodexChatModel.ainvoke``. Receives: thread.started, turn.started,
  turn.completed, turn.failed; ``item.started`` for both proxy MCP and
  native tools; ``item.completed`` only for AgentMessageItem.
- ``ExecHandle.native_futures`` (dict[item_id → asyncio.Future]) —
  awaited by the matching async tool body. Created on
  ``item.started`` for native types, resolved on the matching
  ``item.completed`` with the rendered result string.
- Dropped silently — McpToolCallItem.item.completed (we already have
  the ToolMessage via the proxy MCP HTTP path); GenericItems whose
  type carries no tool semantics (reasoning blocks etc).

All routing is event-driven: ``async for`` over the events generator,
``put`` on the queue, ``set_result`` on the Future. No polling, no
sleep, no while-True with timeout.

See ``docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md`` for the full rationale.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Protocol

from .exec_runner import (
    GenericItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadEvent,
)


# Codex JSONL ``item.type`` values that carry tool-call semantics and
# therefore must be recorded into LangGraph as ``AIMessage.tool_calls``
# + ``ToolMessage`` pairs. The Codex docs and live captures on
# codex-cli 0.130.0 confirm these names; new types can be added in a
# single line as Codex evolves.
NATIVE_TOOL_ITEM_TYPES: frozenset[str] = frozenset({
    "command_execution",
    "apply_patch",          # name observed in API tools=[...] list
    "file_change",          # alternative payload shape on completion
    "web_search",
    "tool_search",
    "custom_tool_call",
    "function_call",
    "function_call_output",
})


# Sentinel pushed onto ainvoke_queue when the pump task finishes for
# any reason (natural end-of-stream, cancellation, or an exception
# inside the generator). ainvoke can use ``isinstance(ev, _EOSType)``
# to recognise it and exit its event loop.
class _EOSType:
    """Singleton sentinel marking pump end-of-stream."""


END_OF_STREAM: _EOSType = _EOSType()


class _PumpableHandle(Protocol):
    """Structural type describing the slice of ``ExecHandle`` the pump
    reaches for. Lets unit tests pass an in-memory stub without
    constructing a real subprocess-backed ExecHandle.
    """

    events: Any  # AsyncGenerator[ThreadEvent, None]
    ainvoke_queue: asyncio.Queue[Any]
    native_futures: dict[str, asyncio.Future[str]]
    # Set by ``event_pump`` to the fatal stream exception (if any) so a
    # consumer that only observes END_OF_STREAM can re-raise the cause.
    pump_error: BaseException | None


# ---------------------------------------------------------------------------
# The pump itself
# ---------------------------------------------------------------------------
async def event_pump(
    handle: _PumpableHandle,
    proxy_mcp_server_name: str,
) -> None:
    """Drain ``handle.events`` forever, routing every event.

    ``proxy_mcp_server_name`` is the name of OUR proxy MCP server
    (typically ``deepworker_<pid>``). Events whose ``McpToolCallItem``
    targets a different server are not "ours" — the pump leaves them
    in the ainvoke queue so the existing diagnostic path stays intact.
    """
    try:
        async for ev in handle.events:
            await route_event(handle, ev, proxy_mcp_server_name)
    except asyncio.CancelledError:
        # Normal teardown (aclose cancels the pump). Not an error to
        # surface — let it propagate without recording.
        raise
    except BaseException as exc:  # noqa: BLE001
        # A fatal error from the stream (e.g. CodexProtocolError when
        # the Codex API rejects the request — wrong model, auth, etc.).
        # Record it on the handle so the ainvoke consumer, which only
        # ever sees the END_OF_STREAM sentinel, can re-raise it instead
        # of returning an empty response and stalling the agent
        # (observed 2026-05-20). We still re-raise so the
        # pump Task's ``.exception()`` stays meaningful.
        if getattr(handle, "pump_error", None) is None:
            handle.pump_error = exc
        raise
    finally:
        # Release every still-parked native future so that any tool
        # body awaiting one wakes up immediately. Using
        # ``set_exception`` (not ``cancel``) so the awaiting coroutine
        # raises a meaningful error that LangGraph's ToolNode can
        # turn into a ``ToolMessage(status="error")``.
        for fut in list(handle.native_futures.values()):
            if not fut.done():
                fut.set_exception(
                    RuntimeError(
                        "codex exec exited before item.completed for "
                        "this native call"
                    )
                )

        # Signal end-of-stream to ainvoke. Best-effort put_nowait —
        # if the queue is full ainvoke can already see the back-log
        # and will not be confused by a missing sentinel; the queue
        # bound (64) means we have at most one accidental drop here.
        with contextlib.suppress(asyncio.QueueFull):
            handle.ainvoke_queue.put_nowait(END_OF_STREAM)


async def route_event(
    handle: _PumpableHandle,
    ev: ThreadEvent,
    proxy_mcp_server_name: str,
) -> None:
    """One-shot routing of a single event. See module docstring for the
    destinations table."""
    # 1) Native item.started — pre-create the future, also deliver
    #    the event to ainvoke so it can emit the AIMessage.tool_call.
    if isinstance(ev, ItemStartedEvent) \
            and isinstance(ev.item, GenericItem) \
            and ev.item.type in NATIVE_TOOL_ITEM_TYPES:
        loop = asyncio.get_running_loop()
        # Pre-create BEFORE delivering to ainvoke so the tool body
        # never races and finds an empty map.
        if ev.item.id not in handle.native_futures:
            handle.native_futures[ev.item.id] = loop.create_future()
        await handle.ainvoke_queue.put(ev)
        return

    # 2) Native item.completed — resolve the parked future. ainvoke
    #    does NOT need this event: the tool body will return on its
    #    own and ToolNode will produce the ToolMessage.
    if isinstance(ev, ItemCompletedEvent) \
            and isinstance(ev.item, GenericItem) \
            and ev.item.type in NATIVE_TOOL_ITEM_TYPES:
        fut = handle.native_futures.get(ev.item.id)
        if fut is not None and not fut.done():
            fut.set_result(native_result_text(ev.item))
        return

    # 3) Proxy MCP item.completed — swallow. ainvoke already learned
    #    the result through the proxy's HTTP handler; re-delivering it
    #    here would be a duplicate.
    if isinstance(ev, ItemCompletedEvent) \
            and isinstance(ev.item, McpToolCallItem) \
            and ev.item.server == proxy_mcp_server_name:
        return

    # 4) AgentMessage item.completed — ainvoke uses this for the
    #    final assistant text. Always deliver.
    # 5) Everything else (ThreadStarted, TurnStarted, McpToolCallItem
    #    ItemStarted for our proxy, TurnCompleted, TurnFailed,
    #    GenericItems we don't classify as native, AgentMessageItem
    #    item.completed) — deliver to ainvoke. Existing logic in
    #    ainvoke handles each.
    await handle.ainvoke_queue.put(ev)


# ---------------------------------------------------------------------------
# Helpers — translate a Codex GenericItem to the model-visible tool
# call shape (tool_call.args) and the result string (ToolMessage.content).
# Both are pure functions of the payload, so they live here next to the
# pump and are reused by ``native_tools`` for matching arg-schema work.
# ---------------------------------------------------------------------------
def native_args(it: GenericItem) -> dict[str, Any]:
    """Project the payload to the subset of fields that represent
    model-visible *input* arguments for this tool call.

    Output fields (``aggregated_output``, ``exit_code``, ``status``,
    ``result``, ``error``) belong in the ToolMessage that follows the
    AIMessage.tool_call, not in the tool_call itself. We strip them
    here.
    """
    p = it.payload
    if it.type == "command_execution":
        return {"command": p.get("command", "")}
    if it.type == "web_search":
        return {"query": p.get("query", "")}
    if it.type == "apply_patch":
        return {"patch": p.get("patch", p.get("input", ""))}
    if it.type == "file_change":
        return {k: v for k, v in p.items()
                if k not in ("aggregated_output", "aggregated_summary",
                             "exit_code", "status", "result", "error")}
    # Generic fallback — strip the universal output field set.
    return {k: v for k, v in p.items()
            if k not in ("aggregated_output", "aggregated_summary",
                         "exit_code", "status", "result", "error")}


def native_result_text(it: GenericItem) -> str:
    """Render the result string the ToolMessage will carry.

    Shape mirrors what Codex itself shows in session logs so dwcli
    history reads naturally.
    """
    p = it.payload
    if it.type == "command_execution":
        rc = p.get("exit_code")
        out = p.get("aggregated_output", "")
        if rc == 0 or rc is None:
            return out
        return f"exit_code={rc} status={p.get('status', '')!r}\n{out}"
    if it.type == "file_change":
        summary = p.get("aggregated_summary")
        if summary:
            return str(summary)
        return json.dumps(p, ensure_ascii=False)
    if it.type == "web_search":
        # Codex puts the search results under a few different keys
        # across versions; prefer the most structured.
        for key in ("result", "results", "aggregated_output"):
            if key in p:
                return json.dumps(p[key], ensure_ascii=False) \
                    if not isinstance(p[key], str) else p[key]
        return json.dumps(p, ensure_ascii=False)
    return json.dumps(p, ensure_ascii=False)


