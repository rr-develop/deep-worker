"""Unit tests for ``codex_provider.event_pump``.

The pump is the single reader of ``ExecHandle.events`` (the JSONL
async generator). Its job is to route every event to exactly one
destination:

- ``ainvoke_queue`` (asyncio.Queue) — events ainvoke needs to see:
  thread.started, turn.started, item.started (for both proxy MCP
  and native), item.completed (only for AgentMessageItem),
  turn.completed, turn.failed.
- ``native_futures`` (dict[item_id → asyncio.Future]) — pre-created
  on ``item.started`` for native item types, resolved on
  ``item.completed`` with the rendered result string.
- Dropped silently — McpToolCallItem.item.completed events (we
  already have the result via proxy MCP), GenericItems of unknown
  type with no tool semantics (reasoning blocks etc).

All routing is event-driven. There is no polling. The Future is
resolved exactly once. ainvoke_queue is bounded.

These tests are async unit tests against an in-memory async
generator that yields scripted events. No subprocess.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider.exec_runner import (  # noqa: E402
    AgentMessageItem,
    GenericItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    McpToolCallItem,
    ThreadEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
)


# ---------------------------------------------------------------------------
# Test helpers — scripted event source + ExecHandle stub.
# ---------------------------------------------------------------------------
async def _scripted_events(
    events: list[ThreadEvent],
) -> AsyncGenerator[ThreadEvent, None]:
    """Async generator that yields scripted events in order, then ends."""
    for ev in events:
        yield ev


class _StubHandle:
    """Minimal stand-in for ``ExecHandle`` exposing the three attributes
    the pump touches: ``events``, ``ainvoke_queue``, ``native_futures``.
    """

    def __init__(self, events: list[ThreadEvent]) -> None:
        self.events = _scripted_events(events)
        self.ainvoke_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, asyncio.Future[str]] = {}
        # Mirror the real ExecHandle: the pump records a fatal stream
        # error here so a consumer that only sees END_OF_STREAM can
        # re-raise it. None until/unless the pump hits a fatal error.
        self.pump_error: BaseException | None = None


PROXY_SERVER_NAME = "deepworker_99999"


@pytest.mark.asyncio
class TestEventPumpRouting:
    async def test_routes_native_started_to_queue_and_creates_future(
        self,
    ) -> None:
        """``item.started`` for a native item:
        - creates ``native_futures[item.id]`` (a fresh asyncio.Future),
        - also enqueues the event into ``ainvoke_queue`` so ainvoke
          can emit the matching AIMessage.tool_call.

        We snapshot ``native_futures`` BEFORE pump's natural end-of-
        stream cleanup runs (finally would set_exception on every
        unresolved future). To do that, we keep the events generator
        open after the ItemStartedEvent so the pump is still inside
        its ``async for``, then we observe state from outside.
        """
        from codex_provider.event_pump import event_pump

        # Use an async generator that yields the two events and then
        # blocks forever, so the pump remains alive long enough to
        # observe the parked Future un-resolved.
        async def _events_then_hang() -> AsyncGenerator[ThreadEvent, None]:
            yield ThreadStartedEvent(thread_id="uuid-A")
            yield ItemStartedEvent(item=GenericItem(
                id="item_0",
                type="command_execution",
                payload={
                    "command": "/bin/bash -lc 'ls /tmp'",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            ))
            await asyncio.Event().wait()   # block — keep pump alive

        handle = _StubHandle([])
        handle.events = _events_then_hang()

        pump = asyncio.create_task(
            event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)
        )

        # Wait event-driven for the Future to appear (the pump puts
        # the started event into the queue right after creating it).
        # We poll the queue (which IS event-driven via .get()).
        seen_started = False
        try:
            while not seen_started:
                ev = await asyncio.wait_for(
                    handle.ainvoke_queue.get(), timeout=2.0
                )
                if isinstance(ev, ItemStartedEvent):
                    seen_started = True

            # At this point the pump processed ItemStartedEvent and
            # is now hung on the next iteration of the generator. The
            # Future MUST be in the map and MUST NOT be resolved.
            assert "item_0" in handle.native_futures
            fut = handle.native_futures["item_0"]
            assert isinstance(fut, asyncio.Future)
            assert not fut.done(), "future must not be resolved yet"
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await pump

    async def test_routes_native_completed_to_future_not_queue(self) -> None:
        """``item.completed`` for a native item must resolve the parked
        Future and NOT be enqueued into ainvoke_queue."""
        from codex_provider.event_pump import event_pump

        handle = _StubHandle([
            ItemStartedEvent(item=GenericItem(
                id="item_0",
                type="command_execution",
                payload={"command": "ls", "status": "in_progress"},
            )),
            ItemCompletedEvent(item=GenericItem(
                id="item_0",
                type="command_execution",
                payload={
                    "command": "ls",
                    "aggregated_output": "file1\nfile2\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            )),
        ])

        await event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)

        # Future resolved with the output
        fut = handle.native_futures["item_0"]
        assert fut.done()
        result = fut.result()
        assert "file1" in result
        assert "file2" in result

        # ItemCompletedEvent for the native tool is NOT in the queue
        types: list[str] = []
        while not handle.ainvoke_queue.empty():
            ev = handle.ainvoke_queue.get_nowait()
            # The EOS sentinel may also appear; convert to type-name
            if isinstance(ev, ThreadEvent):
                types.append(type(ev).__name__)
        # We saw the started event but NOT the completed one
        assert "ItemStartedEvent" in types
        # No ItemCompletedEvent for a native item should reach ainvoke
        assert all(
            not (t == "ItemCompletedEvent")
            for t in types
        ), f"native ItemCompletedEvent leaked into ainvoke queue: {types}"

    async def test_proxy_completed_swallowed(self) -> None:
        """``item.completed`` for an MCP proxy tool is dropped — neither
        enqueued (ainvoke does not need it; the ToolMessage round trip
        already happened) nor used to resolve any native future."""
        from codex_provider.event_pump import event_pump

        handle = _StubHandle([
            ItemStartedEvent(item=McpToolCallItem(
                id="mcp-0",
                server=PROXY_SERVER_NAME,
                tool="execute",
                arguments={"command": "ls"},
                status="in_progress",
            )),
            ItemCompletedEvent(item=McpToolCallItem(
                id="mcp-0",
                server=PROXY_SERVER_NAME,
                tool="execute",
                arguments={"command": "ls"},
                status="completed",
                result={"content": [{"type": "text", "text": "ok"}]},
            )),
        ])

        await event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)

        # Proxy MCP completion does NOT create native futures
        assert "mcp-0" not in handle.native_futures

        # ItemStartedEvent IS in the queue (ainvoke needs it for
        # wait_for_pending_calls); ItemCompletedEvent must NOT be.
        types: list[str] = []
        while not handle.ainvoke_queue.empty():
            ev = handle.ainvoke_queue.get_nowait()
            if isinstance(ev, ThreadEvent):
                types.append(type(ev).__name__)
        assert types.count("ItemStartedEvent") == 1
        assert types.count("ItemCompletedEvent") == 0

    async def test_agent_message_item_completed_reaches_ainvoke(self) -> None:
        """AgentMessage ItemCompletedEvent MUST reach ainvoke — that
        carries the final assistant text. Only TOOL-shaped item
        completions are filtered."""
        from codex_provider.event_pump import event_pump

        handle = _StubHandle([
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="hi")),
            TurnCompletedEvent(),
        ])

        await event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)

        types: list[str] = []
        while not handle.ainvoke_queue.empty():
            ev = handle.ainvoke_queue.get_nowait()
            if isinstance(ev, ThreadEvent):
                types.append(type(ev).__name__)
        assert "ItemCompletedEvent" in types
        assert "TurnCompletedEvent" in types

    async def test_turn_failed_reaches_ainvoke(self) -> None:
        """TurnFailedEvent must propagate to ainvoke so it can raise
        a meaningful error to the LangGraph layer."""
        from codex_provider.event_pump import event_pump

        handle = _StubHandle([
            TurnFailedEvent(error_message="boom"),
        ])

        await event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)

        types: list[str] = []
        while not handle.ainvoke_queue.empty():
            ev = handle.ainvoke_queue.get_nowait()
            if isinstance(ev, ThreadEvent):
                types.append(type(ev).__name__)
        assert "TurnFailedEvent" in types

    async def test_pump_cancellation_releases_futures(self) -> None:
        """If the pump task is cancelled mid-stream (e.g.
        ``ExecHandle.aclose`` was called), every parked native Future
        must be released with an exception so any awaiting tool body
        wakes up immediately. No polling, no orphan await."""
        from codex_provider.event_pump import event_pump

        # Use a generator that NEVER produces its second event so we
        # can cancel the pump while it's blocked on the next yield.
        async def _hung_events() -> AsyncGenerator[ThreadEvent, None]:
            yield ItemStartedEvent(item=GenericItem(
                id="item_5",
                type="command_execution",
                payload={"command": "sleep 9999", "status": "in_progress"},
            ))
            # Block forever — pretend Codex is mid-tool with no result yet
            await asyncio.Event().wait()
            # unreachable

        handle = _StubHandle([])
        handle.events = _hung_events()

        pump = asyncio.create_task(
            event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)
        )

        # Wait until the pump created the native future
        for _ in range(50):
            if "item_5" in handle.native_futures:
                break
            await asyncio.sleep(0.01)
        assert "item_5" in handle.native_futures
        fut = handle.native_futures["item_5"]
        assert not fut.done()

        # Cancel the pump — its finally block must release the Future
        pump.cancel()
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await pump

        assert fut.done(), "pump shutdown must release every parked future"
        # The future was either cancelled or set with an exception —
        # both forms wake the awaiting tool body immediately.
        if not fut.cancelled():
            with pytest.raises(RuntimeError):
                fut.result()


@pytest.mark.asyncio
class TestNativeResultEncoding:
    """``native_result_text`` produces the string ToolMessage.content
    sees. For command_execution we surface aggregated_output verbatim
    on success; on non-zero exit we prepend an exit_code marker."""

    async def test_command_execution_success_returns_output(self) -> None:
        from codex_provider.event_pump import native_result_text

        it = GenericItem(
            id="item_0",
            type="command_execution",
            payload={
                "command": "ls",
                "aggregated_output": "file1\nfile2\n",
                "exit_code": 0,
                "status": "completed",
            },
        )
        result = native_result_text(it)
        assert "file1" in result
        assert "file2" in result

    async def test_command_execution_failure_includes_exit_code(self) -> None:
        from codex_provider.event_pump import native_result_text

        it = GenericItem(
            id="item_0",
            type="command_execution",
            payload={
                "command": "false",
                "aggregated_output": "",
                "exit_code": 2,
                "status": "completed",
            },
        )
        result = native_result_text(it)
        # Failure marker so the model sees what happened
        assert "exit_code" in result
        assert "2" in result


class TestNativeArgs:
    """``native_args`` projects the Codex payload to model-visible
    arguments only — output fields belong in the ToolMessage, not
    the tool_call."""

    def test_command_execution_args(self) -> None:
        from codex_provider.event_pump import native_args

        it = GenericItem(
            id="item_0",
            type="command_execution",
            payload={
                "command": "/bin/bash -lc 'ls /tmp'",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        )
        args = native_args(it)
        assert args == {"command": "/bin/bash -lc 'ls /tmp'"}
        # Output fields must NOT be in args
        assert "aggregated_output" not in args
        assert "exit_code" not in args
        assert "status" not in args

    def test_web_search_args(self) -> None:
        from codex_provider.event_pump import native_args

        it = GenericItem(
            id="item_0",
            type="web_search",
            payload={"query": "deep-worker langchain", "result": "..."},
        )
        args = native_args(it)
        assert args == {"query": "deep-worker langchain"}

    def test_unknown_type_filters_output_fields(self) -> None:
        from codex_provider.event_pump import native_args

        it = GenericItem(
            id="item_0",
            type="something_new",
            payload={
                "key1": "value1",
                "aggregated_output": "drop me",
                "exit_code": 0,
                "status": "drop me too",
            },
        )
        args = native_args(it)
        assert args == {"key1": "value1"}


# Make sure the suppress-test in TurnStartedEvent path doesn't choke
@pytest.mark.asyncio
async def test_turn_started_reaches_ainvoke() -> None:
    """TurnStartedEvent must pass through to ainvoke (it has no
    payload but ainvoke may track it for observability)."""
    from codex_provider.event_pump import event_pump

    handle = _StubHandle([
        TurnStartedEvent(),
    ])
    await event_pump(handle, proxy_mcp_server_name=PROXY_SERVER_NAME)

    types: list[str] = []
    while not handle.ainvoke_queue.empty():
        ev = handle.ainvoke_queue.get_nowait()
        if isinstance(ev, ThreadEvent):
            types.append(type(ev).__name__)
    assert "TurnStartedEvent" in types
