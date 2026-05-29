"""Unit tests for ``codex_provider.native_tools``.

Each native tool is an async BaseTool that:

1. Looks up the live ExecHandle for the current thread_id through
   ``GlobalClientState`` (the handle is stored on
   ``ClientState.current_exec`` by ``CodexChatModel.ainvoke``).
2. Reads the ``tool_call_id`` injected by LangGraph's ToolNode.
3. Pulls the matching ``asyncio.Future`` out of
   ``handle.native_futures``.
4. ``await``s the future. No polling. The pump resolves it when
   Codex emits ``item.completed``.
5. Returns the result string; ToolNode wraps it in a ToolMessage.

These tests construct an in-memory ExecHandle stub with a Future
parked under a known tool_call_id, schedule the tool body, then
resolve the future from a sibling task — proving event-driven
delivery with no busy-wait.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider.client_state import get_global_client_state  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixture — minimal ExecHandle stub the tool body reaches for.
# ---------------------------------------------------------------------------
class _StubHandle:
    def __init__(self) -> None:
        self.native_futures: dict[str, asyncio.Future[str]] = {}


@pytest.fixture(autouse=True)
def isolate_global_state() -> None:
    """Reset per-thread state between tests."""
    get_global_client_state()._states.clear()  # type: ignore[attr-defined]


@pytest.fixture
def thread_id() -> str:
    return "test_thread_native_tools"


@pytest.fixture
async def parked_handle(thread_id: str) -> _StubHandle:
    """Install a stub handle as the live exec for the test thread,
    pre-populated with a parked future for tool_call_id ``"item_0"``."""
    handle = _StubHandle()
    handle.native_futures["item_0"] = asyncio.get_running_loop().create_future()
    state = get_global_client_state().get_or_create(thread_id)
    state.current_exec = handle  # type: ignore[assignment]
    return handle


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _make_tool_call(
    name: str,
    tool_call_id: str,
    **args: Any,
) -> dict[str, Any]:
    """Build the tool_call dict shape LangChain BaseTool.ainvoke
    expects when the tool signature uses InjectedToolCallId."""
    return {
        "args": args,
        "name": name,
        "type": "tool_call",
        "id": tool_call_id,
    }


@pytest.mark.asyncio
async def test_command_execution_returns_on_future_resolution(
    parked_handle: _StubHandle,
    thread_id: str,
) -> None:
    """The tool body parks on the future and returns the value the
    pump (simulated by us here) puts into it. Strictly event-driven."""
    from codex_provider.native_tools import command_execution

    async def _resolver() -> None:
        # Give the tool body a chance to start awaiting
        await asyncio.sleep(0)
        parked_handle.native_futures["item_0"].set_result(
            "file1\nfile2\n"
        )

    # Drive the tool body and the resolver concurrently
    result_task = asyncio.create_task(
        command_execution.ainvoke(
            _make_tool_call("command_execution", "item_0", command="ls"),
            config={"configurable": {"thread_id": thread_id}},
        )
    )
    resolver_task = asyncio.create_task(_resolver())
    result, _ = await asyncio.gather(result_task, resolver_task)

    # When invoked via tool_call form, ainvoke returns a ToolMessage.
    assert hasattr(result, "content")
    assert "file1" in result.content
    assert "file2" in result.content
    # Future was consumed and removed from the map
    assert "item_0" not in parked_handle.native_futures


@pytest.mark.asyncio
async def test_tool_body_propagates_pump_shutdown_exception(
    parked_handle: _StubHandle,
    thread_id: str,
) -> None:
    """If the pump set an exception on the future (e.g. ``codex exec``
    died mid-tool), the tool body must re-raise it. The real ToolNode
    inside LangGraph then converts the exception into a
    ToolMessage(status="error"). Here we test the body's behaviour
    directly (no ToolNode wrapper) — so the exception propagates."""
    from codex_provider.native_tools import command_execution

    async def _fail_future() -> None:
        await asyncio.sleep(0)
        parked_handle.native_futures["item_0"].set_exception(
            RuntimeError("codex exec died mid-tool")
        )

    asyncio.create_task(_fail_future())

    with pytest.raises(RuntimeError, match="codex exec died"):
        await command_execution.ainvoke(
            _make_tool_call("command_execution", "item_0", command="ls"),
            config={"configurable": {"thread_id": thread_id}},
        )


@pytest.mark.asyncio
async def test_apply_patch_tool_body_works_same_way(
    parked_handle: _StubHandle,
    thread_id: str,
) -> None:
    """Every native tool follows the same skeleton; apply_patch's
    body must also park on native_futures and return on resolution."""
    from codex_provider.native_tools import apply_patch

    async def _resolver() -> None:
        await asyncio.sleep(0)
        parked_handle.native_futures["item_0"].set_result("Patched /tmp/x.py")

    result_task = asyncio.create_task(
        apply_patch.ainvoke(
            _make_tool_call("apply_patch", "item_0", patch="*** Begin Patch ***"),
            config={"configurable": {"thread_id": thread_id}},
        )
    )
    resolver_task = asyncio.create_task(_resolver())
    result, _ = await asyncio.gather(result_task, resolver_task)
    assert "Patched" in result.content


@pytest.mark.asyncio
async def test_tool_body_raises_when_future_missing(
    thread_id: str,
) -> None:
    """If LangGraph somehow invokes the tool body for a tool_call_id
    that was never parked (the pump's invariant says this cannot
    happen — Future is pre-created at item.started), the body raises
    so the failure is visible instead of hanging forever."""
    from codex_provider.native_tools import command_execution

    # Install a handle with NO future under that id
    handle = _StubHandle()
    state = get_global_client_state().get_or_create(thread_id)
    state.current_exec = handle  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="No pending native call"):
        await command_execution.ainvoke(
            _make_tool_call("command_execution", "never_parked", command="ls"),
            config={"configurable": {"thread_id": thread_id}},
        )


def test_native_tools_module_exposes_every_native_type() -> None:
    """Each event-pump-recognised native type must have a matching
    BaseTool exported from ``codex_provider.native_tools``. The
    ``NATIVE_TOOLS`` tuple is what ``agent.py`` extends the bound
    tool set with when provider==codex."""
    from codex_provider import native_tools
    from codex_provider.event_pump import NATIVE_TOOL_ITEM_TYPES

    exposed_names = {t.name for t in native_tools.NATIVE_TOOLS}
    # One tool per native item type. The tool name MUST equal the
    # Codex item.type — that's how history shows the real Codex
    # tool names instead of any synthetic prefix.
    assert exposed_names == set(NATIVE_TOOL_ITEM_TYPES), (
        f"native_tools.NATIVE_TOOLS must expose exactly the native "
        f"item types. Missing: {set(NATIVE_TOOL_ITEM_TYPES) - exposed_names}. "
        f"Extra: {exposed_names - set(NATIVE_TOOL_ITEM_TYPES)}."
    )


def test_native_tool_names_are_codex_native_names() -> None:
    """The tool name must literally be the Codex item.type value, so
    LangGraph history reads ``tool_call.name="command_execution"``
    not anything synthetic."""
    from codex_provider.native_tools import command_execution, apply_patch

    assert command_execution.name == "command_execution"
    assert apply_patch.name == "apply_patch"
