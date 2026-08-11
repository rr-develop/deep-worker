"""Integration: a tool call orphaned by a restart must not deadlock the
resumed turn — reproduced with a REAL ProxyMCPServer.

This drives the actual deadlock mechanism end to end through the real
``ChatModel.ainvoke`` and a real ``ProxyMCPServer`` (no mock of the proxy
or the wait logic). Only the CLI message stream is supplied by the test
(the standard technique in ``test_event_based_tool_detection.py``), which
is exactly what the real CLI does on resume: it REPLAYS the dangling
``ToolUseBlock`` from the persisted transcript.

The bug (live mj1_init, 2026-06-15): on resume the replayed
``ToolUseBlock`` carries the orphaned tool_use_id, ``ainvoke`` captures it
and ``await proxy.wait_for_pending_call_by_id(that_id)`` — with NO timeout
— and the matching control-channel call never arrives under that id, so
the turn parks forever.

``test_orphaned_toolcall_deadlock_is_reconciled`` proves the FIX: the
orphaned id is detected from history and marked on the proxy, so the
per-id wait returns immediately and the turn completes.

``test_without_reconcile_the_same_setup_would_deadlock`` proves the test
genuinely exercises the deadlock: the identical setup WITHOUT marking the
orphan parks on ``wait_for_pending_call_by_id`` (caught by a wait_for in
the test — that timeout is the TEST's tripwire, not production behaviour).

See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
from claude_agent_sdk import ResultMessage, TextBlock, ToolUseBlock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata
from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer
from deep_worker.providers.claude_code_provider.session_state import save_session_state

pytestmark = pytest.mark.integration


_ORPHAN_ID = "toolu_ORPHAN_FROM_PRIOR_RUN"
_TOOL_NAME = "history_memory"

_TOOLS_META: list[ToolMetadata] = [
    {
        "name": _TOOL_NAME,
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }
]

# Restart history: the LAST turn requested ``history_memory`` (id
# _ORPHAN_ID) but there is NO ToolMessage for it — orphaned by the kill.
_RESTART_HISTORY = [
    HumanMessage(content="turn 1", id="h1"),
    AIMessage(content="", id="a1",
              tool_calls=[{"id": "c_done", "name": "read_file", "args": {}}]),
    ToolMessage(content="ok", tool_call_id="c_done", id="t1"),
    HumanMessage(content="do the long thing", id="h2"),
    AIMessage(content="", id="a2",
              tool_calls=[{"id": _ORPHAN_ID, "name": _TOOL_NAME, "args": {}}]),
    # process killed here — no ToolMessage for _ORPHAN_ID
    HumanMessage(content="are you back?", id="h3"),
]


def _client_replaying_orphan() -> Mock:
    """A CLI client whose resumed stream REPLAYS the orphaned ToolUseBlock
    (same id as the dangling tool_use in history) — exactly what the real
    CLI does on ``--resume``."""
    c = Mock()
    c.query = AsyncMock()

    async def _recv():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="resuming")],
            model="claude-opus-4",
        )
        # The replayed dangling tool_use from the persisted transcript.
        yield ClaudeAssistantMessage(
            content=[ToolUseBlock(id=_ORPHAN_ID, name=_TOOL_NAME, input={})],
            model="claude-opus-4",
        )
        yield ResultMessage(
            subtype="result", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="s",
            total_cost_usd=0.0, usage={"input_tokens": 1, "output_tokens": 1},
        )

    c.receive_response = _recv
    return c


@pytest.mark.asyncio
async def test_orphaned_toolcall_deadlock_is_reconciled() -> None:
    """With the fix: the resumed turn does NOT deadlock on the orphaned
    tool call — it completes within a tight bound."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    # Expose ``history_memory`` as a proxy tool so the replayed
    # ToolUseBlock is recognized as a proxy tool — that is what makes
    # ainvoke break the stream and reach wait_for_pending_call_by_id.
    model._tools_metadata = list(_TOOLS_META)  # type: ignore[attr-defined]
    tid = "orphan_deadlock_agent"
    save_session_state(tid, sent_count=5, session_uuid="prior-uuid")

    real_proxy = ProxyMCPServer(_TOOLS_META)
    client = _client_replaying_orphan()

    async def fake_get(**kwargs: Any) -> tuple[Mock, ProxyMCPServer]:
        return (client, real_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="resumed-uuid",
        ):
            # Must NOT hang. A deadlock blows past this bound.
            result = await asyncio.wait_for(
                model.ainvoke(
                    _RESTART_HISTORY, config={"configurable": {"thread_id": tid}}
                ),
                timeout=10.0,
            )

    assert result is not None
    # Reaching here at all is the assertion: the resumed turn completed
    # instead of parking on wait_for_pending_call_by_id for the orphan.


@pytest.mark.asyncio
async def test_without_reconcile_the_setup_is_bounded_not_deadlocked() -> None:
    """Control, REWRITTEN 2026-06-17 (variant A): even with orphan-marking
    disabled — so the resumed turn reaches ``wait_for_pending_call_by_id`` for
    an id whose control request never arrives — the turn no longer deadlocks.
    Stage A is bounded by the SINGLE source of truth, so it degrades to a
    NON-error in_progress notice instead of hanging forever (the old mj1_init
    deadlock is now structurally impossible). Proven by mocking ONLY the clock.

    (Previously this test asserted the setup WOULD deadlock — that baseline is
    obsolete now that no tool wait is unbounded. See
    docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.)"""
    from langchain_core.messages import AIMessage

    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    model._tools_metadata = list(_TOOLS_META)  # type: ignore[attr-defined]
    tid = "orphan_deadlock_control"
    save_session_state(tid, sent_count=5, session_uuid="prior-uuid")

    real_proxy = ProxyMCPServer(_TOOLS_META)
    client = _client_replaying_orphan()

    async def fake_get(**kwargs: Any) -> tuple[Mock, ProxyMCPServer]:
        return (client, real_proxy)

    def _noop(_ids: Any) -> None:
        return None

    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]

    async def _advance_clock() -> None:
        # Once ainvoke has parked in the bounded stage-A wait, push the clock
        # past the single-source bound so the SAME timer fires instantly.
        from deepagents.middleware.bounded_tool_call import resolve_tool_wait_seconds
        await asyncio.sleep(0.05)
        skew["v"] = resolve_tool_wait_seconds() + 5.0

    try:
        with patch.object(real_proxy, "mark_orphaned_tool_calls", _noop):
            with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
                with patch(
                    "claude_code_provider.session_restore.resolve_restore_session",
                    return_value="resumed-uuid",
                ):
                    advancer = asyncio.ensure_future(_advance_clock())
                    # Bounded — completes (does NOT hang). Outer cap is a safety
                    # net far above the in-test clock jump.
                    result = await asyncio.wait_for(
                        model.ainvoke(
                            _RESTART_HISTORY,
                            config={"configurable": {"thread_id": tid}},
                        ),
                        timeout=10.0,
                    )
                    await advancer
    finally:
        loop.time = real_time  # type: ignore[method-assign]

    # No deadlock: a real AIMessage came back, never an error.
    assert isinstance(result, AIMessage)
