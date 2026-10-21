"""TDD for the mid-turn pump (fix #2 2026-06-18 — test_grok Shell hang).

When ``_consume_turn`` surfaces an ``AIMessage(tool_calls=[...])`` for a
channel-B built-in, LangGraph's ToolNode invokes the passthrough body which
parks on ``builtin_passthrough_futures[tcid]``. Until fix #2 NOTHING drained
``current_notif_queue`` between that pause and the next ``ainvoke``, so the
reader's ``tool_call_update status=completed`` event was never observed, the
Future never resolved, and the passthrough body hung forever.

The pump is a background asyncio.Task spawned at the mid-turn pause that keeps
draining the same queue: it resolves built-in Futures, accumulates text, and
queues fresh tool_calls / the terminal response for the next ``ainvoke``.

These tests reuse the hermetic fakes from ``test_chat_model.py`` (no real
grok binary, no sockets).
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.acp import chat_model as acp_cm  # noqa: E402
from deep_worker.providers.grok_provider.chat_model import GrokChatModel  # noqa: E402
# The ToolMessage marker: ``GROK_BUILTIN_MARKER`` is still exported by the
# grok module (it is an alias of the shared ``BUILTIN_MARKER`` = "acp_builtin"
# since 3fe834d1), but the production stamper and the production reader both
# use the shared constant — import it from there so a divergence between the
# two spellings fails this test instead of hiding behind the alias.
from deep_worker.providers.acp.builtin_tool_passthrough import (  # noqa: E402
    BUILTIN_MARKER,
)
# The shared fakes live in the sibling test module. This directory has no
# __init__.py (pytest rootdir-relative collection), and the old
# ``deep_worker.providers.grok_provider.tests`` package no longer exists —
# the grok tests moved to the repo-root ``tests/`` tree. Import the sibling
# by adding THIS directory to sys.path.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from test_chat_model import (  # noqa: E402
    _FakeAcpClient,
    _FakeProxy,
    install_client_state,
)


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def model(fake_home, monkeypatch):
    cs = install_client_state(monkeypatch)
    m = GrokChatModel(model="grok-4")
    m._client_state = cs
    m._acp_client = _FakeAcpClient()
    m._proxy_mcp = _FakeProxy()
    m._proxy_started = True
    m._acp_started = True
    m._working_dir = "/tmp"
    return m


def _cfg(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


async def _start_builtin_turn(model, *, command="ls -la", tcid="tc-builtin"):
    """Drive a fresh turn until it pauses on a built-in tool_call. Returns
    (acp, sid, first_ai_message)."""
    acp = model._acp_client
    task = asyncio.create_task(
        model.ainvoke([HumanMessage(content="run shell")], config=_cfg("agentA"))
    )
    await asyncio.sleep(0.01)
    sid = f"sess-{acp.session_counter}"
    acp.push_update(sid, {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": "Shell",
        "rawInput": {"command": command},
    })
    msg: AIMessage = await asyncio.wait_for(task, timeout=1)
    return acp, sid, msg


class TestPumpResolvesBuiltinFuture:
    async def test_pump_resolves_passthrough_future_after_pause(self, model):
        """REGRESSION: the built-in ``tool_call_update status=completed`` is
        pushed AFTER ainvoke returned (while ToolNode runs the passthrough
        body). The pump must drain it and resolve the Future — no hang."""
        acp, sid, msg = await _start_builtin_turn(model)
        assert msg.tool_calls and msg.tool_calls[0]["name"] == "Shell"

        st = model._client_state.get("agentA")
        # The pump must be running now (mid-turn pause spawned it).
        assert st.pump_task is not None and not st.pump_task.done()
        fut = st.builtin_passthrough_futures["tc-builtin"]
        assert not fut.done()

        # The reader pushes the terminal update LATE — only the pump can see it.
        acp.push_update(sid, {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-builtin",
            "status": "completed",
            "content": [{"content": {"type": "text", "text": "total 0\n"}}],
            "rawOutput": {"exit_code": 0},
        })
        # The pump resolves the Future without any further ainvoke.
        result = await asyncio.wait_for(fut, timeout=1)
        assert "total 0" in result["text"]
        assert result["meta"]["exit_code"] == 0


class TestNextAinvokeAdoptsPump:
    async def test_next_ainvoke_picks_up_queued_text_and_finishes(self, model):
        """After the built-in resolves, grok streams a bit more text and ends
        the turn while ToolNode is still finishing. The next ainvoke must
        adopt the pump, inherit the queued text/response, and return the final
        AIMessage."""
        acp, sid, _msg = await _start_builtin_turn(model)
        st = model._client_state.get("agentA")

        # Built-in completes; grok then streams trailing text and ends turn —
        # all observed by the PUMP (no foreground consumer right now).
        acp.push_update(sid, {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-builtin",
            "status": "completed",
            "content": [{"content": {"type": "text", "text": "ok"}}],
            "rawOutput": {"exit_code": 0},
        })
        await asyncio.sleep(0.01)
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "done running"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.sleep(0.01)
        # Pump should have stashed the response.
        assert st.queued_response is not None
        assert "done running" in "".join(st.queued_text)

        # Next ainvoke: the ToolMessage for the built-in arrives, summarization
        # not triggered, mid-turn resume adopts the pump and finalizes.
        messages = [
            HumanMessage(content="run shell"),
            AIMessage(content="", tool_calls=[
                {"id": "tc-builtin", "name": "Shell",
                 "args": {"command": "ls -la"}, "type": "tool_call"}]),
            ToolMessage(content="ok", tool_call_id="tc-builtin",
                        additional_kwargs={BUILTIN_MARKER: True}),
        ]
        final: AIMessage = await asyncio.wait_for(
            model.ainvoke(messages, config=_cfg("agentA")), timeout=1
        )
        assert "done running" in final.content
        # Pump and the in-flight future are fully cleared.
        assert st.pump_task is None
        assert st.pending_session_prompt_future is None

    async def test_next_ainvoke_surfaces_pump_queued_tool_calls(self, model):
        """If grok emits a FRESH tool_call during the pump window, the next
        ainvoke surfaces it as part of its returned AIMessage.tool_calls."""
        acp, sid, _msg = await _start_builtin_turn(model)
        st = model._client_state.get("agentA")

        acp.push_update(sid, {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-builtin",
            "status": "completed",
            "content": [{"content": {"type": "text", "text": "ok"}}],
        })
        await asyncio.sleep(0.01)
        # Grok decides to call another built-in right away (still mid-turn).
        acp.push_update(sid, {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-2",
            "title": "Read",
            "rawInput": {"path": "/etc/hostname"},
        })
        await asyncio.sleep(0.01)
        assert any(tc["id"] == "tc-2" for tc in st.queued_tool_calls)

        messages = [
            HumanMessage(content="run shell"),
            AIMessage(content="", tool_calls=[
                {"id": "tc-builtin", "name": "Shell",
                 "args": {"command": "ls -la"}, "type": "tool_call"}]),
            ToolMessage(content="ok", tool_call_id="tc-builtin",
                        additional_kwargs={BUILTIN_MARKER: True}),
        ]
        nxt: AIMessage = await asyncio.wait_for(
            model.ainvoke(messages, config=_cfg("agentA")), timeout=1
        )
        assert any(tc["id"] == "tc-2" and tc["name"] == "Read"
                   for tc in nxt.tool_calls)


class TestPumpCancellation:
    async def test_summarization_cancels_pump(self, model):
        """A mid-turn pump must be cancelled when its session is dropped by a
        summarization — otherwise it keeps draining a dead queue and the next
        turn inherits the pre-summary session's text.

        Driven through the production route, ``areset_after_summary``
        (SummarizerPort). The fork had no port at all, so the only way to
        reach this reset was ainvoke's implicit shorter-list detection; that
        detection is a safety net now, covered by the sibling test below.
        """
        acp, sid, _msg = await _start_builtin_turn(model)
        st = model._client_state.get("agentA")
        pump = st.pump_task
        assert pump is not None and not pump.done()

        await model.areset_after_summary(thread_id="agentA")

        assert pump.cancelled() or pump.done()
        st2 = model._client_state.get("agentA")
        assert st2 is st  # same slot: the thread id is never rotated
        assert st2.pump_task is None
        assert st2.session_id is None

        # The next turn runs on a brand-new session and is unaffected.
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="fresh start")],
                          config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        new_sid = f"sess-{acp.session_counter}"
        assert new_sid != sid
        acp.push_update(new_sid, {"sessionUpdate": "agent_message_chunk",
                                  "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        final = await asyncio.wait_for(task, timeout=1)
        assert final.content == "ok"

    async def test_safety_net_cancels_pump_and_refuses_to_reprompt(self, model):
        """The ainvoke-side safety net: a shorter message list without a
        preceding ``areset_after_summary``.

        It must still cancel the pump and reset the slot, but it may NOT fall
        through into a fresh session/prompt — that would re-send the whole
        visible history (incident 2026-08-19). So it raises
        ``AcpSummarizationResetRequired`` and the next turn starts clean.
        """
        acp, _sid, _msg = await _start_builtin_turn(model)
        st = model._client_state.get("agentA")
        pump = st.pump_task
        assert pump is not None and not pump.done()
        sessions_before = len(acp.session_new_calls)
        prompts_before = len(acp.prompts)
        # Inflate sent_count so the next (shorter) call trips the net.
        st.sent_count = 50

        # Bounded for the same reason as the sibling assertion in
        # test_chat_model: the regression being pinned is a fall-through into
        # a fresh session/prompt that nothing here resolves, which hangs
        # rather than fails if left unbounded.
        with pytest.raises(acp_cm.AcpSummarizationResetRequired):
            await asyncio.wait_for(
                model.ainvoke([HumanMessage(content="fresh start")],
                              config=_cfg("agentA")),
                timeout=2,
            )

        assert pump.cancelled() or pump.done()
        assert st.pump_task is None
        assert st.session_id is None
        assert st.sent_count == 0
        # No second session and no prompt were fired on the way out.
        assert len(acp.session_new_calls) == sessions_before
        assert len(acp.prompts) == prompts_before

    async def test_channel_a_does_not_start_pump(self, model):
        """A proxy-MCP (channel A) tool_call must NOT spawn a pump — its
        resolution comes via ToolMessage→proxy_mcp.resolve_call, and the next
        ainvoke re-enters _consume_turn directly. The pump is built-in only,
        but the pause path is shared; assert the proxy case leaves no pump
        racing on the queue that resolve_call (not the queue) drives.

        NOTE: the pump is started for ANY mid-turn pause (channel A or B) so
        the queue keeps draining; what matters is that adopting it on the next
        ainvoke is a clean no-op and does not lose the proxy resolution.
        """
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="run")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-a",
            "title": "CallMcpTool",
            "rawInput": {"server": model._proxy_mcp.server_name,
                         "toolName": "execute", "arguments": {"command": "ls"}},
        })
        msg: AIMessage = await asyncio.wait_for(task, timeout=1)
        assert msg.tool_calls[0]["name"] == "execute"

        st = model._client_state.get("agentA")
        # Resolve via the proxy path and finish; the next ainvoke must adopt
        # the pump cleanly and finalize without losing the resolution.
        acp.push_update(sid, {
            "sessionUpdate": "agent_message_chunk",
            "content": {"text": "ran it"},
        })
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.sleep(0.01)

        messages = [
            HumanMessage(content="run"),
            AIMessage(content="", tool_calls=[
                {"id": "tc-a", "name": "execute", "args": {"command": "ls"},
                 "type": "tool_call"}]),
            ToolMessage(content="ls output", tool_call_id="tc-a"),
        ]
        final: AIMessage = await asyncio.wait_for(
            model.ainvoke(messages, config=_cfg("agentA")), timeout=1
        )
        assert any(cid == "tc-a" for cid, _ in model._proxy_mcp.resolved)
        assert "ran it" in final.content
        assert st.pump_task is None

    async def test_start_pump_cancels_the_previous_one(self, model):
        """``_start_pump`` must cancel a live pump before spawning the next
        one, or the notification queue gets TWO consumers and each update is
        delivered to whichever task wins the race.

        This covers the second cancellation site (``_start_pump``'s
        single-consumer guard) — the sibling tests in this class reach the
        other one (``_adopt_pump``, via ``areset_after_summary`` and the
        ainvoke safety net), so without this test the guard is unpinned:
        neutering ``_cancel_pump`` entirely leaves the whole suite green.

        Both arms are asserted, because "the old task is cancelled" alone
        survives a mutation that cancels unconditionally and never spawns a
        replacement:
          * the OLD task ends up cancelled, and
          * a NEW, distinct, live task owns the slot afterwards.
        """
        st = model._client_state.get_or_create("agentA")
        model._start_pump(st, "agentA")
        first = st.pump_task
        assert first is not None and not first.done()

        model._start_pump(st, "agentA")
        second = st.pump_task

        # Arm 2: the slot holds a different, live task.
        assert second is not None
        assert second is not first
        assert not second.done()

        # Arm 1: the displaced task is actually cancelled, not merely dropped
        # on the floor (a leaked task would keep draining the same queue).
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=1)
        assert first.cancelled()

        # Teardown only: ``second`` may already have finished on its own (the
        # fake queue is empty), so this is not an oracle — just don't leak it.
        model._cancel_pump(st)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(second, timeout=1)


class TestNoHangRegression:
    async def test_full_builtin_roundtrip_does_not_hang(self, model):
        """End-to-end regression for the test_grok 2026-06-18 incident: a
        built-in Shell whose terminal update arrives strictly AFTER ainvoke
        returned (only the pump can see it) followed by a ToolMessage resume
        must produce a final AIMessage well within the timeout — never hang."""
        async def scenario():
            acp, sid, msg = await _start_builtin_turn(model)
            # Terminal update arrives only now — during ToolNode execution.
            acp.push_update(sid, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc-builtin",
                "status": "completed",
                "content": [{"content": {"type": "text",
                                         "text": "12 files"}}],
                "rawOutput": {"exit_code": 0},
            })
            await asyncio.sleep(0.01)
            acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                                  "content": {"text": "There are 12 files."}})
            await asyncio.sleep(0.01)
            acp.finish("end_turn")
            await asyncio.sleep(0.01)
            messages = [
                HumanMessage(content="how many files in /tmp"),
                AIMessage(content="", tool_calls=[
                    {"id": "tc-builtin", "name": "Shell",
                     "args": {"command": "ls -la /tmp"},
                     "type": "tool_call"}]),
                ToolMessage(content="12 files", tool_call_id="tc-builtin",
                            additional_kwargs={BUILTIN_MARKER: True}),
            ]
            return await model.ainvoke(messages, config=_cfg("agentA"))

        final: AIMessage = await asyncio.wait_for(scenario(), timeout=2)
        assert "12 files" in final.content
