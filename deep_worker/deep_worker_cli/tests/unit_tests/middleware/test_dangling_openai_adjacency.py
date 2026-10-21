"""Pin: dangling closers must satisfy OpenAI tool-call adjacency (DeepSeek).

Policy: DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md — after
restart, dangling pre-restart tool_calls must be closed without re-execution
and the history must remain valid for the model API.

Production failure (agent_a 2026-08-14): DeepSeek 400
  "insufficient tool messages following tool_calls message"
with history shape:
  AIMessage(tool_calls=[restart_self])
  SystemMessage(NOTICE: restarted)
  HumanMessage(automatic continuation)
  [optional ToolMessage later / at end — NOT adjacent]

Both PatchToolCallsMiddleware and DanglingToolCallMiddleware must leave
ToolMessage(s) immediately after the AIMessage.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_worker_cli.middleware.dangling_tool_calls import DanglingToolCallMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langgraph.types import Overwrite


def _ai_tool(call_id: str, name: str = "restart_self") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "id": call_id, "args": {}}],
    )


def _openai_tool_adjacency_ok(messages: list) -> bool:
    """OpenAI/DeepSeek rule: every tool_call id must be answered by a
    contiguous run of ToolMessages immediately following the AIMessage."""
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        need = {c["id"] for c in m.tool_calls}
        j = i + 1
        got: set[str] = set()
        while j < len(messages) and isinstance(messages[j], ToolMessage):
            got.add(messages[j].tool_call_id)
            j += 1
        if not need <= got:
            return False
    return True


def _restart_poison(call_id: str = "toolu_TEST_RESTART") -> list:
    return [
        HumanMessage(content="please restart"),
        _ai_tool(call_id),
        SystemMessage(content="NOTICE: This agent process was restarted ..."),
        HumanMessage(content="(automatic continuation after restart — ...)"),
    ]


def test_patch_tool_calls_makes_openai_adjacent_after_restart_shape():
    mw = PatchToolCallsMiddleware()
    msgs = _restart_poison()
    out = mw.before_agent({"messages": msgs}, runtime=None)
    assert out is not None
    patched = out["messages"]
    if isinstance(patched, Overwrite):
        patched = patched.value
    assert _openai_tool_adjacency_ok(patched), (
        "PatchToolCallsMiddleware left non-adjacent tool results; "
        f"order={[type(m).__name__ for m in patched]}"
    )


def test_dangling_middleware_makes_openai_adjacent_after_restart_shape():
    """Closers must sit immediately after the AIMessage, not at list end."""
    mw = DanglingToolCallMiddleware()
    msgs = _restart_poison()
    out = mw.before_model({"messages": msgs})
    assert out is not None and out.get("messages")
    closed = out["messages"]
    if isinstance(closed, Overwrite):
        closed = closed.value
    assert _openai_tool_adjacency_ok(closed), (
        "DanglingToolCallMiddleware must Overwrite with ToolMessage "
        "immediately after AIMessage.tool_calls. "
        f"order={[type(m).__name__ for m in closed]}"
    )


def test_patch_hwm_does_not_skip_still_dangling_same_length():
    """If Overwrite does not stick, same-length re-entry must still patch."""
    mw = PatchToolCallsMiddleware()
    msgs = _restart_poison()
    first = mw.before_agent({"messages": msgs}, runtime=None)
    assert first is not None
    # Second call with THE SAME still-poisoned list (Overwrite not applied)
    second = mw.before_agent({"messages": msgs}, runtime=None)
    assert second is not None, (
        "PatchToolCallsMiddleware high-water mark skips re-scan when "
        "len(messages) is unchanged, even if history is still dangling"
    )


def test_patch_does_not_treat_nonadjacent_toolmessage_as_resolved():
    """ToolMessage existing later in history is NOT enough for OpenAI."""
    call_id = "toolu_LATER"
    msgs = [
        HumanMessage(content="x"),
        _ai_tool(call_id),
        SystemMessage(content="NOTICE"),
        HumanMessage(content="continue"),
        ToolMessage(content="cancelled", tool_call_id=call_id, name="restart_self"),
    ]
    mw = PatchToolCallsMiddleware()
    out = mw.before_agent({"messages": msgs}, runtime=None)
    # Must rewrite so ToolMessage is immediately after AIMessage
    assert out is not None, (
        "PatchToolCallsMiddleware treats a later ToolMessage as resolved "
        "and leaves non-adjacent order that DeepSeek rejects"
    )
    patched = out["messages"]
    if isinstance(patched, Overwrite):
        patched = patched.value
    assert _openai_tool_adjacency_ok(patched)


def _unwrap(out):
    msgs = out["messages"]
    if isinstance(msgs, Overwrite):
        msgs = msgs.value
    return list(msgs)


def _assert_existing_closer_moved_once(rewritten, original, call_id):
    """Design §5 / §7.2: move, not copy. Exactly one ToolMessage for
    call_id; it is immediately after its AIMessage; identity and
    content/metadata of the original closer are preserved."""
    tools = [
        m for m in rewritten
        if isinstance(m, ToolMessage) and m.tool_call_id == call_id
    ]
    assert len(tools) == 1, (
        f"expected exactly one ToolMessage id={call_id!r}, got {len(tools)}"
    )
    moved = tools[0]
    ai_idx = next(
        i for i, m in enumerate(rewritten)
        if isinstance(m, AIMessage)
        and any(c.get("id") == call_id for c in (m.tool_calls or ()))
    )
    assert rewritten[ai_idx + 1] is moved, (
        "moved closer is not immediately after its AIMessage"
    )
    assert moved is original, "closer was copied, not moved"
    assert moved.content == original.content
    assert moved.name == original.name
    assert dict(moved.additional_kwargs or {}) == dict(
        original.additional_kwargs or {}
    )


def test_patch_moves_nonadjacent_closer_once_preserving_identity():
    call_id = "toolu_MOVE_PATCH"
    original = ToolMessage(
        content="existing cancelled closer",
        tool_call_id=call_id,
        name="restart_self",
        additional_kwargs={"source": "patch-original"},
    )
    msgs = [
        HumanMessage(content="x"),
        _ai_tool(call_id),
        SystemMessage(content="NOTICE"),
        HumanMessage(content="continue"),
        original,
    ]
    out = PatchToolCallsMiddleware().before_agent({"messages": msgs}, runtime=None)
    assert out is not None
    rewritten = _unwrap(out)
    assert _openai_tool_adjacency_ok(rewritten)
    _assert_existing_closer_moved_once(rewritten, original, call_id)


def test_dangling_moves_nonadjacent_closer_once_preserving_identity():
    """Dangling had no later-closer move test — same contract as Patch."""
    call_id = "toolu_MOVE_DANGLING"
    original = ToolMessage(
        content="existing warning closer",
        tool_call_id=call_id,
        name="restart_self",
        additional_kwargs={"source": "dangling-original"},
    )
    msgs = [
        HumanMessage(content="x"),
        _ai_tool(call_id),
        SystemMessage(content="NOTICE"),
        HumanMessage(content="continue"),
        original,
    ]
    out = DanglingToolCallMiddleware().before_model({"messages": msgs})
    assert out is not None
    rewritten = _unwrap(out)
    assert _openai_tool_adjacency_ok(rewritten)
    _assert_existing_closer_moved_once(rewritten, original, call_id)


def test_exact_checkpoint_shape_still_poisons_deepseek():
    """Live threads.db 2026-08-14 (read-only dump): three
    restart_self AIMessages. Two have no ToolMessage at all; one has
    Patch's 'cancelled' closer *after* System+Human. Patch today fixes
    the first two (Overwrite insert) and leaves the third — that leftover
    is enough for DeepSeek 400.
    """
    id_open_1 = "toolu_018o1xiy5kuypbqwCBKvVRJc"
    id_open_2 = "toolu_01RWZRqKapPcLa4ZV8tjtqRD"
    id_nonadj = "toolu_01Rf5siKsK8Dq9gz1sKPxS45"
    msgs = [
        HumanMessage(content="step1"),
        _ai_tool(id_open_1),
        SystemMessage(content="NOTICE: restarted 09:50:49"),
        HumanMessage(content="(automatic continuation after restart — ...)"),
        HumanMessage(content="step2"),
        _ai_tool(id_open_2),
        SystemMessage(content="NOTICE: restarted 09:51:26"),
        HumanMessage(content="(automatic continuation after restart — ...)"),
        HumanMessage(content="step3"),
        _ai_tool(id_nonadj),
        SystemMessage(content="NOTICE: restarted 10:47:05"),
        HumanMessage(content="(automatic continuation after restart — ...)"),
        ToolMessage(
            content=(
                f"Tool call restart_self with id {id_nonadj} was "
                "cancelled - another message came in before it could be completed."
            ),
            tool_call_id=id_nonadj,
            name="restart_self",
        ),
        SystemMessage(content="NOTICE: restarted 10:47:05"),
        HumanMessage(content="(automatic continuation after restart — ...)"),
        HumanMessage(content="user message after switch to deepseek"),
    ]
    assert not _openai_tool_adjacency_ok(msgs)

    patched_out = PatchToolCallsMiddleware().before_agent(
        {"messages": msgs}, runtime=None
    )
    assert patched_out is not None
    patched = patched_out["messages"]
    if isinstance(patched, Overwrite):
        patched = patched.value
    assert _openai_tool_adjacency_ok(patched), (
        "Patch leaves the non-adjacent cancelled closer in place; "
        f"order={[type(m).__name__ for m in patched]}"
    )
    nonadj_tools = [
        m for m in patched
        if isinstance(m, ToolMessage) and m.tool_call_id == id_nonadj
    ]
    assert len(nonadj_tools) == 1, (
        "agent_1 idx-69 closer must be moved once, not copied"
    )
