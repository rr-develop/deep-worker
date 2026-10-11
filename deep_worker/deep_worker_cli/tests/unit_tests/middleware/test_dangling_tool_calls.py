"""Unit tests for DanglingToolCallMiddleware.

Design doc:
    deep_worker_cli/docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md

Bug B: an AIMessage with tool_calls carried over from a PREVIOUS (interrupted)
session has no matching ToolMessage in the resumed checkpoint. On the next
turn LangGraph would route that unresolved tool_call into ToolNode and
RE-EXECUTE the tool (dangerous, uncontrolled side effects) — observed crashing
codex agents in ``error``.

Fix B: a ``before_model`` middleware that, before the model/ToolNode runs,
finds every dangling tool_call (a ``tool_call.id`` with no matching
``ToolMessage``) and CLOSES it with a synthetic ``ToolMessage`` carrying a
WARNING (outcome-unknown) notice. The tool is never re-invoked.

Invariants under test:
* resolved tail (every tool_call has a ToolMessage) -> returns None.
* one dangling call -> one closer ToolMessage with the right tool_call_id,
  name, ``status="error"`` and the warning markers in additional_kwargs.
* multiple dangling calls (incl. several in one AIMessage) -> one closer each.
* empty / no-AIMessage / fresh-HumanMessage-only state -> None.
* idempotency: feeding the produced closers back in -> None (no re-injection).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Overwrite

# NOTE: import fails until the implementation lands — expected TDD red phase.
from deep_worker_cli.middleware.dangling_tool_calls import (
    DANGLING_MARKER_KEY,
    DanglingToolCallMiddleware,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _state(messages: list[Any] | None) -> dict[str, Any]:
    st: dict[str, Any] = {}
    if messages is not None:
        st["messages"] = messages
    return st


def _invoke(
    mw: DanglingToolCallMiddleware, messages: list[Any] | None
) -> dict[str, Any] | None:
    # runtime is unused by the middleware; None is fine.
    return mw.before_model(_state(messages), None)  # type: ignore[arg-type]


def _ai_with_tool_call(call_id: str, name: str = "apply_patch") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": {}, "type": "tool_call"}],
    )


def _closers(result: dict[str, Any] | None) -> list[ToolMessage]:
    """Synthetic WARNING closers from an Overwrite rewrite (or a raw list)."""
    assert result is not None
    msgs = result["messages"]
    if isinstance(msgs, Overwrite):
        msgs = msgs.value
    return [
        m
        for m in msgs
        if isinstance(m, ToolMessage)
        and (m.additional_kwargs or {}).get(DANGLING_MARKER_KEY)
    ]


# ===========================================================================
# Resolved / no-op cases -> None
# ===========================================================================
class TestNoOp:
    def test_empty_state_returns_none(self) -> None:
        mw = DanglingToolCallMiddleware()
        assert _invoke(mw, None) is None
        assert _invoke(mw, []) is None

    def test_fresh_human_only_returns_none(self) -> None:
        mw = DanglingToolCallMiddleware()
        assert _invoke(mw, [HumanMessage(content="hi")]) is None

    def test_no_aimessage_with_tool_calls_returns_none(self) -> None:
        mw = DanglingToolCallMiddleware()
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="plain answer, no tool calls"),
        ]
        assert _invoke(mw, msgs) is None

    def test_resolved_tool_call_returns_none(self) -> None:
        """A tool_call that HAS a matching ToolMessage is not dangling."""
        mw = DanglingToolCallMiddleware()
        msgs = [
            HumanMessage(content="do it"),
            _ai_with_tool_call("call_1", name="read_file"),
            ToolMessage(content="file body", tool_call_id="call_1", name="read_file"),
            HumanMessage(content="continue"),
        ]
        assert _invoke(mw, msgs) is None


# ===========================================================================
# Dangling cases -> synthetic closer ToolMessage(s)
# ===========================================================================
class TestClosesDangling:
    def test_single_dangling_closed(self) -> None:
        mw = DanglingToolCallMiddleware()
        poison_id = "apply_patch_136_136716975835008"
        msgs = [
            HumanMessage(content="old request"),
            _ai_with_tool_call(poison_id, name="apply_patch"),
            # NO matching ToolMessage — interrupted prior session.
            HumanMessage(content="new turn after restart"),
        ]
        result = _invoke(mw, msgs)
        closers = _closers(result)
        assert len(closers) == 1
        tm = closers[0]
        assert tm.tool_call_id == poison_id
        assert tm.name == "apply_patch"
        # WARNING semantics: status must be error (never success) so the model
        # does not mistake it for a real result.
        assert tm.status == "error"
        # machine-readable warning markers (Q1 + follow-up suggestion).
        assert tm.additional_kwargs.get(DANGLING_MARKER_KEY) is True
        assert tm.additional_kwargs.get("deep_worker_warning") == "outcome_unknown"
        # honest, human-readable warning text mentioning the prior session.
        text = str(tm.content).lower()
        assert "previous session" in text or "prior session" in text

    def test_multiple_dangling_calls_each_closed(self) -> None:
        mw = DanglingToolCallMiddleware()
        msgs = [
            HumanMessage(content="old"),
            _ai_with_tool_call("command_execution_72_127590061854528", name="command_execution"),
            _ai_with_tool_call("apply_patch_129_136716968105280", name="apply_patch"),
            HumanMessage(content="new"),
        ]
        result = _invoke(mw, msgs)
        closers = _closers(result)
        closed_ids = {c.tool_call_id for c in closers}
        assert closed_ids == {
            "command_execution_72_127590061854528",
            "apply_patch_129_136716968105280",
        }

    def test_several_tool_calls_in_one_aimessage(self) -> None:
        mw = DanglingToolCallMiddleware()
        ai = AIMessage(
            content="",
            tool_calls=[
                {"id": "a_1_1", "name": "read_file", "args": {}, "type": "tool_call"},
                {"id": "a_2_2", "name": "grep", "args": {}, "type": "tool_call"},
            ],
        )
        msgs = [HumanMessage(content="x"), ai, HumanMessage(content="y")]
        result = _invoke(mw, msgs)
        closers = _closers(result)
        assert {c.tool_call_id for c in closers} == {"a_1_1", "a_2_2"}

    def test_tool_call_without_id_is_skipped(self) -> None:
        """A malformed tool_call lacking a string ``id`` cannot be closed (there
        is nothing to match a ToolMessage to) — the middleware skips it rather
        than crashing or fabricating an id.

        ``AIMessage`` normalises tool_calls and requires an ``id`` at
        construction time, so we build a valid message first and then inject
        the malformed entries onto ``.tool_calls`` to simulate what a raw
        checkpoint / non-normalised source could carry.
        """
        mw = DanglingToolCallMiddleware()
        ai = _ai_with_tool_call("real_3", name="read_file")
        # Inject malformed tool_calls (None id, missing id key) alongside the
        # well-formed one.
        ai.tool_calls = [
            {"id": None, "name": "apply_patch", "args": {}, "type": "tool_call"},  # type: ignore[typeddict-item]
            {"name": "command_execution", "args": {}, "type": "tool_call"},  # type: ignore[typeddict-item]
            {"id": "real_3", "name": "read_file", "args": {}, "type": "tool_call"},
        ]
        msgs = [HumanMessage(content="x"), ai, HumanMessage(content="y")]
        result = _invoke(mw, msgs)
        closers = _closers(result)
        # Only the well-formed dangling call (with a string id) is closed.
        assert {c.tool_call_id for c in closers} == {"real_3"}

    def test_partial_resolution_only_closes_unresolved(self) -> None:
        """One of two tool_calls already has a ToolMessage; only the other
        unresolved one gets a closer."""
        mw = DanglingToolCallMiddleware()
        ai = AIMessage(
            content="",
            tool_calls=[
                {"id": "resolved_1", "name": "read_file", "args": {}, "type": "tool_call"},
                {"id": "dangling_2", "name": "apply_patch", "args": {}, "type": "tool_call"},
            ],
        )
        msgs = [
            HumanMessage(content="x"),
            ai,
            ToolMessage(content="ok", tool_call_id="resolved_1", name="read_file"),
            HumanMessage(content="y"),
        ]
        result = _invoke(mw, msgs)
        closers = _closers(result)
        assert {c.tool_call_id for c in closers} == {"dangling_2"}


# ===========================================================================
# Idempotency
# ===========================================================================
class TestIdempotency:
    def test_feeding_closers_back_returns_none(self) -> None:
        """After the closers are appended (becoming real ToolMessages), a
        subsequent pass must find nothing dangling -> None."""
        mw = DanglingToolCallMiddleware()
        poison_id = "apply_patch_136_136716975835008"
        ai = _ai_with_tool_call(poison_id, name="apply_patch")
        msgs = [HumanMessage(content="old"), ai, HumanMessage(content="new")]

        result = _invoke(mw, msgs)
        rewritten = result["messages"]
        if isinstance(rewritten, Overwrite):
            rewritten = rewritten.value
        # Simulate the Overwrite landing in state, then a re-entry.
        assert _invoke(mw, list(rewritten)) is None
