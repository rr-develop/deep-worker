"""Tests for ContextLimitGuardMiddleware — the pre-call guarantee.

See ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md`` §3.2.

The guard is the single safety net for the agent_a bug
(2026-08-18, limit 400k → actual 600k). It MUST:

* be **provider-independent** — work whether or not
  ``usage_metadata`` is reported (claude-code does NOT report it
  reliably, which is exactly what caused the original incident);
* enforce a **three-band policy** (Green / Amber / Red / Block) so the
  agent gets one voluntary turn to compact before the hard block;
* be a **no-op** when ``context_limit is None`` (operator did not
  pass ``-l``).

These tests pin every transition. If a future refactor of
``SummarizationMiddleware`` or a provider-side change accidentally
makes the guard a no-op, the suite goes red.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_worker_cli.context_limit.guard import (
    ContextLimitExceeded,
    ContextLimitGuardMiddleware,
    _band_for,
)


# =========================================================================
# _band_for — pure-function classification
# =========================================================================


def _reported(used: int, first_id: str = "h1"):
    """A history whose provider reported ``used`` tokens.

    Bands follow the PROVIDER's number (owner directive 2026-08-25).
    Character counts mean nothing to the guard any more, so a test states
    the reported size explicitly instead of sizing a string and trusting a
    divisor.
    """
    from deep_worker_cli.context_limit.observed_context import (
        annotate_context_observation,
    )

    human = HumanMessage(content="task", id=first_id)
    stamped = AIMessage(content="ok", id="ai-1")
    annotate_context_observation(stamped, used=used, size=None, first_id=first_id)
    return [human, stamped]

def test_band_for_green_below_85_percent() -> None:
    """Below 85% of the limit → Green (no action)."""
    assert _band_for(0, 100_000) == "green"
    assert _band_for(50_000, 100_000) == "green"
    assert _band_for(84_999, 100_000) == "green"


def test_band_for_amber_between_85_and_95_percent() -> None:
    """85% ≤ usage < 95% → Amber (inject hint, model call proceeds)."""
    assert _band_for(85_000, 100_000) == "amber"
    assert _band_for(90_000, 100_000) == "amber"
    assert _band_for(94_999, 100_000) == "amber"


def test_band_for_red_at_or_above_95_percent() -> None:
    """95% ≤ usage ≤ 100% → Red (ToolMessage, NO model call)."""
    assert _band_for(95_000, 100_000) == "red"
    assert _band_for(99_999, 100_000) == "red"
    assert _band_for(100_000, 100_000) == "red"


def test_band_for_block_above_limit() -> None:
    """>100% → Block (ContextLimitExceeded, IDLE state)."""
    assert _band_for(100_001, 100_000) == "block"
    assert _band_for(200_000, 100_000) == "block"


def test_band_for_zero_limit_is_all_block() -> None:
    """Defensive: ``context_limit <= 0`` is treated as Block.

    This shouldn't happen — ``-l`` rejects 0 — but a corrupted
    session_metadata.json or a buggy backfill could leave the field
    at 0. The guard must NOT silently pass through; it must Block.
    """
    assert _band_for(0, 0) == "block"
    assert _band_for(100, 0) == "block"


# =========================================================================
# ContextLimitGuardMiddleware — wrap_model_call integration
# =========================================================================


def _make_request(messages, system_message=None):
    """Build a fake ModelRequest carrying *messages* for the guard tests.

    We avoid importing langchain.agents.middleware.ModelRequest because
    its signature drifts between versions; instead we build the
    minimal duck-typed object the guard actually reads.
    """

    class _FakeRequest:
        def __init__(self, msgs, sys_msg):
            self.messages = msgs
            self.system_message = sys_msg
            self.tools = []

        def override(self, **kwargs):
            new = _FakeRequest(self.messages, self.system_message)
            new.tools = self.tools
            for k, v in kwargs.items():
                setattr(new, k, v)
            return new

    return _FakeRequest(messages, system_message)


def _response(messages):
    """Build a fake ModelResponse carrying *messages* back."""

    class _FakeResponse:
        def __init__(self, msgs):
            self.messages = msgs

    return _FakeResponse(messages)


def test_guard_no_op_when_context_limit_is_none() -> None:
    """Operator did not set -l → guard is a transparent passthrough.

    No hint injected, no block, no side effects. The model call goes
    through unchanged.
    """
    guard = ContextLimitGuardMiddleware(context_limit=None)
    request = _make_request([HumanMessage(content="hi")])
    seen = {}

    def handler(req):
        seen["called"] = True
        return _response(req.messages)

    response = guard.wrap_model_call(request, handler)

    assert seen["called"] is True
    assert response.messages == [HumanMessage(content="hi")]


def test_guard_green_band_passes_through() -> None:
    """A small request within Green band → unchanged passthrough."""
    guard = ContextLimitGuardMiddleware(context_limit=100_000)
    msgs = [HumanMessage(content="hello")]
    request = _make_request(msgs)
    seen = {}

    def handler(req):
        seen["called"] = True
        return _response(req.messages)

    response = guard.wrap_model_call(request, handler)

    assert seen["called"] is True
    assert response.messages == msgs


def test_guard_amber_band_injects_hint_system_message() -> None:
    """A request in Amber band → a SystemMessage hint is prepended.

    The model call STILL proceeds (one voluntary turn to compact). The
    hint is identified by ``additional_kwargs['deep_worker_context_guard']``
    so the agent can recognise it (and not re-act to it).
    """
    guard = ContextLimitGuardMiddleware(context_limit=1000, model_name="claude-sonnet-4.5")
    # Need 0.85 × 1000 = 850 tokens. With claude divisor 2 → 1700 chars.
    # Below 0.95 × 1000 = 950 tokens → 1900 chars (lower bound for Red).
    # 900 reported tokens → 90% of the limit → Amber.
    msgs = _reported(900)
    request = _make_request(msgs)
    seen = {}

    def handler(req):
        seen["called"] = True
        # Capture what the handler actually received.
        seen["received_messages"] = list(req.messages)
        return _response(req.messages)

    response = guard.wrap_model_call(request, handler)

    assert seen["called"] is True, "Amber band must still call the model"
    received = seen["received_messages"]
    assert any(isinstance(m, SystemMessage) for m in received), (
        "Amber band must inject a SystemMessage hint"
    )
    hint = next(m for m in received if isinstance(m, SystemMessage))
    assert "context-guard" in hint.content.lower() or "context" in hint.content.lower()
    assert hint.additional_kwargs.get("deep_worker_context_guard") is True


def test_guard_red_band_injects_tool_message_and_proceeds() -> None:
    """Red band → ToolMessage appended to messages, model call PROCEEDS.

    The model gets one turn to self-correct in-place (call
    compact_history or reduce scope). If the next request still trips
    Red/Block, the hard limit fires. See design §C2 (review): the
    model call MUST happen — otherwise the graph has no AIMessage and
    the agent stalls.
    """
    guard = ContextLimitGuardMiddleware(context_limit=1000, model_name="claude-sonnet-4.5")
    # Without an explicit model_name the guard defaults to ``"unknown"``
    # family which uses divisor 3 — we explicitly name a claude so the
    # heuristic uses divisor 2. Then 2000 chars → 1000 tokens → exactly
    # at limit → Red band.
    msgs = _reported(1000)
    request = _make_request(msgs)
    seen = {}

    def handler(req):
        seen["called"] = True
        seen["received_messages"] = list(req.messages)
        return _response(req.messages)

    response = guard.wrap_model_call(request, handler)

    # Model call DID happen (with injected warning).
    assert seen["called"] is True, (
        "Red band must still call the model — the warning is "
        "INJECTED into the conversation, not a block."
    )
    # The handler received the original messages PLUS a ToolMessage.
    received = seen["received_messages"]
    tool_msgs_in_request = [m for m in received if isinstance(m, ToolMessage)]
    assert tool_msgs_in_request, (
        "Red band must inject a ToolMessage into the messages list"
    )
    err_msg = tool_msgs_in_request[0]
    assert err_msg.additional_kwargs.get("context_overflow_warning") is True
    assert err_msg.additional_kwargs.get("deep_worker_context_guard") is True


def test_guard_red_band_includes_original_messages_unchanged() -> None:
    """Red band: original messages kept verbatim, ToolMessage appended at end.

    The agent's conversation history is preserved (so the model can
    self-correct with full context), with the warning appended last so
    the model sees it as the most recent thing in the conversation.
    """
    guard = ContextLimitGuardMiddleware(context_limit=1000, model_name="claude-sonnet-4.5")
    msgs = _reported(1000)
    request = _make_request(msgs)
    seen = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        return _response(req.messages)

    guard.wrap_model_call(request, handler)

    received = seen["messages"]
    assert len(received) == len(msgs) + 1, (
        f"expected original {len(msgs)} + 1 ToolMessage, got {len(received)}"
    )
    # Original HumanMessage preserved.
    assert received[0].content == "task"
    # ToolMessage appended at the end.
    assert isinstance(received[-1], ToolMessage)
    assert received[-1].additional_kwargs.get("context_overflow_warning") is True


def test_guard_block_band_raises_context_limit_exceeded() -> None:
    """Block band (>L) → ContextLimitExceeded raised, handler NOT called.

    The caller (agent-process) catches this and transitions to IDLE
    (per the no-agent-error-state policy).
    """
    guard = ContextLimitGuardMiddleware(context_limit=1000, model_name="claude-sonnet-4.5")
    # With divisor 2 (claude), 3500 chars → 1750 tokens → over 1000 → Block.
    # 1750 reported tokens against a 1000 limit → Block.
    msgs = _reported(1750)
    request = _make_request(msgs)
    seen = {"called": False}

    def handler(req):
        seen["called"] = True
        return _response(req.messages)

    with pytest.raises(ContextLimitExceeded) as exc_info:
        guard.wrap_model_call(request, handler)

    assert seen["called"] is False
    assert exc_info.value.context_limit == 1000
    assert exc_info.value.estimated_tokens > 1000


def test_guard_no_block_when_estimate_exactly_at_limit() -> None:
    """Edge: estimate == limit triggers Red band (not Block).

    The Red/Block threshold is at the ceiling: estimate == L is still
    within the limit and triggers the Red band's ToolMessage + proceed
    behaviour. Block fires only when estimate STRICTLY exceeds L.
    """
    guard = ContextLimitGuardMiddleware(context_limit=1000, model_name="claude-sonnet-4.5")
    msgs = _reported(1000)  # exactly at the limit → not Block
    request = _make_request(msgs)
    seen = {"called": False, "got_tool_msg": False}

    def handler(req):
        seen["called"] = True
        seen["got_tool_msg"] = any(
            isinstance(m, ToolMessage)
            and m.additional_kwargs.get("context_overflow_warning")
            for m in req.messages
        )
        return _response(req.messages)

    guard.wrap_model_call(request, handler)

    assert seen["called"] is True, "estimate == L is Red -> proceed with warning"
    assert seen["got_tool_msg"] is True, (
        "estimate == L must have a ToolMessage appended to the conversation"
    )


