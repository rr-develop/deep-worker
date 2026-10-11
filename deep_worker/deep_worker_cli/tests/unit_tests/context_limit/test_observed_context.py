"""The provider-report contract: producing, validating and reading a stamp.

This file used to test a counter that mixed a provider observation with a
chars-per-token estimate for the tail, plus the estimator itself. Both are
gone (owner directive 2026-08-25: the provider's number decides, nothing
else, and no fallback is approved for any provider). What survives here is
the part that is still true and still load-bearing: how an observation is
written, when it is valid, and which one wins.

The decision-level invariants — provider-only bands, UNKNOWN staying
unknown, no estimator on the path — live in
``test_provider_context_only.py``.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_worker_cli.context_limit.observed_context import (
    CONTEXT_OBSERVATION_KEY,
    annotate_context_observation,
    context_observation,
    find_context_anchor,
    first_state_message_id,
    provider_context_tokens,
)


def _history(first_id: str = "h1") -> list:
    return [
        SystemMessage(content="sys", id="s0"),
        HumanMessage(content="hi", id=first_id),
    ]


def _stamp(used: int, first_id: str | None, *, size: int | None = None) -> AIMessage:
    msg = AIMessage(content="a", id=f"ai-{used}")
    annotate_context_observation(msg, used=used, size=size, first_id=first_id)
    return msg


# ── identity of a history ────────────────────────────────────────────────
def test_first_state_message_id_skips_leading_system_messages() -> None:
    """The factory prepends the system prompt and the guard may prepend a
    hint; neither is state, so neither may change a history's identity."""
    msgs = [
        SystemMessage(content="sys", id="s0"),
        SystemMessage(content="guard hint", id="s1"),
        HumanMessage(content="hi", id="h1"),
    ]
    assert first_state_message_id(msgs) == "h1"


def test_first_state_message_id_none_for_empty_or_unidentified() -> None:
    assert first_state_message_id([]) is None
    assert first_state_message_id([HumanMessage(content="x")]) is None


# ── writing and reading a stamp ──────────────────────────────────────────
def test_annotate_roundtrip_in_response_metadata() -> None:
    msg = _stamp(1234, "h1", size=200_000)
    obs = msg.response_metadata[CONTEXT_OBSERVATION_KEY]
    assert obs == {"used": 1234, "size": 200_000, "first_id": "h1"}
    assert context_observation(msg) == obs


def test_context_observation_ignores_garbage() -> None:
    for bad in ("not-a-dict", {"used": "many"}, {"used": 0}, {"size": 5}):
        msg = AIMessage(content="a", id="x")
        msg.response_metadata[CONTEXT_OBSERVATION_KEY] = bad
        assert context_observation(msg) is None


# ── which observation counts ─────────────────────────────────────────────
def test_latest_valid_observation_wins() -> None:
    msgs = [*_history(), _stamp(10, "h1"), HumanMessage(content="m", id="h2"), _stamp(20, "h1")]
    assert find_context_anchor(msgs)[1] == 20
    assert provider_context_tokens(msgs) == 20


def test_observation_survives_prepended_system_messages() -> None:
    """A guard hint in front of the history must not invalidate a stamp —
    that is exactly the off-by-one that cost a warm codex session on
    2026-08-25 when a prepended hint changed what the provider counted."""
    msgs = [SystemMessage(content="hint", id="s9"), *_history(), _stamp(42, "h1")]
    assert provider_context_tokens(msgs) == 42


def test_stale_observation_after_summarization_is_ignored() -> None:
    """After summarization the history is replaced; a stamp measuring the
    old conversation must not decide anything about the new one."""
    msgs = [
        SystemMessage(content="sys", id="s0"),
        HumanMessage(content="summary", id="NEW"),
        _stamp(500_000, "OLD"),
    ]
    assert find_context_anchor(msgs) is None
    assert provider_context_tokens(msgs) is None


def test_slice_from_the_middle_never_sees_an_anchor() -> None:
    """A slice has a different first state message, so its stamps belong to
    a different history by construction."""
    full = [*_history(), _stamp(99, "h1"), HumanMessage(content="tail", id="h2")]
    assert provider_context_tokens(full[2:]) is None


def test_observation_without_first_id_is_ignored() -> None:
    msgs = [*_history(), _stamp(77, None)]
    assert provider_context_tokens(msgs) is None
