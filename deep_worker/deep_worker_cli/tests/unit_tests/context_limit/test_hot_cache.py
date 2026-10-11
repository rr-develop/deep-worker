"""Unit tests for hot/cold kv-cache decision."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.context_limit.hot_cache import (
    SESSION_STATEFUL_PROVIDERS,
    STATELESS_PROVIDERS,
    HotColdDecision,
    decide,
    ttl_for,
)


def _ai(*, provider: str = "claude-code", model: str = "fable", received: datetime) -> AIMessage:
    return AIMessage(
        content="ok",
        response_metadata={
            "deep_worker_origin": {
                "provider": provider,
                "model": model,
                "model_id": "claude-fable-5",
                "adapter": None,
                "since": received.isoformat(),
            },
            "deep_worker_usage": {
                "received_at": received.isoformat(),
            },
        },
    )


def test_ttl_default_is_four_minutes() -> None:
    assert ttl_for("claude-code") == 240
    assert ttl_for("unknown-vendor") == 240


def test_hot_same_origin_within_ttl() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    received = now - timedelta(minutes=2)
    d = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        current_model_id="claude-fable-5",
        messages=[HumanMessage(content="q"), _ai(received=received)],
        session_id="sess-1",
    )
    assert d == HotColdDecision(
        True, "hot", 120.0, 240, d.last_origin
    )
    assert d.last_origin is not None
    assert d.last_origin["provider"] == "claude-code"


def test_cold_when_ttl_expired() -> None:
    now = datetime(2026, 8, 19, 15, 14, tzinfo=UTC)
    received = datetime(2026, 8, 19, 11, 7, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=[_ai(received=received)],
        session_id="sess-1",
    )
    assert d.hot is False
    assert d.reason.startswith("ttl_expired:")


def test_cold_on_provider_change() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="grok",
        current_model="fable",
        messages=[_ai(received=now - timedelta(seconds=10))],
        session_id="sess-1",
    )
    assert d.hot is False
    assert d.reason == "provider_changed"


def test_cold_stateful_without_session() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=[_ai(received=now - timedelta(seconds=10))],
        session_id=None,
    )
    assert d.hot is False
    assert d.reason == "no_session"


def test_stateless_hot_without_session() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="deepseek",
        current_model="deepseek-chat",
        messages=[_ai(provider="deepseek", model="deepseek-chat", received=now - timedelta(seconds=10))],
        session_id=None,
    )
    assert d.hot is True


def test_grok_is_session_stateful() -> None:
    """grok is session-stateful — corrected 2026-08-24.

    This test previously asserted the opposite (``test_grok_is_stateless_hot_
    without_session``), which did not merely miss the defect but PINNED it:
    ``GrokChatModel`` sends only ``state_messages[state.sent_count:]`` and
    persists ``grok_session_id`` + ``sent_count`` across restarts, so the
    conversation lives on grok's side.

    The misclassification was not cosmetic. ``_is_session_stateful`` gates the
    guard in ``_ainvoke_identical_prefix``; reading grok as stateless let
    summarization fall through to a plain ``ainvoke``, replaying the entire
    history into a fresh grok session — the exact dump the summarizer port
    exists to prevent, silently, on every summarization.
    """
    assert "grok" in SESSION_STATEFUL_PROVIDERS
    assert "grok" not in STATELESS_PROVIDERS


def test_deepseek_to_grok_is_provider_changed() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="grok",
        current_model="grok-4.6",
        messages=[_ai(provider="deepseek", model="deepseek-v4-flash", received=now - timedelta(seconds=10))],
        session_id=None,
    )
    assert d.hot is False
    assert d.reason == "provider_changed"


def test_zero_ttl_is_always_cold() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=[_ai(received=now - timedelta(seconds=1))],
        session_id="sess-1",
        ttl_overrides={"claude-code": 0},
    )
    assert d.hot is False
    assert d.reason == "always_cold"


def test_star_override_applies_to_other_providers() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    d = decide(
        now=now,
        current_provider="grok",
        current_model="fable",
        messages=[_ai(provider="grok", model="fable", received=now - timedelta(seconds=1))],
        session_id="sess-1",
        ttl_overrides={"*": 0},
    )
    assert d.hot is False
    assert d.reason == "always_cold"
