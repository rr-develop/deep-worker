"""Unit tests for ProvenanceMiddleware origin/usage stamps."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.context_limit.hot_cache import ORIGIN_KEY, USAGE_KEY
from deep_worker_cli.middleware.provenance import (
    ProvenanceMiddleware,
    resolve_provider_name,
    stamp_ai_message,
)


def test_resolve_provider_name_skips_unknown() -> None:
    assert resolve_provider_name(None, "unknown", "  ", "deepseek") == "deepseek"
    assert resolve_provider_name("unknown", None) is None
    assert resolve_provider_name("Grok") == "grok"


def test_stamp_does_not_write_unknown_provider() -> None:
    msg = AIMessage(content="one")
    stamp_ai_message(
        msg,
        request_messages=[],
        provider_name="unknown",
        model_name="deepseek-v4-flash",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=10,
    )
    assert ORIGIN_KEY not in msg.response_metadata
    assert USAGE_KEY in msg.response_metadata


def test_stamp_uses_cli_provider_not_unknown() -> None:
    msg = AIMessage(content="one")
    stamp_ai_message(
        msg,
        request_messages=[],
        provider_name="deepseek",
        model_name="deepseek-v4-flash",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=10,
    )
    assert msg.response_metadata[ORIGIN_KEY]["provider"] == "deepseek"


def test_origin_stamped_on_first_ai_only() -> None:
    first = AIMessage(content="one")
    stamp_ai_message(
        first,
        request_messages=[HumanMessage(content="hi")],
        provider_name="claude-code",
        model_name="fable",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=10,
    )
    assert first.response_metadata[ORIGIN_KEY]["provider"] == "claude-code"
    second = AIMessage(content="two")
    stamp_ai_message(
        second,
        request_messages=[HumanMessage(content="hi"), first],
        provider_name="claude-code",
        model_name="fable",
        model_id=first.response_metadata[ORIGIN_KEY].get("model_id"),
        request_started_at="2026-08-19T15:00:01+00:00",
        latency_ms=11,
    )
    assert ORIGIN_KEY not in second.response_metadata
    assert USAGE_KEY in second.response_metadata


def test_origin_stamped_again_on_provider_change() -> None:
    first = AIMessage(content="one")
    stamp_ai_message(
        first,
        request_messages=[],
        provider_name="claude-code",
        model_name="fable",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=10,
    )
    failover = AIMessage(content="two")
    stamp_ai_message(
        failover,
        request_messages=[first],
        provider_name="grok",
        model_name="grok-4",
        request_started_at="2026-08-19T15:01:00+00:00",
        latency_ms=12,
    )
    assert failover.response_metadata[ORIGIN_KEY]["provider"] == "grok"


def test_usage_copies_context_observation() -> None:
    msg = AIMessage(
        content="ok",
        response_metadata={"deep_worker_context": {"used": 481811, "size": 1_000_000}},
    )
    stamp_ai_message(
        msg,
        request_messages=[],
        provider_name="claude-code",
        model_name="fable",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=42,
        provider_session_id="f44ee866",
    )
    usage = msg.response_metadata[USAGE_KEY]
    assert usage["context_used"] == 481811
    assert usage["context_size"] == 1_000_000
    assert usage["provider_session_id"] == "f44ee866"
    assert usage["latency_ms"] == 42
    assert usage["source"] == "acp_usage_update"


def test_middleware_wraps_handler() -> None:
    mw = ProvenanceMiddleware(provider_name="deepseek", model_name="deepseek-chat")
    request = SimpleNamespace(messages=[HumanMessage(content="q")])
    out = mw.wrap_model_call(request, lambda _r: AIMessage(content="a"))
    assert isinstance(out, AIMessage)
    assert out.response_metadata[ORIGIN_KEY]["provider"] == "deepseek"
    assert USAGE_KEY in out.response_metadata


def test_request_payload_does_not_contain_deep_worker_stamps() -> None:
    """§9.1: provider request body must not carry deep_worker_* stamps."""
    import json

    from langchain_core.messages import convert_to_openai_messages

    from deepagents.middleware.message_timestamps import (
        CREATED_AT_KEY,
        format_messages_with_timestamps,
        MonotonicUtcClock,
    )

    first = AIMessage(content="one")
    stamp_ai_message(
        first,
        request_messages=[HumanMessage(content="hi")],
        provider_name="deepseek",
        model_name="deepseek-chat",
        request_started_at="2026-08-19T15:00:00+00:00",
        latency_ms=10,
    )
    first.additional_kwargs[CREATED_AT_KEY] = "2026-08-19T15:00:00.000000+00:00"
    payload = convert_to_openai_messages(
        [HumanMessage(content="hi"), first]
    )
    dumped = json.dumps(payload)
    assert "deep_worker_" not in dumped
    clock = MonotonicUtcClock()
    acp_text = format_messages_with_timestamps([HumanMessage(content="hi"), first], clock)
    assert "deep_worker_" not in acp_text
