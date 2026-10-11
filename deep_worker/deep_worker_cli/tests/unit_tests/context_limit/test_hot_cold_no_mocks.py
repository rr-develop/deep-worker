"""Integration-style tests without mocks: real stamps + detector.

Not a live LLM call — exercises the real serialize/deserialize/decide
path on real LangChain messages.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware.summarization_history import (
    deserialize_summarized_messages_jsonl,
    serialize_summarized_messages_jsonl,
)
from deep_worker_cli.context_limit.hot_cache import decide
from deep_worker_cli.middleware.provenance import stamp_ai_message


def test_origin_survives_offload_and_drives_cold_after_ttl() -> None:
    received = datetime(2026, 8, 19, 11, 7, 13, tzinfo=UTC)
    ai = AIMessage(content="hello")
    stamp_ai_message(
        ai,
        request_messages=[HumanMessage(content="hi")],
        provider_name="claude-code",
        model_name="fable",
        request_started_at=received.isoformat(),
        latency_ms=100,
        provider_session_id="f44ee866",
    )
    ai.response_metadata["deep_worker_usage"]["received_at"] = received.isoformat()

    blob = serialize_summarized_messages_jsonl(
        [HumanMessage(content="hi"), ai],
        summarized_at="2026-08-19T15:16:00+00:00",
        thread_id="rg1_worker",
        reason="cold",
        provider_session_id="f94ef7a6",
    )
    header = blob.splitlines()[0]
    assert '"kind":"deepagents.summarization_history.segment"' in header
    assert '"provider": "claude-code"' in header or '"provider":"claude-code"' in header

    restored = deserialize_summarized_messages_jsonl(blob)
    assert restored[1].response_metadata["deep_worker_origin"]["provider"] == "claude-code"
    assert restored[1].response_metadata["deep_worker_usage"]["context_used"] is None

    now = datetime(2026, 8, 19, 15, 14, tzinfo=UTC)
    decision = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=restored,
        session_id="f44ee866",
    )
    assert decision.hot is False
    assert decision.reason.startswith("ttl_expired")


def test_hot_when_last_reply_is_recent() -> None:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    ai = AIMessage(content="ok")
    stamp_ai_message(
        ai,
        request_messages=[],
        provider_name="claude-code",
        model_name="fable",
        request_started_at=(now - timedelta(seconds=30)).isoformat(),
        latency_ms=5,
    )
    ai.response_metadata["deep_worker_usage"]["received_at"] = (
        now - timedelta(seconds=30)
    ).isoformat()
    d = decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=[ai],
        session_id="live",
    )
    assert d.hot is True
    assert d.reason == "hot"
