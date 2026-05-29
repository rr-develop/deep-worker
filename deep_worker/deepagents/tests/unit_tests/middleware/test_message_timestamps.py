from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deepagents.middleware.message_timestamps import (
    CREATED_AT_KEY,
    MessageTimestampMiddleware,
    MonotonicUtcClock,
    ensure_message_created_at,
    format_messages_with_timestamps,
)


def test_monotonic_clock_emits_utc_and_strictly_increases() -> None:
    base = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
    clock = MonotonicUtcClock(now=lambda: base)

    first = clock.next()
    second = clock.next()

    assert first == "2026-05-28T12:00:00.000000+00:00"
    assert second == "2026-05-28T12:00:00.000001+00:00"
    assert datetime.fromisoformat(second) > datetime.fromisoformat(first)


def test_ensure_message_created_at_stamps_messages_and_tool_calls() -> None:
    values = iter([
        datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC),
        datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC),
    ])
    clock = MonotonicUtcClock(now=lambda: next(values))
    msg = AIMessage(
        content="running grep",
        id="a1",
        tool_calls=[{"id": "call_1", "name": "grep", "args": {"pattern": "x"}}],
    )

    ensure_message_created_at(msg, clock)

    assert msg.additional_kwargs[CREATED_AT_KEY] == "2026-05-28T12:00:00.000000+00:00"
    assert msg.additional_kwargs["tool_call_created_at"] == {
        "call_1": msg.additional_kwargs[CREATED_AT_KEY]
    }


def test_existing_created_at_is_preserved_when_monotonic() -> None:
    existing = "2026-05-28T12:00:05.000000+00:00"
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))
    msg = HumanMessage(content="hello", additional_kwargs={CREATED_AT_KEY: existing})

    ensure_message_created_at(msg, clock)

    assert msg.additional_kwargs[CREATED_AT_KEY] == existing
    assert clock.next() == "2026-05-28T12:00:05.000001+00:00"


def test_middleware_stamps_model_and_tool_outputs() -> None:
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))
    middleware = MessageTimestampMiddleware(clock=clock)

    ai = middleware.wrap_model_call(None, lambda _request: AIMessage(content="answer"))
    tool = middleware.wrap_tool_call(None, lambda _request: ToolMessage(content="ok", tool_call_id="call_1"))
    command = middleware.wrap_tool_call(
        None,
        lambda _request: Command(update={"messages": [ToolMessage(content="cmd", tool_call_id="call_2")]}),
    )

    assert ai.additional_kwargs[CREATED_AT_KEY] == "2026-05-28T12:00:00.000000+00:00"
    assert tool.additional_kwargs[CREATED_AT_KEY] == "2026-05-28T12:00:00.000001+00:00"
    assert command.update["messages"][0].additional_kwargs[CREATED_AT_KEY] == "2026-05-28T12:00:00.000002+00:00"


def test_format_messages_with_timestamps_includes_each_message_and_tool_call() -> None:
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "read_file", "args": {"file_path": "/tmp/a"}}]),
        ToolMessage(content="body", tool_call_id="call_1", name="read_file"),
    ]

    text = format_messages_with_timestamps(messages, clock)

    assert "[2026-05-28T12:00:00.000000+00:00] Human: hi" in text
    assert "[2026-05-28T12:00:00.000001+00:00] AI:" in text
    assert "[2026-05-28T12:00:00.000001+00:00] Tool call read_file (call_1):" in text
    assert "[2026-05-28T12:00:00.000002+00:00] Tool result read_file (call_1): body" in text
