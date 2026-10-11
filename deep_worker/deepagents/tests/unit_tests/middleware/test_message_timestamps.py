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


def test_existing_created_at_is_never_rewritten_when_older_than_clock() -> None:
    """Owner R1: real receipt time is immutable even if it breaks monotonicity."""
    existing = "2020-01-01T00:00:00.000000+00:00"
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))
    msg = HumanMessage(content="old", additional_kwargs={CREATED_AT_KEY: existing})

    ensure_message_created_at(msg, clock)

    assert msg.additional_kwargs[CREATED_AT_KEY] == existing
    # Clock still observes the old stamp so NEW stamps sort after it.
    assert clock.next() == "2026-05-28T12:00:00.000000+00:00"


def test_created_at_survives_repeated_before_model_and_offload() -> None:
    """§9.1: N before_model + offload/deserialize must not rewrite created_at."""
    from deepagents.middleware.summarization_history import (
        deserialize_summarized_messages_jsonl,
        serialize_summarized_messages_jsonl,
    )

    existing = "2026-08-01T00:00:00.000000+00:00"
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 8, 19, 15, 0, 0, tzinfo=UTC))
    middleware = MessageTimestampMiddleware(clock=clock)
    msg = HumanMessage(content="keep", additional_kwargs={CREATED_AT_KEY: existing})
    for _ in range(5):
        middleware.before_model({"messages": [msg]})
    assert msg.additional_kwargs[CREATED_AT_KEY] == existing

    blob = serialize_summarized_messages_jsonl([msg])
    restored = deserialize_summarized_messages_jsonl(blob)
    assert restored[0].additional_kwargs[CREATED_AT_KEY] == existing
    blob2 = serialize_summarized_messages_jsonl(restored)
    restored2 = deserialize_summarized_messages_jsonl(blob2)
    assert restored2[0].additional_kwargs[CREATED_AT_KEY] == existing


def test_before_model_does_not_rewrite_existing_stamps() -> None:
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 8, 19, 15, 0, 0, tzinfo=UTC))
    middleware = MessageTimestampMiddleware(clock=clock)
    stamped = [
        HumanMessage(
            content=f"m{i}",
            additional_kwargs={CREATED_AT_KEY: f"2026-08-01T00:00:{i:02d}.000000+00:00"},
        )
        for i in range(5)
    ]
    before = [m.additional_kwargs[CREATED_AT_KEY] for m in stamped]
    middleware.before_model({"messages": stamped})
    after = [m.additional_kwargs[CREATED_AT_KEY] for m in stamped]
    assert after == before


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

    # Formatter is read-only: missing stamps render as "?" and messages stay unstamped.
    assert "[?] Human: hi" in text
    assert "[?] AI:" in text
    assert "[?] Tool call read_file (call_1):" in text
    assert "[?] Tool result read_file (call_1): body" in text
    assert CREATED_AT_KEY not in messages[0].additional_kwargs


def test_stamp_response_terminates_on_endlessly_nested_result() -> None:
    """A value that mints a fresh ``.result`` forever must not recurse forever.

    Regression for a related internal incident analysis (not included here): a
    ``MagicMock`` response satisfied ``getattr(value, "result")`` at every depth,
    so stamping recursed to ``RecursionError``. Formatting that traceback under
    pytest allocated 19.5 GB and locked the host up twice.
    """
    from unittest.mock import MagicMock

    from deepagents.middleware.message_timestamps import _stamp_response

    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))

    # Must return, not raise RecursionError, and not hang.
    assert _stamp_response(MagicMock(), clock) is not None


def test_stamp_response_visits_each_node_of_a_result_cycle_once() -> None:
    """A genuine cycle must be recognised, not merely survived.

    The depth limit alone would also terminate here, so asserting termination
    proves nothing about the cycle guard. What distinguishes them is work done:
    with identity tracking each node is visited once, without it every node is
    re-walked until the depth limit runs out.
    """
    from deepagents.middleware.message_timestamps import _MAX_STAMP_DEPTH, _stamp_response

    visits: list[str] = []

    class Node:
        def __init__(self, name: str) -> None:
            self.name = name
            self._result: object | None = None

        @property
        def result(self) -> object | None:
            visits.append(self.name)
            return self._result

    first, second = Node("first"), Node("second")
    first._result = second
    second._result = first

    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))

    assert _stamp_response(first, clock) is first

    # One probe per distinct node. Without the cycle guard this would run until
    # the depth limit, i.e. _MAX_STAMP_DEPTH probes.
    assert visits == ["first", "second"]
    assert len(visits) < _MAX_STAMP_DEPTH


def test_stamp_response_still_reaches_messages_through_nesting() -> None:
    """The bound must not break ordinary unwrapping."""
    from deepagents.middleware.message_timestamps import _stamp_response

    class Wrapper:
        def __init__(self, result: object) -> None:
            self.result = result

    message = AIMessage(content="hi")
    clock = MonotonicUtcClock(now=lambda: datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC))

    _stamp_response(Wrapper(Wrapper([message])), clock)

    assert message.additional_kwargs[CREATED_AT_KEY] == "2026-05-28T12:00:00.000000+00:00"
