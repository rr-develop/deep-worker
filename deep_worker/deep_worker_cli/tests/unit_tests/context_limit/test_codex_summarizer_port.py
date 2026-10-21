"""Behaviour of ``CodexChatModel``'s SummarizerPort.

The contract test next door proves the two methods EXIST and are async. That is
not enough: a port that exists but dumps history, or that reports success on an
empty summary, is worse than no port at all. These tests pin what it does.

Context: codex is session-stateful, so the middleware refuses the generic
identical-prefix fallback for it. Before this port existed that refusal made
every summarization on codex deterministically fatal — agent_a died with
``SummarizationFailedError: summarization port unavailable on a
session-stateful provider (CodexChatModel)`` on 2026-08-23 and, being already
past its context limit, re-died on every subsequent turn.
"""

from __future__ import annotations

from typing import Any

import pytest

from providers.codex_provider.chat_model import CodexChatModel
from providers.codex_provider.exec_runner import (
    AgentMessageItem,
    ItemCompletedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)


class _FakeQueue:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    async def get(self) -> Any:
        if not self._events:
            raise AssertionError("summary turn consumed more events than provided")
        return self._events.pop(0)


class _FakeHandle:
    def __init__(self, events: list[Any]) -> None:
        self.ainvoke_queue = _FakeQueue(events)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeRunner:
    """Captures the ExecArgs of every spawn so the test can assert on them."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.spawns: list[Any] = []
        self.last_handle: _FakeHandle | None = None

    async def spawn(self, args: Any, **kwargs: Any) -> _FakeHandle:
        self.spawns.append((args, kwargs))
        self.last_handle = _FakeHandle(self._events)
        return self.last_handle


def _ok_events(text: str = "THE SUMMARY") -> list[Any]:
    return [
        ItemCompletedEvent(item=AgentMessageItem(id="i1", text=text)),
        TurnCompletedEvent(usage={"input_tokens": 10, "output_tokens": 2}),
    ]


def _model(runner: _FakeRunner) -> CodexChatModel:
    model = CodexChatModel.__new__(CodexChatModel)
    object.__setattr__(model, "_exec_runner", runner)
    object.__setattr__(model, "_working_dir", "/tmp")
    object.__setattr__(model, "_sandbox_mode", "workspace-write")
    object.__setattr__(model, "_approval_policy", "never")
    object.__setattr__(model, "_model_name", None)
    object.__setattr__(model, "_disabled_features", ())
    object.__setattr__(model, "_config_overrides", ())
    return model


class _Decision:
    def __init__(self, hot: bool) -> None:
        self.hot = hot


class _State:
    def __init__(self, uuid: str | None) -> None:
        self.codex_session_uuid = uuid


class _ClientState:
    def __init__(self, uuid: str | None) -> None:
        self._state = _State(uuid)
        self.reset_calls: list[str] = []

    def get_or_create(self, thread_id: str) -> _State:
        return self._state

    def record_summarization(self, thread_id: str) -> None:
        self.reset_calls.append(thread_id)


@pytest.mark.asyncio
async def test_hot_resumes_the_rollout_and_sends_no_history() -> None:
    """The whole point of the hot path: instruction only, no message dump.

    Codex reloads its own transcript from
    ``CODEX_HOME/sessions/<uuid>.jsonl`` when resumed, so re-sending the
    history would be pure waste — and is exactly the dump the port exists to
    avoid.
    """
    runner = _FakeRunner(_ok_events())
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    result = await model.asummarize(
        thread_id="agent_1",
        messages=[{"role": "user", "content": "SENTINEL_HISTORY"}],
        instruction="Summarize.",
        decision=_Decision(hot=True),
    )

    args, _kwargs = runner.spawns[0]
    assert args.session_uuid == "uuid-live", "hot path must resume the rollout"
    assert args.prompt == "Summarize.", "hot path must send the instruction alone"
    assert "SENTINEL_HISTORY" not in args.prompt
    assert result.path == "hot"
    assert result.text == "THE SUMMARY"
    assert result.provider_session_id == "uuid-live"


@pytest.mark.asyncio
async def test_cold_starts_a_fresh_session_and_includes_the_dump() -> None:
    runner = _FakeRunner(_ok_events())
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    from langchain_core.messages import HumanMessage

    result = await model.asummarize(
        thread_id="agent_1",
        messages=[HumanMessage(content="SENTINEL_HISTORY")],
        instruction="Summarize.",
        decision=_Decision(hot=False),
    )

    args, _kwargs = runner.spawns[0]
    assert args.session_uuid is None, "cold path must NOT resume the live rollout"
    assert "SENTINEL_HISTORY" in args.prompt
    assert result.path == "cold"


@pytest.mark.asyncio
async def test_hot_without_a_session_falls_back_to_cold() -> None:
    """A stale "hot" belief must not summarize an empty session.

    ``decide()`` can report hot from message timestamps alone, while the uuid
    is gone (fresh process, or an earlier reset). Resuming nothing would make
    Codex summarize an empty transcript and return a confident, empty-headed
    summary.
    """
    runner = _FakeRunner(_ok_events())
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState(None))

    from langchain_core.messages import HumanMessage

    result = await model.asummarize(
        thread_id="agent_1",
        messages=[HumanMessage(content="SENTINEL_HISTORY")],
        instruction="Summarize.",
        decision=_Decision(hot=True),
    )

    args, _kwargs = runner.spawns[0]
    assert result.path == "cold"
    assert args.session_uuid is None
    assert "SENTINEL_HISTORY" in args.prompt


@pytest.mark.asyncio
async def test_summary_turn_carries_no_proxy_mcp_server() -> None:
    """The summary turn must not be able to call deep-worker tools."""
    runner = _FakeRunner(_ok_events())
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    await model.asummarize(
        thread_id="t",
        messages=[],
        instruction="Summarize.",
        decision=_Decision(hot=True),
    )

    args, _kwargs = runner.spawns[0]
    assert args.mcp_server_name is None
    assert args.mcp_server_port is None


@pytest.mark.asyncio
async def test_empty_summary_raises_rather_than_returning_it() -> None:
    """POLICIES §12: failure is fatal and loud, never a plausible-looking value.

    The caller deletes the history this summary replaces. An empty summary
    returned as success destroys the conversation silently.
    """
    runner = _FakeRunner(_ok_events(text="   "))
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    with pytest.raises(RuntimeError, match="empty summary"):
        await model.asummarize(
            thread_id="t",
            messages=[],
            instruction="Summarize.",
            decision=_Decision(hot=True),
        )


@pytest.mark.asyncio
async def test_failed_turn_raises() -> None:
    runner = _FakeRunner([TurnFailedEvent(error_message="boom")])
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    with pytest.raises(RuntimeError, match="boom"):
        await model.asummarize(
            thread_id="t",
            messages=[],
            instruction="Summarize.",
            decision=_Decision(hot=True),
        )


@pytest.mark.asyncio
async def test_handle_is_closed_even_when_the_turn_fails() -> None:
    """Otherwise every failed summarization leaks a codex subprocess."""
    runner = _FakeRunner([TurnFailedEvent(error_message="boom")])
    model = _model(runner)
    object.__setattr__(model, "_client_state", _ClientState("uuid-live"))

    with pytest.raises(RuntimeError):
        await model.asummarize(
            thread_id="t",
            messages=[],
            instruction="Summarize.",
            decision=_Decision(hot=True),
        )

    assert runner.last_handle is not None
    assert runner.last_handle.closed is True


@pytest.mark.asyncio
async def test_reset_delegates_to_record_summarization() -> None:
    """Clears the uuid + sent_count and persists, so a restart right after a
    summarization cannot reload the stale pre-summary rollout."""
    runner = _FakeRunner(_ok_events())
    model = _model(runner)
    client_state = _ClientState("uuid-live")
    object.__setattr__(model, "_client_state", client_state)

    await model.areset_after_summary(thread_id="agent_1")
    await model.areset_after_summary(thread_id="agent_1")  # idempotent

    assert client_state.reset_calls == ["agent_1", "agent_1"]
