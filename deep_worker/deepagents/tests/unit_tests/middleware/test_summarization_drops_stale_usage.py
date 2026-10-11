"""A preserved tail must not carry the PRE-summary prompt measurement.

Incident 2026-08-30. Summarization worked: the history was
offloaded and replaced by a summary plus a short tail. But the first
message of that tail was an ``AIMessage`` still carrying
``usage_metadata.input_tokens = 450132`` — the size of the prompt that
produced it, measured against the history that summarization had just
thrown away.

``provider_context_tokens`` reads exactly that field as "the context the
provider reported for this history" (``observed_context.py``), so the
context guard saw 450132 against a six-message state, blocked the turn,
and the agent deadlocked: no model call meant no fresh measurement, so
the next attempt read the same stale number. Sixteen identical
``ContextLimitExceeded: estimated 450132`` errors, unchanged while the
history grew.

The stamped ``deep_worker_context`` observation is already immune — it
carries a ``first_id`` validity key and is discarded when the history it
measured no longer exists. ``usage_metadata`` has no such key, so the
measurement has to be dropped at the point the tail is preserved.

Per-turn accounting is unaffected: ``ProvenanceMiddleware`` copies these
numbers into ``response_metadata['deep_worker_usage']`` on the way out of
the model, long before summarization runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from deepagents.middleware.summarization import SummarizationMiddleware

from .test_summarization_middleware import MockBackend, make_mock_model

# The real numbers from the incident.
STALE_INPUT_TOKENS = 450_132


def _history() -> list:
    """A conversation whose LAST exchange will survive as the kept tail."""
    return [
        HumanMessage(content="old question", id="h1"),
        AIMessage(content="old answer", id="a1"),
        HumanMessage(content="recent question", id="h2"),
        AIMessage(
            content="recent answer",
            id="a2",
            usage_metadata={
                "input_tokens": STALE_INPUT_TOKENS,
                "output_tokens": 2_937,
                "total_tokens": STALE_INPUT_TOKENS + 2_937,
            },
        ),
    ]


def _middleware() -> SummarizationMiddleware:
    model = make_mock_model()
    model.ainvoke = AsyncMock(return_value=MagicMock(text="SUMMARY"))
    return SummarizationMiddleware(
        model=model,
        backend=MockBackend(),
        trigger=("messages", 100),
        keep=("messages", 2),
    )


@pytest.mark.asyncio
async def test_preserved_tail_drops_pre_summary_usage_metadata() -> None:
    """The kept tail must not report the size of a discarded history."""
    result = await _middleware().apply_requested_summary("t1", _history())

    assert result.summarized is True
    assert result.replacement_messages is not None

    for msg in result.replacement_messages:
        usage = getattr(msg, "usage_metadata", None)
        assert not (usage or {}).get("input_tokens"), (
            "a preserved message still carries the pre-summary prompt size; "
            "the context guard will read it as the size of the NEW history"
        )


@pytest.mark.asyncio
async def test_guard_sees_no_provider_report_for_the_new_history() -> None:
    """End to end: the very read that caused the deadlock.

    Drives the real summarizer and then the real
    ``provider_context_tokens``. Before the fix this returned 450132 for a
    history of three messages.
    """
    from deep_worker_cli.context_limit.observed_context import (
        provider_context_tokens,
    )

    result = await _middleware().apply_requested_summary("t1", _history())
    assert result.replacement_messages is not None

    reported = provider_context_tokens(result.replacement_messages)

    assert reported != STALE_INPUT_TOKENS, (
        f"the guard read the pre-summary measurement ({reported}) as the "
        "size of the post-summary history — this is the deadlock"
    )
    # UNKNOWN is the correct answer: nothing has measured the NEW history
    # yet. Estimating here is forbidden (owner directive 2026-08-25).
    assert reported is None


def _auto_middleware() -> SummarizationMiddleware:
    """Configured so ``before_model`` summarizes on its own."""
    model = make_mock_model()
    model.invoke = MagicMock(return_value=MagicMock(text="SUMMARY", content="SUMMARY"))
    return SummarizationMiddleware(
        model=model,
        backend=MockBackend(),
        trigger=("messages", 2),
        keep=("messages", 2),
    )


def test_automatic_trigger_also_drops_the_stale_measurement() -> None:
    """The OTHER entrance — review finding, 2026-08-30.

    A tail is chosen in two places: the requested path
    (``_arun_compaction``) and the automatic trigger (``before_model``).
    The first fix cleaned only the requested one, leaving the identical
    deadlock reachable whenever summarization fired on its own — which is
    the path that actually runs in production, since the trigger is
    configured from the context size and nothing disables it.
    """
    state = {"messages": _history()}

    result = _auto_middleware().before_model(state, None)

    assert result is not None, "the trigger should have summarized"
    kept = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
    for msg in kept:
        usage = getattr(msg, "usage_metadata", None)
        assert not (usage or {}).get("input_tokens"), (
            "the automatic path kept the pre-summary prompt size"
        )


def test_automatic_trigger_leaves_no_provider_report() -> None:
    """Same end-to-end read as the requested path, through before_model."""
    from deep_worker_cli.context_limit.observed_context import (
        provider_context_tokens,
    )

    result = _auto_middleware().before_model({"messages": _history()}, None)
    assert result is not None
    kept = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]

    reported = provider_context_tokens(kept)

    assert reported != STALE_INPUT_TOKENS, (
        f"the guard read the pre-summary measurement ({reported}) after an "
        "automatic summarization — the deadlock, through the other entrance"
    )
    assert reported is None


@pytest.mark.asyncio
async def test_provenance_copy_of_the_numbers_survives() -> None:
    """Dropping the field must not cost the per-turn accounting.

    ``ProvenanceMiddleware`` stamps ``deep_worker_usage`` on the way out of
    the model; that copy is what the web UI and token stats read, and it
    has to be untouched by the tail cleanup.
    """
    history = _history()
    history[-1].response_metadata = {
        "deep_worker_usage": {
            "input_tokens": STALE_INPUT_TOKENS,
            "output_tokens": 2_937,
            "source": "api_usage_metadata",
        }
    }

    result = await _middleware().apply_requested_summary("t1", history)
    assert result.replacement_messages is not None

    preserved_usage = [
        (m.response_metadata or {}).get("deep_worker_usage")
        for m in result.replacement_messages
        if (getattr(m, "response_metadata", None) or {}).get("deep_worker_usage")
    ]
    assert preserved_usage, "the provenance copy was destroyed with the raw field"
    assert preserved_usage[0]["input_tokens"] == STALE_INPUT_TOKENS
