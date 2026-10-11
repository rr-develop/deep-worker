"""Context size comes from the PROVIDER. Never from an estimate.

Owner directive 2026-08-25, restated after it was violated in production:

    "I HAVE ALWAYS DEMANDED HONEST DATA FROM THE PROVIDER AND IT IS EXACTLY
    THAT DATA THAT MUST BE USED. AT THE MOMENT THERE ARE NO EXCEPTIONS. Any
    fallback requires my direct and explicit consent, and I have not given it."

What this replaces. The guard's own docstring used to say it is
"provider-independent: it does NOT consult usage_metadata", and both the
guard and the summarizer ran on a chars-per-token table. Measured cost on
a production agent, 2026-08-25: the estimator claimed 214613 tokens against ``-l
250000`` (86%, amber) while codex was reading 90k — a 2.4x fiction. The
amber band prepends a hint SystemMessage; the codex provider then recorded
that inflated length as ``sent_count``, and on the next turn
``sent_count > len(messages)`` fired its summarization detector, dropping a
warm codex session and restarting its context from zero. Nobody had
summarized anything. The same class of failure is on record twice before
(a related internal design document, not included here: heuristic 151k vs
real 468k — "the status was right; the summarizer was blind").

The rule now:

* the only admissible number is one the provider reported — the same number
  the status bar shows;
* when no provider number exists yet, the answer is UNKNOWN. Not an
  estimate, not a guess: nothing decides on fiction. The guard passes the
  turn through and says so out loud; the summarizer does not fire;
* every provider must report. A provider that cannot is a defect to fix in
  that provider, not a licence to estimate.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_worker_cli.context_limit.observed_context import (
    annotate_context_observation,
    provider_context_tokens,
)


# The identity of a history is the id of its first NON-system message: the
# factory prepends the system prompt and the guard may prepend a hint, and
# neither is state (see observed_context.first_state_message_id).
FIRST_STATE_ID = "m1"


def _history() -> list:
    return [
        SystemMessage(content="sys", id="m0"),
        HumanMessage(content="hello", id=FIRST_STATE_ID),
    ]


def _stamped(
    used: int, *, first_id: str = FIRST_STATE_ID, size: int | None = None
) -> AIMessage:
    msg = AIMessage(content="ok", id="a1")
    annotate_context_observation(msg, used=used, size=size, first_id=first_id)
    return msg


def _stamped_raw(used: int, *, first_id: str = FIRST_STATE_ID) -> AIMessage:
    """A stamp written by hand — for values the annotator rejects."""
    return AIMessage(
        content="ok",
        id="a1",
        response_metadata={
            "deep_worker_context": {"used": used, "size": None, "first_id": first_id}
        },
    )


class TestProviderNumberIsTheOnlySource:
    def test_acp_style_stamp_is_used_verbatim(self) -> None:
        msgs = [*_history(), _stamped(90_000)]
        assert provider_context_tokens(msgs) == 90_000

    def test_usage_metadata_counts_as_a_provider_report(self) -> None:
        """Standard API providers (deepseek/openai/...) report the prompt
        size of every call in ``usage_metadata.input_tokens``. That IS the
        context the provider read; ignoring it and estimating instead was
        the same lie in a different place."""
        msg = AIMessage(
            content="ok",
            id="a1",
            usage_metadata={
                "input_tokens": 41_234,
                "output_tokens": 10,
                "total_tokens": 41_244,
            },
        )
        assert provider_context_tokens([*_history(), msg]) == 41_234

    def test_latest_report_wins(self) -> None:
        msgs = [
            *_history(),
            _stamped(10_000),
            HumanMessage(content="more", id="m2"),
            _stamped(25_000),
        ]
        assert provider_context_tokens(msgs) == 25_000

    def test_no_report_means_unknown_not_a_guess(self) -> None:
        assert provider_context_tokens(_history()) is None

    def test_a_stamp_for_a_different_history_is_not_ours(self) -> None:
        """A stamp carries the id of the first message of the history it
        measured. After summarization the history is replaced, so an old
        stamp measures a conversation that no longer exists."""
        stale = _stamped(500_000, first_id="OLD")
        assert provider_context_tokens([*_history(), stale]) is None

    def test_zero_and_negative_reports_are_not_reports(self) -> None:
        assert provider_context_tokens([*_history(), _stamped_raw(0)]) is None
        assert provider_context_tokens([*_history(), _stamped_raw(-5)]) is None
        msg = AIMessage(
            content="x",
            id="a1",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 1,
                "total_tokens": 1,
            },
        )
        assert provider_context_tokens([*_history(), msg]) is None


class TestNoEstimatorOnTheDecisionPath:
    """The estimator must not be reachable from the guard or the summarizer.

    Asserted structurally: as long as the module can be imported and called
    from those paths, the next incident is one refactor away."""

    def test_guard_does_not_import_the_estimator(self) -> None:
        import inspect

        from deep_worker_cli.context_limit import guard

        src = inspect.getsource(guard)
        assert "estimate_request_tokens" not in src
        assert "heuristic" not in src.lower()

    def test_observed_context_does_not_import_the_estimator(self) -> None:
        import inspect

        from deep_worker_cli.context_limit import observed_context

        src = inspect.getsource(observed_context)
        assert "estimate_request_tokens" not in src
        assert "heuristic" not in src.lower()

    def test_the_estimator_module_is_gone(self) -> None:
        import importlib.util

        assert (
            importlib.util.find_spec("deep_worker_cli.context_limit.token_estimator")
            is None
        ), (
            "token_estimator still exists — a chars-per-token table that "
            "outranked the provider's own count is exactly what was ordered "
            "removed, not merely bypassed."
        )

    def test_agent_builds_a_provider_only_counter(self) -> None:
        import inspect

        from deep_worker_cli import agent as agent_mod

        src = inspect.getsource(agent_mod)
        assert "heuristic_token_counter_for" not in src
        assert "provider_context_token_counter" in src


class TestSummarizerTrigger:
    """``SummarizationMiddleware`` takes a ``token_counter``; ours reports
    the provider number, or 0 when nothing has been reported — 0 keeps the
    trigger silent instead of firing on a number nobody measured."""

    def test_counter_returns_the_provider_number(self) -> None:
        from deep_worker_cli.context_limit.observed_context import (
            provider_context_token_counter,
        )

        counter = provider_context_token_counter()
        assert counter([*_history(), _stamped(123_456)]) == 123_456

    def test_counter_is_zero_when_unknown(self) -> None:
        from deep_worker_cli.context_limit.observed_context import (
            provider_context_token_counter,
        )

        counter = provider_context_token_counter()
        assert counter(_history()) == 0

    def test_persisted_provider_floor_is_still_provider_data(self) -> None:
        """After a restart the checkpoint may hold no stamp while the
        provider's own state does (production agent, 2026-08-19: the CLI held 468k
        and the summarizer saw nothing). A persisted PROVIDER reading is not
        an estimate — it is the same number, read from where it was kept."""
        from deep_worker_cli.context_limit.observed_context import (
            provider_context_token_counter,
        )

        counter = provider_context_token_counter(floor=lambda: 468_000)
        assert counter(_history()) == 468_000
        # A live report always wins over the persisted one.
        assert counter([*_history(), _stamped(12_000)]) == 12_000


class _FakeRequest:
    """The duck-typed shape the guard actually reads."""

    def __init__(self, messages):
        self.messages = messages
        self.system_message = None
        self.tools = []

    def override(self, **kwargs):
        new = _FakeRequest(self.messages)
        new.tools = self.tools
        new.system_message = self.system_message
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


class TestGuardDecidesOnProviderNumbers:
    def _band(self, messages, limit: int = 100_000) -> str:
        from deep_worker_cli.context_limit.guard import (
            ContextLimitGuardMiddleware,
        )

        guard = ContextLimitGuardMiddleware(context_limit=limit)
        return guard._decide(_FakeRequest(messages)).band

    def test_bands_use_the_provider_number(self) -> None:
        assert self._band([*_history(), _stamped(50_000)]) == "green"
        assert self._band([*_history(), _stamped(88_000)]) == "amber"
        assert self._band([*_history(), _stamped(96_000)]) == "red"
        assert self._band([*_history(), _stamped(120_000)]) == "block"

    def test_usage_metadata_drives_the_bands_too(self) -> None:
        """A standard API provider reports in ``usage_metadata``; that is a
        provider number and must decide exactly like an ACP stamp."""
        msg = AIMessage(
            content="ok",
            id="a1",
            usage_metadata={
                "input_tokens": 96_000,
                "output_tokens": 5,
                "total_tokens": 96_005,
            },
        )
        assert self._band([*_history(), msg]) == "red"

    def test_unknown_context_never_blocks_and_never_guesses(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No provider number yet (first turn of a session, or a provider
        that has not reported). Estimating here is forbidden; blocking a
        working agent over a number nobody has is worse. Pass through — and
        say so, so a provider that never reports becomes visible instead of
        silently unguarded."""
        with caplog.at_level(logging.WARNING):
            band = self._band(_history())
        assert band == "green"
        assert "no provider" in caplog.text.lower()

    def test_a_huge_untracked_history_is_still_not_estimated(self) -> None:
        """The old code charged ~1M chars as tokens and would have blocked.
        Without a provider report there is nothing to decide on."""
        big = [
            SystemMessage(content="x" * 1_000_000, id="m0"),
            HumanMessage(content="y" * 1_000_000, id=FIRST_STATE_ID),
        ]
        assert self._band(big) == "green"

    def test_a_stale_stamp_does_not_decide(self) -> None:
        """After summarization the history is new; the old stamp measured a
        conversation that no longer exists and must not blow the guard up."""
        assert self._band([*_history(), _stamped(999_999, first_id="OLD")]) == "green"
