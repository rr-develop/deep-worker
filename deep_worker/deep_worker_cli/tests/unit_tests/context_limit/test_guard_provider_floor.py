"""The guard must not go blind on providers that keep history in the CLI.

Measured on the live agent_a (grok, `-l 250000`), 2026-08-25:

  * the LangGraph state at guard time holds ONE message — the turn's new
    ``HumanMessage``. The conversation lives inside the grok CLI session,
    not in our message list (checkpoint ``1f1a0af1-db70``: ``n=2`` after the
    turn, ``n=0`` for every intermediate write);
  * therefore no ``deep_worker_context`` stamp is ever anchored to it: the
    stamp written at the end of turn N carries ``first_id=b50f955b``, while
    the state the guard sees on turn N+1 starts at ``4d10180b``;
  * so ``provider_context_tokens`` is permanently ``None`` and the guard
    logged "no provider context report ... passing the turn through
    UNGUARDED" while the provider itself reported 275704 against a 250000
    limit — 110% of the limit, unguarded.

The summarizer already survives this: ``provider_context_token_counter``
takes a ``floor`` reading the provider's OWN persisted gauge
(``acp_session_state.json``: ``last_context_used``). POLICIES §13 blesses
that number explicitly — "A persisted provider number is still the
provider's number, not an estimate". The guard was simply never given the
same source, and that asymmetry is the whole defect: one consumer of the
same fact can see it and the other cannot.

⚠ NOT an estimate. The floor is a callable supplied by the wiring and its
value comes from the provider. These tests pin that the guard NEVER
computes a number: with no floor and no stamp the answer stays UNKNOWN.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.context_limit.guard import ContextLimitGuardMiddleware
from deep_worker_cli.context_limit.observed_context import (
    annotate_context_observation,
)


class _Request:
    """Minimal stand-in for the middleware request object."""

    def __init__(self, messages: list) -> None:
        self.messages = messages

    def override(self, **_kwargs):
        return self


def _guard(limit: int | None = 250_000, floor=None) -> ContextLimitGuardMiddleware:
    return ContextLimitGuardMiddleware(
        context_limit=limit, model_name="grok", provider_floor=floor
    )


class TestGuardReadsThePersistedProviderGauge:
    """A CLI-session provider reports through the gauge, not the messages."""

    def test_floor_is_used_when_no_stamp_is_anchored(self) -> None:
        """RED: today this request is UNGUARDED at 110% of the limit."""
        guard = _guard(floor=lambda: 275_704)
        request = _Request([HumanMessage(content="next task", id="4d10180b")])

        decision = guard._decide(request)

        assert decision.band == "block", (
            f"band={decision.band}: the guard ignored the provider's own "
            "persisted gauge (275704 against a 250000 limit) and passed the "
            "turn through"
        )

    def test_a_live_stamp_beats_the_persisted_floor(self) -> None:
        """The floor is the fallback, never the winner.

        It is by construction older than a stamp anchored to THIS history —
        after a summarization it can still describe the collapsed session
        for a moment (``record_summarization`` nulls it, but the ordering is
        not ours to assume). A stamp on the request wins.
        """
        first = HumanMessage(content="q", id="m1")
        answer = AIMessage(content="a", id="m2")
        annotate_context_observation(answer, used=9_000, size=None, first_id="m1")
        guard = _guard(floor=lambda: 275_704)

        decision = guard._decide(_Request([first, answer]))

        assert decision.band == "green", (
            f"band={decision.band}: the stale floor overrode a live stamp of "
            "9000 tokens and would block an agent that has just summarized"
        )

    def test_no_floor_and_no_stamp_stays_unknown(self, caplog) -> None:
        """POLICIES §13: UNKNOWN is a real answer. The guard must not guess."""
        guard = _guard(floor=None)
        with caplog.at_level(logging.WARNING):
            decision = guard._decide(_Request([HumanMessage(content="q", id="m1")]))

        assert decision.band == "green"
        assert any(
            "no provider context report" in r.getMessage() for r in caplog.records
        ), "the UNGUARDED pass-through must stay loud"

    def test_a_broken_floor_is_unknown_not_a_number(self) -> None:
        """A floor that raises must not become a fabricated count."""

        def boom() -> int:
            raise RuntimeError("state file unreadable")

        guard = _guard(floor=boom)
        decision = guard._decide(_Request([HumanMessage(content="q", id="m1")]))
        assert decision.band == "green"

    @pytest.mark.parametrize("bad", [None, 0, -5, True, "275704", 12.5])
    def test_only_a_positive_int_counts_as_a_report(self, bad) -> None:
        """``True`` is an int in Python; a string is a shape error. Neither
        is a token count, and coercing either would be inventing one."""
        guard = _guard(floor=lambda: bad)
        decision = guard._decide(_Request([HumanMessage(content="q", id="m1")]))
        assert decision.band == "green"

    def test_the_floor_drives_every_band(self) -> None:
        """Not just Block: the amber hint and the red tool message must fire
        off the same source, or a CLI-session agent gets no warning at all
        before it is blocked."""
        for used, expected in ((100_000, "green"), (220_000, "amber"),
                               (240_000, "red"), (275_704, "block")):
            guard = _guard(floor=lambda u=used: u)
            decision = guard._decide(_Request([HumanMessage(content="q", id="m1")]))
            assert decision.band == expected, (
                f"used={used} classified as {decision.band}, expected {expected}"
            )

    def test_no_limit_still_short_circuits(self) -> None:
        """Without ``-l`` the guard is off and must not even call the floor."""
        calls: list[int] = []

        def counting() -> int:
            calls.append(1)
            return 999_999

        guard = _guard(limit=None, floor=counting)
        decision = guard._decide(_Request([HumanMessage(content="q", id="m1")]))
        assert decision.band == "green"
        assert not calls, "the guard consulted the provider gauge with no limit set"


class TestWiring:
    """The agent factory must hand the guard the same source the summarizer
    already uses — otherwise the fix is a parameter nobody passes."""

    def test_create_cli_agent_passes_a_provider_floor(self) -> None:
        import ast
        import inspect
        import textwrap

        from deep_worker_cli import agent as agent_mod

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(agent_mod.create_cli_agent))
        )
        passed = [
            kw
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ContextLimitGuardMiddleware"
            for kw in node.keywords
            if kw.arg == "provider_floor"
        ]
        assert passed, (
            "create_cli_agent builds the guard without provider_floor: a "
            "grok/ACP agent stays UNGUARDED no matter what the guard can do"
        )

    def test_the_guard_and_the_summarizer_share_one_source(self) -> None:
        """Two readers of the same fact must not drift apart again."""
        import ast
        import inspect
        import textwrap

        from deep_worker_cli import agent as agent_mod

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(agent_mod.create_cli_agent))
        )
        guard_arg = None
        summarizer_arg = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            for kw in node.keywords:
                if name == "ContextLimitGuardMiddleware" and kw.arg == "provider_floor":
                    guard_arg = getattr(kw.value, "id", None)
                if name == "provider_context_token_counter" and kw.arg == "floor":
                    summarizer_arg = getattr(kw.value, "id", None)
        assert guard_arg is not None and guard_arg == summarizer_arg, (
            f"guard floor={guard_arg!r}, summarizer floor={summarizer_arg!r} — "
            "the two consumers must read the same provider gauge"
        )
