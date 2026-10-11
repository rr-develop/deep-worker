"""Regression: for tokens-semantics records without a snapshot, the
``context_used`` fallback MUST prefer ``prompt_tokens`` over
``total_tokens``.

Owner reasoning (2026-08-22): ``prompt_tokens`` is the FULL context the
provider read on the last turn (system + tools schemas + full history +
new user turn — one contiguous blob shipped to the API).  That number
is directly comparable with ``-l`` and matches what the guard checks.

``total_tokens = prompt_tokens + completion_tokens`` is a transport
metric (what the model read AND wrote for that turn).  It is NOT the
"current context size" — the completion part is either already inside
the next turn's ``prompt_tokens`` (if the response is kept in history)
or gone (if trimmed).  Using it as ``context_used`` overstates by
``completion_tokens`` and confuses the operator.

Live evidence — codex agent (``-l 250k`` cap):
  prompt_tokens=505_788, completion_tokens=1_498, total_tokens=507_286
Old UI rendered ``Context: 507.3k tokens`` (via ``total_tokens`` fallback).
Correct rendering: ``Context: 505.8k tokens`` (via ``prompt_tokens``),
``Last turn: 507.3k tokens`` as a separate row.

Providers that never expose ``prompt_tokens`` (e.g. minimax adapter)
keep the old ``total_tokens`` fallback — the second test below pins
that we do NOT regress the 2026-08-17 ``Context: -`` bug for them.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from deep_worker_cli.stats.token_stats import (
    KIND_AGENT,
    STATUS_OK,
    TokenUsageRecorder,
    USAGE_SEMANTICS_TOKENS,
    UsageRecord,
    build_record,
    latest_context_by_kind,
    read_stats_file,
)


class TestPromptTokensPreferredOverTotalTokens:
    def test_context_used_is_prompt_tokens_when_available(
        self, tmp_path: Path
    ) -> None:
        """Live codex payload: prompt_tokens=505_788, total_tokens=507_286.
        Fallback must give context_used=505_788 (not 507_286)."""
        recorder = TokenUsageRecorder(tmp_path)
        cb = recorder.make_usage_event_callback(
            provider="codex",
            kind=KIND_AGENT,
            model_name="gpt-5.6-sol",
        )

        async def _fire() -> None:
            await cb(event="invocation_start")
            await cb(
                event="terminal",
                status=STATUS_OK,
                usage={
                    "input_tokens": 505_788,
                    "output_tokens": 1_498,
                    "cached_input_tokens": 456_704,
                    "total_tokens": 507_286,
                },
                session_id="s1",
                turn_id="item_0",
            )

        asyncio.run(_fire())
        records = read_stats_file(recorder.path).records
        snap = latest_context_by_kind(records)
        assert KIND_AGENT in snap, "no agent snapshot surfaced"
        agent = snap[KIND_AGENT]
        assert agent.context_used == 505_788, (
            f"UI's Context must show the full input the provider read "
            f"(prompt_tokens), NOT the transport total (prompt+completion); "
            f"got context_used={agent.context_used!r}"
        )
        # The per-turn total stays as ``total_tokens`` for the ``Last turn``
        # UI row — the two numbers now have distinct meanings and distinct
        # values, so the operator sees two rows with different data.
        assert agent.total_tokens == 507_286, (
            f"Last turn field must carry the per-turn total (prompt+completion); "
            f"got total_tokens={agent.total_tokens!r}"
        )
        assert agent.usage_semantics == USAGE_SEMANTICS_TOKENS
        # Sanity: NOT the transport total.
        assert agent.context_used != agent.total_tokens, (
            "context_used and total_tokens must diverge when the provider "
            "reports both — that is the whole point of this fix"
        )

    def test_fallback_stays_on_total_tokens_when_prompt_tokens_absent(
        self,
    ) -> None:
        """Defensive branch: an older / minimalist provider that reports
        only ``total_tokens`` and no ``prompt_tokens`` must still get a
        Context reading — the historical fallback
        ``total_tokens → context_used`` MUST stay for that shape,
        otherwise the 2026-08-17 ``Context: -`` regression comes back
        for whichever adapter starts feeding us such rows.

        Constructed via a hand-built UsageRecord because live prod data
        no longer produces this shape (every prod tokens row we grep
        carries prompt_tokens too — checked 2026-08-22).  Still worth
        pinning because the fallback is a contract with future
        adapters."""
        # Hand-built record: tokens semantics, prompt_tokens=None,
        # total_tokens=42_100. Mirrors the shape a minimalist future
        # OpenAI-compat adapter would emit if it only surfaced totals.
        rec = UsageRecord(
            record_id="rec-1",
            event_key=None,
            session_cumulative=None,
            ts="2026-08-22T12:00:00.000Z",
            provider="future_adapter",
            model="",
            kind=KIND_AGENT,
            status=STATUS_OK,
            usage_semantics=USAGE_SEMANTICS_TOKENS,
            provider_turn_id=None,
            provider_session_id=None,
            prompt_tokens=None,
            completion_tokens=None,
            cache_hit_tokens=None,
            cache_miss_tokens=None,
            total_tokens=42_100,
            context_used=None,
            context_size=None,
        )
        snap = latest_context_by_kind([rec])
        agent = snap.get(KIND_AGENT)
        assert agent is not None, "record must produce a snapshot via fallback"
        assert agent.context_used == 42_100, (
            f"minimax-style fallback (no prompt_tokens) must use "
            f"total_tokens for context_used, else the 2026-08-17 "
            f"``Context: -`` bug is back; got {agent.context_used!r}"
        )
        assert agent.total_tokens == 42_100

    def test_zero_prompt_tokens_is_junk_and_falls_through_to_total(
        self,
    ) -> None:
        """``prompt_tokens=0`` on a turn that reports a non-zero total is
        not a measurement — it is a junk row (no provider genuinely reads
        an empty context and still emits 42_100 total tokens).  The
        fallback must treat it as absent and use ``total_tokens``, so the
        operator sees a usable number instead of ``Context: 0 tokens``.

        Pins the ``> 0`` in the branch condition (as opposed to a plain
        ``is not None``) — reviewer request, 2026-08-22.
        """
        rec = UsageRecord(
            record_id="rec-zero",
            event_key=None,
            session_cumulative=None,
            ts="2026-08-22T12:00:00.000Z",
            provider="junk_adapter",
            model="",
            kind=KIND_AGENT,
            status=STATUS_OK,
            usage_semantics=USAGE_SEMANTICS_TOKENS,
            provider_turn_id=None,
            provider_session_id=None,
            prompt_tokens=0,
            completion_tokens=None,
            cache_hit_tokens=None,
            cache_miss_tokens=None,
            total_tokens=42_100,
            context_used=None,
            context_size=None,
        )
        snap = latest_context_by_kind([rec])
        agent = snap.get(KIND_AGENT)
        assert agent is not None, "junk row must still yield a snapshot"
        assert agent.context_used == 42_100, (
            f"prompt_tokens=0 alongside a non-zero total is a junk row; the "
            f"fallback must skip it and use total_tokens, got "
            f"{agent.context_used!r}"
        )

    def test_prompt_tokens_wins_even_when_total_tokens_larger(
        self, tmp_path: Path
    ) -> None:
        """Behavioural guard for the priority: if a future refactor
        swaps the two branches back or re-adds ``total_tokens`` first,
        this test fires immediately.  Distinct values (100k vs 150k) so
        an off-by-one confusion of the fields is impossible.

        Deliberately paired with the structural guard below: this one
        pins the OUTCOME, that one pins the SOURCE ORDER (so a refactor
        that keeps the outcome by accident still cannot silently drop
        the documented priority)."""
        recorder = TokenUsageRecorder(tmp_path)
        cb = recorder.make_usage_event_callback(
            provider="codex", kind=KIND_AGENT, model_name="gpt-5.6-sol",
        )

        async def _fire() -> None:
            await cb(event="invocation_start")
            await cb(
                event="terminal",
                status=STATUS_OK,
                usage={
                    "input_tokens": 100_000,
                    "output_tokens": 50_000,
                    "total_tokens": 150_000,
                },
                session_id="s1",
                turn_id="item_0",
            )

        asyncio.run(_fire())
        records = read_stats_file(recorder.path).records
        snap = latest_context_by_kind(records)
        agent = snap[KIND_AGENT]
        # PRIORITY ORDER: prompt_tokens > total_tokens. Never the reverse.
        assert agent.context_used == 100_000, (
            f"priority regression: latest_context_by_kind picked "
            f"{agent.context_used!r}; must be prompt_tokens (100000), "
            f"NOT total_tokens (150000)"
        )


class TestFallbackPriorityIsStructural:
    """Structural (AST) guard on the fallback branch ORDER.

    The behavioural tests above pin the outcome for the payloads we
    know about.  This one pins the SOURCE ORDER inside
    ``latest_context_by_kind``: ``prompt_tokens`` must be consulted in
    a branch that executes BEFORE the ``total_tokens`` branch.

    Why both: a refactor could keep the current tests green while
    quietly reintroducing ``total_tokens`` as the primary source for
    some payload shape we did not enumerate (e.g. by testing
    ``total_tokens is not None`` first and only falling through to
    ``prompt_tokens`` when the total is absent).  Every prod row today
    carries BOTH fields, so such an inversion would be invisible to a
    value-based test that only feeds rows where the current logic
    happens to agree.
    """

    def test_prompt_tokens_branch_precedes_total_tokens_branch(self) -> None:
        import ast
        import inspect

        from deep_worker_cli.stats import token_stats

        tree = ast.parse(inspect.getsource(token_stats.latest_context_by_kind))

        # Collect, in source order, the line numbers at which each
        # attribute is READ (ast.Load) off a record object.
        prompt_lines: list[int] = []
        total_lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.ctx, ast.Load):
                continue
            if node.attr == "prompt_tokens":
                prompt_lines.append(node.lineno)
            elif node.attr == "total_tokens":
                total_lines.append(node.lineno)

        assert prompt_lines, (
            "latest_context_by_kind no longer reads ``prompt_tokens`` at "
            "all — the documented primary source for the tokens-semantics "
            "context fallback was removed"
        )
        assert total_lines, (
            "latest_context_by_kind no longer reads ``total_tokens`` — the "
            "minimax-style fallback of last resort was removed"
        )

        # The FIRST read of prompt_tokens must come before the first
        # read of total_tokens: that is the priority, expressed
        # structurally and independent of any particular payload.
        assert min(prompt_lines) < min(total_lines), (
            f"fallback priority inverted in source: first ``prompt_tokens`` "
            f"read is at line {min(prompt_lines)}, first ``total_tokens`` "
            f"read at line {min(total_lines)}; ``prompt_tokens`` (the full "
            f"context the provider read) MUST be consulted first, with "
            f"``total_tokens`` only as the last-resort fallback"
        )
