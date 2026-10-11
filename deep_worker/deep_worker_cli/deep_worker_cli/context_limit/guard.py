"""Pre-call context-limit guard middleware.

See ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md`` §3.2.

This is the safety net for the rg1_worker 2026-08-18 bug: an explicit
``-l L`` must never be exceeded, regardless of whether the provider
emits ``usage_metadata`` (claude-code often does not). The guard runs
BEFORE every model invocation and decides which of four bands the
upcoming request falls into:

* **Green** (``< 0.85 × L``): pass through, no action.
* **Amber** (``0.85 × L ≤ usage < 0.95 × L``): inject a one-shot
  SystemMessage hint that tells the model "you're approaching the
  limit, consider invoking ``compact_history``". The model call
  proceeds; one voluntary turn to compact.
* **Red** (``0.95 × L ≤ usage ≤ L``): inject a structured
  ``ToolMessage`` with ``error="context_overflow_warning"`` into the
  conversation; the model call **proceeds** on this turn so the model
  sees the warning AND the result of its decision on the same turn.
  If the next request is still in Red, Block will trip.
* **Block** (``usage > L``): raise :class:`ContextLimitExceeded`. The
  agent-process catches this and transitions to ``IDLE`` (per the
  no-agent-error-state policy). NO LLM call is made. NO auto-restart.

The guard decides on the number the PROVIDER reported — the same number
the status bar shows — and on nothing else. Owner directive 2026-08-25:

    "I HAVE ALWAYS DEMANDED HONEST DATA FROM THE PROVIDER, AND IT IS
    EXACTLY THAT DATA THAT MUST BE USED. AT THE MOMENT THERE ARE NO
    EXCEPTIONS."

This docstring used to claim the opposite ("provider-independent: it does
NOT consult usage_metadata"), and the chars-per-token estimator behind that
claim cost a production incident on 2026-08-25: it reported 214613 tokens
against ``-l 250000`` (86%, amber) while codex was reading 90k. The amber
hint it injected then tripped the codex provider's shrink detector and
dropped a warm session. Estimation is not a safety net here; it is a
second, louder source of wrong answers.

When no provider number exists yet (the first call of a session, or a
provider that has not reported one), the context is UNKNOWN. The guard then
passes the turn through and logs it: guessing is forbidden, and blocking a
working agent over a number nobody has measured would be worse. A provider
that never reports is a defect in that provider — the log line is how it
becomes visible instead of silently unguarded.
"""

from __future__ import annotations

from collections.abc import Callable

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

from deep_worker_cli.context_limit.observed_context import provider_context_tokens


@dataclass(frozen=True)
class _Decision:
    """Result of ``ContextLimitGuardMiddleware._decide``.

    Pure data — carries the band classification and the (possibly
    modified) request to forward to the LLM handler. I/O-free: no
    call to the handler, no exception raising. Both
    ``wrap_model_call`` (sync) and ``awrap_model_call`` (async) own
    the I/O themselves so neither path carries duplicated control
    flow.

    Attributes:
        band: One of ``"green"``, ``"amber"``, ``"red"``, ``"block"``.
            Green/Amber/Red forward to the handler with a (possibly)
            modified request. Block raises ``ContextLimitExceeded`` —
            the entry point that owns the raise.
        new_request: The request to pass to the LLM handler. Equals
            the input request unchanged for Green; carries the hint
            (Amber) or overflow ToolMessage (Red) appended otherwise.
        estimated: Estimated token count for the request. Used by the
            entry point when raising ``ContextLimitExceeded`` so the
            caller sees the value that triggered the Block.
    """

    band: str
    new_request: Any
    estimated: int = 0

logger = logging.getLogger(__name__)

# Band thresholds as fractions of the context limit. Conservative
# numbers chosen so the guard enters Amber *before* the LangChain
# SummarizationMiddleware's trigger (which fires at 0.85). The guard
# owns the FIRST alarm; the summarizer is the second.
_GREEN_FRACTION = 0.85   # < this → pass through
_AMBER_FRACTION = 0.95   # < this → inject hint (between green and amber)

# Tool-message marker keys (kept here as constants so the tool-call
# surface can grep for them without string-matching the content).
CONTEXT_GUARD_MARKER = "deep_worker_context_guard"
CONTEXT_OVERFLOW_ERROR = "context_overflow_warning"


class ContextLimitExceeded(Exception):
    """Raised by the guard when an upcoming request would exceed the limit.

    The agent-process catches this and transitions to ``IDLE``; the
    operator decides what to do next. Auto-restart is intentionally
    NOT performed (it would re-load the same bloated history — see
    design §3.2.2 "Block").
    """

    def __init__(
        self,
        *,
        context_limit: int,
        estimated_tokens: int,
        model_family: str,
    ) -> None:
        self.context_limit = context_limit
        self.estimated_tokens = estimated_tokens
        self.model_family = model_family
        super().__init__(
            f"Context-limit exceeded: estimated {estimated_tokens} tokens for "
            f"{model_family} request exceeds operator-set limit of "
            f"{context_limit} tokens."
        )


def _band_for(estimated: int, limit: int) -> str:
    """Pure-function classification of the upcoming request.

    Returns one of ``"green" | "amber" | "red" | "block"``. Defensive
    against ``limit <= 0`` (corrupted config / buggy backfill): the
    guard defaults to Block so a malformed config cannot let
    oversize requests through silently.
    """
    if limit <= 0:
        return "block"
    if estimated > limit:
        return "block"
    if estimated >= int(limit * _AMBER_FRACTION):
        return "red"
    if estimated >= int(limit * _GREEN_FRACTION):
        return "amber"
    return "green"


class ContextLimitGuardMiddleware(AgentMiddleware):
    """Pre-call guard — the single source of truth for the limit check.

    Mount this middleware FIRST in the agent's middleware list, ahead
    of ``SummarizationMiddleware``, so its decision is final before
    any other middleware injects state.

    Attributes:
        context_limit: Operator-set limit in tokens. ``None`` disables
            the guard entirely (no-op passthrough).
        model_name: The model identifier — used for the per-family
            log lines and status records. Defaults to ``"unknown"``.
        calibration_path: Optional path to a JSONL file where every
            estimate is appended. Used by the calibration loop
            (design §3.2.1) to track observed-vs-estimated drift.
    """

    name = "ContextLimitGuardMiddleware"  # required by LangChain factory

    def __init__(
        self,
        *,
        context_limit: int | None,
        model_name: str | None = None,
        calibration_path: Path | None = None,
        provider_floor: Callable[[], int | None] | None = None,
    ) -> None:
        self.context_limit = context_limit
        self.model_name = model_name or "unknown"
        self.provider_floor = provider_floor

    def _provider_context(self, request: Any) -> int | None:
        """The provider-reported context for this request, or ``None``.

        Two provider channels, in order of freshness:

        1. a ``deep_worker_context`` stamp anchored to THIS message list;
        2. ``provider_floor()`` — the provider's own persisted gauge
           (``acp_session_state.json``: ``last_context_used``), which is the
           source ``provider_context_token_counter`` already gives the
           summarizer.

        The second is not a convenience. A CLI-session provider keeps the
        conversation inside its own process, so our message list holds only
        the current exchange and NO stamp is ever anchored to it: measured
        on rg1_worker (grok) 2026-08-25, the stamp said ``first_id=b50f955b``
        while the state the guard saw began at ``4d10180b``. The guard
        logged "passing the turn through UNGUARDED" while the provider
        itself reported 275704 against ``-l 250000``. POLICIES §13: a
        persisted provider number is still the provider's number.

        ``None`` is a real answer — UNKNOWN — and it is NOT convertible into
        a number by estimating. Defensive against malformed requests and a
        broken floor: a shape error is also UNKNOWN, never a fabricated
        count.
        """
        try:
            messages = list(getattr(request, "messages", []) or [])
        except Exception:  # noqa: BLE001
            logger.exception("context-guard: unreadable request messages")
            return None
        reported = provider_context_tokens(messages)
        if reported is not None:
            return reported
        if self.provider_floor is None:
            return None
        try:
            persisted = self.provider_floor()
        except Exception:  # noqa: BLE001 — a broken gauge must not decide
            logger.exception("context-guard: provider gauge unreadable")
            return None
        if isinstance(persisted, bool) or not isinstance(persisted, int):
            return None
        return persisted if persisted > 0 else None

    def _hint_message(self, *, estimated: int) -> SystemMessage:
        """Build the Amber-band SystemMessage hint.

        Phrased so the model understands it can choose to invoke
        ``compact_history`` voluntarily (Phase 3) to stay under the
        limit on the next turn. The ``deep_worker_context_guard``
        marker lets the agent recognise the hint and not act on it
        twice if summarisation re-runs the same hook.
        """
        fraction = estimated / self.context_limit if self.context_limit else 0
        content = (
            "[context-guard] You are approaching the operator-set context "
            f"limit: ~{int(fraction * 100)}% of {self.context_limit:,} tokens "
            f"would be consumed by the next request (~{estimated:,} estimated "
            "tokens). Consider invoking the `compact_history` tool to "
            "summarise earlier turns before continuing. This hint will be "
            "delivered at most once per turn."
        )
        return SystemMessage(
            content=content,
            additional_kwargs={CONTEXT_GUARD_MARKER: True},
        )

    def _overflow_tool_message(self, *, estimated: int) -> ToolMessage:
        """Build the Red-band ToolMessage returned to the caller.

        Encoded as ``status="error"`` so it is rendered visibly to the
        model on the next turn. The machine-readable marker is in
        ``additional_kwargs`` so structured consumers (the agent's
        tool-dispatch layer, future web-UI event channel) can
        distinguish a context-guard warning from a real tool error.
        """
        fraction = estimated / self.context_limit if self.context_limit else 0
        content = (
            f"[context-guard] Overflow warning: ~{int(fraction * 100)}% of "
            f"{self.context_limit:,} tokens (~{estimated:,} estimated). "
            f"Action required: invoke `compact_history` or reduce scope "
            f"before the next request. If the next request still exceeds "
            f"the limit, the guard will Block (raise ContextLimitExceeded "
            f"and transition to IDLE)."
        )
        return ToolMessage(
            content=content,
            tool_call_id="context-guard-overflow",
            name="context_guard",
            status="error",
            additional_kwargs={
                CONTEXT_GUARD_MARKER: True,
                CONTEXT_OVERFLOW_ERROR: True,
                "context_limit": self.context_limit,
                "estimated_tokens": estimated,
            },
        )

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Passthrough — the guard is pre-model, not on tool execution.

        LangChain's agent factory checks for the presence of both
        ``wrap_model_call`` and ``wrap_tool_call`` to classify a
        middleware as "full". Defining a passthrough keeps the
        classification happy without changing the guard's behaviour
        (the guard's job is pre-call, not during tool execution; tool
        results arrive back through ``wrap_model_call`` on the next
        pass).
        """
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Async passthrough — see :meth:`wrap_tool_call`."""
        return await handler(request)

    # ── Band decision (pure, I/O-free) ─────────────────────────────────
    # The decision logic is the SAME for sync and async paths. We
    # extract it so neither ``wrap_model_call`` nor ``awrap_model_call``
    # carries duplicated control flow — only the I/O (await vs
    # blocking return) is method-local. This is the async-native
    # shape: one source of truth for the decision, two thin I/O
    # adapters.

    def _decide(self, request: Any) -> "_Decision":
        """Pure decision step: classify the request, build the payloads.

        Returns a ``_Decision`` describing which band the request
        belongs to and what (if anything) needs to be added to the
        conversation before forwarding to the LLM handler.

        I/O-free — does not call the handler, does not raise
        ``ContextLimitExceeded``. The two entry points (``wrap_model_call``
        and ``awrap_model_call``) own the I/O and the raise.
        """
        if self.context_limit is None:
            return _Decision(band="green", new_request=request)

        reported = self._provider_context(request)
        if reported is None:
            # UNKNOWN — see the module docstring. Pass through, loudly.
            logger.warning(
                "context-guard: no provider context report for this request "
                "(limit=%s) — passing the turn through UNGUARDED. The "
                "provider must report its context size; estimating it here "
                "is forbidden (owner directive 2026-08-25).",
                self.context_limit,
            )
            return _Decision(band="green", new_request=request)

        estimated = reported
        band = _band_for(estimated, self.context_limit)

        if band == "green":
            return _Decision(band="green", new_request=request)

        if band == "amber":
            logger.info(
                "context-guard: amber band, estimated=%d, limit=%d (%.0f%%)",
                estimated, self.context_limit,
                100 * estimated / self.context_limit,
            )
            hint = self._hint_message(estimated=estimated)
            return _Decision(
                band="amber",
                new_request=request.override(messages=[hint, *request.messages]),
                estimated=estimated,
            )

        if band == "red":
            logger.warning(
                "context-guard: red band, estimated=%d, limit=%d (%.0f%%) — "
                "injecting ToolMessage and proceeding",
                estimated, self.context_limit,
                100 * estimated / self.context_limit,
            )
            tool_msg = self._overflow_tool_message(estimated=estimated)
            return _Decision(
                band="red",
                new_request=request.override(
                    messages=[*request.messages, tool_msg]
                ),
                estimated=estimated,
            )

        # band == "block"
        logger.error(
            "context-guard: block band, estimated=%d, limit=%d — "
            "raising ContextLimitExceeded",
            estimated, self.context_limit,
        )
        return _Decision(
            band="block",
            new_request=request,
            estimated=estimated,
        )

    # ── Sync entry point ──────────────────────────────────────────────
    # Kept for symmetry with the LangChain middleware contract. This
    # project NEVER invokes the agent synchronously in production —
    # every code path uses ``ainvoke`` / ``astream`` — but the sync
    # hook must still be present because the LangChain factory checks
    # ``m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call``
    # to classify middleware. The body is three lines because the
    # decision lives in ``_decide``.

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Sync LangChain middleware hook. Not used in production
        (this project always invokes async), but required for the
        factory's middleware-classification check. See :meth:`awrap_model_call`
        for the canonical implementation."""
        decision = self._decide(request)
        if decision.band == "block":
            raise ContextLimitExceeded(
                context_limit=self.context_limit,
                estimated_tokens=decision.estimated,
                model_family=self.model_name,
            )
        return handler(decision.new_request)

    # ── Async entry point ─────────────────────────────────────────────
    # The CANONICAL path. Production always uses
    # ``agent.ainvoke`` / ``agent.astream`` (RPyC handler, TaskManager,
    # full-dialog mode). rg1_worker 2026-08-18 regression proved that
    # omitting this method leaves the agent unable to invoke ANY
    # model — the LangChain factory inherits ``AgentMiddleware.awrap_model_call``
    # which raises ``NotImplementedError`` and breaks the whole chain.

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Async LangChain middleware hook — the canonical implementation.

        Mirrors :meth:`_decide`'s decision step and owns the I/O:
        ``await handler(new_request)`` so the inner LLM call stays
        non-blocking. ``ContextLimitExceeded`` is raised here (not in
        ``_decide``) so the failure mode is identical between sync and
        async paths — caller catches and transitions to IDLE.
        """
        decision = self._decide(request)
        if decision.band == "block":
            raise ContextLimitExceeded(
                context_limit=self.context_limit,
                estimated_tokens=decision.estimated,
                model_family=self.model_name,
            )
        return await handler(decision.new_request)