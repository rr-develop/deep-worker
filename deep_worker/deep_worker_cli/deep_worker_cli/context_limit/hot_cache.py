"""Hot/cold kv-cache decision for summarization (DESIGN_SUMMARIZATION_HOT_COLD §4)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage

ORIGIN_KEY = "deep_worker_origin"
USAGE_KEY = "deep_worker_usage"
CREATED_AT_KEY = "created_at"

# Conservative defaults: Anthropic prompt-cache TTL is 5 min; subtract slack
# for session/load + network so we do not fire a "hot" summarize into an
# already-expired prefix (that costs ~2.5× a honest cold dump).
HOT_CACHE_TTL_DEFAULTS_SEC: dict[str, int] = {
    "claude-code": 240,
    "claude-code-sdk": 240,
    "anthropic": 240,
    "codex": 240,
    "openai": 240,
    "grok": 240,
    "google": 240,
    "deepseek": 240,
    "*": 240,
}

STATELESS_PROVIDERS: frozenset[str] = frozenset(
    {"deepseek", "openai", "anthropic", "google", "minimax"}
)

# Providers that keep the conversation on THEIR side (a CLI subprocess
# session). Re-invoking such a model replays the whole history into a new
# provider session, so callers that lose the summarizer port must fail
# loudly rather than fall through to ``ainvoke``.
#
# This is an ALLOW-LIST on purpose. The complement of STATELESS_PROVIDERS
# is not the answer: it would classify every unknown string — and every
# test double whose attribute access returns a truthy mock — as
# session-stateful, which is exactly how the first version of
# ``_is_session_stateful`` broke seven unrelated summarization tests
# (reviewer B2, 2026-08-21).
# INVARIANT: every provider listed here MUST implement
# ``summarizer_port.SummarizerPort`` (both halves). Membership without the port
# is a deterministic crash — the middleware refuses the generic fallback and
# raises ``SummarizationFailedError`` (agent_a/codex, 2026-08-23). Absence
# while session-stateful is WORSE: the provider sails past the guard and dumps
# its whole history into a fresh session, silently. ``grok`` sat in
# STATELESS_PROVIDERS in exactly that state until 2026-08-24 despite sending
# only ``messages[sent_count:]`` and persisting its session id across restarts,
# which is the definition of session-stateful.
#
# The pairing is enforced by test, not by convention.
SESSION_STATEFUL_PROVIDERS: frozenset[str] = frozenset(
    {"claude-code", "claude-code-sdk", "codex", "acp", "grok"}
)


@dataclass(frozen=True)
class HotColdDecision:
    hot: bool
    reason: str
    elapsed_sec: float | None
    n_sec: int
    last_origin: dict[str, Any] | None


@dataclass(frozen=True)
class SummaryResult:
    """Return of ``SummarizerPort.asummarize`` (DESIGN_SUMMARIZATION_HOT_COLD §5.1)."""

    text: str
    path: str
    provider_session_id: str | None = None
    usage: dict[str, Any] | None = None


def ttl_for(provider: str | None, overrides: Mapping[str, int] | None = None) -> int:
    """Resolve N for ``provider``. ``0`` means always-cold."""
    name = (provider or "").strip().lower()
    if overrides:
        if name in overrides:
            return int(overrides[name])
        if "*" in overrides:
            return int(overrides["*"])
    env_specific = os.environ.get(f"DEEP_WORKER_HOT_CACHE_TTL_SEC_{name.upper().replace('-', '_')}")
    if env_specific is not None and env_specific.strip() != "":
        return int(env_specific)
    env_all = os.environ.get("DEEP_WORKER_HOT_CACHE_TTL_SEC")
    if env_all is not None and env_all.strip() != "":
        return int(env_all)
    if name in HOT_CACHE_TTL_DEFAULTS_SEC:
        return HOT_CACHE_TTL_DEFAULTS_SEC[name]
    return HOT_CACHE_TTL_DEFAULTS_SEC["*"]


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _usable_origin(origin: Any) -> dict[str, Any] | None:
    if not isinstance(origin, dict):
        return None
    provider = str(origin.get("provider") or "").strip().lower()
    if not provider or provider == "unknown":
        return None
    out = dict(origin)
    out["provider"] = provider
    return out


def origin_of(message: BaseMessage) -> dict[str, Any] | None:
    meta = getattr(message, "response_metadata", None) or {}
    found = _usable_origin(meta.get(ORIGIN_KEY))
    if found is not None:
        return found
    extra = getattr(message, "additional_kwargs", None) or {}
    return _usable_origin(extra.get(ORIGIN_KEY))


def effective_origin(messages: list[Any]) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for message in messages:
        if not isinstance(message, BaseMessage):
            continue
        found = origin_of(message)
        if found is not None:
            last = found
    return last


def _received_at(message: BaseMessage) -> datetime | None:
    meta = getattr(message, "response_metadata", None) or {}
    usage = meta.get(USAGE_KEY)
    if isinstance(usage, dict):
        parsed = parse_iso(usage.get("received_at"))
        if parsed is not None:
            return parsed
    extra = getattr(message, "additional_kwargs", None) or {}
    return parse_iso(extra.get(CREATED_AT_KEY))


def last_provider_message_at(
    messages: list[Any],
    *,
    state_last_response_at: str | None = None,
) -> datetime | None:
    latest: datetime | None = None
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        stamp = _received_at(message)
        if stamp is not None and (latest is None or stamp > latest):
            latest = stamp
    state_stamp = parse_iso(state_last_response_at)
    if state_stamp is not None and (latest is None or state_stamp > latest):
        return state_stamp
    return latest


def decide(
    *,
    now: datetime | None = None,
    current_provider: str,
    current_model: str | None,
    current_model_id: str | None = None,
    messages: list[Any],
    state_last_response_at: str | None = None,
    session_id: str | None = None,
    ttl_overrides: Mapping[str, int] | None = None,
) -> HotColdDecision:
    """Pure hot/cold decision. No I/O."""
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    current_provider = str(current_provider or "").strip().lower()
    n_sec = ttl_for(current_provider, ttl_overrides)
    last_origin = effective_origin(messages)
    last_at = last_provider_message_at(
        messages, state_last_response_at=state_last_response_at
    )
    elapsed: float | None = None
    if last_at is not None:
        elapsed = (moment - last_at).total_seconds()

    if n_sec <= 0:
        return HotColdDecision(False, "always_cold", elapsed, n_sec, last_origin)

    if last_origin is None:
        reason = "no_origin"
        return HotColdDecision(False, reason, elapsed, n_sec, last_origin)

    if last_origin.get("provider") != current_provider:
        return HotColdDecision(False, "provider_changed", elapsed, n_sec, last_origin)
    if (last_origin.get("model") or None) != (current_model or None):
        return HotColdDecision(False, "model_changed", elapsed, n_sec, last_origin)
    if (
        current_model_id
        and last_origin.get("model_id")
        and last_origin.get("model_id") != current_model_id
    ):
        return HotColdDecision(False, "model_changed", elapsed, n_sec, last_origin)

    if last_at is None:
        return HotColdDecision(False, "no_timestamp", elapsed, n_sec, last_origin)
    if elapsed is not None and elapsed >= n_sec:
        return HotColdDecision(
            False, f"ttl_expired:{int(elapsed)}", elapsed, n_sec, last_origin
        )

    stateless = current_provider in STATELESS_PROVIDERS
    if not stateless and not session_id:
        return HotColdDecision(False, "no_session", elapsed, n_sec, last_origin)

    return HotColdDecision(True, "hot", elapsed, n_sec, last_origin)
