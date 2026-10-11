"""ACP provider usage-limit detection for deep-worker failover.

Failover chain (``runtime/provider_chain.py``):
``claude-code-sdk → codex → grok → deepseek``. When an ACP-backed agent hits an
account-level cap, ``AcpUsageLimitError`` triggers ``_schedule_failover``
to advance to the next provider.

"Account-level exhaustion" covers TWO shapes that both mean "this account
cannot serve the request and retrying will not help — fail over now":

  * Subscription/rate caps — "usage limit", "rate limit", "quota exceeded" …
  * **Billing / payment (HTTP 402)** — the provider API returns ``Payment
    Required`` when the account is out of credit. Billing failures are
    terminal for the account, so they must be treated exactly like a usage
    cap (grok incident 2026-07-03 / 2026-07-04).

Detection is conservative and marker-based. Transient failures (5xx,
timeouts, connection resets) are deliberately NOT matched — they are
retryable and must never burn a failover step.

Provider-specific markers (e.g. Claude Code's ``"You've hit your session
limit · resets <TIME>"``) live in the provider profile; the generic markers
below apply to every ACP provider.
"""
from __future__ import annotations

import re

_GENERIC_USAGE_LIMIT_MARKERS = (
    re.compile(r"usage\s+limit", re.IGNORECASE),
    re.compile(r"rate\s+limit", re.IGNORECASE),
    re.compile(r"quota\s+exceeded", re.IGNORECASE),
    re.compile(r"subscription\s+limit", re.IGNORECASE),
    re.compile(r"hit your .* limit", re.IGNORECASE),
    # Billing / payment exhaustion (HTTP 402). These are account-terminal:
    # retrying the same request cannot succeed, so failover must fire.
    re.compile(r"payment\s+required", re.IGNORECASE),
    re.compile(r"\b402\b", re.IGNORECASE),
    re.compile(r"insufficient\s+(?:credit|credits|balance|funds)", re.IGNORECASE),
    re.compile(r"out\s+of\s+credits?", re.IGNORECASE),
    re.compile(r"credit\s+balance\s+is\s+too\s+low", re.IGNORECASE),
    # ``billing`` alone is too broad ("billing question answered" is not a
    # failure). Require a failure-context word so only genuine billing errors
    # match.
    re.compile(
        r"billing\s+(?:error|issue|problem|failure|required|declined)"
        r"|(?:no\s+active|invalid|failed)\s+billing",
        re.IGNORECASE,
    ),
)


class AcpUsageLimitError(RuntimeError):
    """Raised when the ACP CLI reports an account-level cap (usage OR billing)."""


def is_usage_limit_text(text: str, extra_markers: tuple[re.Pattern, ...] = ()) -> bool:
    """Return True iff ``text`` looks like an account-level exhaustion
    (subscription/rate cap OR billing/payment failure)."""
    if not text or not text.strip():
        return False
    patterns = _GENERIC_USAGE_LIMIT_MARKERS + tuple(extra_markers)
    return any(pattern.search(text) for pattern in patterns)


def usage_limit_error_from_text(
    text: str, extra_markers: tuple[re.Pattern, ...] = ()
) -> AcpUsageLimitError | None:
    """Build an ``AcpUsageLimitError`` when ``text`` matches, else ``None``."""
    if not is_usage_limit_text(text, extra_markers):
        return None
    return AcpUsageLimitError(f"ACP usage limit reached — {text}")
