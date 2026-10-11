"""Grok provider usage-limit detection for deep-worker failover.

Failover chain (``runtime/provider_chain.py``):
``claude-code-sdk → codex → grok → deepseek``. When a grok agent hits an
account-level cap, ``GrokUsageLimitError`` triggers ``_schedule_failover``
to advance to ``deepseek``.

"Account-level exhaustion" covers TWO shapes that both mean "this account
cannot serve the request and retrying will not help — fail over now":

  * Subscription/rate caps — "usage limit", "rate limit", "quota exceeded" …
  * **Billing / payment (HTTP 402)** — the xAI API returns ``Payment
    Required`` when the account is out of credit. Incident 2026-07-03: this
    was NOT recognised, so the grok provider produced empty ``AIMessage``s
    instead of failing over to deepseek. Billing failures are terminal for
    the account, so they must be treated exactly like a usage cap.

Detection is conservative and marker-based. Transient failures (5xx,
timeouts, connection resets) are deliberately NOT matched — they are
retryable and must never burn a failover step.
"""
from __future__ import annotations

import re

_USAGE_LIMIT_MARKERS = (
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
    # match (review nit, 2026-07-03). The extractor only ever sees
    # ``reply["error"]["message"]``, but tightening keeps false failovers out
    # even if the error text is generic.
    re.compile(
        r"billing\s+(?:error|issue|problem|failure|required|declined)"
        r"|(?:no\s+active|invalid|failed)\s+billing",
        re.IGNORECASE,
    ),
)


class GrokUsageLimitError(RuntimeError):
    """Raised when the Grok CLI reports an account-level cap (usage OR billing)."""


def is_grok_usage_limit_text(text: str) -> bool:
    """Return True iff ``text`` looks like a grok account-level exhaustion
    (subscription/rate cap OR billing/payment failure)."""
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _USAGE_LIMIT_MARKERS)


def grok_usage_limit_error_from_text(text: str) -> GrokUsageLimitError | None:
    """Build a ``GrokUsageLimitError`` when ``text`` matches, else ``None``."""
    if not is_grok_usage_limit_text(text):
        return None
    return GrokUsageLimitError(f"Grok usage limit reached — {text}")