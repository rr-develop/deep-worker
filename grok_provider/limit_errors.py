"""Grok provider usage-limit detection for deep-worker failover.

Failover chain (``runtime/provider_chain.py``):
``claude-code → codex → grok → deepseek``. When a grok agent hits its
subscription cap, ``GrokUsageLimitError`` triggers ``_schedule_failover``
to advance to ``deepseek``.

Marker strings are best-effort until a live incident pins the exact grok
0.2.54+ wording; detection is conservative (multiple phrases).
"""
from __future__ import annotations

import re

_USAGE_LIMIT_MARKERS = (
    re.compile(r"usage\s+limit", re.IGNORECASE),
    re.compile(r"rate\s+limit", re.IGNORECASE),
    re.compile(r"quota\s+exceeded", re.IGNORECASE),
    re.compile(r"subscription\s+limit", re.IGNORECASE),
    re.compile(r"hit your .* limit", re.IGNORECASE),
)


class GrokUsageLimitError(RuntimeError):
    """Raised when the Grok CLI reports an account-level usage cap."""


def is_grok_usage_limit_text(text: str) -> bool:
    """Return True iff ``text`` looks like a grok usage-limit failure."""
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _USAGE_LIMIT_MARKERS)


def grok_usage_limit_error_from_text(text: str) -> GrokUsageLimitError | None:
    """Build a ``GrokUsageLimitError`` when ``text`` matches, else ``None``."""
    if not is_grok_usage_limit_text(text):
        return None
    return GrokUsageLimitError(f"Grok usage limit reached — {text}")