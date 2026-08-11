"""TDD for grok usage-limit detection (failover hook)."""

from __future__ import annotations

import pytest

from deep_worker.providers.grok_provider.limit_errors import (
    GrokUsageLimitError,
    grok_usage_limit_error_from_text,
    is_grok_usage_limit_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "You've hit your usage limit. Resets at 10pm UTC.",
        "Grok error: rate limit exceeded for this account",
        "quota exceeded on SuperGrok subscription",
        "subscription limit reached — try again later",
    ],
)
def test_usage_limit_markers_match(text: str) -> None:
    assert is_grok_usage_limit_text(text)
    err = grok_usage_limit_error_from_text(text)
    assert isinstance(err, GrokUsageLimitError)
    assert "usage limit" in str(err).lower() or "rate limit" in str(err).lower()


@pytest.mark.parametrize(
    "text",
    [
        # The exact incident wording (2026-07-03): xAI API returns 402.
        "Payment Required",
        "payment required",
        "402 Payment Required",
        '{"code": 402, "message": "Payment Required"}',
        "Error: HTTP 402 Payment Required",
        "your credit balance is too low to run this model",
        "insufficient credits on this account",
        "insufficient balance",
        "insufficient funds",
        "you are out of credits",
        "billing issue: please add a payment method",
        "account has no active billing",
    ],
)
def test_payment_and_billing_markers_match(text: str) -> None:
    """A billing/402 failure is an account-level exhaustion → must failover,
    exactly like a usage cap. Incident 2026-07-03: grok API returned
    'Payment Required' and the provider produced empty AIMessages instead of
    failing over to deepseek."""
    assert is_grok_usage_limit_text(text)
    err = grok_usage_limit_error_from_text(text)
    assert isinstance(err, GrokUsageLimitError)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "permission denied for tool Shell",
        "connection reset by peer",
        "grok refused to continue",
        # Transient / non-billing failures MUST NOT be treated as exhaustion —
        # they are retryable and must not burn a failover step.
        "500 Internal Server Error",
        "502 Bad Gateway",
        "request timed out",
        "temporary network error, please retry",
        "the payment was processed successfully",  # 'payment' but not a failure
        # 'billing' alone must NOT match — only failure-context billing does
        # (review nit, 2026-07-03).
        "billing question answered",
        "your billing address was updated",
    ],
)
def test_non_limit_text_does_not_match(text: str) -> None:
    assert not is_grok_usage_limit_text(text)
    assert grok_usage_limit_error_from_text(text) is None