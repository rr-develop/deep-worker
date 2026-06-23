"""TDD for grok usage-limit detection (failover hook)."""

from __future__ import annotations

import pytest

from grok_provider.limit_errors import (
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
        "",
        "permission denied for tool Shell",
        "connection reset by peer",
        "grok refused to continue",
    ],
)
def test_non_limit_text_does_not_match(text: str) -> None:
    assert not is_grok_usage_limit_text(text)
    assert grok_usage_limit_error_from_text(text) is None