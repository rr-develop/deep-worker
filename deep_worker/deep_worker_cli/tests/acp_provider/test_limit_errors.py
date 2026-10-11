"""Unit tests for ACP usage-limit detection (failover trigger).

Covers the generic markers (usage/rate/quota/billing/402) plus the
profile-supplied extra markers (Claude Code's "You've hit your session
limit · resets <TIME>" banner, live-verified 0.68.0).
"""

from __future__ import annotations

import re

from deep_worker.providers.acp import limit_errors

_CLAUDE_EXTRA = (
    re.compile(r"You['\u2019]ve hit your session limit"),
    re.compile(r"session limit \u00b7 resets", re.IGNORECASE),
)


class TestIsUsageLimitText:
    def test_empty_text_false(self) -> None:
        assert not limit_errors.is_usage_limit_text("")
        assert not limit_errors.is_usage_limit_text("   ")

    def test_generic_markers(self) -> None:
        for text in (
            "Usage limit reached",
            "You have hit your rate limit",
            "quota exceeded for this model",
            "subscription limit reached",
            "hit your context limit",
            "HTTP 402 Payment Required",
            "insufficient credits",
            "out of credits",
            "credit balance is too low",
            "billing error: card declined",
        ):
            assert limit_errors.is_usage_limit_text(text), text

    def test_transient_failures_not_matched(self) -> None:
        for text in (
            "connection reset by peer",
            "timeout after 30s",
            "500 internal server error",
            "502 bad gateway",
            "billing question answered",
        ):
            assert not limit_errors.is_usage_limit_text(text), text

    def test_claude_extra_markers(self) -> None:
        assert limit_errors.is_usage_limit_text(
            "You've hit your session limit · resets 7pm (Europe/Amsterdam)",
            _CLAUDE_EXTRA,
        )
        assert limit_errors.is_usage_limit_text(
            "Session limit · resets 12:00", _CLAUDE_EXTRA
        )

    def test_extra_markers_only_apply_when_passed(self) -> None:
        # Without the claude profile markers the "session limit · resets"
        # banner is not matched by the GENERIC set either (no
        # usage/rate/quota/hit-your token in this phrasing).
        assert not limit_errors.is_usage_limit_text(
            "Session limit · resets 12:00"
        )


class TestUsageLimitErrorFromText:
    def test_match_builds_error(self) -> None:
        err = limit_errors.usage_limit_error_from_text("rate limit hit")
        assert err is not None
        assert isinstance(err, limit_errors.AcpUsageLimitError)

    def test_no_match_returns_none(self) -> None:
        assert limit_errors.usage_limit_error_from_text("all good") is None
