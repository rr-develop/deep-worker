"""A usage cap is ACCOUNT-level, so the tracker must key by account.

Owner directive (2026-08-17):

    "claude-code-sdk and claude-code are two of our implementations of the
     same provider, and their limits are the same."

``provider_chain.PROVIDER_ACCOUNT_GROUPS`` already groups the two Claude
wrappers under ``claude-subscription``, and ``failover_candidates`` honours
it. ``AvailabilityTracker`` did not: it wrote a flat ``_data[provider]``
key, so a cap observed through the ACP wrapper left the SDK wrapper looking
available. Failover would then hop to the *same* Claude subscription and
re-fail immediately.

No migration is written for pre-existing provider-keyed records: the data
is short-lived (a cap resolves within a day) and a stale key simply stops
matching. That is a deliberate decision, pinned by
``test_legacy_provider_keyed_record_is_not_destructive`` below.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker


LATER = datetime(2099, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def tracker(tmp_path):
    return AvailabilityTracker(data_dir=tmp_path)


class TestClaudeWrappersShareOneAccount:
    """The two Claude wrappers bill the same subscription."""

    def test_acp_limit_blocks_the_sdk_wrapper(self, tracker):
        tracker.record_failure(
            provider="claude-code",
            marker_text="You've hit your session limit · resets 11pm (Europe/Amsterdam)",
            agent_name="unit",
            available_after_utc=LATER,
        )

        assert tracker.is_available("claude-code") is False
        assert tracker.is_available("claude-code-sdk") is False, (
            "same Claude subscription — a cap seen through the ACP wrapper "
            "must also block the SDK wrapper, or failover hops straight "
            "back into the exhausted account"
        )

    def test_sdk_limit_blocks_the_acp_wrapper(self, tracker):
        tracker.record_failure(
            provider="claude-code-sdk",
            marker_text="You've hit your session limit · resets 11pm (Europe/Amsterdam)",
            agent_name="unit",
            available_after_utc=LATER,
        )

        assert tracker.is_available("claude-code") is False, (
            "symmetry: the direction of the observation must not matter"
        )

    def test_restore_eligibility_is_shared_too(self, tracker):
        tracker.record_failure(
            provider="claude-code",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )

        assert tracker.is_restore_eligible("claude-code-sdk") is False

    def test_clear_lifts_the_block_for_the_whole_account(self, tracker):
        """Recovery is account-wide, exactly like the cap was.

        ``clear()`` runs when a call finally succeeds on a provider, which
        proves the shared subscription is serving again. Keyed by provider
        instead of account, the block on the sibling wrapper would outlive
        the outage itself — a provider fenced off with nothing left to
        expire it, because the success that should have cleared it landed
        under a different key.

        Note the asymmetry on purpose: the failure is recorded through one
        wrapper and cleared through the OTHER.
        """
        tracker.record_failure(
            provider="claude-code",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )
        assert tracker.is_available("claude-code-sdk") is False

        tracker.clear("claude-code-sdk")

        assert tracker.is_available("claude-code") is True, (
            "a success on the sibling wrapper proves the shared account "
            "recovered — the block must lift for both"
        )
        assert tracker.is_available("claude-code-sdk") is True

    def test_clear_does_not_touch_other_accounts(self, tracker):
        """Clearing one account must not unblock an unrelated one."""
        tracker.record_failure(
            provider="claude-code",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )
        tracker.record_failure(
            provider="codex",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )

        tracker.clear("claude-code")

        assert tracker.is_available("claude-code") is True
        assert tracker.is_available("codex") is False, (
            "codex bills a different account and must stay blocked"
        )

    def test_unrelated_providers_are_untouched(self, tracker):
        """Grouping must not leak across accounts."""
        tracker.record_failure(
            provider="claude-code",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )

        assert tracker.is_available("codex") is True
        assert tracker.is_available("grok") is True
        assert tracker.is_available("deepseek") is True

    def test_provider_without_a_group_keeps_its_own_key(self, tracker):
        """An unknown provider is its own account — no accidental sharing."""
        tracker.record_failure(
            provider="brand-new-vendor",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )

        assert tracker.is_available("brand-new-vendor") is False
        assert tracker.is_available("another-vendor") is True

    def test_legacy_provider_keyed_record_is_not_destructive(self, tmp_path):
        """A record written under the old scheme must not blow anything up.

        We deliberately ship no migration, and this test states the exact
        consequence rather than pretending there is none: a legacy
        provider-keyed record STOPS MATCHING (it is not read under the new
        account-group key) and ages out on its own. The cost is bounded —
        at worst one extra failover attempt against a provider that is
        still capped, which the next failure re-records under the correct
        key. What must NOT happen is a crash, a wiped file, or a legacy key
        being silently reinterpreted as some other account's verdict.
        """
        legacy = tmp_path / "provider_availability.json"
        legacy.write_text(
            '{"claude-code-sdk": {"available_after_utc": "2099-01-01T00:00:00+00:00",'
            ' "last_failure_marker": "old", "last_failure_at_utc": "2026-01-01T00:00:00+00:00",'
            ' "last_failure_agent": "legacy"},'
            ' "grok": {"available_after_utc": "2099-01-01T00:00:00+00:00",'
            ' "last_failure_marker": "real", "last_failure_at_utc": "2026-01-01T00:00:00+00:00",'
            ' "last_failure_agent": "agent_1"}}'
        )

        tracker = AvailabilityTracker(data_dir=tmp_path)

        # Legacy keys no longer match — the documented, accepted cost.
        assert tracker.is_available("grok") is True
        assert tracker.is_available("claude-code") is True

        # Nothing was destroyed: the raw entries survive on disk, so a
        # human (or a later migration, should we ever want one) can still
        # see what happened.
        raw = legacy.read_text()
        assert "legacy" in raw and "real" in raw

        # And a fresh failure records correctly under the account key,
        # which is what actually restores protection.
        tracker.record_failure(
            provider="claude-code-sdk",
            marker_text="limit",
            agent_name="unit",
            available_after_utc=LATER,
        )
        assert tracker.is_available("claude-code") is False


class TestConfigDrivenFallbackHours:
    """Owner: minimax gets a 3h fallback once a cap is seen."""

    def test_minimax_falls_back_to_three_hours(self):
        from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

        observed = datetime(2026, 8, 17, 1, 36, tzinfo=timezone.utc)
        resolution = parse_limit_reset(
            "minimax",
            "Token Plan usage limit reached: Upgrade your Token Plan or "
            "purchase Credits for more usage. (2056)",
            observed_at=observed,
        )

        assert resolution.available_after_utc == observed + timedelta(hours=3), (
            "minimax states no reset time and exposes no quota endpoint or "
            "rate-limit header, so the operator-chosen 3h fallback is the "
            "only signal we have"
        )
        assert resolution.is_fallback is True
        assert resolution.source == "fallback", (
            "must not be labelled provider_message — the provider gave us "
            "no time at all"
        )

    def test_the_three_hours_live_in_exactly_one_place(self):
        """No second copy of the constant may drift from the table."""
        from deep_worker_cli.runtime import availability_tracker

        assert availability_tracker._PROVIDER_FALLBACK_HOURS["minimax"] == 3
