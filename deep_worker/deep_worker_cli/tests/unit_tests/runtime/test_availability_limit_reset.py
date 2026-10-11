from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def test_claude_reset_timezone_from_real_log_format() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 4, 18, 0, 38, tzinfo=ZoneInfo("Europe/Amsterdam"))

    result = parse_limit_reset(
        "claude-code-sdk",
        "You've hit your session limit · resets 6:50pm (Europe/Amsterdam)",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 7, 4, 16, 50, tzinfo=timezone.utc)
    assert result.source == "provider_message"
    assert result.is_fallback is False


def test_claude_reset_rolls_to_next_day_when_local_time_already_passed() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 6, 28, 23, 57, 40, tzinfo=ZoneInfo("Europe/Amsterdam"))

    result = parse_limit_reset(
        "claude-code-sdk",
        "You've hit your session limit · resets 1am (Europe/Amsterdam)",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 6, 28, 23, 0, tzinfo=timezone.utc)
    assert result.source == "provider_message"


def test_codex_absolute_date_from_real_log_format() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 6, 17, 20, 44, 29, tzinfo=ZoneInfo("Europe/Amsterdam"))

    result = parse_limit_reset(
        "codex",
        "Codex stream error: You've hit your usage limit. Upgrade to Pro, "
        "visit https://chatgpt.com/codex/settings/usage to purchase more credits "
        "or try again at Jun 18th, 2026 12:39 AM.",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 6, 17, 22, 39, tzinfo=timezone.utc)
    assert result.source == "provider_message"


def test_codex_time_only_uses_local_timezone_and_rolls_forward() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 3, 3, 19, 25, tzinfo=ZoneInfo("Europe/Amsterdam"))

    result = parse_limit_reset(
        "codex",
        "Codex stream error: You've hit your usage limit. Please try again at 7:20 AM.",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 7, 3, 5, 20, tzinfo=timezone.utc)
    assert result.source == "provider_message"


def test_codex_relative_hours_are_duration_not_clock_time() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)

    result = parse_limit_reset(
        "codex",
        "Codex stream error: You've hit your usage limit. Please try again in 2 hours.",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)
    assert result.source == "provider_message"


def test_codex_relative_minutes_are_duration_not_clock_time() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)

    result = parse_limit_reset(
        "codex",
        "Codex stream error: You've hit your usage limit. Please try again in 30 minutes.",
        observed_at=observed,
    )

    assert result.available_after_utc == datetime(2026, 7, 4, 12, 30, tzinfo=timezone.utc)
    assert result.source == "provider_message"


def test_grok_402_without_reset_uses_rough_24h_fallback() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 4, 14, 44, 51, tzinfo=timezone.utc)

    result = parse_limit_reset(
        "grok",
        "Grok usage limit reached — Internal error | API error (status 402 Payment Required): "
        "Grok Build usage balance exhausted",
        observed_at=observed,
    )

    assert result.available_after_utc == observed + timedelta(hours=24)
    assert result.source == "fallback"
    assert result.is_fallback is True
    assert "rough fallback" in result.note


def test_claude_unparseable_reset_uses_rough_6h_fallback() -> None:
    from deep_worker_cli.runtime.availability_tracker import parse_limit_reset

    observed = datetime(2026, 7, 4, 14, 44, 51, tzinfo=timezone.utc)

    result = parse_limit_reset("claude-code-sdk", "You've hit your session limit · resets soonish", observed_at=observed)

    assert result.available_after_utc == observed + timedelta(hours=6)
    assert result.source == "fallback"
    assert result.is_fallback is True


def test_tracker_persists_limit_reset_source_and_note(tmp_path: Path) -> None:
    from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker, parse_limit_reset

    observed = datetime(2026, 7, 4, 14, 44, 51, tzinfo=timezone.utc)
    reset = parse_limit_reset("grok", "Payment Required", observed_at=observed)

    tracker = AvailabilityTracker(data_dir=tmp_path)
    tracker.record_failure(
        provider="grok",
        marker_text="Payment Required",
        agent_name="dwtest",
        available_after_utc=reset.available_after_utc,
        available_after_source=reset.source,
        available_after_note=reset.note,
    )

    tracker2 = AvailabilityTracker(data_dir=tmp_path)
    info = tracker2.limit_info("grok")

    assert info is not None
    assert info["available_after_source"] == "fallback"
    assert "rough fallback" in info["available_after_note"]
