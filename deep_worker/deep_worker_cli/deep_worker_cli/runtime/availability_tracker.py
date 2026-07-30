"""Availability tracker for provider failover.

Persists failure records to ``~/.deep_worker/provider_availability.json``
so that supervisor restarts don't forget reset times. All agents on the
same host share one tracker — a usage cap that one agent triggers is
immediately known to all others.

Schema::

    {
      "claude-code": {
        "available_after_utc": "2026-06-04T18:10:00+00:00",   # null if unparseable
        "last_failure_marker": "You've hit your session limit · resets 7pm (Europe/Amsterdam)",
        "last_failure_at_utc": "2026-06-04T13:55:11+00:00",
        "last_failure_agent": "<agent-name>"
      } | null,
      "codex": null,
      "grok": null,
      "deepseek": null
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as dtparser

logger = logging.getLogger(__name__)

_PROVIDER_FALLBACK_HOURS: dict[str, int] = {
    "claude-code": 6,
    "codex": 6,
    "grok": 24,
}

_RESET_PATTERNS = (
    re.compile(r"\bresumes?\s+(?:at|around)?\s*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\btry\s+again\s+(?:at|around|in)?\s*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\bresets?\s+(?:at|around)?\s*(.+?)(?:\.|$)", re.IGNORECASE),
)
_RELATIVE_PATTERN = re.compile(
    r"\b(?:try\s+again\s+)?(?:in\s+)?(\d+)\s*(minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_TZ_PARENS_PATTERN = re.compile(r"\(([^()]+/[A-Za-z0-9_+\-]+)\)")
_ORDINAL_DAY_PATTERN = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)
_DATE_COMPONENT_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LimitResetResolution:
    available_after_utc: datetime | None
    source: str
    note: str
    is_fallback: bool


def parse_limit_reset(
    provider: str,
    marker_text: str,
    *,
    observed_at: datetime | None = None,
) -> LimitResetResolution:
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    parsed = _parse_provider_reset_time(marker_text, now)
    if parsed is not None:
        return LimitResetResolution(
            available_after_utc=parsed,
            source="provider_message",
            note="Parsed from provider usage-limit message.",
            is_fallback=False,
        )

    fallback_hours = _PROVIDER_FALLBACK_HOURS.get(provider)
    if fallback_hours is None:
        return LimitResetResolution(
            available_after_utc=None,
            source="unknown",
            note="No provider reset time was parseable and no fallback is configured.",
            is_fallback=False,
        )

    return LimitResetResolution(
        available_after_utc=now.astimezone(timezone.utc) + timedelta(hours=fallback_hours),
        source="fallback",
        note=(
            f"No provider reset time was parseable; using {fallback_hours}h "
            f"rough fallback for {provider}. This is not an exact recovery "
            "time and may be wrong for this account."
        ),
        is_fallback=True,
    )


def _parse_provider_reset_time(marker_text: str, observed_at: datetime) -> datetime | None:
    marker = marker_text.strip()
    if not marker:
        return None

    for pattern in _RESET_PATTERNS:
        match = pattern.search(marker)
        if not match:
            continue
        raw_when = match.group(1).strip()
        relative = _parse_relative_reset(raw_when, observed_at)
        if relative is not None:
            return relative
        parsed = _parse_absolute_or_time_reset(raw_when, observed_at)
        if parsed is not None:
            return parsed
    return None


def _parse_relative_reset(raw_when: str, observed_at: datetime) -> datetime | None:
    match = _RELATIVE_PATTERN.search(raw_when)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    delta = timedelta(minutes=amount) if unit.startswith(("min", "minute")) else timedelta(hours=amount)
    return observed_at.astimezone(timezone.utc) + delta


def _parse_absolute_or_time_reset(raw_when: str, observed_at: datetime) -> datetime | None:
    tz = _timezone_from_message(raw_when) or observed_at.tzinfo or datetime.now().astimezone().tzinfo or timezone.utc
    cleaned = _TZ_PARENS_PATTERN.sub("", raw_when).strip()
    cleaned = _ORDINAL_DAY_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.rstrip(" .")
    if not cleaned:
        return None

    observed_local = observed_at.astimezone(tz)
    default = observed_local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    try:
        parsed = dtparser.parse(cleaned, default=default)
    except (ValueError, TypeError, OverflowError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    parsed_local = parsed.astimezone(tz)

    if not _has_date_component(cleaned) and parsed_local <= observed_local:
        parsed_local = parsed_local + timedelta(days=1)

    return parsed_local.astimezone(timezone.utc)


def _timezone_from_message(text: str) -> ZoneInfo | None:
    match = _TZ_PARENS_PATTERN.search(text)
    if not match:
        return None
    try:
        return ZoneInfo(match.group(1))
    except ZoneInfoNotFoundError:
        return None


def _has_date_component(text: str) -> bool:
    return bool(_DATE_COMPONENT_PATTERN.search(text))


class AvailabilityTracker:
    """Tracks per-provider availability based on observed failure markers."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path.home() / Path("~/.deep_worker").name
        self._file = data_dir / "provider_availability.json"
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(
        self,
        provider: str,
        marker_text: str,
        agent_name: str,
        available_after_utc: datetime | None = None,
        *,
        available_after_source: str = "unknown",
        available_after_note: str = "",
    ) -> bool:
        """Record a failure for ``provider``.

        Returns True if this is the FIRST failure record for an ongoing
        outage (was previously available, or no record existed). Returns
        False if the provider is already in a recorded failure state
        (idempotency guard — callers should only trigger failover once
        per logical outage).

        When ``available_after_utc`` is None the reset time could not be
        parsed — the provider is treated as "available now" for failover
        (so we can still advance) but "do not auto-restore" for the
        restore-on-idle trigger.
        """
        now = datetime.now(timezone.utc)
        was_available = self.is_available(provider)

        self._data[provider] = {
            "available_after_utc": (
                available_after_utc.isoformat() if available_after_utc else None
            ),
            "available_after_source": available_after_source,
            "available_after_note": available_after_note,
            "last_failure_marker": marker_text,
            "last_failure_at_utc": now.isoformat(),
            "last_failure_agent": agent_name,
        }
        self._persist()
        return was_available

    def is_available(self, provider: str) -> bool:
        """True if the provider has no failure record OR
        the reset moment is in the past (or unknown / unparseable).

        ``None`` available_after_utc (== unknown reset) returns True for
        failover decisions (we can still try), but the caller should NOT
        auto-restore to a provider with unknown reset (see
        ``is_restore_eligible``).
        """
        record = self._data.get(provider)
        if record is None:
            return True
        raw = record.get("available_after_utc")
        if raw is None:
            # Unknown reset — assume available for failover
            return True
        try:
            reset = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return True
        return reset <= datetime.now(timezone.utc)

    def is_restore_eligible(self, provider: str) -> bool:
        """True only when restore-on-idle may switch back to ``provider``.

        Stricter than ``is_available``: a provider with an active failure
        record and no confirmed past reset is NOT eligible — including
        when ``available_after_utc`` is unknown (``None``).
        """
        record = self._data.get(provider)
        if record is None:
            return True
        raw = record.get("available_after_utc")
        if raw is None:
            return False
        try:
            reset = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return False
        return reset <= datetime.now(timezone.utc)

    def next_available(self, provider: str) -> datetime | None:
        """When will this provider become available, or None if unknown /
        no record.

        The caller uses this to decide whether restore-on-idle can fire:
        a return of None means "don't auto-restore, the reset window is
        unknown".
        """
        record = self._data.get(provider)
        if record is None:
            return None
        raw = record.get("available_after_utc")
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None

    def limit_info(self, provider: str) -> dict[str, Any] | None:
        """Return the raw persisted limit record for readonly CLI/UI surfaces."""
        record = self._data.get(provider)
        if record is None:
            return None
        return dict(record)

    def clear(self, provider: str) -> None:
        """Remove the failure record for ``provider``.

        Called when an invocation succeeds on that provider (proving it
        is back), or when restore-on-idle completes.
        """
        self._data.pop(provider, None)
        self._persist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            if self._file.exists():
                raw = self._file.read_text()
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("could not load provider_availability: %s", exc)
        return {}

    def _persist(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False, default=str)
            )
            tmp.chmod(0o600)
            tmp.replace(self._file)
            self._file.chmod(0o600)
        except OSError as exc:
            logger.error("could not persist provider_availability: %s", exc)
