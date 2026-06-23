"""Availability tracker for provider failover.

Persists failure records to ``provider_availability.json`` under each
user's data directory so that supervisor restarts don't forget reset
times. All agents on the same host share one tracker — a usage cap that
one agent triggers is immediately known to all others.

Schema::

    {
      "claude-code": {
        "available_after_utc": "2026-06-04T18:10:00+00:00",   # null if unparseable
        "last_failure_marker": "You've hit your session limit · resets 7pm (Europe/Amsterdam)",
        "last_failure_at_utc": "2026-06-04T13:55:11+00:00",
        "last_failure_agent": "<agent_name>"
      } | null,
      "codex": null,
      "grok": null,
      "deepseek": null
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
