"""Unit tests for ACP session-state persistence (DESIGN §3.5).

Covers the provider/fingerprint scoping (a mismatch → fresh-session
behaviour), the atomic owner-only write (0o700 dir / 0o600 file), the
``:tm`` thread-base mapping, and corrupt-file degradation.
"""

from __future__ import annotations

import json
import os

import pytest

from deep_worker.providers.acp import session_state as ss

_FP = {"adapter": "@agentclientprotocol/claude-agent-acp", "adapter_version": "0.68.0", "session-capability": "load-v1"}


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:  # type: ignore[name-defined]
    """Point deep_worker_home at a temp dir so tests never touch ~/.deep_worker."""
    monkeypatch.setattr(ss, "deep_worker_home", lambda: tmp_path)
    return tmp_path


class TestThreadBase:
    def test_plain_thread_id_unchanged(self) -> None:
        assert ss._thread_base("agent-1") == "agent-1"

    def test_tm_suffix_stripped(self) -> None:
        assert ss._thread_base("agent-1:tm") == "agent-1"

    def test_epoch_suffix_stripped(self) -> None:
        assert ss._thread_base("agent-1__epoch_123") == "agent-1"


class TestSaveLoadRoundtrip:
    def test_save_then_load_returns_state(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="sess-abc",
            sent_count=4,
            fingerprint=_FP,
        )
        loaded = ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=_FP
        )
        assert loaded == {
            "sent_count": 4,
            "session_id": "sess-abc",
            "last_response_at": None,
            "last_context_used": None,
            "model_id": None,
        }

    def test_file_permissions_owner_only(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id=None,
            sent_count=0,
            fingerprint=_FP,
        )
        path = ss.session_state_path("agent-1")
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600
        assert (path.parent.stat().st_mode & 0o777) == 0o700

    def test_write_is_atomic_no_tmp_leftover(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="s1",
            sent_count=1,
            fingerprint=_FP,
        )
        leftovers = [
            p.name for p in home.iterdir() if p.suffix == ".tmp"
        ]
        assert leftovers == []

    def test_other_threads_preserved(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="s1",
            sent_count=1,
            fingerprint=_FP,
        )
        ss.save_session_state(
            "agent-2",
            provider="claude-code",
            session_id="s2",
            sent_count=2,
            fingerprint=_FP,
        )
        loaded = ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=_FP
        )
        assert loaded is not None
        assert loaded["sent_count"] == 1
        assert loaded["session_id"] == "s1"


class TestFingerprintScoping:
    def test_provider_mismatch_is_fresh(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="s1",
            sent_count=1,
            fingerprint=_FP,
        )
        assert (
            ss.load_session_state("agent-1", provider="grok", fingerprint=_FP) is None
        )

    def test_fingerprint_mismatch_is_fresh(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="s1",
            sent_count=1,
            fingerprint=_FP,
        )
        other_fp = dict(_FP, adapter_version="0.69.0")
        assert (
            ss.load_session_state(
                "agent-1", provider="claude-code", fingerprint=other_fp
            )
            is None
        )

    def test_fingerprint_none_skips_check(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="s1",
            sent_count=1,
            fingerprint=_FP,
        )
        loaded = ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=None
        )
        assert loaded is not None


class TestCorruptFile:
    def test_corrupt_json_reads_as_empty(self, home: Path) -> None:
        path = ss.session_state_path("agent-1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        assert ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=_FP
        ) is None

    def test_missing_file_returns_none(self, home: Path) -> None:
        assert ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=_FP
        ) is None

    def test_last_response_fields_round_trip(self, home: Path) -> None:
        ss.save_session_state(
            "agent-1",
            provider="claude-code",
            session_id="sess-abc",
            sent_count=4,
            fingerprint=_FP,
            last_response_at="2026-08-19T11:07:13+00:00",
            last_context_used=468146,
            model_id="claude-fable-5",
        )
        loaded = ss.load_session_state(
            "agent-1", provider="claude-code", fingerprint=_FP
        )
        assert loaded is not None
        assert loaded["last_response_at"] == "2026-08-19T11:07:13+00:00"
        assert loaded["last_context_used"] == 468146
        assert loaded["model_id"] == "claude-fable-5"
