"""TDD for persisting per-thread Claude session state (sent_count + session
uuid hint) across agent restarts.

DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.4: today ``sent_count`` lives only in
the in-memory ClientManager, so a restart silently resets it to 0 and the
broken empty-session replay path runs. Persisting ``sent_count`` (and the
last-known Claude session uuid as a *hint* for locating the resumable
transcript) makes restart deterministic.

Storage: a small JSON file named ``claude_session_state.json`` co-located
with the agent's data dir (the ``:tm`` / ``__epoch_N`` suffixes are stripped
so the TaskManager thread and the main agent share the agent's data dir but
keep per-thread entries).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker.providers.claude_code_provider.session_state import (
    invalidate_session_state,
    load_session_state,
    save_session_state,
    session_state_path,
)


def test_path_strips_tm_and_epoch_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    # main agent, :tm, and an epoch-rotated id all map to the SAME data dir.
    assert session_state_path("user_thread").parent.name == "user_thread"
    assert session_state_path("user_thread:tm").parent.name == "user_thread"
    assert session_state_path("user_thread__epoch_3").parent.name == "user_thread"
    assert session_state_path("user_thread:tm__epoch_2").parent.name == "user_thread"
    assert session_state_path("user_thread").name == "claude_session_state.json"


def test_save_then_load_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentX", sent_count=12, session_uuid="abc-123")
    state = load_session_state("agentX")
    assert state is not None
    assert state["sent_count"] == 12
    assert state["session_uuid"] == "abc-123"


def test_load_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    assert load_session_state("never_saved") is None


def test_per_thread_entries_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main agent and its :tm thread keep SEPARATE sent_count/uuid even
    though they share the same data dir / state file."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentY", sent_count=5, session_uuid="main-uuid")
    save_session_state("agentY:tm", sent_count=99, session_uuid="tm-uuid")

    main = load_session_state("agentY")
    tm = load_session_state("agentY:tm")
    assert main is not None and tm is not None
    assert main["sent_count"] == 5 and main["session_uuid"] == "main-uuid"
    assert tm["sent_count"] == 99 and tm["session_uuid"] == "tm-uuid"
    # Same underlying file.
    assert session_state_path("agentY") == session_state_path("agentY:tm")


def test_overwrite_updates_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentZ", sent_count=1, session_uuid="u1")
    save_session_state("agentZ", sent_count=7, session_uuid="u2")
    state = load_session_state("agentZ")
    assert state is not None
    assert state["sent_count"] == 7
    assert state["session_uuid"] == "u2"


def test_session_uuid_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sent_count can be persisted before we know the session uuid."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentW", sent_count=3, session_uuid=None)
    state = load_session_state("agentW")
    assert state is not None
    assert state["sent_count"] == 3
    assert state["session_uuid"] is None


def test_corrupt_file_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt state file must not crash the agent — degrade to None
    (restart falls back to current behavior)."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    p = session_state_path("agentC")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json")
    assert load_session_state("agentC") is None


def test_invalidate_removes_only_target_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentR", sent_count=8, session_uuid="poisoned")
    save_session_state("agentR:tm", sent_count=3, session_uuid="tm-clean")

    invalidate_session_state("agentR")

    assert load_session_state("agentR") is None
    assert load_session_state("agentR:tm") == {
        "sent_count": 3,
        "session_uuid": "tm-clean",
    }


def test_invalidate_missing_and_corrupt_are_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    invalidate_session_state("missing")
    path = session_state_path("corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json")
    invalidate_session_state("corrupt")


def test_invalidate_write_error_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    save_session_state("agentO", sent_count=1, session_uuid="poisoned")

    def fail_write(self: Path, data: str, *args: object, **kwargs: object) -> int:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)
    invalidate_session_state("agentO")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
