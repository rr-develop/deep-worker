"""TDD for the restore-selection orchestrator (DESIGN §3.3).

``resolve_restore_session(thread_id, messages, cwd)`` decides, on a fresh
claude-code session with history to carry over, which Claude session to
resume:

  1. find a candidate session file (hint = persisted session uuid);
  2. if its tail MATCHES the agent's history tail -> resume it (Approach A);
  3. else synthesize a transcript from the history -> resume that (Approach B).

Returns the session uuid to pass as ``ClaudeAgentOptions.resume`` (or None if
there is nothing to restore / the path should fall back to a fresh session).
Never raises — any failure degrades to None (current behavior).

Tested with the seams (candidate lookup, tail read, synthesize) mocked; the
real lookup/synthesis are covered by their own module tests + live experiments
(E1/E2/E3).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import claude_code_provider.session_restore as sr

# Aliases matching the seam signatures, so the monkeypatched lambdas below are
# fully typed (pyright strict: no reportUnknownLambdaType).
_Hint = str | None
_Entry = dict[str, Any] | None


def _hist() -> list[BaseMessage]:
    return [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="a1 final", id="a1"),
    ]


def test_no_history_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sr.resolve_restore_session("agent", [], cwd="/x") is None


def test_matching_candidate_is_resumed_approach_A(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate found and its tail matches → resume it; do NOT synthesize."""
    synth_called = {"n": 0}

    def _hint(tid: str) -> _Hint:
        return "cand-uuid"

    def _cand(uuid: _Hint, cwd: str) -> _Entry:
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "a1 final"}],
            },
        }

    def _match(msg: BaseMessage, entry: dict[str, Any]) -> bool:
        return True

    def _synth(msgs: list[BaseMessage], cwd: str) -> _Hint:
        synth_called["n"] += 1
        return "synth-uuid"

    monkeypatch.setattr(sr, "_load_uuid_hint", _hint)
    monkeypatch.setattr(sr, "_candidate_tail_entry", _cand)
    monkeypatch.setattr(sr, "_tail_matches", _match)
    monkeypatch.setattr(sr, "_synthesize", _synth)

    result = sr.resolve_restore_session("agent", _hist(), cwd="/x")
    assert result == "cand-uuid"
    assert synth_called["n"] == 0, "must NOT synthesize when a candidate matches"


def test_divergent_candidate_falls_back_to_synthesis_approach_B(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate found but tail does NOT match → synthesize and resume that."""

    def _hint(tid: str) -> _Hint:
        return "stale-uuid"

    def _cand(uuid: _Hint, cwd: str) -> _Entry:
        return {"type": "user", "message": {"role": "user", "content": "different"}}

    def _match(msg: BaseMessage, entry: dict[str, Any]) -> bool:
        return False

    def _synth(msgs: list[BaseMessage], cwd: str) -> _Hint:
        return "synth-uuid"

    monkeypatch.setattr(sr, "_load_uuid_hint", _hint)
    monkeypatch.setattr(sr, "_candidate_tail_entry", _cand)
    monkeypatch.setattr(sr, "_tail_matches", _match)
    monkeypatch.setattr(sr, "_synthesize", _synth)

    result = sr.resolve_restore_session("agent", _hist(), cwd="/x")
    assert result == "synth-uuid"


def test_no_candidate_synthesizes_approach_B(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No candidate at all (e.g. cross-provider — never ran on claude-code) →
    synthesize from the LangGraph history."""

    def _hint(tid: str) -> _Hint:
        return None

    def _cand(uuid: _Hint, cwd: str) -> _Entry:
        return None

    def _match(msg: BaseMessage, entry: dict[str, Any]) -> bool:
        return False

    def _synth(msgs: list[BaseMessage], cwd: str) -> _Hint:
        return "synth-uuid"

    monkeypatch.setattr(sr, "_load_uuid_hint", _hint)
    monkeypatch.setattr(sr, "_candidate_tail_entry", _cand)
    monkeypatch.setattr(sr, "_tail_matches", _match)
    monkeypatch.setattr(sr, "_synthesize", _synth)

    result = sr.resolve_restore_session("agent", _hist(), cwd="/x")
    assert result == "synth-uuid"


def test_synthesis_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """If synthesis yields nothing/raises, fall back to None (fresh session),
    never crash."""

    def _hint(tid: str) -> _Hint:
        return None

    def _cand(uuid: _Hint, cwd: str) -> _Entry:
        return None

    def _match(msg: BaseMessage, entry: dict[str, Any]) -> bool:
        return False

    def _synth(msgs: list[BaseMessage], cwd: str) -> _Hint:
        return None

    monkeypatch.setattr(sr, "_load_uuid_hint", _hint)
    monkeypatch.setattr(sr, "_candidate_tail_entry", _cand)
    monkeypatch.setattr(sr, "_tail_matches", _match)
    monkeypatch.setattr(sr, "_synthesize", _synth)

    assert sr.resolve_restore_session("agent", _hist(), cwd="/x") is None


def test_unexpected_error_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any unexpected exception in the path must degrade to None, not crash
    the agent turn."""

    def _boom(tid: str) -> _Hint:
        raise RuntimeError("disk gone")

    monkeypatch.setattr(sr, "_load_uuid_hint", _boom)

    assert sr.resolve_restore_session("agent", _hist(), cwd="/x") is None


def test_tail_is_last_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The match is performed against the LAST history message (the tail)."""
    seen: dict[str, Any] = {}

    def _hint(tid: str) -> _Hint:
        return "u"

    def _cand(uuid: _Hint, cwd: str) -> _Entry:
        return {"x": 1}

    def _match(msg: BaseMessage, entry: dict[str, Any]) -> bool:
        seen["msg"] = msg
        return True

    def _synth(msgs: list[BaseMessage], cwd: str) -> _Hint:
        return "s"

    monkeypatch.setattr(sr, "_load_uuid_hint", _hint)
    monkeypatch.setattr(sr, "_candidate_tail_entry", _cand)
    monkeypatch.setattr(sr, "_tail_matches", _match)
    monkeypatch.setattr(sr, "_synthesize", _synth)

    hist = _hist()
    sr.resolve_restore_session("agent", hist, cwd="/x")
    assert seen["msg"] is hist[-1], "tail-match must use the last history message"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
