"""TDD for _resolve_persisted_grok_display_model — seeds the TUI display
with the modelId from grok's per-session log so the user does not see
"unknown" until the first session/prompt response. Mirror of the codex
helper. Hermetic — monkeypatches Path.home() to a tmpdir.
"""
from __future__ import annotations

import json
from pathlib import Path

from deep_worker_cli.process.agent_process import (
    _extract_grok_model_id,
    _resolve_persisted_grok_display_model,
)
from deep_worker_cli.storage_paths import deep_worker_home


def _write_session(
    home: Path, thread_id: str, session_id: str, cwd_segment: str, files: dict
) -> None:
    state_dir = deep_worker_home(home) / "agents" / thread_id
    state_dir.mkdir(parents=True)
    (state_dir / "grok_session_state.json").write_text(
        json.dumps({thread_id: {"sent_count": 5, "session_id": session_id}}),
        encoding="utf-8",
    )
    session_dir = home / ".grok" / "sessions" / cwd_segment / session_id
    session_dir.mkdir(parents=True)
    for name, body in files.items():
        (session_dir / name).write_text(body, encoding="utf-8")


def test_returns_model_from_updates_jsonl(tmp_path: Path, monkeypatch) -> None:
    thread = "grok-agent"
    sid = "019ed7a0-afa8-7730-86b8-6a5817010dca"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    # Real grok format — modelId inside session/update _meta.
    upd = json.dumps(
        {
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {"sessionUpdate": "agent_message_chunk"},
                "_meta": {"modelId": "grok-composer-2.5-fast"},
            },
        }
    )
    _write_session(tmp_path, thread, sid, "%2Ftmp", {"updates.jsonl": upd + "\n"})
    assert _resolve_persisted_grok_display_model(thread) == "grok-composer-2.5-fast"


def test_returns_model_from_chat_history_jsonl(tmp_path: Path, monkeypatch) -> None:
    thread = "grok-agent"
    sid = "019ed7a0-afa8-7730-86b8-6a5817010dca"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    # chat_history.jsonl uses model_id at top-level.
    ch = json.dumps({"model_id": "grok-4-fast", "model_fingerprint": "fp_xxx"})
    _write_session(tmp_path, thread, sid, "%2Ftmp", {"chat_history.jsonl": ch + "\n"})
    assert _resolve_persisted_grok_display_model(thread) == "grok-4-fast"


def test_returns_latest_model_in_file(tmp_path: Path, monkeypatch) -> None:
    """If multiple updates carry a modelId, the LAST one wins (grok-CLI may
    upgrade mid-session)."""
    thread = "grok-agent"
    sid = "019ed7a0-afa8-7730-86b8-6a5817010dca"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    body = (
        json.dumps({"_meta": {"modelId": "grok-composer-2.5-fast"}}) + "\n"
        + json.dumps({"_meta": {"modelId": "grok-composer-2.5-final"}}) + "\n"
    )
    _write_session(tmp_path, thread, sid, "%2Ftmp", {"updates.jsonl": body})
    assert _resolve_persisted_grok_display_model(thread) == "grok-composer-2.5-final"


def test_rejects_unknown(tmp_path: Path, monkeypatch) -> None:
    thread = "grok-agent"
    sid = "uuid-unknown"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    body = json.dumps({"_meta": {"modelId": "unknown"}}) + "\n"
    _write_session(tmp_path, thread, sid, "%2Ftmp", {"updates.jsonl": body})
    assert _resolve_persisted_grok_display_model(thread) is None


def test_returns_none_when_no_state_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert _resolve_persisted_grok_display_model("no-such-agent") is None


def test_returns_none_when_session_dir_missing(tmp_path: Path, monkeypatch) -> None:
    thread = "grok-agent"
    sid = "missing-sid"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    state_dir = deep_worker_home(tmp_path) / "agents" / thread
    state_dir.mkdir(parents=True)
    (state_dir / "grok_session_state.json").write_text(
        json.dumps({thread: {"sent_count": 5, "session_id": sid}}),
        encoding="utf-8",
    )
    # ~/.grok/sessions does not exist
    assert _resolve_persisted_grok_display_model(thread) is None


def test_tolerates_corrupt_jsonl_lines(tmp_path: Path, monkeypatch) -> None:
    thread = "grok-agent"
    sid = "019ed7a0-afa8-7730-86b8-6a5817010dca"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    body = (
        "not valid json\n"
        + json.dumps({"_meta": {"modelId": "grok-composer-2.5-fast"}})
        + "\nanother bad line"
    )
    _write_session(tmp_path, thread, sid, "%2Ftmp", {"updates.jsonl": body})
    assert _resolve_persisted_grok_display_model(thread) == "grok-composer-2.5-fast"


def test_extract_grok_model_id_recurses() -> None:
    """The extractor should find modelId in any nesting (list, dict)."""
    assert _extract_grok_model_id({"a": {"b": {"modelId": "X"}}}) == "X"
    assert _extract_grok_model_id([{"x": 1}, {"model_id": "Y"}]) == "Y"
    assert _extract_grok_model_id({"no": "model here"}) is None
    assert _extract_grok_model_id("plain string") is None
