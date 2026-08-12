from __future__ import annotations

import json
from pathlib import Path

from deep_worker_cli.process.agent_process import _resolve_persisted_codex_display_model

from deep_worker_cli.storage_paths import deep_worker_home
def test_resolve_persisted_codex_display_model_from_saved_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    thread_id = "codex-agent"
    session_uuid = "019e60cf-9b61-7200-bcd1-99358db17e0c"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    state_dir = deep_worker_home(tmp_path) / thread_id
    state_dir.mkdir(parents=True)
    (state_dir / "codex_session_state.json").write_text(
        json.dumps({thread_id: {"sent_count": 7, "session_uuid": session_uuid}}),
        encoding="utf-8",
    )

    session_dir = tmp_path / ".codex" / "sessions" / "2026" / "05" / "26"
    session_dir.mkdir(parents=True)
    (session_dir / f"rollout-2026-05-26T10-00-00-{session_uuid}.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": session_uuid}})
        + "\n"
        + json.dumps({"type": "turn_context", "payload": {"model": "gpt-real-default"}})
        + "\n",
        encoding="utf-8",
    )

    assert _resolve_persisted_codex_display_model(thread_id) == "gpt-real-default"


def test_resolve_persisted_codex_display_model_rejects_unknown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    thread_id = "codex-agent"
    session_uuid = "uuid-unknown"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    state_dir = deep_worker_home(tmp_path) / thread_id
    state_dir.mkdir(parents=True)
    (state_dir / "codex_session_state.json").write_text(
        json.dumps({thread_id: {"sent_count": 7, "session_uuid": session_uuid}}),
        encoding="utf-8",
    )

    session_dir = tmp_path / ".codex" / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / f"rollout-{session_uuid}.jsonl").write_text(
        json.dumps({"type": "turn_context", "payload": {"model": "unknown"}}) + "\n",
        encoding="utf-8",
    )

    assert _resolve_persisted_codex_display_model(thread_id) is None
