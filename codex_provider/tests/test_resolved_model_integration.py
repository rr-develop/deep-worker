from __future__ import annotations

import json
from pathlib import Path

from codex_provider.chat_model import resolve_model_from_session_log


def test_resolve_model_from_real_codex_session_layout(tmp_path: Path) -> None:
    thread_id = "019e6100-e176-7c01-ae9b-8995d28a7ebd"
    session_dir = tmp_path / "sessions" / "2026" / "05" / "26"
    session_dir.mkdir(parents=True)
    session_file = session_dir / f"rollout-2026-05-26T09-00-00-{thread_id}.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": thread_id}}),
                json.dumps({"type": "event_msg", "payload": {"type": "task_started"}}),
                json.dumps({"type": "turn_context", "payload": {"model": "gpt-session-default"}}),
            ]
        )
        + "\n"
    )

    assert resolve_model_from_session_log(thread_id, codex_home=tmp_path) == "gpt-session-default"
