"""TDD for synthesizing a Claude CLI session transcript (.jsonl) from a
LangGraph message history.

This is Approach B of DESIGN_HISTORY_RESTORE_ON_RESTART.md: when there is no
existing Claude session to resume (e.g. cross-provider migration — the agent
ran on deepseek/codex and we now continue under claude-code), we build a valid
transcript file from the LangGraph history so the CLI can ``resume`` it and
treat it as prior context (NO tool re-execution — verified by experiment E2).

Schema (verified against a real ~/.claude/projects/.../*.jsonl, CLI 2.1.150):
every line is a JSON object with: uuid, parentUuid (linked-list chain; first
is null), sessionId (same on every line), type ("user"/"assistant"),
message{role, content}, timestamp (ISO-8601 ...Z), cwd, version, gitBranch,
isSidechain=false. User entries add promptId/userType/entrypoint/
permissionMode; assistant entries add requestId.

Mapping (DESIGN §3.2):
  HumanMessage           -> user,  content = <str>
  AIMessage (text)       -> assistant, content = [{type:text, text}]
  AIMessage (tool_calls) -> assistant, content = [..text.., {type:tool_use, id, name, input}]
  ToolMessage            -> user,  content = [{type:tool_result, tool_use_id, content}]
  SystemMessage          -> skipped (forwarded via system_prompt)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from claude_code_provider.session_synthesis import (
    synthesize_session_file,
    session_project_dir,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_conversation(path: Path) -> list[dict[str, Any]]:
    """The conversation entries only — skips the leading ``queue-operation``
    meta lines the CLI requires for resume (see synthesize_session_file)."""
    return [e for e in _read_jsonl(path) if e.get("type") != "queue-operation"]


def test_writes_file_under_sanitized_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point HOME at a tmp dir so ~/.claude/projects is isolated.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]

    msgs: list[BaseMessage] = [HumanMessage(content="hello", id="h1")]
    session_id = synthesize_session_file(msgs, cwd="/path/to/working_dir")

    proj = session_project_dir("/path/to/working_dir")
    sf = proj / f"{session_id}.jsonl"
    assert sf.exists(), f"transcript must be written at {sf}"
    # cwd '/path/to/working_dir' sanitizes (non-alnum -> '-') to
    # '-path-to-working-dir'.
    assert proj.name == "-path-to-working-dir"


def test_human_message_becomes_user_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [HumanMessage(content="what is 2+2?", id="h1")]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")

    assert len(entries) == 1
    e = entries[0]
    assert e["type"] == "user"
    assert e["message"]["role"] == "user"
    assert e["message"]["content"] == "what is 2+2?"
    assert e["parentUuid"] is None  # first entry
    assert e["sessionId"] == sid
    assert e["isSidechain"] is False
    for k in ("uuid", "timestamp", "cwd", "version", "gitBranch"):
        assert k in e


def test_parent_uuid_chain_links_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="a1", id="a1"),
        HumanMessage(content="q2", id="h2"),
    ]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")

    assert len(entries) == 3
    assert entries[0]["parentUuid"] is None
    assert entries[1]["parentUuid"] == entries[0]["uuid"]
    assert entries[2]["parentUuid"] == entries[1]["uuid"]
    # All uuids unique.
    uuids = [e["uuid"] for e in entries]
    assert len(set(uuids)) == 3


def test_ai_message_with_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    ai = AIMessage(
        content="Let me read it.",
        id="a1",
        tool_calls=[{"name": "read_file", "args": {"path": "/cfg"}, "id": "tool_1"}],
    )
    msgs: list[BaseMessage] = [HumanMessage(content="go", id="h1"), ai]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")

    asst = entries[1]
    assert asst["type"] == "assistant"
    assert asst["message"]["role"] == "assistant"
    blocks = asst["message"]["content"]
    types = [b["type"] for b in blocks]
    assert "text" in types and "tool_use" in types
    tu = next(b for b in blocks if b["type"] == "tool_use")
    assert tu["id"] == "tool_1"
    assert tu["name"] == "read_file"
    assert tu["input"] == {"path": "/cfg"}


def test_assistant_message_has_full_cli_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (observed 2026-05-25): a synthesized assistant entry's
    ``message`` must carry the same fields a REAL CLI transcript writes —
    ``type``/``id``/``model``/``role``/``content``/``stop_reason``/``usage`` —
    not just ``role``+``content``. Without them the CLI hangs at ``initialize``
    when resuming the synthesized session and the agent "forgets everything".
    Reference captured from a real ~/.claude/projects/.../<uuid>.jsonl
    (CLI 2.1.150) assistant entry.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi", id="h1"),
        AIMessage(content="hello there", id="a1"),
    ]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")
    asst_msg = entries[1]["message"]

    # Fields the CLI reads to reconstruct an assistant turn on resume.
    assert asst_msg["type"] == "message"
    assert asst_msg["role"] == "assistant"
    assert isinstance(asst_msg.get("id"), str) and asst_msg["id"]
    assert isinstance(asst_msg.get("model"), str) and asst_msg["model"]
    assert "stop_reason" in asst_msg
    assert "stop_sequence" in asst_msg
    assert "usage" in asst_msg and isinstance(asst_msg["usage"], dict)
    # content stays the block list we already produced.
    assert asst_msg["content"][0]["type"] == "text"


def test_starts_with_queue_operation_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (observed 2026-05-25): the CLI's ``--resume`` only
    RECOGNIZES a transcript that begins with the ``queue-operation`` meta
    entries it writes itself (enqueue → dequeue). A file starting straight with
    a ``user`` entry is rejected ("No conversation found with session ID").
    Verified live (CLI 2.1.150) — prepending these two lines is what makes a
    synthesized transcript resumable. The enqueue carries the first user
    message; both are standalone (no uuid/parentUuid).
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [
        HumanMessage(content="first question", id="h1"),
        AIMessage(content="an answer", id="a1"),
    ]
    sid = synthesize_session_file(msgs, cwd="/x")
    all_entries = _read_jsonl(session_project_dir("/x") / f"{sid}.jsonl")

    assert all_entries[0]["type"] == "queue-operation"
    assert all_entries[0]["operation"] == "enqueue"
    assert all_entries[0]["content"] == "first question"
    assert all_entries[0]["sessionId"] == sid
    assert all_entries[1]["type"] == "queue-operation"
    assert all_entries[1]["operation"] == "dequeue"
    # The first CONVERSATION entry follows the prefix.
    assert all_entries[2]["type"] == "user"


def test_tool_message_becomes_user_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [
        HumanMessage(content="go", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "read_file", "args": {}, "id": "tool_1"}]),
        ToolMessage(content="port=8472", tool_call_id="tool_1", id="t1"),
    ]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")

    tr_entry = entries[2]
    assert tr_entry["type"] == "user"  # tool_result lives in a user-role entry
    block = tr_entry["message"]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tool_1"
    assert "8472" in (block["content"] if isinstance(block["content"], str) else str(block["content"]))


def test_system_message_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    msgs: list[BaseMessage] = [
        SystemMessage(content="you are an assistant"),
        HumanMessage(content="hi", id="h1"),
    ]
    sid = synthesize_session_file(msgs, cwd="/x")
    entries = _read_conversation(session_project_dir("/x") / f"{sid}.jsonl")

    # SystemMessage is forwarded via system_prompt, never a transcript entry.
    assert len(entries) == 1
    assert entries[0]["message"]["content"] == "hi"


def test_returns_unique_session_id_each_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    s1 = synthesize_session_file([HumanMessage(content="a", id="h1")], cwd="/x")
    s2 = synthesize_session_file([HumanMessage(content="a", id="h1")], cwd="/x")
    assert s1 != s2
    # Both files exist.
    assert (session_project_dir("/x") / f"{s1}.jsonl").exists()
    assert (session_project_dir("/x") / f"{s2}.jsonl").exists()


def test_empty_history_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    # No restorable messages -> nothing to synthesize.
    assert synthesize_session_file([], cwd="/x") is None
    assert synthesize_session_file([SystemMessage(content="sys")], cwd="/x") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
