"""TDD for importing a Claude CLI session transcript into a LangGraph history.

Moved out of ``claude_code_sdk_provider`` (DESIGN_IMPORT_CLAUDE_SESSION.md):
parsing a Claude Code CLI transcript into ``list[BaseMessage]`` has no
dependency on ``claude_agent_sdk`` or on any provider's runtime, so its tests
live here, next to the module (``deep_worker_cli.claude_session_import``).

Inverse of session_synthesis.py. Parses a real Claude ``.jsonl`` transcript
into ``list[BaseMessage]``:

  user  (content=str)                         -> HumanMessage
  assistant (text + tool_use blocks)          -> AIMessage(content, tool_calls)
  user  (content=[tool_result ...])           -> ToolMessage(tool_call_id=...)

Whitelist: only ``type in {user, assistant}`` with a ``message`` object are
imported; every other entry type (system, attachment, last-prompt,
permission-mode, file-history-snapshot, queue-operation, ai-title, unknown) and
any malformed line is ignored — never crash. Entries are ordered by the
``parentUuid`` chain.
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
    ToolMessage,
)

from deep_worker_cli.claude_session_import import (
    parse_claude_transcript,
    resolve_transcript_path,
    session_project_dir,
)


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _chain(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link entries into a parentUuid chain, assigning uuids in order."""
    parent: str | None = None
    out: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        e = dict(e)
        e.setdefault("uuid", f"u{i}")
        e["parentUuid"] = parent
        e.setdefault("sessionId", "sess-1")
        e.setdefault("isSidechain", False)
        out.append(e)
        parent = e["uuid"]
    return out


def _user(content: Any) -> dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assistant(content: Any) -> dict[str, Any]:
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def test_user_string_becomes_human_message(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([_user("what is 2+2?")]))
    msgs = parse_claude_transcript(f)
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "what is 2+2?"


def test_assistant_text_becomes_ai_message(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([_assistant([{"type": "text", "text": "the answer is 4"}])]))
    msgs = parse_claude_transcript(f)
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert "the answer is 4" in msgs[0].content
    assert not msgs[0].tool_calls


def test_assistant_tool_use_becomes_tool_calls(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([
        _assistant([
            {"type": "text", "text": "let me read it"},
            {"type": "tool_use", "id": "tool_1", "name": "read_file",
             "input": {"path": "/cfg"}, "caller": "agent"},
        ]),
    ]))
    msgs = parse_claude_transcript(f)
    # AIMessage + a placeholder ToolMessage for the unmatched tool_use.
    ai = msgs[0]
    assert isinstance(ai, AIMessage)
    assert "let me read it" in ai.content
    assert len(ai.tool_calls) == 1
    tc = ai.tool_calls[0]
    assert tc["id"] == "tool_1"
    assert tc["name"] == "read_file"
    assert tc["args"] == {"path": "/cfg"}
    assert isinstance(msgs[1], ToolMessage) and msgs[1].tool_call_id == "tool_1"


def test_tool_result_becomes_tool_message(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([
        _assistant([{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]),
        _user([{"type": "tool_result", "tool_use_id": "t1", "content": "port=8472"}]),
    ]))
    msgs = parse_claude_transcript(f)
    assert len(msgs) == 2
    tm = msgs[1]
    assert isinstance(tm, ToolMessage)
    assert tm.tool_call_id == "t1"
    assert "8472" in (tm.content if isinstance(tm.content, str) else str(tm.content))


def test_multiple_tool_use_one_assistant(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([
        _assistant([
            {"type": "tool_use", "id": "a", "name": "x", "input": {}},
            {"type": "tool_use", "id": "b", "name": "y", "input": {}},
        ]),
        _user([{"type": "tool_result", "tool_use_id": "a", "content": "ra"}]),
        _user([{"type": "tool_result", "tool_use_id": "b", "content": "rb"}]),
    ]))
    msgs = parse_claude_transcript(f)
    ai = msgs[0]
    assert isinstance(ai, AIMessage)
    assert [tc["id"] for tc in ai.tool_calls] == ["a", "b"]
    assert isinstance(msgs[1], ToolMessage) and msgs[1].tool_call_id == "a"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "b"


def test_whitelist_ignores_service_entry_types(tmp_path: Path) -> None:
    """Real transcripts carry many non-conversation types — ignore all but
    user/assistant, and never crash on malformed lines."""
    f = tmp_path / "s.jsonl"
    entries = _chain([
        {"type": "system", "content": "boot"},
        {"type": "file-history-snapshot", "snapshot": {}},
        _user("hello"),
        {"type": "queue-operation", "op": "x"},
        {"type": "attachment", "data": "..."},
        _assistant([{"type": "text", "text": "hi"}]),
        {"type": "ai-title", "title": "t"},
        {"type": "permission-mode", "mode": "plan"},
        {"type": "last-prompt", "prompt": "..."},
    ])
    # Append a malformed (non-JSON) line + an entry with null message.
    f.write_text(
        "\n".join(json.dumps(e) for e in entries)
        + "\nthis is not json\n"
        + json.dumps({"type": "user", "message": None}) + "\n"
    )
    msgs = parse_claude_transcript(f)
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert msgs[0].content == "hello"


def test_sidechain_entries_skipped(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    entries = _chain([_user("main task"), _assistant([{"type": "text", "text": "ok"}])])
    # a sub-agent (Task) entry that happens to be in the same file
    entries.append({
        "type": "assistant", "uuid": "sc1", "parentUuid": None,
        "isSidechain": True,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "subtask"}]},
    })
    _write_jsonl(f, entries)
    msgs = parse_claude_transcript(f)
    assert len(msgs) == 2
    assert all("subtask" not in str(m.content) for m in msgs)


def test_parent_uuid_chain_ordering(tmp_path: Path) -> None:
    """Even if file lines are out of order, the parentUuid chain defines order."""
    chained = _chain([_user("first"), _assistant([{"type": "text", "text": "second"}]),
                      _user("third")])
    shuffled = [chained[2], chained[0], chained[1]]  # scramble file order
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, shuffled)
    msgs = parse_claude_transcript(f)
    assert [str(m.content) for m in msgs] == ["first", "second", "third"]


def test_dangling_tool_use_gets_placeholder(tmp_path: Path) -> None:
    """A tool_use with no matching tool_result must get a synthesized
    placeholder ToolMessage so the history is well-formed for any provider."""
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([
        _user("do it"),
        _assistant([{"type": "tool_use", "id": "dangling", "name": "x", "input": {}}]),
    ]))
    msgs = parse_claude_transcript(f)
    assert isinstance(msgs[-1], ToolMessage)
    assert msgs[-1].tool_call_id == "dangling"


def test_tool_result_is_error_maps_status(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([
        _assistant([{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]),
        _user([{"type": "tool_result", "tool_use_id": "t1",
                "content": "boom", "is_error": True}]),
    ]))
    msgs = parse_claude_transcript(f)
    tm = msgs[1]
    assert isinstance(tm, ToolMessage)
    assert tm.status == "error"


def test_empty_or_no_conversation_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "s.jsonl"
    _write_jsonl(f, _chain([{"type": "system", "content": "x"}]))
    assert parse_claude_transcript(f) == []


def test_round_trip_with_synthesize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """synthesize then parse ≈ original (locks the inverse against the writer).

    The writer (``synthesize_session_file``) legitimately still lives in
    ``claude_code_sdk_provider`` — only the reader moved here — so this test
    alone needs that (excluded-from-distribution) package.
    """
    pytest.importorskip(
        "deep_worker.providers.claude_code_sdk_provider",
        reason="claude_code_sdk_provider excluded from the public distribution",
    )
    from deep_worker.providers.claude_code_sdk_provider.session_synthesis import (
        synthesize_session_file,
    )

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    original: list[BaseMessage] = [
        HumanMessage(content="go", id="h1"),
        AIMessage(content="reading", id="a1",
                  tool_calls=[{"id": "t1", "name": "read_file", "args": {"p": "/x"}}]),
        ToolMessage(content="result-data", tool_call_id="t1", id="tm1"),
        AIMessage(content="done", id="a2"),
    ]
    sid = synthesize_session_file(original, cwd="/proj/Demo")
    assert sid is not None
    f = session_project_dir("/proj/Demo") / f"{sid}.jsonl"
    parsed = parse_claude_transcript(f)
    # Same sequence of message types and key fields.
    assert [type(m).__name__ for m in parsed] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
    assert parsed[0].content == "go"
    ai, tm = parsed[1], parsed[2]
    assert isinstance(ai, AIMessage) and isinstance(tm, ToolMessage)
    assert ai.tool_calls[0]["name"] == "read_file"
    assert ai.tool_calls[0]["args"] == {"p": "/x"}
    assert tm.tool_call_id == "t1"
    assert "result-data" in str(tm.content)
    assert parsed[3].content == "done"


class TestResolveTranscriptPath:
    def test_explicit_path_passthrough(self, tmp_path: Path) -> None:
        f = tmp_path / "x.jsonl"
        f.write_text("{}\n")
        assert resolve_transcript_path(str(f), cwd=None) == f

    def test_uuid_via_project_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        proj = session_project_dir("/proj/Demo")
        proj.mkdir(parents=True, exist_ok=True)
        uuid = "11111111-1111-1111-1111-111111111111"
        (proj / f"{uuid}.jsonl").write_text("{}\n")
        got = resolve_transcript_path(uuid, cwd="/proj/Demo")
        assert got == proj / f"{uuid}.jsonl"

    def test_uuid_via_glob_when_cwd_unknown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        proj = session_project_dir("/proj/Other")
        proj.mkdir(parents=True, exist_ok=True)
        uuid = "22222222-2222-2222-2222-222222222222"
        (proj / f"{uuid}.jsonl").write_text("{}\n")
        got = resolve_transcript_path(uuid, cwd=None)
        assert got == proj / f"{uuid}.jsonl"

    def test_not_found_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        with pytest.raises(FileNotFoundError):
            resolve_transcript_path("33333333-3333-3333-3333-333333333333", cwd=None)

    def test_ambiguous_uuid_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        uuid = "44444444-4444-4444-4444-444444444444"
        for cwd in ("/proj/A", "/proj/B"):
            p = session_project_dir(cwd)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"{uuid}.jsonl").write_text("{}\n")
        with pytest.raises(ValueError):
            resolve_transcript_path(uuid, cwd=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
