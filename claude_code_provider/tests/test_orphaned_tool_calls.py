"""Unit tests for orphaned-tool-call detection.

An orphaned tool call is one the model requested (AIMessage.tool_calls)
that has NO matching ToolMessage result in the conversation history —
the "process died after the model asked for the tool but before the
result was produced" state that deadlocks the agent on restart.

See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from claude_code_provider.orphaned_tool_calls import (
    OrphanedToolCall,
    find_orphaned_tool_calls,
    find_orphaned_tool_calls_in_transcript,
)


class _SM:
    """Minimal SessionMessage stand-in: ``.type`` + ``.message`` dict."""

    def __init__(self, type: str, content: list[dict[str, object]]) -> None:
        self.type = type
        self.message = {"role": type, "content": content}


def test_transcript_no_dangling_tool_use() -> None:
    transcript = [
        _SM("user", [{"type": "text", "text": "hi"}]),
        _SM("assistant", [{"type": "tool_use", "id": "tu_1", "name": "execute"}]),
        _SM("user", [{"type": "tool_result", "tool_use_id": "tu_1"}]),
        _SM("assistant", [{"type": "text", "text": "done"}]),
    ]
    assert find_orphaned_tool_calls_in_transcript(transcript) == []


def test_transcript_dangling_tool_use_is_orphan() -> None:
    # Killed mid-call: tool_use with NO following tool_result.
    transcript = [
        _SM("user", [{"type": "text", "text": "run sleep"}]),
        _SM("assistant", [{"type": "tool_use", "id": "tu_X", "name": "execute"}]),
        # process died here
    ]
    assert find_orphaned_tool_calls_in_transcript(transcript) == [
        OrphanedToolCall(id="tu_X", name="execute")
    ]


def test_transcript_partial_resolution() -> None:
    transcript = [
        _SM("assistant", [
            {"type": "tool_use", "id": "tu_A", "name": "read_file"},
            {"type": "tool_use", "id": "tu_B", "name": "execute"},
        ]),
        _SM("user", [{"type": "tool_result", "tool_use_id": "tu_A"}]),
        # tu_B never resolved
    ]
    orphans = find_orphaned_tool_calls_in_transcript(transcript)
    assert [o.id for o in orphans] == ["tu_B"]


def test_transcript_ignores_non_dict_messages() -> None:
    class _Weird:
        type = "assistant"
        message = "not a dict"
    assert find_orphaned_tool_calls_in_transcript([_Weird()]) == []


def _ai_with_tool_calls(*calls: tuple[str, str]) -> AIMessage:
    """AIMessage carrying tool_calls as (id, name) pairs."""
    return AIMessage(
        content="",
        tool_calls=[
            {"id": cid, "name": name, "args": {}} for cid, name in calls
        ],
    )


def test_no_messages_no_orphans() -> None:
    assert find_orphaned_tool_calls([]) == []


def test_resolved_tool_call_is_not_orphan() -> None:
    msgs = [
        HumanMessage(content="do it"),
        _ai_with_tool_calls(("toolu_A", "read_file")),
        ToolMessage(content="result", tool_call_id="toolu_A"),
    ]
    assert find_orphaned_tool_calls(msgs) == []


def test_unresolved_tool_call_is_orphan() -> None:
    msgs = [
        HumanMessage(content="do it"),
        _ai_with_tool_calls(("toolu_A", "history_memory")),
        # no ToolMessage for toolu_A — process died mid-call
    ]
    orphans = find_orphaned_tool_calls(msgs)
    assert orphans == [OrphanedToolCall(id="toolu_A", name="history_memory")]


def test_partial_resolution_only_unresolved_are_orphans() -> None:
    msgs = [
        HumanMessage(content="do it"),
        _ai_with_tool_calls(
            ("toolu_A", "read_file"),
            ("toolu_B", "execute"),
            ("toolu_C", "grep"),
        ),
        ToolMessage(content="ok", tool_call_id="toolu_B"),
        # A and C never resolved
    ]
    orphans = find_orphaned_tool_calls(msgs)
    assert {o.id for o in orphans} == {"toolu_A", "toolu_C"}
    assert {o.name for o in orphans} == {"read_file", "grep"}


def test_multiple_turns_match_by_id() -> None:
    msgs = [
        HumanMessage(content="turn 1"),
        _ai_with_tool_calls(("toolu_1", "read_file")),
        ToolMessage(content="r1", tool_call_id="toolu_1"),
        HumanMessage(content="turn 2"),
        _ai_with_tool_calls(("toolu_2", "execute")),
        # toolu_2 orphaned by restart
    ]
    assert find_orphaned_tool_calls(msgs) == [
        OrphanedToolCall(id="toolu_2", name="execute")
    ]


def test_systemmessages_and_plain_ai_ignored() -> None:
    msgs = [
        SystemMessage(content="prompt"),
        HumanMessage(content="hi"),
        AIMessage(content="hello, no tools here"),
    ]
    assert find_orphaned_tool_calls(msgs) == []


def test_orphan_preserves_first_seen_name_for_duplicate_ids() -> None:
    # Defensive: if the same id somehow appears twice (it shouldn't),
    # detection must not crash and must not double-report.
    msgs = [
        _ai_with_tool_calls(("toolu_X", "read_file")),
        _ai_with_tool_calls(("toolu_X", "read_file")),
    ]
    orphans = find_orphaned_tool_calls(msgs)
    assert [o.id for o in orphans] == ["toolu_X"]
