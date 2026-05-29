"""TDD for tail-matching a candidate Claude session against the agent's
LangGraph history.

Approach A of DESIGN_HISTORY_RESTORE_ON_RESTART.md is made SAFE by validating
a candidate session file by CONTENT before resuming it (we do not trust the
session uuid, which rotates/forks). The gate: the candidate transcript's last
visible message must match the agent's history tail (the message at
``sent_count - 1``, or the last message). Match ⇒ resume that file; mismatch
or absent ⇒ synthesize from history instead.

The comparison must tolerate the LangGraph↔CLI representation gap:
  * role (user/assistant) must match;
  * a content fingerprint must match — text compared normalized; for an
    assistant turn carrying tool calls, the set of tool_use ids must match.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from claude_code_provider.session_tail_match import (
    history_tail_fingerprint,
    transcript_tail_fingerprint,
    tails_match,
)


# --- fingerprint of a LangGraph message ------------------------------------

def test_human_message_fingerprint() -> None:
    fp = history_tail_fingerprint(HumanMessage(content="hello world", id="h1"))
    assert fp["role"] == "user"
    assert "hello world" in fp["text"]


def test_assistant_text_fingerprint() -> None:
    fp = history_tail_fingerprint(AIMessage(content="the answer is 42", id="a1"))
    assert fp["role"] == "assistant"
    assert "the answer is 42" in fp["text"]


def test_assistant_tool_call_fingerprint() -> None:
    ai = AIMessage(
        content="reading",
        id="a1",
        tool_calls=[{"name": "read_file", "args": {}, "id": "tool_9"}],
    )
    fp = history_tail_fingerprint(ai)
    assert fp["role"] == "assistant"
    assert fp["tool_use_ids"] == {"tool_9"}


# --- fingerprint of a CLI transcript entry (SessionMessage-like) -----------

def _user_entry(text: str) -> dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant_entry(text: str, tool_uses: list[str] | None = None) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for tid in (tool_uses or []):
        blocks.append({"type": "tool_use", "id": tid, "name": "x", "input": {}})
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}}


def test_transcript_user_fingerprint() -> None:
    fp = transcript_tail_fingerprint(_user_entry("hello world"))
    assert fp["role"] == "user"
    assert "hello world" in fp["text"]


def test_transcript_assistant_tool_fingerprint() -> None:
    fp = transcript_tail_fingerprint(_assistant_entry("reading", ["tool_9"]))
    assert fp["role"] == "assistant"
    assert fp["tool_use_ids"] == {"tool_9"}


# --- the match gate --------------------------------------------------------

def test_matching_tails_match() -> None:
    msg = AIMessage(content="the answer is 42", id="a1")
    entry = _assistant_entry("the answer is 42")
    assert tails_match(msg, entry) is True


def test_divergent_text_does_not_match() -> None:
    msg = AIMessage(content="the answer is 42", id="a1")
    entry = _assistant_entry("a completely different answer")
    assert tails_match(msg, entry) is False


def test_role_mismatch_does_not_match() -> None:
    msg = HumanMessage(content="same text", id="h1")
    entry = _assistant_entry("same text")
    assert tails_match(msg, entry) is False


def test_assistant_tool_ids_must_match() -> None:
    msg = AIMessage(content="reading", id="a1",
                    tool_calls=[{"name": "x", "args": {}, "id": "tool_9"}])
    same = _assistant_entry("reading", ["tool_9"])
    diff = _assistant_entry("reading", ["tool_OTHER"])
    assert tails_match(msg, same) is True
    assert tails_match(msg, diff) is False


def test_whitespace_normalized() -> None:
    """Round-trip / formatting differences in whitespace must not block a
    match (the content survives JSON but trailing newlines etc. may differ)."""
    msg = AIMessage(content="hello world\n", id="a1")
    entry = _assistant_entry("hello world")
    assert tails_match(msg, entry) is True


def test_tool_message_tail_matches_tool_result_entry() -> None:
    msg = ToolMessage(content="port=8472", tool_call_id="tool_1", id="t1")
    entry = {
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tool_1", "content": "port=8472"}
        ]},
    }
    assert tails_match(msg, entry) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
