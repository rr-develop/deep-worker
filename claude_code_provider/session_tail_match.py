"""Validate a candidate Claude session transcript against the agent's
LangGraph history by comparing their TAILS.

Approach A of ``docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md`` resumes an
existing CLI session, but we do NOT trust the session uuid (it rotates on
summarization and forks on resume — verified). Instead we VALIDATE a candidate
file by content: its last visible message must match the agent's history tail.
Match ⇒ safe to resume that file; mismatch/absent ⇒ synthesize from history.

The comparison tolerates the LangGraph↔CLI representation gap:
  * ``role`` (user/assistant) must be equal;
  * ``text`` is compared after whitespace normalization (trailing newlines and
    formatting differ across the round-trip);
  * for assistant turns carrying tool calls, the SET of ``tool_use`` ids must
    be equal (ids are stable across the representation).

A "fingerprint" is a small dict ``{role, text, tool_use_ids}`` derived from
either a LangGraph ``BaseMessage`` or a CLI transcript entry / ``SessionMessage``.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip ends."""
    return _WS.sub(" ", text).strip()


def _blocks_text_and_tools(content: Any) -> tuple[str, set[str]]:
    """Extract concatenated text + the set of tool_use ids from a content that
    is either a string or a list of content blocks."""
    if isinstance(content, str):
        return content, set()
    text_parts: list[str] = []
    tool_ids: set[str] = set()
    if isinstance(content, list):
        for blk in content:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                text_parts.append(str(blk.get("text", "")))
            elif btype == "tool_use":
                tid = blk.get("id")
                if tid:
                    tool_ids.add(str(tid))
            elif btype == "tool_result":
                c = blk.get("content")
                text_parts.append(c if isinstance(c, str) else str(c))
    return " ".join(text_parts), tool_ids


def history_tail_fingerprint(msg: BaseMessage) -> dict[str, Any]:
    """Fingerprint a LangGraph message."""
    if isinstance(msg, HumanMessage):
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"role": "user", "text": _norm(text), "tool_use_ids": set()}
    if isinstance(msg, ToolMessage):
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"role": "user", "text": _norm(text), "tool_use_ids": set()}
    if isinstance(msg, AIMessage):
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        tool_ids = {tc.get("id", "") for tc in (msg.tool_calls or []) if tc.get("id")}
        return {"role": "assistant", "text": _norm(text), "tool_use_ids": tool_ids}
    if isinstance(msg, SystemMessage):
        # System messages are not transcript entries; never a tail to match.
        return {"role": "system", "text": "", "tool_use_ids": set()}
    text = getattr(msg, "content", "")
    return {"role": "?", "text": _norm(str(text)), "tool_use_ids": set()}


def transcript_tail_fingerprint(entry: Any) -> dict[str, Any]:
    """Fingerprint a CLI transcript entry (a raw dict) or an SDK
    ``SessionMessage`` (has ``.type`` / ``.message``)."""
    if isinstance(entry, dict):
        etype = entry.get("type")
        message = entry.get("message") or {}
    else:
        # SessionMessage-like object.
        etype = getattr(entry, "type", None)
        message = getattr(entry, "message", None) or {}
    role = "user" if etype == "user" else ("assistant" if etype == "assistant" else "?")
    content = message.get("content") if isinstance(message, dict) else None
    text, tool_ids = _blocks_text_and_tools(content)
    return {"role": role, "text": _norm(text), "tool_use_ids": tool_ids}


def tails_match(history_msg: BaseMessage, transcript_entry: Any) -> bool:
    """True iff the agent's history tail and the candidate transcript tail are
    the same message (role + normalized text + tool_use id set)."""
    h = history_tail_fingerprint(history_msg)
    t = transcript_tail_fingerprint(transcript_entry)
    if h["role"] != t["role"]:
        return False
    if h["text"] != t["text"]:
        return False
    if h["tool_use_ids"] != t["tool_use_ids"]:
        return False
    return True
