"""Detect tool calls orphaned by an agent restart.

When a deep-worker agent is killed/restarted while a tool call is in
flight, the LangGraph conversation history ends up with an
``AIMessage`` whose ``tool_calls`` entry has NO matching ``ToolMessage``
result — the model asked for the tool, but the process died before the
result was produced/persisted.

On the next invoke the claude-code provider resumes the CLI session,
which replays that dangling ``tool_use`` and deadlocks waiting for a
result that can never arrive (id mismatch between the replayed
``ToolUseBlock`` and the fresh control-channel ``tools/call``). See
``docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md`` and the
``project_mj1_tooluseid_mismatch_deadlock_2026_06_15`` root-cause memo.

This module provides the pure, deterministic detector used to reconcile
those orphans BEFORE anything can wait on them — no timeouts, no
polling, no heuristics. The LangGraph history is the source of truth:
a requested tool_call id with no resolving ToolMessage is orphaned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass(frozen=True)
class OrphanedToolCall:
    """A tool call the model requested but for which no result exists."""

    id: str
    name: str


def _iter_requested_calls(
    messages: Iterable[BaseMessage],
) -> Iterable[tuple[str, str]]:
    """Yield ``(id, name)`` for every tool call requested in an AIMessage.

    ``AIMessage.tool_calls`` is the normalized list of
    ``{"id", "name", "args", ...}`` dicts that langchain_core exposes.
    Entries without an id are skipped — an id is required to ever match a
    ToolMessage, and a tool_use without an id is a separate (upstream)
    invariant violation, not an orphan we can reconcile.
    """
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls or ():
            cid = call.get("id")
            if not cid:
                continue
            name = call.get("name") or ""
            yield cid, name


def _resolved_ids(messages: Iterable[BaseMessage]) -> set[str]:
    """Set of tool_call ids that have a resolving ToolMessage."""
    resolved: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            resolved.add(msg.tool_call_id)
    return resolved


def find_orphaned_tool_calls_in_transcript(
    transcript: Iterable[Any],
) -> list[OrphanedToolCall]:
    """Find dangling ``tool_use`` blocks in a Claude CLI session transcript.

    This is the AUTHORITATIVE source for the restart deadlock: when the
    agent is killed mid-tool-call, the CLI's own ``.jsonl`` transcript
    ends with an assistant ``tool_use`` block that has NO following user
    ``tool_result`` — and on resume the CLI replays that dangling
    ``tool_use``, which deadlocks the turn (the LangGraph checkpoint, by
    contrast, never recorded the interrupted tool turn at all — verified
    live 2026-06-15). So orphan detection MUST run against the transcript
    that is being resumed, not the LangGraph history.

    ``transcript`` is a sequence of SDK ``SessionMessage``-like objects,
    each exposing ``.type`` ('user'|'assistant') and ``.message`` (a dict
    with a ``content`` list of blocks). A block is a dict with ``type``
    ('text'|'tool_use'|'tool_result'), plus ``id`` (on tool_use) and
    ``tool_use_id`` (on tool_result).

    Returns the tool_use ids that were requested but never got a
    tool_result, in order of appearance, de-duplicated.
    """
    requested: list[tuple[str, str]] = []
    resolved: set[str] = set()
    for sm in transcript:
        msg = getattr(sm, "message", None)
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                bid = block.get("id")
                if bid:
                    requested.append((bid, block.get("name") or ""))
            elif btype == "tool_result":
                tuid = block.get("tool_use_id")
                if tuid:
                    resolved.add(tuid)

    orphans: list[OrphanedToolCall] = []
    seen: set[str] = set()
    for bid, name in requested:
        if bid in resolved or bid in seen:
            continue
        seen.add(bid)
        orphans.append(OrphanedToolCall(id=bid, name=name))
    return orphans


def find_orphaned_tool_calls(
    messages: Sequence[BaseMessage],
) -> list[OrphanedToolCall]:
    """Return tool calls requested in history with no resolving ToolMessage.

    Deterministic and order-preserving: orphans are returned in the order
    their requesting AIMessage appears. A given id is reported at most
    once (first occurrence wins) even if it somehow appears twice.
    """
    resolved = _resolved_ids(messages)
    orphans: list[OrphanedToolCall] = []
    seen: set[str] = set()
    for cid, name in _iter_requested_calls(messages):
        if cid in resolved or cid in seen:
            continue
        seen.add(cid)
        orphans.append(OrphanedToolCall(id=cid, name=name))
    return orphans
