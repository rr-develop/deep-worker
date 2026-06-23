"""Tools for accessing pre-summarization history from markdown summary files.

Three tools that give any agent read-only access to its own history stored
in ``summaries/<thread_id>.md`` files under each agent's local data directory.

These tools work for ALL agents — no ``--history-memory`` flag required.

File format (append-only markdown sections)::

    ## Summarized at <iso-8601-ts>

    <evicted messages as get_buffer_string output>
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# Matches the section header the middleware writes
_HEADER_RE = re.compile(r"^## Summarized at (\S+)\s*$", re.MULTILINE)


def _resolve_path(
    assistant_id: str | None,
    thread_id: str | None,
    home: str | None,
) -> Path | None:
    """Resolve the path to the thread's markdown summary file.

    Returns None if assistant_id or thread_id is missing (can't resolve).
    """
    if not assistant_id or not thread_id:
        return None
    home_path = Path(home) if home else Path.home()
    return home_path / Path("~/.deep_worker").name / assistant_id / "summaries" / f"{thread_id}.md"


def _extract_runtime_info(
    runtime: Any,
    working_dir: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Extract assistant_id, thread_id, and home from runtime or working_dir.

    Returns (assistant_id, thread_id, home_str).
    """
    if runtime is not None:
        state = getattr(runtime, "state", {})
        config = getattr(runtime, "config", {})
        assistant_id = state.get("history_access_assistant_id") if isinstance(state, dict) else None
        home = state.get("history_access_home") if isinstance(state, dict) else None
        thread_id = (
            config.get("configurable", {}).get("thread_id")
            if isinstance(config, dict)
            else None
        )
        return assistant_id, thread_id, home
    if working_dir:
        return None, None, working_dir
    return None, None, None


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (timestamp, section_body) tuples.

    Returns empty list if no headers found. Non-header preamble is dropped.
    """
    starts = [(m.start(), m.group(1)) for m in _HEADER_RE.finditer(text)]
    if not starts:
        return []
    sections: list[tuple[str, str]] = []
    for i, (start, ts) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end]
        # Remove the header line itself, keep only the body content
        body_lines = body.split("\n", 1)
        body_text = body_lines[1] if len(body_lines) > 1 else ""
        sections.append((ts, body_text.strip()))
    return sections


def _parse_ts(ts: str) -> datetime | None:
    """Parse ISO-8601 timestamp string. Returns timezone-aware datetime or None."""
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _make_message(ts: str, body: str) -> dict[str, str | None]:
    """Create a message dict from a section."""
    snippet = body[:300] if body else ""
    return {
        "summarized_at": ts,
        "content_snippet": snippet,
    }


# ── Tool 1: history_range ────────────────────────────────────────────────


@tool
def history_range(
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Return the range of summarised history sections (earliest, latest, count).

    Tells you the time range and total number of pre-summarization history
    sections stored in the agent's summary file.

    Use this to:
    - Quickly check how much pre-summarization history exists
    - Find timestamp boundaries before calling history_at
    - Understand the scope of available records

    Returns:
        JSON: {"earliest": "<iso-8601>" or null, "latest": "<iso-8601>" or null, "count": <int>}
    """
    assistant_id, thread_id, home = _extract_runtime_info(runtime, working_dir)
    path = _resolve_path(assistant_id, thread_id, home)

    if path is None or not path.exists() or path.stat().st_size == 0:
        return json.dumps({"earliest": None, "latest": None, "count": 0})

    text = path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    if not sections:
        return json.dumps({"earliest": None, "latest": None, "count": 0})

    return json.dumps({
        "earliest": sections[0][0],
        "latest": sections[-1][0],
        "count": len(sections),
    })


# ── Tool 2: history_at ───────────────────────────────────────────────────


@tool
def history_at(
    timestamp: str,
    before: int = 1,
    at_or_after: int = 2,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Find history sections around a given ISO-8601 timestamp.

    Returns messages BEFORE and AT-or-AFTER the specified timestamp using
    proper datetime comparison (not string comparison), so it handles
    ``+00:00``, ``Z``, and microsecond-precision timestamps correctly.

    Args:
        timestamp: ISO-8601 timestamp to search around
                   (e.g. "2026-04-22T15:40:18+00:00").
        before: Number of sections to return BEFORE the timestamp (default 1).
        at_or_after: Number of sections AT or AFTER the timestamp (default 2).

    Returns:
        JSON: {"messages": [...], "indices": {"start": <int>, "end": <int>}}
    """
    assistant_id, thread_id, home = _extract_runtime_info(runtime, working_dir)
    path = _resolve_path(assistant_id, thread_id, home)

    if path is None or not path.exists() or path.stat().st_size == 0:
        return json.dumps({"messages": [], "indices": {"start": 0, "end": 0}})

    target_dt = _parse_ts(timestamp)
    if target_dt is None:
        return json.dumps({"messages": [], "indices": {"start": 0, "end": 0}})

    text = path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    if not sections:
        return json.dumps({"messages": [], "indices": {"start": 0, "end": 0}})

    # Find the first index where section timestamp >= target
    pivot = len(sections)
    for i, (ts, _body) in enumerate(sections):
        section_dt = _parse_ts(ts)
        if section_dt is not None and section_dt >= target_dt:
            pivot = i
            break

    # Collect before entries (before the pivot)
    before_start = max(0, pivot - before)
    before_entries = sections[before_start:pivot]

    # Collect at_or_after entries (from pivot)
    after_end = min(len(sections), pivot + at_or_after)
    after_entries = sections[pivot:after_end]

    # Combine in chronological order
    result_entries = before_entries + after_entries
    messages = [_make_message(ts, body) for ts, body in result_entries]
    start_idx = before_start
    end_idx = before_start + len(result_entries)

    return json.dumps({
        "messages": messages,
        "indices": {"start": start_idx, "end": end_idx},
    })


# ── Tool 3: history_slice ────────────────────────────────────────────────


@tool
def history_slice(
    start: int,
    end: int,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Return a slice of history sections by index (0-based, end-exclusive).

    Python slice convention: ``history_slice(1, 3)`` returns sections at
    indices 1 and 2.

    Negative indices are clamped to 0 (not Python-negative-index semantics).

    Args:
        start: 0-based start index (clamped to 0 if negative).
        end: exclusive end index (clamped to section count if beyond file).

    Returns:
        JSON: {"messages": [...], "indices": {"start": <int>, "end": <int>}}
    """
    assistant_id, thread_id, home = _extract_runtime_info(runtime, working_dir)
    path = _resolve_path(assistant_id, thread_id, home)

    if path is None or not path.exists() or path.stat().st_size == 0:
        return json.dumps({"messages": [], "indices": {"start": max(0, start), "end": max(0, start)}})

    text = path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    if not sections:
        return json.dumps({"messages": [], "indices": {"start": max(0, start), "end": max(0, start)}})

    # Clamp indices
    count = len(sections)
    safe_start = max(0, min(start, count))
    safe_end = max(safe_start, min(end, count))

    if safe_start >= count or safe_end <= safe_start:
        return json.dumps({"messages": [], "indices": {"start": safe_start, "end": safe_start}})

    selected = sections[safe_start:safe_end]
    messages = [_make_message(ts, body) for ts, body in selected]
    return json.dumps({
        "messages": messages,
        "indices": {"start": safe_start, "end": safe_end},
    })
