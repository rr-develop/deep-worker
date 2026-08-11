"""Restore-selection orchestrator (DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.3).

On a fresh claude-code session that has LangGraph history to carry over,
decide which Claude session to ``resume``:

  1. find a candidate session file (hint = persisted session uuid);
  2. if its tail MATCHES the agent's history tail → resume it (Approach A);
  3. else synthesize a transcript from the history → resume that (Approach B).

Returns the session uuid for ``ClaudeAgentOptions.resume`` (or ``None`` to fall
back to a fresh session). NEVER raises — any failure degrades to ``None`` so a
restore problem can never break an agent turn (the worst case is today's
behavior).

The seams (``_load_uuid_hint``, ``_candidate_tail_entry``, ``_tail_matches``,
``_synthesize``) are module-level so they can be unit-tested in isolation;
their real implementations are covered by the per-module tests
(``session_state``, ``session_tail_match``, ``session_synthesis``) and the
live experiments E1/E2/E3.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import BaseMessage


def _load_uuid_hint(thread_id: str) -> str | None:
    """Persisted last-known Claude session uuid for this thread (a hint only —
    the tail-match is authoritative)."""
    from .session_state import load_session_state  # noqa: PLC0415

    state = load_session_state(thread_id)
    return state.get("session_uuid") if state else None


def _candidate_tail_entry(session_uuid: str | None, cwd: str) -> Any | None:
    """The last visible message of the candidate Claude session transcript,
    or ``None`` if there is no usable candidate.

    Uses the persisted uuid hint to locate the file via the SDK's session
    reader. (A future enhancement could scan-by-cwd when the hint is absent;
    for now an absent/invalid hint simply yields no candidate → synthesis.)
    """
    if not session_uuid:
        return None
    try:
        from claude_agent_sdk import get_session_messages  # noqa: PLC0415

        msgs = get_session_messages(session_uuid, directory=cwd)
        if not msgs:
            return None
        return msgs[-1]
    except Exception:  # noqa: BLE001 — best-effort; no candidate on any error
        return None


def _tail_matches(history_tail: BaseMessage, transcript_tail: Any) -> bool:
    from .session_tail_match import tails_match  # noqa: PLC0415

    return tails_match(history_tail, transcript_tail)


def _synthesize(messages: list[BaseMessage], cwd: str) -> str | None:
    from .session_synthesis import (  # noqa: PLC0415
        synthesize_session_file,
    )

    return synthesize_session_file(messages, cwd=cwd)


def resolve_restore_session(
    thread_id: str, messages: list[BaseMessage], cwd: str
) -> str | None:
    """Decide which session uuid (if any) to resume for ``thread_id``.

    See module docstring. Returns the uuid for ``resume`` or ``None``.
    """
    try:
        if not messages:
            return None

        history_tail = messages[-1]
        hint = _load_uuid_hint(thread_id)
        candidate = _candidate_tail_entry(hint, cwd)

        if candidate is not None and _tail_matches(history_tail, candidate):
            # Approach A — the candidate transcript IS our conversation.
            return hint

        # Approach B — no candidate or it diverged: build one from history.
        return _synthesize(messages, cwd)
    except Exception as exc:  # noqa: BLE001
        # Restore is an optimization; never let it break the turn.
        print(
            f"[claude_code_provider] history restore skipped for "
            f"{thread_id}: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        return None
