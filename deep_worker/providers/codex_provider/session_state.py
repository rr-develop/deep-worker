"""Persist per-``agent_thread_id`` Codex session state
(``sent_count`` + ``codex_session_uuid``) across agent restarts.

See ``docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md``. Today the Codex provider
keeps ``sent_count`` and ``codex_session_uuid`` only in the in-memory
``CodexClientState`` singleton, so a restart resets them to 0 / ``None`` and
the first ``ainvoke`` replays the WHOLE LangGraph history into a brand-new
(empty) Codex session — dropping every AIMessage and tool result, so Codex
"forgets" everything but the raw user prompts.

Persisting ``sent_count`` (so only the unsent tail is mirrored) and the Codex
session uuid (so spawn resumes the existing session via
``codex exec resume <uuid>``) makes restart deterministic. Unlike claude-code-sdk
the uuid is authoritative, NOT a hint: Codex assigns it once in its first
``thread.started`` event and it is stable for the session's life, and Codex
owns the transcript store at ``CODEX_HOME/sessions/<uuid>.jsonl``. A
stale/missing session is handled by the resume fallback in
``CodexChatModel.ainvoke`` (DESIGN §6.1), not here.

Storage: the per-agent state file (``codex_session_state.json``) under the
agent's data directory, a dict
keyed by full ``agent_thread_id`` (so the main agent and its ``:tm`` thread
keep separate entries while sharing the agent's data dir). The ``:tm`` (and a
defensive ``__epoch_N``) suffix is stripped ONLY to choose the directory.

The filename is ``codex_session_state.json`` — deliberately distinct from
claude-code-sdk's ``claude_session_state.json`` so the two providers never collide
in a shared agent data dir.

All functions are best-effort: a corrupt/unreadable file degrades to ``None``
and never raises (restart then falls back to current behaviour).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from deep_worker_cli.storage_paths import agent_root

_STATE_FILENAME = "codex_session_state.json"


def _thread_base(thread_id: str) -> str:
    """The agent data-dir name for an ``agent_thread_id``.

    ``agent_name`` / ``agent_name:tm`` all live under the agent's data
    directory. Strip the ``:tm`` role suffix; the base
    before it is the agent name. Codex has no ``__epoch_N`` rotation suffix
    today, but we strip it defensively (harmless no-op for current ids) so a
    future rotation scheme cannot silently split an agent's state across two
    files.
    """
    base = thread_id
    # Strip epoch suffix first (it can follow the :tm part) — defensive.
    if "__epoch_" in base:
        base = base.split("__epoch_", 1)[0]
    if base.endswith(":tm"):
        base = base[: -len(":tm")]
    return base


def session_state_path(thread_id: str) -> Path:
    """Path to the state file for the agent owning ``thread_id``."""
    data_dir = agent_root(_thread_base(thread_id), create=False)
    return data_dir / _STATE_FILENAME


def _read_all(thread_id: str) -> dict[str, Any]:
    """Read the whole per-agent state dict (keyed by full thread_id). Returns
    ``{}`` on a missing/corrupt file."""
    path = session_state_path(thread_id)
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_session_state(
    thread_id: str, *, sent_count: int, session_uuid: str | None
) -> None:
    """Persist ``sent_count`` and ``session_uuid`` for ``thread_id``.

    Best-effort: never raises. Other threads' entries in the same file are
    preserved (read-modify-write), and the write is atomic (temp-file replace)
    so a crash mid-write cannot leave a half-written file.
    """
    try:
        path = session_state_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        all_state = _read_all(thread_id)
        all_state[thread_id] = {
            "sent_count": sent_count,
            "session_uuid": session_uuid,
        }
        tmp = path.with_suffix(f".{thread_id.replace(':', '_')}.tmp")
        tmp.write_text(json.dumps(all_state, indent=2))
        tmp.replace(path)
    except OSError:
        # Persistence is an optimization for deterministic restart; failing to
        # write must never break the agent.
        pass


def load_session_state(thread_id: str) -> dict[str, Any] | None:
    """Load the persisted state for ``thread_id``, or ``None`` if absent /
    unreadable.

    Returns ``{"sent_count": int, "session_uuid": str | None}``.
    """
    entry = _read_all(thread_id).get(thread_id)
    if not isinstance(entry, dict) or "sent_count" not in entry:
        return None
    return {
        "sent_count": entry.get("sent_count", 0),
        "session_uuid": entry.get("session_uuid"),
    }
