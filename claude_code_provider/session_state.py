"""Persist per-thread Claude session state (sent_count + session-uuid hint)
across agent restarts.

DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.4. Today ``sent_count`` lives only in
the in-memory ``ClientManager``, so a restart resets it to 0 and the broken
empty-session replay path runs. Persisting it (and the last-known Claude
session uuid as a *hint* for locating the resumable transcript) makes restart
deterministic. The uuid is only a hint — the tail-match (``session_tail_match``)
is authoritative — so a stale/rotated uuid can never cause a wrong resume.

Storage: the per-agent state file (``claude_session_state.json``) under the
agent's data directory, a dict
keyed by full thread_id (so the main agent and its ``:tm`` thread keep
separate entries while sharing the agent's data dir). The ``:tm`` and
``__epoch_N`` suffixes are stripped ONLY to choose the directory.

All functions are best-effort: a corrupt/unreadable file degrades to ``None``
and never raises (restart then falls back to current behavior).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STATE_FILENAME = "claude_session_state.json"


def _thread_base(thread_id: str) -> str:
    """The agent data-dir name for a thread_id.

    ``agent_name`` / ``agent_name:tm`` / ``agent_name__epoch_3`` /
    ``agent_name:tm__epoch_2`` all live under the agent's data directory.
    Strip the ``:tm`` role suffix and any ``__epoch_N`` rotation suffix; the
    base before them is the agent name.
    """
    base = thread_id
    # Strip epoch suffix first (it can follow the :tm part).
    if "__epoch_" in base:
        base = base.split("__epoch_", 1)[0]
    if base.endswith(":tm"):
        base = base[: -len(":tm")]
    return base


def session_state_path(thread_id: str) -> Path:
    """Path to the state file for the agent owning ``thread_id``."""
    data_dir = Path.home() / Path("~/.deep_worker").name / _thread_base(thread_id)
    return data_dir / _STATE_FILENAME


def _read_all(thread_id: str) -> dict[str, Any]:
    """Read the whole per-agent state dict (keyed by full thread_id). Returns
    ``{}`` on missing/corrupt file."""
    path = session_state_path(thread_id)
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_session_state(
    thread_id: str, *, sent_count: int, session_uuid: str | None
) -> None:
    """Persist ``sent_count`` and the ``session_uuid`` hint for ``thread_id``.

    Best-effort: never raises. Other threads' entries in the same file are
    preserved (read-modify-write).
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
    unreadable."""
    entry = _read_all(thread_id).get(thread_id)
    if not isinstance(entry, dict) or "sent_count" not in entry:
        return None
    return {
        "sent_count": entry.get("sent_count", 0),
        "session_uuid": entry.get("session_uuid"),
    }


def invalidate_session_state(thread_id: str) -> None:
    """Remove the persisted resume hint for a poisoned Claude session."""
    try:
        path = session_state_path(thread_id)
        all_state = _read_all(thread_id)
        if thread_id not in all_state:
            return
        del all_state[thread_id]
        tmp = path.with_suffix(f".{thread_id.replace(':', '_')}.tmp")
        tmp.write_text(json.dumps(all_state, indent=2))
        tmp.replace(path)
    except OSError:
        pass
