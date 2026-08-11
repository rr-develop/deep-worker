"""Persist per-``agent_thread_id`` grok session state
(``sent_count`` + ``grok_session_id``) across agent restarts.

Mirror of ``codex_provider/session_state.py``. Without this a restart
resets ``sent_count`` to 0 and ``grok_session_id`` to ``None``, so the
first ``ainvoke`` would replay the WHOLE LangGraph history into a brand
new grok session — dropping every AIMessage and tool result. Persisting
``sent_count`` (so only the unsent tail is mirrored) and the grok session
id (so the restart can ``session/load`` the existing session) makes the
restart deterministic. A stale/missing session is handled by the
``session/load`` fallback in ``GrokChatModel.ainvoke`` (DESIGN §6.1),
not here.

Storage: ``~/.deep_worker/<thread-base>/grok_session_state.json``, a dict
keyed by full ``agent_thread_id`` (so the main agent and its ``:tm``
thread keep separate entries while sharing the agent's data dir). The
filename is deliberately distinct from ``codex_session_state.json`` /
``claude_session_state.json`` so the providers never collide in a shared
agent data dir.

Filesystem permissions (POLICIES.md §4 / DESIGN §6.1, a related issue): the data
dir is created ``0o700`` and the state file ``0o600`` — owner-only. NO
777/666. The write is atomic: a temp file is written, ``chmod 0o600``,
then ``os.replace``'d into place, so a crash mid-write cannot leave a
half-written or world-readable file.

All functions are best-effort: a corrupt/unreadable file degrades to
``None`` and never raises (the restart then falls back to fresh-session
behaviour).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_STATE_FILENAME = "grok_session_state.json"
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _thread_base(thread_id: str) -> str:
    """The agent data-dir name for an ``agent_thread_id``.

    ``agent-1`` / ``agent-1:tm`` all live under the agent's data
    directory in the user's home. Strip the ``:tm`` role suffix; the base
    before it is the agent name. The ``__epoch_N`` suffix (if a future
    rotation scheme adds one) is stripped defensively so an agent's state
    can never silently split across two files.
    """
    base = thread_id
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
    """Read the whole per-agent state dict (keyed by full thread_id).
    Returns ``{}`` on a missing/corrupt file."""
    path = session_state_path(thread_id)
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _ensure_dir(path: Path) -> None:
    """Create the parent dir with owner-only perms (0o700).

    ``mkdir(mode=...)`` is subject to umask, so we ``chmod`` explicitly
    afterwards to guarantee exactly 0o700 regardless of the process umask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, _DIR_MODE)
    except OSError:
        pass


def save_session_state(
    thread_id: str, *, sent_count: int, session_id: str | None
) -> None:
    """Persist ``sent_count`` and ``session_id`` for ``thread_id``.

    Best-effort: never raises. Other threads' entries in the same file are
    preserved (read-modify-write). The write is atomic and owner-only: the
    temp file is chmod'd 0o600 before being ``os.replace``'d into place.
    """
    try:
        path = session_state_path(thread_id)
        _ensure_dir(path)
        all_state = _read_all(thread_id)
        all_state[thread_id] = {
            "sent_count": sent_count,
            "session_id": session_id,
        }
        tmp = path.with_suffix(f".{thread_id.replace(':', '_')}.tmp")
        tmp.write_text(json.dumps(all_state, indent=2))
        try:
            os.chmod(tmp, _FILE_MODE)
        except OSError:
            pass
        os.replace(tmp, path)
        # Defensive: if the destination pre-existed with looser perms,
        # os.replace keeps the source's perms, but re-assert to be safe.
        try:
            os.chmod(path, _FILE_MODE)
        except OSError:
            pass
    except OSError:
        # Persistence is an optimization for deterministic restart; failing
        # to write must never break the agent.
        pass


def load_session_state(thread_id: str) -> dict[str, Any] | None:
    """Load the persisted state for ``thread_id``, or ``None`` if absent /
    unreadable.

    Returns ``{"sent_count": int, "session_id": str | None}``.
    """
    entry = _read_all(thread_id).get(thread_id)
    if not isinstance(entry, dict) or "sent_count" not in entry:
        return None
    return {
        "sent_count": entry.get("sent_count", 0),
        "session_id": entry.get("session_id"),
    }
