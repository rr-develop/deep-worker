"""Persist per-``agent_thread_id`` ACP session state across agent restarts.

Mirror of ``codex_provider/session_state.py`` / ``grok_provider/session_state.py``
with provider scoping (DESIGN §3.5): one file per agent data dir
(``acp_session_state.json``) whose per-``thread_id`` entries carry the
provider profile id AND a session fingerprint ``{adapter, adapter_version,
session-capability}``. On load, a fingerprint mismatch is treated as "no
persisted session" (fresh-session behaviour) so a CLI adapter upgrade can
never resume a session it cannot faithfully continue.

Without persistence a restart resets ``sent_count`` to 0 and ``session_id``
to ``None``, so the first ``ainvoke`` would replay the WHOLE LangGraph
history into a brand new session — dropping every AIMessage and tool result.

Filesystem permissions (POLICIES.md §4): the data dir is created ``0o700``
and the state file ``0o600`` — owner-only. NO 777/666. The write is atomic:
a temp file is written, ``chmod 0o600``, then ``os.replace``'d into place.

All functions are best-effort: a corrupt/unreadable file degrades to
``None`` and never raises (the restart then falls back to fresh-session
behaviour).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from deep_worker_cli.storage_paths import agent_root, deep_worker_home

_STATE_FILENAME = "acp_session_state.json"
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
    data_dir = agent_root(_thread_base(thread_id), create=False)
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
    """Create the parent dir with owner-only perms (0o700)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, _DIR_MODE)
    except OSError:
        pass


def save_session_state(
    thread_id: str,
    *,
    provider: str,
    session_id: str | None,
    sent_count: int,
    fingerprint: dict[str, Any] | None,
    last_response_at: str | None = None,
    last_context_used: int | None = None,
    model_id: str | None = None,
) -> None:
    """Persist the ACP session entry for ``thread_id`` (provider-scoped).

    Best-effort: never raises. Other threads' entries in the same file are
    preserved (read-modify-write). The write is atomic and owner-only: the
    temp file is chmod'd 0o600 before being ``os.replace``'d into place.
    """
    try:
        path = session_state_path(thread_id)
        _ensure_dir(path)
        all_state = _read_all(thread_id)
        entry: dict[str, Any] = {
            "provider": provider,
            "session_id": session_id,
            "sent_count": sent_count,
        }
        if fingerprint is not None:
            entry["fingerprint"] = fingerprint
        if last_response_at is not None:
            entry["last_response_at"] = last_response_at
        if last_context_used is not None:
            entry["last_context_used"] = last_context_used
        if model_id is not None:
            entry["model_id"] = model_id
        all_state[thread_id] = entry
        tmp = path.with_suffix(f".{thread_id.replace(':', '_')}.tmp")
        tmp.write_text(json.dumps(all_state, indent=2))
        try:
            os.chmod(tmp, _FILE_MODE)
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            os.chmod(path, _FILE_MODE)
        except OSError:
            pass
    except OSError:
        pass


def load_session_state(
    thread_id: str, *, provider: str, fingerprint: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Load the persisted state for ``thread_id``, or ``None`` when absent /
    unreadable / provider-mismatched / fingerprint-mismatched.

    Returns ``{"sent_count": int, "session_id": str | None}``.
    """
    entry = _read_all(thread_id).get(thread_id)
    if not isinstance(entry, dict) or "sent_count" not in entry:
        return None
    if entry.get("provider") != provider:
        return None
    stored_fp = entry.get("fingerprint")
    if fingerprint is not None and stored_fp != fingerprint:
        # Adapter/version/capability drift → the persisted session is not
        # trustworthy. Treat as fresh (DESIGN §5.2).
        return None
    return {
        "sent_count": entry.get("sent_count", 0),
        "session_id": entry.get("session_id"),
        "last_response_at": entry.get("last_response_at"),
        "last_context_used": entry.get("last_context_used"),
        "model_id": entry.get("model_id"),
    }
