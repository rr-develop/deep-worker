"""Process-global per-``agent_thread_id`` state for the Codex provider.

Why a singleton? LangGraph rebuilds ``CodexChatModel`` instances during
pickling / deserialization. If ``sent_count`` lived on the model
instance, it would be lost on every rebuild and the provider would
re-send the full history each turn — exactly what we forbid (§0.4 of
DESIGN_CODEX_PROVIDER.md). So state is decoupled from the model and
keyed by the deep-worker ``agent_thread_id`` (which equals
``self.name`` in ``agent_process.py``).

This module is intentionally tiny — two scalars per agent_thread_id
plus a possibly-live ExecHandle. See ``custom_types.ThreadState``.

CRITICAL: ``agent_thread_id`` (the dict key) is read-only input. We
never mutate it. On LangGraph summarization we reset the *contents*
of the slot (drop ``codex_session_uuid``, set ``sent_count=0``), but
the key — and therefore the deep-worker LangGraph thread it
corresponds to — is untouched. See lesson L3 in §12.
"""
from __future__ import annotations

from codex_provider.custom_types import ThreadState


class CodexClientState:
    """Per-agent_thread_id state container. Use :func:`get_global_client_state`
    for the process-wide singleton."""

    def __init__(self) -> None:
        self._states: dict[str, ThreadState] = {}

    def get(self, agent_thread_id: str) -> ThreadState | None:
        return self._states.get(agent_thread_id)

    def get_or_create(self, agent_thread_id: str) -> ThreadState:
        st = self._states.get(agent_thread_id)
        if st is None:
            st = ThreadState()
            self._states[agent_thread_id] = st
        return st

    def record_summarization(self, agent_thread_id: str) -> None:
        """Apply the summarization reset: drop Codex session, zero
        ``sent_count``. **Does not touch the key.**"""
        st = self._states.get(agent_thread_id)
        if st is None:
            # Defensive: if there's no state yet, just create an empty one
            self._states[agent_thread_id] = ThreadState()
            return
        st.codex_session_uuid = None
        st.sent_count = 0
        # current_exec is *not* cleared here — that's the caller's job
        # (only ``ainvoke`` knows whether the current exec is still
        # consumable). Tests of summarization only exercise the cursor
        # reset; current_exec lifecycle is tested in chat_model tests.

    def forget(self, agent_thread_id: str) -> None:
        """Remove a slot entirely. Intended for tests and explicit
        cleanup (no production caller in current code; matches
        ``claude_code_provider.client_manager.cleanup_client``)."""
        self._states.pop(agent_thread_id, None)


# ---------------------------------------------------------------------------
# Process-global singleton accessor
# ---------------------------------------------------------------------------
_GLOBAL_STATE: CodexClientState | None = None


def get_global_client_state() -> CodexClientState:
    """Return the process-global :class:`CodexClientState`.

    Lazily constructed on first call. Same instance is returned forever
    after — survives LangGraph's model rebuilds because it lives at
    module scope, not on any model instance.
    """
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        _GLOBAL_STATE = CodexClientState()
    return _GLOBAL_STATE
