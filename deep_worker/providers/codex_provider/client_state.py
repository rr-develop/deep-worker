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

from .custom_types import ThreadState


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
            # First access in THIS process for this agent_thread_id. Lazily
            # seed ``sent_count`` + ``codex_session_uuid`` from disk so a
            # restart is deterministic: the first ainvoke then mirrors only
            # ``messages[sent_count:]`` (not the whole history) and resumes
            # the existing Codex session by uuid. While the process is hot the
            # slot already exists, so we never re-read the disk (no clobbering
            # of live in-memory values). Best-effort — never raises.
            # See docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.2.
            st = ThreadState()
            try:
                from .session_state import (  # noqa: PLC0415
                    load_session_state,
                )

                loaded = load_session_state(agent_thread_id)
                if loaded is not None:
                    st.sent_count = int(loaded["sent_count"])
                    st.codex_session_uuid = loaded.get("session_uuid")
            except Exception:  # noqa: BLE001 — best-effort; default to fresh
                pass
            self._states[agent_thread_id] = st
        return st

    def persist(self, agent_thread_id: str) -> None:
        """Persist the slot's current ``sent_count`` + ``codex_session_uuid``
        to disk so the next process restart can resume.

        Best-effort: never raises. A no-op if the slot does not exist (nothing
        to persist yet). See docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.2.
        """
        st = self._states.get(agent_thread_id)
        if st is None:
            return
        try:
            from .session_state import (  # noqa: PLC0415
                save_session_state,
            )

            save_session_state(
                agent_thread_id,
                sent_count=st.sent_count,
                session_uuid=st.codex_session_uuid,
            )
        except Exception:  # noqa: BLE001 — persistence must never break a turn
            pass

    def record_summarization(self, agent_thread_id: str) -> None:
        """Apply the summarization reset: drop Codex session, zero
        ``sent_count``, and PERSIST the reset. **Does not touch the key.**

        Persisting the reset is required so a restart that happens right after
        a summarization does not reload the stale pre-summary uuid (DESIGN
        §3.2). Persistence is best-effort.
        """
        st = self._states.get(agent_thread_id)
        if st is None:
            # Defensive: if there's no state yet, just create an empty one
            self._states[agent_thread_id] = ThreadState()
            self.persist(agent_thread_id)
            return
        st.codex_session_uuid = None
        st.sent_count = 0
        # current_exec is *not* cleared here — that's the caller's job
        # (only ``ainvoke`` knows whether the current exec is still
        # consumable). Tests of summarization only exercise the cursor
        # reset; current_exec lifecycle is tested in chat_model tests.
        self.persist(agent_thread_id)

    def forget(self, agent_thread_id: str) -> None:
        """Remove a slot entirely. Intended for tests and explicit
        cleanup (no production caller in current code; matches
        ``claude_code_sdk_provider.client_manager.cleanup_client``)."""
        self._states.pop(agent_thread_id, None)


# ---------------------------------------------------------------------------
# Process-global singleton accessor
# ---------------------------------------------------------------------------
# pyright treats ALL_CAPS module-level names as ``Final`` and refuses to
# let us reassign them. The singleton needs lazy construction on first
# access, so use a lowercase name (it's module-private anyway).
_global_state: CodexClientState | None = None


def get_global_client_state() -> CodexClientState:
    """Return the process-global :class:`CodexClientState`.

    Lazily constructed on first call. Same instance is returned forever
    after — survives LangGraph's model rebuilds because it lives at
    module scope, not on any model instance.
    """
    global _global_state
    if _global_state is None:
        _global_state = CodexClientState()
    return _global_state
