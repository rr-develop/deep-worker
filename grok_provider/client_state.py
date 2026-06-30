"""Process-global per-``agent_thread_id`` state for the grok provider.

Why a singleton? LangGraph rebuilds ``GrokChatModel`` instances during
pickling / deserialization. If ``sent_count`` lived on the model instance
it would be lost on every rebuild and the provider would re-send the full
history each turn — exactly what we forbid (DESIGN §0.4). So state is
decoupled from the model and keyed by the deep-worker ``agent_thread_id``
(which equals ``self.name`` in ``agent_process.py``).

Beyond the per-thread ``ThreadState`` slots this also holds a reverse
``grok_session_id -> agent_thread_id`` index. The single shared ACP
reader_task (one per agent process) uses ``thread_for_session`` to route a
``session/update`` notification — which carries ``params["sessionId"]`` —
back to the owning ``ThreadState`` (DESIGN §11A.4, §12).

CRITICAL: ``agent_thread_id`` (the dict key) is read-only input. We never
mutate it. On LangGraph summarization we reset the slot CONTENTS (drop
``grok_session_id``, zero ``sent_count``) and evict the now-dead session
from the reverse index — but the key, and therefore the deep-worker
LangGraph thread it corresponds to, is untouched (lesson L3, DESIGN §12).
"""
from __future__ import annotations

from grok_provider.custom_types import ThreadState


class GrokClientState:
    """Per-agent_thread_id state container plus the sessionId reverse index.
    Use :func:`get_global_client_state` for the process-wide singleton."""

    def __init__(self) -> None:
        self._states: dict[str, ThreadState] = {}
        # grok_session_id -> agent_thread_id (reverse index for the reader).
        self._session_index: dict[str, str] = {}

    # -- slot access --------------------------------------------------------
    def get(self, agent_thread_id: str) -> ThreadState | None:
        return self._states.get(agent_thread_id)

    def get_or_create(self, agent_thread_id: str) -> ThreadState:
        st = self._states.get(agent_thread_id)
        if st is None:
            # First access in THIS process for this agent_thread_id. Lazily
            # seed ``sent_count`` + ``grok_session_id`` from disk so a restart
            # is deterministic: the first ainvoke then mirrors only
            # ``messages[sent_count:]`` (not the whole history) and resumes
            # the existing grok session by id. While the process is hot the
            # slot already exists, so we never re-read disk (no clobbering of
            # live in-memory values). Best-effort — never raises.
            st = ThreadState()
            try:
                from grok_provider.session_state import (  # noqa: PLC0415
                    load_session_state,
                )

                loaded = load_session_state(agent_thread_id)
                if loaded is not None:
                    st.sent_count = int(loaded["sent_count"])
                    st.grok_session_id = loaded.get("session_id")
                    # Re-seed the reverse index so a restart's first
                    # session/load can still be routed by the reader.
                    if st.grok_session_id:
                        self._session_index[st.grok_session_id] = agent_thread_id
            except Exception:  # noqa: BLE001 — best-effort; default to fresh
                pass
            self._states[agent_thread_id] = st
        return st

    # -- sessionId reverse index -------------------------------------------
    def register_session(self, grok_session_id: str, agent_thread_id: str) -> None:
        """Record ``grok_session_id -> agent_thread_id`` so the reader can
        route ``session/update`` notifications. Called right after
        ``session/new`` returns (DESIGN §10 step 10)."""
        self._session_index[grok_session_id] = agent_thread_id

    def thread_for_session(self, grok_session_id: str) -> ThreadState | None:
        """Resolve the ``ThreadState`` that owns ``grok_session_id``, or
        ``None`` if unknown (e.g. a late update for an evicted session)."""
        agent_thread_id = self._session_index.get(grok_session_id)
        if agent_thread_id is None:
            return None
        return self._states.get(agent_thread_id)

    def _evict_session(self, grok_session_id: str | None) -> None:
        if grok_session_id is not None:
            self._session_index.pop(grok_session_id, None)

    # -- persistence --------------------------------------------------------
    def persist(self, agent_thread_id: str) -> None:
        """Persist the slot's current ``sent_count`` + ``grok_session_id`` to
        disk so the next restart can resume. Best-effort: never raises. No-op
        if the slot does not exist."""
        st = self._states.get(agent_thread_id)
        if st is None:
            return
        try:
            from grok_provider.session_state import (  # noqa: PLC0415
                save_session_state,
            )

            save_session_state(
                agent_thread_id,
                sent_count=st.sent_count,
                session_id=st.grok_session_id,
            )
        except Exception:  # noqa: BLE001 — persistence must never break a turn
            pass

    # -- summarization (L3) -------------------------------------------------
    def record_summarization(self, agent_thread_id: str) -> None:
        """Apply the summarization reset: drop the grok session, zero
        ``sent_count``, evict the stale session id from the reverse index,
        and PERSIST the reset. **Does not touch the key.**

        Evicting the reverse-index entry prevents the reader from routing a
        late ``session/update`` for the now-dead session to this slot.
        Persisting the reset is required so a restart right after a
        summarization does not reload the stale pre-summary session id.
        """
        st = self._states.get(agent_thread_id)
        if st is None:
            # Defensive: no state yet — create an empty one and persist.
            self._states[agent_thread_id] = ThreadState()
            self.persist(agent_thread_id)
            return
        self._evict_session(st.grok_session_id)
        st.grok_session_id = None
        st.sent_count = 0
        st.forwarded_first_system_chars = 0
        st.forwarded_extra_system_keys.clear()
        # pending_session_prompt_future / current_notif_queue are NOT cleared
        # here — that is the caller's concern (the in-flight turn, if any,
        # owns them). builtin_passthrough_futures likewise belongs to the
        # live turn.
        self.persist(agent_thread_id)


# ---------------------------------------------------------------------------
# Process-global singleton.
# ---------------------------------------------------------------------------
_GLOBAL_CLIENT_STATE: GrokClientState | None = None


def get_global_client_state() -> GrokClientState:
    """Return the process-wide :class:`GrokClientState` singleton."""
    global _GLOBAL_CLIENT_STATE
    if _GLOBAL_CLIENT_STATE is None:
        _GLOBAL_CLIENT_STATE = GrokClientState()
    return _GLOBAL_CLIENT_STATE
