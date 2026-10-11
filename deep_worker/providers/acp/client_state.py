"""Process-global per-``agent_thread_id`` state for the ACP provider layer.

Why a singleton? LangGraph rebuilds ``AcpChatModel`` instances during
pickling / deserialization. If ``sent_count`` lived on the model instance
it would be lost on every rebuild and the provider would re-send the full
history each turn — exactly what we forbid (DESIGN §0.4). So state is
decoupled from the model and keyed by the deep-worker ``agent_thread_id``
(which equals ``self.name`` in ``agent_process.py``).

Beyond the per-thread ``ThreadState`` slots this also holds a reverse
``session_id -> agent_thread_id`` index. The single shared ACP
reader_task (one per agent process) uses ``thread_for_session`` to route a
``session/update`` notification — which carries ``params["sessionId"]`` —
back to the owning ``ThreadState``.

CRITICAL: ``agent_thread_id`` (the dict key) is read-only input. We never
mutate it. On LangGraph summarization we reset the slot CONTENTS (drop
``session_id``, zero ``sent_count``) and evict the now-dead session from
the reverse index — but the key, and therefore the deep-worker LangGraph
thread it corresponds to, is untouched (lesson L3, DESIGN §12).
"""
from __future__ import annotations

from .custom_types import ThreadState


class AcpClientState:
    """Per-agent_thread_id state container plus the sessionId reverse index.
    Use :func:`get_global_client_state` for the process-wide singleton."""

    def __init__(self) -> None:
        self._states: dict[str, ThreadState] = {}
        # session_id -> agent_thread_id (reverse index for the reader).
        self._session_index: dict[str, str] = {}
        # Provider profile id + session fingerprint, supplied by the model
        # via ``set_profile`` before the first load (persistence scoping).
        self._provider: str | None = None
        self._fingerprint: dict | None = None

    def set_profile(self, provider: str, fingerprint: dict | None = None) -> None:
        """Record the provider profile id + session fingerprint used to scope
        persisted session state (DESIGN §3.5). Called once at model init."""
        self._provider = provider
        self._fingerprint = fingerprint

    # -- slot access --------------------------------------------------------
    def get(self, agent_thread_id: str) -> ThreadState | None:
        return self._states.get(agent_thread_id)

    def get_or_create(self, agent_thread_id: str) -> ThreadState:
        st = self._states.get(agent_thread_id)
        if st is None:
            # First access in THIS process for this agent_thread_id. Lazily
            # seed ``sent_count`` + ``session_id`` from disk so a restart
            # is deterministic: the first ainvoke then mirrors only
            # ``messages[sent_count:]`` (not the whole history) and resumes
            # the existing session by id. While the process is hot the
            # slot already exists, so we never re-read disk (no clobbering of
            # live in-memory values). Best-effort — never raises.
            st = ThreadState()
            try:
                from .session_state import (  # noqa: PLC0415
                    load_session_state,
                )

                if self._provider is not None:
                    loaded = load_session_state(
                        agent_thread_id,
                        provider=self._provider,
                        fingerprint=self._fingerprint,
                    )
                    if loaded is not None:
                        st.sent_count = int(loaded["sent_count"])
                        st.session_id = loaded.get("session_id")
                        st.last_response_at = loaded.get("last_response_at")
                        used = loaded.get("last_context_used")
                        st.last_context_used = int(used) if isinstance(used, int) else None
                        st.last_model_id = loaded.get("model_id")
                        # Re-seed the reverse index so a restart's first
                        # session/load can still be routed by the reader.
                        if st.session_id:
                            self._session_index[st.session_id] = agent_thread_id
            except Exception:  # noqa: BLE001 — best-effort; default to fresh
                pass
            self._states[agent_thread_id] = st
        return st

    # -- sessionId reverse index -------------------------------------------
    def register_session(self, session_id: str, agent_thread_id: str) -> None:
        """Record ``session_id -> agent_thread_id`` so the reader can route
        ``session/update`` notifications. Called right after ``session/new``
        returns."""
        self._session_index[session_id] = agent_thread_id

    def thread_for_session(self, session_id: str) -> ThreadState | None:
        """Resolve the ``ThreadState`` that owns ``session_id``, or
        ``None`` if unknown (e.g. a late update for an evicted session)."""
        agent_thread_id = self._session_index.get(session_id)
        if agent_thread_id is None:
            return None
        return self._states.get(agent_thread_id)

    def _evict_session(self, session_id: str | None) -> None:
        if session_id is not None:
            self._session_index.pop(session_id, None)

    # -- CLI crash recovery (DESIGN §5.3) -----------------------------------
    def on_cli_crash(self, err: BaseException) -> None:
        """Deterministic recoverable-crash transition for ALL threads sharing
        the dead CLI subprocess.

        Fails every outstanding waiter so nothing hangs: builtin passthrough
        Futures (the ToolNode is blocked on them — without this they would
        never resolve) and, via the client, prompt/request/readiness futures.
        Also resets per-thread ``session_live_in_process`` so the next turn
        creates a FRESH session instead of ``session/load``-ing one that died
        with the process.
        """
        for st in self._states.values():
            for fut in list(st.builtin_passthrough_futures.values()):
                if not fut.done():
                    fut.set_exception(err)
            st.builtin_passthrough_futures.clear()
            st.session_live_in_process = False

    # -- persistence --------------------------------------------------------
    def persist(self, agent_thread_id: str) -> None:
        """Persist the slot's current ``sent_count`` + ``session_id`` to disk
        so the next restart can resume. Best-effort: never raises. No-op if
        the slot does not exist or no provider profile is set."""
        st = self._states.get(agent_thread_id)
        if st is None or self._provider is None:
            return
        try:
            from .session_state import (  # noqa: PLC0415
                save_session_state,
            )

            save_session_state(
                agent_thread_id,
                provider=self._provider,
                session_id=st.session_id,
                sent_count=st.sent_count,
                fingerprint=self._fingerprint,
                last_response_at=st.last_response_at,
                last_context_used=st.last_context_used,
                model_id=st.last_model_id,
            )
        except Exception:  # noqa: BLE001 — persistence must never break a turn
            pass

    def persist_state(self, st: ThreadState) -> None:
        """Persist the slot that owns ``st``, looked up by identity.

        The context-gauge stamp (``_stamp_context_observation``) only holds
        the ``ThreadState``, not the key. Without this the freshly written
        ``last_context_used`` / ``last_response_at`` sat in memory until the
        NEXT turn's persist — so a ``--restart`` right after a completed
        turn lost the gauge, and ``extra_floor`` came back blind. That is
        the exact blindness the 2026-08-19 incident turned on.
        """
        for key, candidate in self._states.items():
            if candidate is st:
                self.persist(key)
                return

    # -- summarization (L3) -------------------------------------------------
    def record_summarization(self, agent_thread_id: str) -> None:
        """Apply the summarization reset: drop the CLI session, zero
        ``sent_count``, clear the context gauge, evict the stale session id
        from the reverse index, and PERSIST the reset. **Does not touch the
        key.**

        Evicting the reverse-index entry prevents the reader from routing a
        late ``session/update`` for the now-dead session to this slot.
        Persisting the reset is required so a restart right after a
        summarization does not reload the stale pre-summary session id.

        The gauge (``last_context_used`` / ``last_response_at``) MUST be
        nulled here: it describes the pre-summary session that no longer
        exists. ``extra_floor`` reads it, so leaving 468k behind makes the
        very next turn — or a ``--restart`` before it — re-trigger
        summarization on a context that was already collapsed (reviewer B1,
        2026-08-21).
        """
        st = self._states.get(agent_thread_id)
        if st is None:
            # Defensive: no state yet — create an empty one and persist.
            self._states[agent_thread_id] = ThreadState()
            self.persist(agent_thread_id)
            return
        self._evict_session(st.session_id)
        st.session_id = None
        st.sent_count = 0
        st.last_context_used = None
        st.last_response_at = None
        st.forwarded_first_system_chars = 0
        st.forwarded_extra_system_keys.clear()
        self.clear_queued_turn_state(st)
        self.persist(agent_thread_id)

    @staticmethod
    def clear_queued_turn_state(st: ThreadState) -> None:
        """Drop mid-turn residue so a post-summary turn cannot resume a dead session.

        DESIGN_SUMMARIZATION_HOT_COLD §5.1: the 2026-08-19 race left
        ``queued_tool_calls`` from the dropped session; mid-turn resume then
        swallowed the summarizer output as the user-facing reply.
        """
        st.queued_text.clear()
        st.queued_tool_calls.clear()
        st.queued_response = None
        fut = st.pending_session_prompt_future
        st.pending_session_prompt_future = None
        if fut is not None and not fut.done():
            fut.cancel()
        st.current_notif_queue = None
        st.pump_task = None


# ---------------------------------------------------------------------------
# Process-global singleton.
# ---------------------------------------------------------------------------
_GLOBAL_CLIENT_STATE: AcpClientState | None = None


def get_global_client_state() -> AcpClientState:
    """Return the process-wide :class:`AcpClientState` singleton."""
    global _GLOBAL_CLIENT_STATE
    if _GLOBAL_CLIENT_STATE is None:
        _GLOBAL_CLIENT_STATE = AcpClientState()
    return _GLOBAL_CLIENT_STATE
