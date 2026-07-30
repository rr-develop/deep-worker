"""Tests for ``SummarizationMiddleware.apply_pseudo_summary`` (startup query).

See ``deep_worker/deep_worker_cli/docs/DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.4.

Contract under test:

  * ``apply_pseudo_summary(thread_id, summary_text, current_messages)`` is a
    PURE PRODUCER: it evicts ``current_messages`` to the JSONL history for
    ``thread_id`` and returns ``(new_messages, file_path)``. It does NOT write
    to any checkpointer and does NOT emit ``RemoveMessage`` (Blocker-2 — the
    caller owns the wipe).

  * The offload targets ``<history_prefix>/<thread_id>.jsonl`` via an explicit
    ``thread_id`` argument — NOT via langgraph's ``get_config()`` contextvar
    (Blocker-1, option B). It must resolve the right path with NO graph run
    and NO config-context.

  * The additive ``thread_id`` override on ``_get_history_path`` /
    ``_offload_to_backend`` / ``_aoffload_to_backend`` defaults to ``None``,
    preserving the existing contextvar behaviour for ``before_model`` callers
    (regression guard).

  * Empty history (after filtering previous summaries) → skip offload, no
    JSONL reference in the summary-shape message, ``file_path=None`` (§4.4
    empty-history invariant).

  * Backend must be a concrete instance; a factory backend is a hard-fail
    (Gap-4).

  * Backend write failure → ``HistoryOffloadError`` (fail-loud, ae16f7c).
"""

from __future__ import annotations

from typing import Any, Callable, cast

import pytest
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage

from deepagents.middleware.summarization import (
    HistoryOffloadError,
    SummarizationMiddleware,
)

from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    make_conversation_messages,
    make_mock_model,
    mock_get_config,
)


def _make_middleware(backend: Any) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=make_mock_model(),
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
        history_path_prefix="/conversation_history",
    )


# ---------------------------------------------------------------------------
# Blocker-1: explicit thread_id override → correct JSONL path, no graph
# ---------------------------------------------------------------------------


class TestApplyPseudoSummaryOffloadPath:
    @pytest.mark.asyncio
    async def test_apply_pseudo_summary_offload_path_uses_explicit_thread_id(
        self,
    ) -> None:
        """Called with an explicit thread_id and NO config-context, the offload
        must write to ``.../agent7.jsonl`` (not ``session_<uuid>.jsonl``)."""
        backend = MockBackend()
        mw = _make_middleware(backend)

        messages = make_conversation_messages(num_old=4, num_recent=0)
        new_msgs, file_path = await mw.apply_pseudo_summary(
            "agent7", "start fresh", cast("list[AnyMessage]", messages)
        )

        assert file_path == "/conversation_history/agent7.jsonl"
        # The write actually targeted that path.
        assert backend.write_calls, "expected a write to the offload file"
        written_path = backend.write_calls[0][0]
        assert written_path == "/conversation_history/agent7.jsonl"
        assert "session_" not in written_path
        # Summary-shape message references the JSONL file.
        assert len(new_msgs) == 1
        assert "/conversation_history/agent7.jsonl" in new_msgs[0].content

    @pytest.mark.asyncio
    async def test_offload_helpers_default_thread_id_none_preserves_contextvar_behaviour(
        self,
    ) -> None:
        """Regression guard: with ``thread_id=None`` the helpers resolve via
        ``_get_thread_id()`` (contextvar) exactly as before — existing
        ``before_model`` callers are unaffected."""
        backend = MockBackend()
        mw = _make_middleware(backend)

        # No explicit thread_id → uses contextvar (mocked).
        with mock_get_config("legacy-thread"):
            path = mw._get_history_path()
        assert path == "/conversation_history/legacy-thread.jsonl"

        # And the explicit override wins when provided.
        assert (
            mw._get_history_path("override-thread")
            == "/conversation_history/override-thread.jsonl"
        )


# ---------------------------------------------------------------------------
# Blocker-2: pure producer — no RemoveMessage, no checkpointer mutation
# ---------------------------------------------------------------------------


class TestApplyPseudoSummaryPureProducer:
    @pytest.mark.asyncio
    async def test_apply_pseudo_summary_does_not_emit_remove_message(self) -> None:
        backend = MockBackend()
        mw = _make_middleware(backend)
        messages = make_conversation_messages(num_old=4, num_recent=0)

        new_msgs, _ = await mw.apply_pseudo_summary(
            "agentX", "go", cast("list[AnyMessage]", messages)
        )
        assert all(not isinstance(m, RemoveMessage) for m in new_msgs)
        assert len(new_msgs) == 1
        assert isinstance(new_msgs[0], HumanMessage)

    @pytest.mark.asyncio
    async def test_apply_pseudo_summary_returns_summary_shape_message(self) -> None:
        backend = MockBackend()
        mw = _make_middleware(backend)
        messages = make_conversation_messages(num_old=4, num_recent=0)

        new_msgs, _ = await mw.apply_pseudo_summary(
            "agentX", "the-sum-query-text", cast("list[AnyMessage]", messages)
        )
        assert new_msgs[0].additional_kwargs.get("lc_source") == "summarization"
        assert "the-sum-query-text" in new_msgs[0].content


# ---------------------------------------------------------------------------
# Empty-history invariant
# ---------------------------------------------------------------------------


class TestApplyPseudoSummaryEmptyHistory:
    @pytest.mark.asyncio
    async def test_empty_history_skips_offload_no_reference_in_summary(self) -> None:
        backend = MockBackend()
        mw = _make_middleware(backend)

        new_msgs, file_path = await mw.apply_pseudo_summary("agentX", "hello", [])

        assert file_path is None
        assert not backend.write_calls, "must not write JSONL for empty history"
        assert not backend.edit_calls
        assert len(new_msgs) == 1
        # No dangling file link in the summary-shape message.
        assert ".jsonl" not in new_msgs[0].content
        assert "hello" in new_msgs[0].content

    @pytest.mark.asyncio
    async def test_only_previous_summary_messages_are_treated_as_empty(self) -> None:
        """A history containing ONLY previous summary messages filters to empty
        → skip offload, no reference."""
        backend = MockBackend()
        mw = _make_middleware(backend)
        messages = make_conversation_messages(
            num_old=0, num_recent=0, include_previous_summary=True
        )

        new_msgs, file_path = await mw.apply_pseudo_summary(
            "agentX", "hello", cast("list[AnyMessage]", messages)
        )
        assert file_path is None
        assert not backend.write_calls
        assert ".jsonl" not in new_msgs[0].content


# ---------------------------------------------------------------------------
# Gap-4: factory backend hard-fail
# ---------------------------------------------------------------------------


class TestApplyPseudoSummaryBackendResolution:
    @pytest.mark.asyncio
    async def test_apply_pseudo_summary_hard_fails_when_backend_is_factory(
        self,
    ) -> None:
        """When the middleware was built with a factory backend (a Callable,
        not a BackendProtocol instance), apply_pseudo_summary hard-fails with
        RuntimeError — a ToolRuntime is unavailable outside a graph run."""

        def factory(_runtime: Any) -> Any:  # a factory, not a BackendProtocol
            return MockBackend()

        mw = _make_middleware(cast("Any", factory))
        messages = make_conversation_messages(num_old=4, num_recent=0)

        with pytest.raises(RuntimeError):
            await mw.apply_pseudo_summary(
                "agentX", "go", cast("list[AnyMessage]", messages)
            )


# ---------------------------------------------------------------------------
# Fail-loud on backend write failure
# ---------------------------------------------------------------------------


class TestApplyPseudoSummaryFailLoud:
    @pytest.mark.asyncio
    async def test_raises_history_offload_error_on_backend_failure(self) -> None:
        backend = MockBackend(should_fail=True, error_message="disk full")
        mw = _make_middleware(backend)
        messages = make_conversation_messages(num_old=4, num_recent=0)

        with pytest.raises(HistoryOffloadError) as exc:
            await mw.apply_pseudo_summary(
                "agentX", "go", cast("list[AnyMessage]", messages)
            )
        assert "agentX.jsonl" in str(exc.value)
        assert "disk full" in str(exc.value)
