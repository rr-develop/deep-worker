"""Fail-loud semantics for the pre-summarization history offload.

See ``deep_worker/docs/DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md``.

Contract under test (both sync and async paths):

  * ``_offload_to_backend`` / ``_aoffload_to_backend`` return ``str``
    (the file path) on success and RAISE ``HistoryOffloadError`` on any
    failure path. They never return ``None`` any more.

  * ``before_model`` / ``abefore_model`` no longer emit
    ``warnings.warn(...)`` on offload failure. They let
    ``HistoryOffloadError`` propagate so that the agent runtime catch-all
    (``FATAL_EXCEPTION_TYPES``) transitions the agent into
    ``AgentStatus.ERROR``.

  * The raise happens BEFORE ``_create_summary`` — i.e. no summarization
    call is made and no LLM tokens are burned when the offload cannot be
    persisted.

  * The error message contains the target path and the underlying cause
    so the operator can act on it (permissions / disk / locked file / …).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents.middleware.summarization import HistoryOffloadError, SummarizationMiddleware
from deepagents.middleware.summarization_history import serialize_summarized_messages_jsonl

# Reuse the well-tested MockBackend / helpers from the existing test module.
from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    make_conversation_messages,
    make_mock_runtime,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


def _make_mock_model_recording_calls() -> MagicMock:
    """Return a mock model that records whether it was invoked for a summary.

    We want to prove that ``_create_summary`` is NOT called when the offload
    raises — the raise must happen first, so no tokens are burned.
    """
    m = MagicMock()
    m.invoke = MagicMock(return_value=MagicMock(text="SHOULD_NOT_BE_CALLED"))
    m.ainvoke = MagicMock(return_value=MagicMock(text="SHOULD_NOT_BE_CALLED"))
    return m


# ---------------------------------------------------------------------------
# HistoryOffloadError class shape
# ---------------------------------------------------------------------------


class TestHistoryOffloadErrorClass:
    def test_is_runtime_error_subclass(self) -> None:
        """``HistoryOffloadError`` inherits from ``RuntimeError`` so plain
        ``except RuntimeError`` still catches it, but its class NAME
        (``HistoryOffloadError``) is what the runtime matches on for the
        fatal-vs-recoverable classification."""
        assert issubclass(HistoryOffloadError, RuntimeError)

    def test_class_name_is_stable_for_fatal_lookup(self) -> None:
        # The name is the public contract with ``FATAL_EXCEPTION_TYPES``.
        assert HistoryOffloadError.__name__ == "HistoryOffloadError"


# ---------------------------------------------------------------------------
# Sync path: _offload_to_backend and before_model
# ---------------------------------------------------------------------------


class TestSyncOffloadRaises:
    def test_backend_write_returns_error_raises_history_offload_error(self) -> None:
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            middleware._offload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        # Actionable: mentions the path AND the underlying cause.
        assert "/conversation_history/" in str(exc.value)
        assert "Storage unavailable" in str(exc.value)

    def test_backend_write_returns_none_raises_history_offload_error(self) -> None:
        """Explicit coverage: backend method returning ``None`` (no result object)
        must be surfaced with ``backend returned None`` in the error message."""

        class NoneReturningBackend(MockBackend):
            def write(self, path: str, content: str):  # type: ignore[override]
                return None  # type: ignore[return-value]

            def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False):  # type: ignore[override]  # noqa: FBT001, FBT002
                return None  # type: ignore[return-value]

        backend = NoneReturningBackend()
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError, match="backend returned None") as exc:
            middleware._offload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert "/conversation_history/" in str(exc.value)

    def test_backend_write_raises_exception_wrapped_in_history_offload_error(self) -> None:
        backend = MockBackend(write_raises=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            middleware._offload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        # The chained cause is the original exception.
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "Mock write exception" in str(exc.value)

    def test_backend_edit_raises_exception_wrapped_in_history_offload_error(self) -> None:
        existing = serialize_summarized_messages_jsonl(
            [HumanMessage(content="prev", id="prev-1")]
        )
        backend = MockBackend(existing_content=existing, write_raises=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            middleware._offload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "Mock edit exception" in str(exc.value)

    def test_before_model_propagates_history_offload_error(self) -> None:
        """``before_model`` must NOT swallow the error into a warning any
        more. It re-raises so the runtime catch-all can flip the agent to
        ``AgentStatus.ERROR``."""
        backend = MockBackend(should_fail=True, error_message="Disk full")
        model = _make_mock_model_recording_calls()
        middleware = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Disk full"):
            middleware.before_model(state, runtime)

        # Prove the summary model was NEVER invoked — no tokens burned,
        # no partial state mutation.
        assert model.invoke.call_count == 0

    def test_no_warnings_emitted_on_offload_failure(self, recwarn: pytest.WarningsRecorder) -> None:
        """The old ``warnings.warn("Offloading conversation history to
        backend failed…")`` must be gone. If it comes back we've regressed
        to silent-failure."""
        backend = MockBackend(should_fail=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError):
            middleware.before_model(state, runtime)

        # No UserWarning about offload failure — that would mean we're
        # back to the silent-failure contract.
        offload_warnings = [
            w for w in recwarn.list
            if "Offloading conversation history to backend failed" in str(w.message)
        ]
        assert offload_warnings == [], (
            f"Expected no offload-failure warnings, got: {offload_warnings}"
        )


# ---------------------------------------------------------------------------
# Async path: _aoffload_to_backend and abefore_model
# ---------------------------------------------------------------------------


class TestAsyncOffloadRaises:
    @pytest.mark.anyio
    async def test_backend_awrite_returns_error_raises_history_offload_error(self) -> None:
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            await middleware._aoffload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert "/conversation_history/" in str(exc.value)
        assert "Storage unavailable" in str(exc.value)

    @pytest.mark.anyio
    async def test_backend_awrite_returns_none_raises_history_offload_error(self) -> None:
        """Async explicit coverage for ``backend returned None`` message."""

        class NoneReturningBackend(MockBackend):
            async def awrite(self, path: str, content: str):  # type: ignore[override]
                return None  # type: ignore[return-value]

            async def aedit(self, path: str, old_string: str, new_string: str, replace_all: bool = False):  # type: ignore[override]  # noqa: FBT001, FBT002
                return None  # type: ignore[return-value]

        backend = NoneReturningBackend()
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError, match="backend returned None") as exc:
            await middleware._aoffload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert "/conversation_history/" in str(exc.value)

    @pytest.mark.anyio
    async def test_backend_awrite_raises_exception_wrapped_in_history_offload_error(self) -> None:
        backend = MockBackend(write_raises=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            await middleware._aoffload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "Mock awrite exception" in str(exc.value)

    @pytest.mark.anyio
    async def test_backend_aedit_raises_exception_wrapped_in_history_offload_error(self) -> None:
        existing = serialize_summarized_messages_jsonl(
            [HumanMessage(content="prev", id="prev-1")]
        )
        backend = MockBackend(existing_content=existing, write_raises=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        with pytest.raises(HistoryOffloadError) as exc:
            await middleware._aoffload_to_backend(
                backend, [HumanMessage(content="m1", id="h1")]
            )
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "Mock aedit exception" in str(exc.value)

    @pytest.mark.anyio
    async def test_abefore_model_propagates_history_offload_error(self) -> None:
        backend = MockBackend(should_fail=True, error_message="Permission denied")
        model = _make_mock_model_recording_calls()
        middleware = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Permission denied"):
            await middleware.abefore_model(state, runtime)

        # Async summary model was NEVER invoked — no tokens burned.
        assert model.ainvoke.call_count == 0

    @pytest.mark.anyio
    async def test_async_no_warnings_on_offload_failure(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        backend = MockBackend(should_fail=True)
        middleware = SummarizationMiddleware(
            model=_make_mock_model_recording_calls(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError):
            await middleware.abefore_model(state, runtime)

        offload_warnings = [
            w for w in recwarn.list
            if "Offloading conversation history to backend failed" in str(w.message)
        ]
        assert offload_warnings == [], (
            f"Expected no offload-failure warnings, got: {offload_warnings}"
        )
