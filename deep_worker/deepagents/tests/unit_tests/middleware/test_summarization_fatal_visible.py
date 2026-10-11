"""POLICIES.md §12 — summarization must never fail silently.

Incident 2026-08-23. When producing the summary failed on the AUTOMATIC path
(``abefore_model``), the middleware returned the exception text as if it were
the summary:

    return f"Error generating summary: {e!s}", None

``_build_new_messages_with_path`` then wrote that string into state and
``abefore_model`` returned it alongside ``RemoveMessage(REMOVE_ALL_MESSAGES)``,
deleting the real history behind the suppressed error. Agents woke up after a
restart holding ``<summary>Error generating summary: …</summary>`` instead of
their context, with the agent status still IDLE and nothing reporting a fault.

Fourteen confirmed corruptions across four agents — counted as
marker-inside-the-summary-wrapper, not raw grep hits (most other matches are
agents discussing the bug). See POLICIES.md §12.

The behaviour was deliberate and pinned by a passing test
(``test_automatic_model_failure_soft_errors_after_offload``), which asserted
the error string landed in the replacement message. That test WAS the defect;
it is replaced by the invariants below.

Two invariants, both required:
  1. the error string never reaches state, and the failure is a fatal exception;
  2. history is not evicted — no ``RemoveMessage(REMOVE_ALL_MESSAGES)``.

Invariant 2 is the critical one: erasing the history is what makes the failure
unrecoverable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import RemoveMessage

from deepagents.middleware.summarization import (
    SummarizationFailedError,
    SummarizationMiddleware,
)

from .test_summarization_middleware import (
    MockBackend,
    make_conversation_messages,
    make_mock_model,
    make_mock_runtime,
    mock_get_config,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

FORBIDDEN = "Error generating summary"


def _load_sibling_fatal_classifier() -> Any:
    """Load ``is_fatal_exception`` from the deep_worker_cli in THIS repo.

    ``deepagents`` and ``deep_worker_cli`` are separate packages in the same
    repository. An installed/sibling-worktree copy of deep_worker_cli may
    shadow the in-repo one on sys.path, so importing it normally would test
    the taxonomy of a tree we are not editing — the fix would look broken
    here and the regression would look fixed there.
    """
    import importlib.util
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[4]
        / "deep_worker_cli"
        / "deep_worker_cli"
        / "exceptions.py"
    )
    if not src.is_file():  # pragma: no cover - layout guard
        pytest.fail(f"deep_worker_cli/exceptions.py not found at {src}")
    spec = importlib.util.spec_from_file_location("_dw_exceptions_under_test", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_fatal_exception


def _build(model: MagicMock, backend: MockBackend, *, trigger: int = 5) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", trigger),
        keep=("messages", 2),
    )


def _failing_model(exc: BaseException) -> MagicMock:
    model = make_mock_model()
    model.ainvoke = AsyncMock(side_effect=exc)
    return model


def _model_with_failing_port(exc: BaseException) -> MagicMock:
    """A session-stateful model whose summarizer port raises.

    This is the shape that actually broke in production: the ACP/codex
    providers expose ``asummarize`` and the failure happens inside it.
    """
    model = make_mock_model()
    model.ainvoke = AsyncMock(side_effect=AssertionError("must not re-invoke the session"))
    type(model).asummarize = AsyncMock(side_effect=exc)
    return model


# ---------------------------------------------------------------------------
# Invariant 1 — the error text never becomes the summary
# ---------------------------------------------------------------------------


class TestErrorStringNeverEntersState:
    @pytest.mark.asyncio
    async def test_automatic_path_raises_instead_of_soft_returning(self) -> None:
        """The automatic path must fail loud, exactly like the requested path."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        backend = MockBackend()
        middleware = _build(_failing_model(RuntimeError("model down")), backend)

        with mock_get_config("auto-fail-thread"), pytest.raises(Exception) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert FORBIDDEN not in str(excinfo.value), (
            "the failure must surface as an exception, not as summary text"
        )

    @pytest.mark.asyncio
    async def test_raised_error_is_the_fatal_class(self) -> None:
        """Fatal by class NAME — this is what flips the agent to ERROR.

        The taxonomy is loaded from the deep_worker_cli source that sits in
        THIS repository, not from whichever copy happens to be on sys.path:
        running the deepagents suite from a sibling worktree otherwise
        classifies against a stale table and reports a fix as broken.
        """
        is_fatal_exception = _load_sibling_fatal_classifier()

        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(_failing_model(RuntimeError("model down")), MockBackend())

        with mock_get_config("auto-fail-thread"), pytest.raises(SummarizationFailedError) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert is_fatal_exception(excinfo.value), (
            "SummarizationFailedError must be in FATAL_EXCEPTION_TYPES so the "
            "runtime catch-all transitions the agent to AgentStatus.ERROR"
        )

    @pytest.mark.asyncio
    async def test_underlying_cause_is_preserved(self) -> None:
        """The operator must be able to see WHY summarization failed."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(_failing_model(RuntimeError("upstream 503")), MockBackend())

        with mock_get_config("auto-fail-thread"), pytest.raises(SummarizationFailedError) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert "upstream 503" in str(excinfo.value)
        assert excinfo.value.__cause__ is not None, "chain the original exception"

    @pytest.mark.asyncio
    async def test_summarizer_port_failure_also_fails_loud(self) -> None:
        """The production shape: failure inside ``asummarize`` itself."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(_model_with_failing_port(RuntimeError("session/load failed")), MockBackend())

        with mock_get_config("port-fail-thread"), pytest.raises(SummarizationFailedError) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert FORBIDDEN not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_offload_failure_keeps_its_own_class(self) -> None:
        """``HistoryOffloadError`` must not be re-wrapped as a summary failure.

        The two classes tell the operator WHICH stage lost data: offload failed
        (history never reached disk — the dangerous one) versus summary failed
        (history is safe, only the summary is missing). Both are fatal, so a
        re-wrap stays green on every "is it fatal" assertion while destroying
        that distinction. The offload currently runs outside the try block that
        raises ``SummarizationFailedError``; this pins that arrangement.
        """
        from deepagents.middleware.summarization import HistoryOffloadError

        messages = make_conversation_messages(num_old=6, num_recent=2)
        backend = MockBackend(write_raises=True)
        middleware = _build(make_mock_model(), backend)

        with mock_get_config("offload-fail-thread"), pytest.raises(Exception) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert isinstance(excinfo.value, HistoryOffloadError), (
            "an offload failure must keep its own class, not be relabelled "
            f"as a summary failure (got {type(excinfo.value).__name__})"
        )

    @pytest.mark.asyncio
    async def test_missing_port_on_stateful_provider_stays_fatal(self) -> None:
        """The exact message the affected agents received.

        A session-stateful provider without ``asummarize`` must refuse to dump
        — and that refusal must reach the operator as a fault, not as a summary.
        """
        messages = make_conversation_messages(num_old=6, num_recent=2)
        model = make_mock_model()
        model.ainvoke = AsyncMock(return_value=MagicMock(text="should never be called"))
        # Real provider identity: ``_is_session_stateful`` is a deliberate
        # allow-list keyed on the provider string, so a bare mock attribute
        # would take the ordinary (stateless) path and prove nothing.
        model._deep_worker_provider = "claude-code"
        model.profile = {"provider": "claude-code", "max_input_tokens": 100000}
        assert not hasattr(type(model), "asummarize"), "port must be absent for this case"
        middleware = _build(model, MockBackend())

        with mock_get_config("stateful-thread"), pytest.raises(Exception) as excinfo:
            await middleware.abefore_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert FORBIDDEN not in str(excinfo.value)


class TestSyncPathIsFixedToo:
    """The SYNC ``before_model`` carried a second, complete copy of the incident.

    ``before_model`` calls ``self._create_summary``, which deepagents does not
    override — it resolves to the langchain base class, whose body still ends:

        except Exception as e:
            return f"Error generating summary: {e!s}"

    So the sync path offloaded, received the error STRING as the summary, wrapped
    it via ``_build_new_messages_with_path`` and returned it next to
    ``RemoveMessage(REMOVE_ALL_MESSAGES)``. Byte-for-byte the incident.

    Production reaches only ``ainvoke``/``astream``, so this is latent rather
    than live — but §12 says a failure ANYWHERE in the summarization path raises,
    and "currently uninvoked" is not a fix. Any sync ``.invoke()`` reaches it.

    Note the AST guard in this file cannot see this: it scans THIS module's
    source, and the offending literal lives in the installed base class. It
    passed while the sync path still produced the marker — a guard giving false
    confidence, which is why these tests drive behaviour instead.
    """

    def test_sync_before_model_raises_instead_of_summarising_the_error(self) -> None:
        messages = make_conversation_messages(num_old=6, num_recent=2)
        model = make_mock_model()
        model.invoke = MagicMock(side_effect=RuntimeError("model down"))
        middleware = _build(model, MockBackend())

        with mock_get_config("sync-fail-thread"), pytest.raises(SummarizationFailedError) as excinfo:
            middleware.before_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        # The base class's marker may legitimately appear INSIDE the exception
        # text — quoting what the base class returned is useful diagnostics.
        # What the policy forbids is the marker reaching STATE as a summary,
        # which the sibling tests below assert. So the check here is that the
        # failure surfaced as a raise carrying the real cause.
        assert "model down" in str(excinfo.value)
        assert str(excinfo.value).startswith("Failed to generate conversation summary")

    def test_sync_path_does_not_evict_history(self) -> None:
        """No ``RemoveMessage`` and no error text in state on the sync path."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        model = make_mock_model()
        model.invoke = MagicMock(side_effect=RuntimeError("model down"))
        middleware = _build(model, MockBackend())

        result: Any = None
        with mock_get_config("sync-no-erase-thread"):
            try:
                result = middleware.before_model(
                    cast("AgentState[Any]", {"messages": messages}),
                    make_mock_runtime(),
                )
            except SummarizationFailedError:
                result = None

        if result is None:
            return  # failed loud without touching state — correct

        emitted = result.get("messages", [])
        assert not any(isinstance(m, RemoveMessage) for m in emitted), (
            "sync path must not evict history when the summary failed"
        )
        for m in emitted:
            assert FORBIDDEN not in str(getattr(m, "content", ""))

    def test_sync_success_path_still_summarises(self) -> None:
        """Control: the sync path must keep working when the model is healthy."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(make_mock_model("A real summary."), MockBackend())

        with mock_get_config("sync-ok-thread"):
            result = middleware.before_model(
                cast("AgentState[Any]", {"messages": messages}),
                make_mock_runtime(),
            )

        assert result is not None
        contents = " ".join(str(getattr(m, "content", "")) for m in result["messages"])
        assert "A real summary." in contents
        assert FORBIDDEN not in contents


# ---------------------------------------------------------------------------
# Invariant 2 — history is NOT evicted when summarization fails
# ---------------------------------------------------------------------------


class TestHistoryIsNotErasedOnFailure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_factory",
        [
            pytest.param(lambda: _failing_model(RuntimeError("model down")), id="ainvoke"),
            pytest.param(
                lambda: _model_with_failing_port(RuntimeError("port down")), id="asummarize"
            ),
        ],
    )
    async def test_no_remove_all_messages_after_failure(self, model_factory: Any) -> None:
        """``RemoveMessage(REMOVE_ALL_MESSAGES)`` must not be emitted.

        This is what made the incident unrecoverable: the real history was
        deleted in the same state update that installed the error string.
        """
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(model_factory(), MockBackend())

        result: Any = None
        with mock_get_config("no-erase-thread"):
            try:
                result = await middleware.abefore_model(
                    cast("AgentState[Any]", {"messages": messages}),
                    make_mock_runtime(),
                )
            except Exception:  # noqa: BLE001 — the raise is invariant 1's business
                result = None

        if result is None:
            return  # failed loud without touching state — correct

        emitted = result.get("messages", [])
        assert not any(isinstance(m, RemoveMessage) for m in emitted), (
            "history must not be evicted when summarization failed"
        )

    @pytest.mark.asyncio
    async def test_original_messages_survive_the_failure(self) -> None:
        """The caller's message list is left intact and reusable."""
        messages = make_conversation_messages(num_old=6, num_recent=2)
        original_ids = [m.id for m in messages]
        middleware = _build(_failing_model(RuntimeError("model down")), MockBackend())

        with mock_get_config("survive-thread"):
            with pytest.raises(Exception):  # noqa: B017, PT011
                await middleware.abefore_model(
                    cast("AgentState[Any]", {"messages": messages}),
                    make_mock_runtime(),
                )

        assert [m.id for m in messages] == original_ids, (
            "summarization failure must not mutate the live history"
        )


# ---------------------------------------------------------------------------
# The forbidden literal, checked end-to-end on the real state channel
# ---------------------------------------------------------------------------


class TestForbiddenLiteralNeverReachesState:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("model down"),
            TimeoutError("summarize timed out"),
            ValueError("bad payload"),
        ],
    )
    async def test_no_state_message_contains_the_marker(self, exc: BaseException) -> None:
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = _build(_failing_model(exc), MockBackend())

        produced: list[Any] = []
        with mock_get_config("marker-thread"):
            try:
                result = await middleware.abefore_model(
                    cast("AgentState[Any]", {"messages": messages}),
                    make_mock_runtime(),
                )
            except Exception:  # noqa: BLE001
                result = None
            if result:
                produced = result.get("messages", [])

        for m in produced:
            assert FORBIDDEN not in str(getattr(m, "content", "")), (
                f"error text leaked into state as a summary: {m}"
            )

    def test_production_source_has_no_soft_return(self) -> None:
        """AST guard: no ``return`` of the marker string anywhere in the module.

        A behavioural test alone would not catch the string being reintroduced
        on a path this suite does not exercise.
        """
        import ast
        from pathlib import Path

        import deepagents.middleware.summarization as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if FORBIDDEN in sub.value:
                        offenders.append(node.lineno)
        assert not offenders, (
            f"summarization.py returns the forbidden marker at lines {offenders}; "
            "a summarization failure must raise, never return error text as a summary "
            "(POLICIES.md §12)"
        )
