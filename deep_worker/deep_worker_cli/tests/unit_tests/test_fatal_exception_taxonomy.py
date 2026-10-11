"""Invariants for the fatal-exception taxonomy (``deep_worker_cli.exceptions``).

``FATAL_EXCEPTION_TYPES`` and ``is_fatal_exception`` existed in THREE verbatim
copies in this module (2026-08-23). Python keeps the last definition, so the
first two were dead code that still looked authoritative: adding a new fatal
class to the first copy would have had no effect while appearing applied.

The structural test below is the guard against that regression returning. The
behavioural tests pin what the taxonomy must classify.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deep_worker_cli.exceptions import (
    FATAL_EXCEPTION_TYPES,
    is_fatal_exception,
    is_provider_api_error,
)

_EXCEPTIONS_SRC = (
    Path(__file__).resolve().parents[2] / "deep_worker_cli" / "exceptions.py"
)


def _toplevel_definition_count(name: str) -> int:
    """Count top-level assignments/defs binding ``name`` in exceptions.py."""
    tree = ast.parse(_EXCEPTIONS_SRC.read_text(encoding="utf-8"))
    count = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                count += 1
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                count += 1
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    count += 1
    return count


class TestSingleCanonicalDefinition:
    """Each taxonomy symbol is defined exactly once."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "FATAL_EXCEPTION_TYPES",
            "is_fatal_exception",
            "PROVIDER_API_ERROR_TYPES",
            "is_provider_api_error",
            "RATE_LIMIT_ERROR_TYPES",
            "is_rate_limit_error",
            "CheckpointCorruptedError",
            "ProviderAuthFatalError",
            "AgentNotInitialisedError",
        ],
    )
    def test_defined_exactly_once(self, symbol: str) -> None:
        count = _toplevel_definition_count(symbol)
        assert count == 1, (
            f"{symbol} has {count} top-level definitions in exceptions.py. "
            "Duplicate definitions are dead code that still reads as "
            "authoritative — editing the wrong copy is a silent no-op."
        )


class TestFatalClassification:
    """Behavioural contract of ``is_fatal_exception``."""

    @pytest.mark.parametrize(
        "name",
        [
            "ClaudeUsageLimitError",
            "AcpUsageLimitError",
            "CodexUsageLimitError",
            "GrokUsageLimitError",
            "CheckpointCorruptedError",
            "ProviderAuthFatalError",
            "AgentNotInitialisedError",
            "HistoryOffloadError",
            "SummarizationFailedError",
        ],
    )
    def test_listed_names_are_fatal(self, name: str) -> None:
        exc = type(name, (RuntimeError,), {})()
        assert is_fatal_exception(exc), f"{name} must transition the agent to ERROR"

    def test_matched_by_name_not_by_import(self) -> None:
        """Provider-agnostic: a locally-defined class of the right NAME matches."""
        local = type("CodexUsageLimitError", (Exception,), {})()
        assert is_fatal_exception(local)

    @pytest.mark.parametrize(
        "exc",
        [ValueError("x"), RuntimeError("x"), KeyError("x"), TimeoutError("x")],
    )
    def test_ordinary_errors_are_recoverable(self, exc: BaseException) -> None:
        assert not is_fatal_exception(exc)

    def test_fatal_and_recoverable_sets_are_disjoint(self) -> None:
        """A class cannot be both fatal and a recoverable provider error."""
        for name in FATAL_EXCEPTION_TYPES:
            exc = type(name, (RuntimeError,), {})()
            assert not is_provider_api_error(exc), (
                f"{name} is fatal; it must not also be classified as a "
                "recoverable provider API error"
            )


class TestSummarizationFailureIsFatal:
    """POLICIES.md §12 — a summarization failure is fatal and visible."""

    def test_summarization_failed_error_is_registered(self) -> None:
        assert "SummarizationFailedError" in FATAL_EXCEPTION_TYPES

    def test_middleware_class_name_matches_the_taxonomy_entry(self) -> None:
        """The name in the table must match the class the middleware raises.

        Classification is by class NAME, so a rename on either side silently
        downgrades the failure to "recoverable" — exactly the silent path the
        policy forbids.
        """
        from deepagents.middleware.summarization import (
            HistoryOffloadError,
            SummarizationFailedError,
        )

        assert SummarizationFailedError.__name__ in FATAL_EXCEPTION_TYPES
        assert HistoryOffloadError.__name__ in FATAL_EXCEPTION_TYPES
        assert is_fatal_exception(SummarizationFailedError("boom"))
        assert is_fatal_exception(HistoryOffloadError("boom"))
