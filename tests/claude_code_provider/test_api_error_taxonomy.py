"""Taxonomy tests for :class:`ClaudeApiResponseError`.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.1 (exception
type), §5.1 T1–T3.

The runtime routes exception handling by CLASS NAME through the two
frozensets in :mod:`deep_worker_cli.exceptions`
(``FATAL_EXCEPTION_TYPES`` / ``PROVIDER_API_ERROR_TYPES``).
``ClaudeApiResponseError`` must live in the RECOVERABLE bucket only.
"""

from __future__ import annotations

from deep_worker.providers.claude_code_provider.chat_model import ClaudeApiResponseError
from deep_worker_cli.exceptions import (
    is_fatal_exception,
    is_provider_api_error,
)


def test_T1_is_provider_api_error():
    assert is_provider_api_error(ClaudeApiResponseError("boom")) is True


def test_T2_is_not_fatal():
    assert is_fatal_exception(ClaudeApiResponseError("boom")) is False


def test_T3_structured_fields_preserved_and_str_non_empty():
    """str(exc) uses the message; structured fields stay on the instance."""
    exc = ClaudeApiResponseError(
        "real API text",
        assistant_error="invalid_request",
        result_subtype="error_during_execution",
        result_errors=("upstream 503",),
        stop_reason="stop_sequence",
    )
    assert str(exc) == "real API text"
    assert exc.assistant_error == "invalid_request"
    assert exc.result_subtype == "error_during_execution"
    assert exc.result_errors == ("upstream 503",)
    assert exc.stop_reason == "stop_sequence"


def test_defaults_are_neutral():
    """Instantiable with only the message; all structured fields default to None/()."""
    exc = ClaudeApiResponseError("x")
    assert str(exc) == "x"
    assert exc.assistant_error is None
    assert exc.result_subtype is None
    assert exc.result_errors == ()
    assert exc.stop_reason is None


def test_result_errors_is_tuple_even_if_list_passed():
    exc = ClaudeApiResponseError("x", result_errors=["a", "b"])
    assert exc.result_errors == ("a", "b")
    assert isinstance(exc.result_errors, tuple)
