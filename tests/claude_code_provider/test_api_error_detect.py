"""Unit tests for :mod:`claude_code_provider.api_error_detect`.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.2 (detector),
§5.1 (unit pin-tests U1–U10, D1–D13, T1–T3).
"""

from __future__ import annotations

from typing import Any

from deep_worker.providers.claude_code_provider.api_error_detect import (
    ApiErrorDetection,
    UsageVerdict,
    classify_usage,
    detect_api_error,
)


# ---------------------------------------------------------------------------
# The verbatim production 400 text (§1.2 of the design) — used across D-tests.
# ---------------------------------------------------------------------------
REAL_400 = (
    "API Error: 400 messages.18.content.0.tool_result.content.1: "
    "Input tag 'image_url' found using 'type' does not match any of the "
    "expected tags: 'document', 'image', 'search_result', 'text', "
    "'tool_reference'"
)

ALL_ZERO = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

NONZERO_USAGE = {
    "input_tokens": 12,
    "output_tokens": 34,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


# ===========================================================================
# classify_usage — U1..U10  (BLOCKER: absent usage is NOT zero)
# ===========================================================================


def test_U1_none_is_unknown():
    assert classify_usage(None) is UsageVerdict.UNKNOWN


def test_U2_empty_is_unknown():
    assert classify_usage({}) is UsageVerdict.UNKNOWN


def test_U3_only_unrecognised_keys_is_unknown():
    assert classify_usage({"service_tier": "standard", "speed": None}) is UsageVerdict.UNKNOWN


def test_U4_string_counter_is_unknown():
    assert classify_usage({"input_tokens": "0"}) is UsageVerdict.UNKNOWN


def test_U5_bool_counter_is_unknown():
    # bool is a subclass of int in Python — explicitly excluded.
    assert classify_usage({"input_tokens": True}) is UsageVerdict.UNKNOWN
    assert classify_usage({"input_tokens": False}) is UsageVerdict.UNKNOWN


def test_U6_all_recognised_zero_is_known_zero():
    assert classify_usage(ALL_ZERO) is UsageVerdict.KNOWN_ZERO


def test_U7_single_recognised_zero_is_known_zero():
    assert classify_usage({"input_tokens": 0}) is UsageVerdict.KNOWN_ZERO


def test_U8_any_recognised_nonzero_is_nonzero():
    assert classify_usage({"input_tokens": 12, "output_tokens": 0}) is UsageVerdict.NONZERO


def test_U9_unknown_field_is_ignored():
    assert classify_usage({"input_tokens": 0, "unknown_field": 999}) is UsageVerdict.KNOWN_ZERO


def test_U10_negative_int_is_unknown():
    # Round-2 REQUIRED nit (a): malformed, never treated as zero.
    assert classify_usage({"input_tokens": -1}) is UsageVerdict.UNKNOWN
    assert classify_usage({"input_tokens": 0, "output_tokens": -5}) is UsageVerdict.UNKNOWN


# ===========================================================================
# detect_api_error — D1..D13
# ===========================================================================


def _detect(**overrides: Any):
    """detect_api_error with sensible defaults; override just what a test needs."""
    kwargs: dict[str, Any] = dict(
        response_text="",
        assistant_error=None,
        result_is_error=False,
        result_subtype=None,
        result_errors=(),
        result_text=None,
        stop_reason=None,
        usage=None,
    )
    kwargs.update(overrides)
    return detect_api_error(**kwargs)


def test_D1_marker_and_known_zero_detected():
    """Production case (§1.2): marker + all-zero usage + no flag."""
    d = _detect(response_text=REAL_400, usage=ALL_ZERO)
    assert isinstance(d, ApiErrorDetection)
    # Precedence 1 — marker-carrying response_text wins.
    assert d.message == REAL_400


def test_D2_marker_but_nonzero_usage_not_detected():
    """Answer that legitimately quotes an API error — bills tokens."""
    assert _detect(response_text=REAL_400, usage=NONZERO_USAGE) is None


def test_D3_marker_but_usage_none_not_detected():
    """BLOCKER pin: absent usage is UNKNOWN, must not fire rule 4."""
    assert _detect(response_text=REAL_400, usage=None) is None


def test_D4_marker_but_usage_empty_not_detected():
    """BLOCKER pin: {} is UNKNOWN."""
    assert _detect(response_text=REAL_400, usage={}) is None


def test_D5_marker_but_only_unrecognised_usage_not_detected():
    assert _detect(response_text=REAL_400, usage={"service_tier": "standard"}) is None


def test_D6_normal_answer_with_known_zero_not_detected():
    """No marker → rule 4 does not apply; no other signal → not detected."""
    assert _detect(response_text="Here is the answer.", usage=ALL_ZERO) is None


def test_D7_assistant_error_with_normal_text_detected():
    """Rule 1: structured SDK signal alone is sufficient."""
    d = _detect(response_text="Some partial reply", assistant_error="invalid_request", usage=NONZERO_USAGE)
    assert d is not None
    assert d.assistant_error == "invalid_request"
    # MAJOR 1 pin: str(exc) is NEVER the partial assistant prose.
    assert "Some partial reply" not in d.message
    assert d.message  # non-empty
    assert "invalid_request" in d.message


def test_D8_assistant_error_with_empty_text_detected_non_empty_message():
    """MAJOR 1 pin: no empty ClaudeApiResponseError('')."""
    d = _detect(response_text="", assistant_error="invalid_request")
    assert d is not None
    assert d.message  # non-empty
    assert "invalid_request" in d.message


def test_D9_result_is_error_with_normal_text_and_nonzero_usage_detected():
    """Round-2 REQUIRED nit (c): single SDK bool wins."""
    d = _detect(
        response_text="Ordinary looking assistant text.",
        result_is_error=True,
        usage=NONZERO_USAGE,
    )
    assert d is not None
    assert d.message  # non-empty; falls to synthesised description


def test_D10_result_is_error_with_result_text_and_errors_detected():
    """MAJOR 1 precedence 2: result_text wins over joined errors when response_text is empty."""
    d = _detect(
        response_text="",
        result_is_error=True,
        result_errors=("upstream 503",),
        result_text="Retryable",
    )
    assert d is not None
    assert d.message == "Retryable"
    assert d.result_errors == ("upstream 503",)


def test_D10b_result_is_error_no_result_text_falls_to_errors():
    """MAJOR 1 precedence 3: joined errors used when no result_text."""
    d = _detect(
        response_text="",
        result_is_error=True,
        result_errors=("upstream 503", "retry later"),
    )
    assert d is not None
    assert "upstream 503" in d.message
    assert "retry later" in d.message


def test_D11_result_is_error_with_marker_text_prefers_marker():
    """MAJOR 1 precedence 1: marker-carrying response_text beats result_text."""
    d = _detect(
        response_text=REAL_400,
        result_is_error=True,
        result_text="Retryable",
    )
    assert d is not None
    assert d.message == REAL_400


def test_D12_enrichment_fields_alone_without_is_error_not_detected():
    """Round-2 REQUIRED nit (c): enrichment fields alone are NOT proof.

    result_subtype/result_errors without result_is_error=True must not
    trigger; the SDK's single terminal-error bool is authoritative.
    """
    assert _detect(
        response_text="Ordinary looking assistant text.",
        result_is_error=False,
        result_subtype="error_during_execution",
        result_errors=("something bad",),
        usage=NONZERO_USAGE,
    ) is None


def test_D13_malformed_inputs_never_raise():
    """Detector never raises."""
    # bytes, list, weird nested types — all must return None quietly.
    assert _detect(response_text="", usage=b"\x00\x01") is None
    assert _detect(response_text="", usage=[("input_tokens", 0)]) is None  # list, not mapping
    assert _detect(response_text="", usage={"input_tokens": {"nested": 0}}) is None
    # No crash from weird result_errors either.
    assert _detect(response_text="", result_is_error=False, result_errors=None) is None or True  # tolerant


# ===========================================================================
# Detector structured fields carried on ApiErrorDetection
# ===========================================================================


def test_detection_carries_structured_fields():
    d = _detect(
        response_text=REAL_400,
        assistant_error="invalid_request",
        result_is_error=True,
        result_subtype="error_during_execution",
        result_errors=("upstream 503",),
        stop_reason="stop_sequence",
        usage=ALL_ZERO,
    )
    assert d is not None
    assert d.assistant_error == "invalid_request"
    assert d.result_subtype == "error_during_execution"
    assert d.result_errors == ("upstream 503",)
    assert d.stop_reason == "stop_sequence"
    assert d.message == REAL_400  # precedence 1
