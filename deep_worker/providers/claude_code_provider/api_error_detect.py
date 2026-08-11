"""Detect Claude CLI turns that carry an upstream API error instead of an answer.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.2 / §5.1.

The detector is pure, side-effect free, and NEVER raises. The caller
(``chat_model.ainvoke``) decides whether to consult it based on the
authoritative ``returning_tool_calls`` state (§3.2b); once consulted, the
detector classifies the terminal turn on its own merits — prior tool blocks
in the same turn do NOT suppress a terminal API error.

Return value: ``ApiErrorDetection`` when an error is detected, ``None``
otherwise. ``ApiErrorDetection.message`` is the pre-built exception message
per the §3.2 precedence ladder — the caller feeds it directly into
``ClaudeApiResponseError``.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The CLI repacks upstream Anthropic HTTP-4xx/5xx errors into a single
# ``TextBlock`` whose text starts with this prefix. Observed across recorded
# incident transcripts.
_API_ERROR_PREFIX = "API Error:"

# Recognised integer token counters — taken from the observed transcript
# schema (§1.2a of the design). Any other key in ``usage`` is ignored for
# the verdict.
_TOKEN_COUNTER_KEYS: frozenset[str] = frozenset({
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
})


# ---------------------------------------------------------------------------
# Usage classification (BLOCKER fix)
# ---------------------------------------------------------------------------

class UsageVerdict(enum.Enum):
    """Three-valued verdict on the terminal turn's ``usage`` payload.

    * ``KNOWN_ZERO`` — at least one recognised integer counter is present,
      every present recognised counter is an ``int`` (not ``bool``), every
      such counter is ``0``.
    * ``NONZERO`` — at least one recognised integer counter is present and
      positive (``> 0``). A real assistant answer always bills input tokens,
      so this proves the turn is a real answer.
    * ``UNKNOWN`` — nothing usable to judge on: ``None``, ``{}``, only
      unrecognised keys, all recognised keys have unusable types (str,
      ``bool``, ...), or any recognised counter is negative (malformed).

    Callers must NOT collapse ``UNKNOWN`` into ``KNOWN_ZERO`` — that would
    misclassify a legitimate text-only reply whose transport did not record
    ``usage`` as an error (reviewer BLOCKER, round 2).
    """

    KNOWN_ZERO = "known_zero"
    NONZERO = "nonzero"
    UNKNOWN = "unknown"


def classify_usage(usage: Mapping[str, Any] | None) -> UsageVerdict:
    """Classify a ``usage`` mapping. NEVER raises."""
    if usage is None:
        return UsageVerdict.UNKNOWN
    usable_zeros = 0
    for key in _TOKEN_COUNTER_KEYS:
        if key not in usage:
            continue
        value = usage[key]
        # ``bool`` is a subclass of ``int`` — explicitly exclude it.
        if isinstance(value, bool):
            continue
        if not isinstance(value, int):
            continue
        if value < 0:
            # Malformed — SDK should never emit a negative token count. Treat
            # as unusable (round-2 REQUIRED nit a). A single malformed
            # counter poisons the whole verdict: we do NOT trust that "the
            # other counters are fine" because we do not know why this one
            # is negative.
            return UsageVerdict.UNKNOWN
        if value > 0:
            return UsageVerdict.NONZERO
        # value == 0
        usable_zeros += 1

    if usable_zeros > 0:
        return UsageVerdict.KNOWN_ZERO
    return UsageVerdict.UNKNOWN


# ---------------------------------------------------------------------------
# Text marker
# ---------------------------------------------------------------------------

def looks_like_api_error_text(text: str | None) -> bool:
    """True iff the WHOLE turn text is a CLI-repacked upstream API error."""
    if not isinstance(text, str):
        return False
    return text.strip().startswith(_API_ERROR_PREFIX)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApiErrorDetection:
    """Structured detection returned by :func:`detect_api_error`.

    ``message`` is the pre-built exception message per §3.2 precedence:
    marker-carrying ``response_text`` → ``result_text`` → joined
    ``result_errors`` → synthesised description from structured fields.
    Never empty; never ordinary partial assistant prose.
    """

    message: str
    assistant_error: str | None
    result_subtype: str | None
    result_errors: tuple[str, ...]
    stop_reason: str | None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def _coerce_errors(errors: Any) -> tuple[str, ...]:
    """Normalise ``result_errors`` into a tuple of str; never raise."""
    if errors is None:
        return ()
    if isinstance(errors, str):
        return (errors,)
    if isinstance(errors, Iterable):
        try:
            return tuple(str(e) for e in errors)
        except Exception:
            return ()
    return ()


def _build_message(
    *,
    response_text: str | None,
    result_text: str | None,
    result_errors: tuple[str, ...],
    assistant_error: str | None,
    result_subtype: str | None,
    stop_reason: str | None,
) -> str:
    """§3.2 precedence — always returns a non-empty string."""
    # 1. marker-carrying response_text
    if isinstance(response_text, str) and looks_like_api_error_text(response_text):
        stripped = response_text.strip()
        if stripped:
            return stripped
    # 2. result_text
    if isinstance(result_text, str) and result_text.strip():
        return result_text.strip()
    # 3. joined result_errors
    joined = "; ".join(e for e in result_errors if e)
    if joined:
        return joined
    # 4. synthesised description from structured fields
    parts: list[str] = []
    if assistant_error:
        parts.append(f"assistant_error={assistant_error}")
    if result_subtype:
        parts.append(f"subtype={result_subtype}")
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    tail = f" ({', '.join(parts)})" if parts else ""
    return f"Claude API error{tail}"


def detect_api_error(
    *,
    response_text: str | None,
    assistant_error: str | None,
    result_is_error: bool,
    result_subtype: str | None = None,
    result_errors: Any = (),
    result_text: str | None = None,
    stop_reason: str | None = None,
    usage: Mapping[str, Any] | None = None,
) -> ApiErrorDetection | None:
    """Return a structured detection, or ``None``. NEVER raises.

    Decision rule (§3.2):

    1. ``assistant_error`` set → error.
    2. Else ``result_is_error`` is True → error.
    3. Else marker text AND ``classify_usage`` is ``KNOWN_ZERO`` → error.
    4. Else → not an error.

    ``result_subtype`` / ``result_errors`` / ``result_text`` / ``stop_reason``
    are ENRICHMENT ONLY — used to build the message text, never as
    independent proof of error (round-2 REQUIRED nit c).
    """
    try:
        errors_tuple = _coerce_errors(result_errors)

        # Rule 1 — assistant_error is a truthy structured SDK signal.
        if assistant_error:
            return ApiErrorDetection(
                message=_build_message(
                    response_text=response_text,
                    result_text=result_text,
                    result_errors=errors_tuple,
                    assistant_error=str(assistant_error),
                    result_subtype=result_subtype,
                    stop_reason=stop_reason,
                ),
                assistant_error=str(assistant_error),
                result_subtype=result_subtype,
                result_errors=errors_tuple,
                stop_reason=stop_reason,
            )

        # Rule 2 — the SDK's single terminal-error bool.
        if bool(result_is_error):
            return ApiErrorDetection(
                message=_build_message(
                    response_text=response_text,
                    result_text=result_text,
                    result_errors=errors_tuple,
                    assistant_error=None,
                    result_subtype=result_subtype,
                    stop_reason=stop_reason,
                ),
                assistant_error=None,
                result_subtype=result_subtype,
                result_errors=errors_tuple,
                stop_reason=stop_reason,
            )

        # Rule 3 — marker AND measured KNOWN_ZERO usage (conservative fallback).
        if looks_like_api_error_text(response_text) and classify_usage(usage) is UsageVerdict.KNOWN_ZERO:
            return ApiErrorDetection(
                message=_build_message(
                    response_text=response_text,
                    result_text=result_text,
                    result_errors=errors_tuple,
                    assistant_error=None,
                    result_subtype=result_subtype,
                    stop_reason=stop_reason,
                ),
                assistant_error=None,
                result_subtype=result_subtype,
                result_errors=errors_tuple,
                stop_reason=stop_reason,
            )

        return None
    except Exception:
        # Contract: never raise. On any unforeseen malformed input, treat as
        # "not detected" so the caller falls through to the existing
        # normal-answer path.
        return None
