"""Unit tests for grok_provider.arg_compat.

Contract per docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md
§3.2 — narrow compat layer for live-observed Grok arg mismatches.
"""
from __future__ import annotations

import pytest

from grok_provider.arg_compat import normalize_tool_args


# ---------------------------------------------------------------------------
# KEY_ALIASES — read_file(target_file=...) → read_file(file_path=...)
# ---------------------------------------------------------------------------


def test_read_file_target_file_renamed_to_file_path() -> None:
    out = normalize_tool_args("read_file", {"target_file": "/x"})
    assert out == {"file_path": "/x"}


def test_read_file_no_op_when_correct_key() -> None:
    out = normalize_tool_args("read_file", {"file_path": "/x"})
    assert out == {"file_path": "/x"}


def test_alias_collision_keeps_canonical_drops_alias() -> None:
    """Both target_file and file_path present → keep file_path, drop alias.

    Reviewer clarification #1: do not merge, do not overwrite, do not raise.
    """
    out = normalize_tool_args(
        "read_file",
        {"target_file": "/from_alias", "file_path": "/canonical"},
    )
    assert out == {"file_path": "/canonical"}


def test_read_file_preserves_unrelated_keys_alongside_rename() -> None:
    out = normalize_tool_args(
        "read_file",
        {"target_file": "/x", "offset": 100, "limit": 50},
    )
    assert out == {"file_path": "/x", "offset": 100, "limit": 50}


# ---------------------------------------------------------------------------
# VALUE_COERCERS — execute(timeout="20s") → execute(timeout=20)
# ---------------------------------------------------------------------------


def test_execute_timeout_string_seconds_coerced() -> None:
    out = normalize_tool_args("execute", {"command": "sleep 1", "timeout": "20s"})
    assert out == {"command": "sleep 1", "timeout": 20}


def test_execute_timeout_string_minutes_coerced() -> None:
    out = normalize_tool_args("execute", {"command": "x", "timeout": "2m"})
    assert out == {"command": "x", "timeout": 120}


def test_execute_timeout_string_hours_coerced() -> None:
    out = normalize_tool_args("execute", {"command": "x", "timeout": "1h"})
    assert out == {"command": "x", "timeout": 3600}


def test_execute_timeout_string_ms_coerced_ceil_at_least_1() -> None:
    """500ms → 1 second (never truncate a sub-second budget to zero)."""
    out = normalize_tool_args("execute", {"command": "x", "timeout": "500ms"})
    assert out == {"command": "x", "timeout": 1}


def test_execute_timeout_string_ms_over_second() -> None:
    """1500ms → ceil(1.5) → 2 seconds."""
    out = normalize_tool_args("execute", {"command": "x", "timeout": "1500ms"})
    assert out == {"command": "x", "timeout": 2}


def test_execute_timeout_int_passes_through() -> None:
    out = normalize_tool_args("execute", {"command": "x", "timeout": 30})
    assert out == {"command": "x", "timeout": 30}


def test_execute_timeout_float_coerced_to_int() -> None:
    out = normalize_tool_args("execute", {"command": "x", "timeout": 45.7})
    assert out == {"command": "x", "timeout": 45}


def test_execute_timeout_bare_numeric_string_coerced() -> None:
    """'20' (no suffix) → int(20). Reviewer note: optional-but-add coverage."""
    out = normalize_tool_args("execute", {"command": "x", "timeout": "20"})
    assert out == {"command": "x", "timeout": 20}


def test_garbage_timeout_string_passthrough() -> None:
    """Unknown shape → return unchanged so pydantic surfaces the real type error.

    Do NOT invent a value; the point is to keep coercion narrow and safe.
    """
    out = normalize_tool_args("execute", {"command": "x", "timeout": "forever"})
    assert out == {"command": "x", "timeout": "forever"}


def test_execute_no_timeout_key_no_op() -> None:
    out = normalize_tool_args("execute", {"command": "x"})
    assert out == {"command": "x"}


# ---------------------------------------------------------------------------
# General contract
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_args_unchanged() -> None:
    out = normalize_tool_args("some_other_tool", {"foo": "bar", "n": 1})
    assert out == {"foo": "bar", "n": 1}


def test_empty_args_returns_empty() -> None:
    out = normalize_tool_args("read_file", {})
    assert out == {}


def test_none_args_treated_as_empty() -> None:
    """Defensive: some code paths pass None instead of {}. Must not raise."""
    out = normalize_tool_args("read_file", None)  # type: ignore[arg-type]
    assert out == {}


def test_original_dict_not_mutated() -> None:
    """normalize_tool_args must return a NEW dict (no side effects on caller)."""
    src = {"target_file": "/x", "offset": 100}
    out = normalize_tool_args("read_file", src)
    assert src == {"target_file": "/x", "offset": 100}
    assert out == {"file_path": "/x", "offset": 100}


def test_normalize_never_raises_on_odd_input() -> None:
    """Contract: never raises. Coercers swallow their own failures."""
    # A coercer that receives a completely wrong type must not raise.
    out = normalize_tool_args("execute", {"command": "x", "timeout": [1, 2, 3]})
    # List is not in the coercer's accepted shapes → passthrough.
    assert out == {"command": "x", "timeout": [1, 2, 3]}
