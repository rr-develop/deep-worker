"""Narrow argument-compatibility layer for grok tool_calls.

Grok's model regularly emits deep-worker tool_calls with Cursor-style
argument keys (``read_file(target_file=...)``) and human-readable
values (``execute(timeout='20s')``). Our tool schemas declare
``file_path`` / ``timeout: int``, so pydantic ``ValidationError`` fires
during tool dispatch — which, under the old status callback, flipped
the whole agent to ERROR (see the DESIGN doc below).

This module is the single place where such mismatches are rewritten
into the canonical shape our tools expect. It is deliberately
**narrow**: only patterns actually observed in production Grok logs
are handled. New mismatches must be added explicitly with a
verbatim-payload from a real run (POLICIES.md §7 / §9).

The hook site is ``grok_provider.chat_model.GrokChatModel._map_tool_call``
CASE A only (the proxy-MCP branch). Built-in passthrough (CASE B)
forwards ``rawInput`` verbatim by design (see
``builtin_tool_passthrough.py``) — do NOT plug this in there.

See docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md.
"""
from __future__ import annotations

import math
from typing import Any, Callable

# ---------------------------------------------------------------------------
# KEY_ALIASES — {tool_name: {wrong_key: canonical_key}}
#
# Live-observed sources (grep the agent error logs for
# "validation error for <tool>" + inspect the offending input_value=):
#
#   * read_file: target_file → file_path
#     (Cursor's read_file schema exposes ``target_file``; Grok's model
#     borrows that name.)
# ---------------------------------------------------------------------------
KEY_ALIASES: dict[str, dict[str, str]] = {
    "read_file": {"target_file": "file_path"},
}


# ---------------------------------------------------------------------------
# VALUE_COERCERS — {tool_name: {key: coercer}}
#
# Each coercer takes the raw value and returns a corrected one. It MUST NOT
# raise; on any unexpected shape it returns the input unchanged so pydantic
# surfaces the real type error (never invent a value).
# ---------------------------------------------------------------------------


def _duration_to_int_seconds(value: Any) -> Any:
    """Best-effort coerce a duration to whole seconds.

    Accepts:
      - int / bool → int(value)  (bool is a subclass of int; forward as-is)
      - float → int(value) (truncation is fine for seconds granularity)
      - "20"        → 20            (bare numeric string)
      - "20s"       → 20
      - "2m"        → 120
      - "1h"        → 3600
      - "500ms"     → 1  (never truncate a sub-second budget to 0)
      - "1500ms"    → 2  (math.ceil(x/1000))

    Anything else — return the input verbatim so downstream pydantic
    validation surfaces the real problem instead of a silent coercion.
    """
    # bool is a subclass of int — treat as int.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    # Bare integer / float string, no suffix.
    try:
        return int(s)
    except ValueError:
        pass
    # Suffixed forms. Order matters — "ms" MUST be tested before "s".
    try:
        if s.endswith("ms"):
            ms_str = s[:-2].strip()
            ms = float(ms_str)
            if ms <= 0:
                return value
            return max(1, math.ceil(ms / 1000.0))
        if s.endswith("s"):
            return int(float(s[:-1].strip()))
        if s.endswith("m"):
            return int(float(s[:-1].strip()) * 60)
        if s.endswith("h"):
            return int(float(s[:-1].strip()) * 3600)
    except (ValueError, TypeError):
        return value
    return value


VALUE_COERCERS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "execute": {"timeout": _duration_to_int_seconds},
}


def normalize_tool_args(tool_name: str, args: Any) -> dict[str, Any]:
    """Return a new dict with alias renames + value coercions applied.

    Contract:
      * Never raises. Odd/None inputs → ``{}`` or verbatim passthrough.
      * Never mutates the caller's dict (always returns a fresh copy).
      * Alias collision (both wrong and canonical keys present):
        keep the canonical value verbatim, drop the alias without
        touching it. Deterministic — no dict-insertion-order surprises.
      * Unknown tool / unknown key → pass through unchanged.

    See docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md
    §3.2 for the rationale and the alias-collision rule (reviewer #1).
    """
    if not isinstance(args, dict):
        return {}
    # Fresh copy so we never mutate the caller's dict.
    out: dict[str, Any] = dict(args)

    # 1) Key aliases.
    aliases = KEY_ALIASES.get(tool_name)
    if aliases:
        for alias_key, canonical_key in aliases.items():
            if alias_key not in out:
                continue
            if canonical_key in out:
                # Collision: keep canonical, drop alias.
                del out[alias_key]
            else:
                out[canonical_key] = out.pop(alias_key)

    # 2) Value coercers.
    coercers = VALUE_COERCERS.get(tool_name)
    if coercers:
        for key, coerce in coercers.items():
            if key not in out:
                continue
            try:
                out[key] = coerce(out[key])
            except Exception:  # noqa: BLE001 — contract: never raise
                # Leave the value untouched so pydantic surfaces the
                # real type error.
                pass

    return out
