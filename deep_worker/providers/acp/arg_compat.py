"""Narrow argument-compatibility layer for ACP provider tool_calls.

ACL-backed CLIs (Grok, Claude Code, …) regularly emit deep-worker tool_calls
with Cursor-style argument keys (``read_file(target_file=...)``) and
human-readable values (``execute(timeout='20s')``). Our tool schemas declare
``file_path`` / ``timeout: int``, so pydantic ``ValidationError`` fires
during tool dispatch.

This module is the single place where such mismatches are rewritten into the
canonical shape our tools expect. It is deliberately **narrow**: only
patterns actually observed in production logs are handled. New mismatches
must be added explicitly with a verbatim-payload from a real run
(POLICIES.md §7 / §9).

The hook site is ``AcpChatModel._map_tool_call`` CASE A only (the proxy-MCP
branch). Built-in passthrough (CASE B) forwards ``rawInput`` verbatim by
design.
"""
from __future__ import annotations

import math
from typing import Any, Callable

# ---------------------------------------------------------------------------
# KEY_ALIASES — {tool_name: {wrong_key: canonical_key}}
# ---------------------------------------------------------------------------
KEY_ALIASES: dict[str, dict[str, str]] = {
    "read_file": {"target_file": "file_path"},
}


# ---------------------------------------------------------------------------
# VALUE_COERCERS — {tool_name: {key: coercer}}
# ---------------------------------------------------------------------------
def _duration_to_int_seconds(value: Any) -> Any:
    """Best-effort coerce a duration to whole seconds.

    Accepts int/float/bool, bare numeric strings, and ``Ns``/``Nm``/``Nh``/
    ``Nms`` suffixed forms (never truncate a sub-second budget to 0).
    Anything else — return the input verbatim so downstream pydantic
    validation surfaces the real problem instead of a silent coercion.
    """
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
    try:
        return int(s)
    except ValueError:
        pass
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
        touching it. Deterministic.
      * Unknown tool / unknown key → pass through unchanged.
    """
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = dict(args)

    aliases = KEY_ALIASES.get(tool_name)
    if aliases:
        for alias_key, canonical_key in aliases.items():
            if alias_key not in out:
                continue
            if canonical_key in out:
                del out[alias_key]
            else:
                out[canonical_key] = out.pop(alias_key)

    coercers = VALUE_COERCERS.get(tool_name)
    if coercers:
        for key, coerce in coercers.items():
            if key not in out:
                continue
            try:
                out[key] = coerce(out[key])
            except Exception:  # noqa: BLE001 — contract: never raise
                pass

    return out
