"""MCPToolNameMangler — an exact bidirectional bijection of MCP tool names.

Contract (see docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md §3):

* ``mangle(canonical)``: a pure function, replaces EVERY ``.`` and ``/``
  with ``__`` (double underscore). No other distortions.
* ``MCPToolNameMangler.add(canonical) -> mangled``: registers the tool in the
  bijection; a collision on the final name → ``MCPToolNameCollisionError``.
* ``to_mangled(canonical)`` / ``to_canonical(mangled)``: lookup, returns
  the argument as-is when absent
  (``mangled = forward.get(canonical, canonical)``).
* Only actually mangled names go into ``forward``/``reverse``
  (if ``mangle(canonical) == canonical``, we do not write to the map).
"""
from __future__ import annotations

import pytest

from deep_worker_cli.external_mcp.mangler import (
    MCPToolNameCollisionError,
    MCPToolNameMangler,
    mangle,
)


# ─────────────────────────── pure mangle() function ──────────────────────────


def test_mangle_replaces_dot_with_double_underscore() -> None:
    assert mangle("browser.list_tabs") == "browser__list_tabs"


def test_mangle_replaces_slash_with_double_underscore() -> None:
    assert mangle("user/profile/update") == "user__profile__update"


def test_mangle_handles_multiple_separators() -> None:
    # Both dots and slashes, and their combinations.
    assert mangle("chrome.browser.status") == "chrome__browser__status"
    assert mangle("admin/tools/list") == "admin__tools__list"
    assert mangle("ns.sub/item.method") == "ns__sub__item__method"


def test_mangle_leaves_safe_names_unchanged() -> None:
    # The name is already in ^[a-zA-Z0-9_-]+$ — no distortion.
    assert mangle("create_issue") == "create_issue"
    assert mangle("getUser") == "getUser"
    assert mangle("DATA_EXPORT_v2") == "DATA_EXPORT_v2"
    assert mangle("a-b-c") == "a-b-c"
    assert mangle("plain") == "plain"


def test_mangle_preserves_existing_double_underscore_in_source() -> None:
    # No "escaping": a double underscore in the source name stays as-is.
    # (If it collides with a dot/slash mapping — add() catches that,
    # see below.)
    assert mangle("a__b") == "a__b"


def test_mangle_is_idempotent_on_already_mangled() -> None:
    # mangle(mangle(x)) == mangle(x) — there are no dots/slashes in the output.
    once = mangle("ns.tool")
    twice = mangle(once)
    assert once == twice == "ns__tool"


# ──────────────────────── MCPToolNameMangler — add ───────────────────────────


def test_mangler_returns_canonical_for_safe_name_and_does_not_store_map() -> None:
    m = MCPToolNameMangler()
    out = m.add("create_issue")
    assert out == "create_issue"
    # Unmangled names are NOT stored in the map.
    assert m.to_mangled("create_issue") == "create_issue"
    assert m.to_canonical("create_issue") == "create_issue"
    # Internal: forward must be empty for unmangled names.
    assert "create_issue" not in m.forward
    assert "create_issue" not in m.reverse


def test_mangler_stores_bijection_only_for_mangled_names() -> None:
    m = MCPToolNameMangler()
    out = m.add("chrome.browser.status")
    assert out == "chrome__browser__status"
    # Only mangled names go into the map.
    assert m.forward == {"chrome.browser.status": "chrome__browser__status"}
    assert m.reverse == {"chrome__browser__status": "chrome.browser.status"}


def test_mangler_lookup_returns_argument_unchanged_for_unknown() -> None:
    # Contract: mangled = forward.get(canonical, canonical)
    # and the reverse — the same.
    m = MCPToolNameMangler()
    # Nothing registered — both lookups return the argument as-is.
    assert m.to_mangled("never_added") == "never_added"
    assert m.to_canonical("never_added") == "never_added"


def test_mangler_to_mangled_and_to_canonical_roundtrip_after_add() -> None:
    m = MCPToolNameMangler()
    mangled = m.add("a.b.c")
    assert mangled == "a__b__c"
    assert m.to_mangled("a.b.c") == "a__b__c"
    assert m.to_canonical("a__b__c") == "a.b.c"


def test_mangler_idempotent_re_add_of_same_canonical_is_noop() -> None:
    # Re-adding the same canonical is not a collision — it just
    # returns the same mangled (important for retry-safe init).
    m = MCPToolNameMangler()
    first = m.add("a.b")
    second = m.add("a.b")
    assert first == second == "a__b"
    assert m.forward == {"a.b": "a__b"}
    assert m.reverse == {"a__b": "a.b"}


# ─────────────────────────────── collisions ───────────────────────────────────


def test_collision_two_canonicals_yield_same_mangled() -> None:
    """`a.b` and `a/b` both mangle to `a__b` — fail-loud."""
    m = MCPToolNameMangler()
    m.add("a.b")
    with pytest.raises(MCPToolNameCollisionError):
        m.add("a/b")


def test_collision_mangled_first_then_literal_with_same_name() -> None:
    """First `a.b`→`a__b` is registered; then a literal tool
    with the name `a__b` (mangle = identity) → collision."""
    m = MCPToolNameMangler()
    m.add("a.b")
    with pytest.raises(MCPToolNameCollisionError):
        m.add("a__b")


def test_collision_literal_first_then_mangled_into_same_name() -> None:
    """First the literal `a__b` (not written to the map, but
    `_all_mangled` stores it); then `a.b`→`a__b` → collision.
    This scenario is the main argument for the `_all_mangled` set."""
    m = MCPToolNameMangler()
    m.add("a__b")
    with pytest.raises(MCPToolNameCollisionError):
        m.add("a.b")


def test_collision_error_carries_both_names() -> None:
    m = MCPToolNameMangler()
    m.add("ns.tool")
    with pytest.raises(MCPToolNameCollisionError) as ei:
        m.add("ns/tool")
    msg = str(ei.value)
    assert "ns/tool" in msg
    assert "ns__tool" in msg


def test_non_colliding_additions_coexist_in_one_mangler() -> None:
    m = MCPToolNameMangler()
    a = m.add("ns.a")
    b = m.add("ns.b")
    c = m.add("plain")
    d = m.add("other/x")
    assert (a, b, c, d) == ("ns__a", "ns__b", "plain", "other__x")
    assert m.to_canonical("ns__a") == "ns.a"
    assert m.to_canonical("ns__b") == "ns.b"
    # `plain` was not mangled — it is not in the map, the lookup returns the argument.
    assert m.to_canonical("plain") == "plain"
    assert m.to_canonical("other__x") == "other/x"


def test_mangler_independent_instances_do_not_share_state() -> None:
    m1 = MCPToolNameMangler()
    m2 = MCPToolNameMangler()
    m1.add("a.b")
    # m2 knows nothing — `a.b` registers freely without a collision.
    assert m2.add("a.b") == "a__b"
    assert "a.b" not in m1.forward or m1.forward["a.b"] == "a__b"
