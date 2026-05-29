"""Unit tests for ``_dedupe_tools_by_name_priority`` (Fix A).

Design doc:
    deep_worker_cli/docs/DESIGN_FIX_TOOL_NAME_PRIORITY_AND_DANGLING_TOOLCALLS.md

Bug A: for a Codex agent the bound tool set contains BOTH base/proxy tools and
the Codex native passthrough tools. ``web_search`` exists in both; LangGraph's
ToolNode resolves a name by "last registration wins", and native is appended
last, so the native passthrough silently overrides the proxy tool.

Fix A: a single, order-preserving "keep first by name" de-dup applied in
``create_cli_agent`` before ``create_deep_agent``. Base/proxy tools are
assembled first, so they win; native is a fallback for names not already
present. User decision Q3: on a collision the BASE tool wins.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

# NOTE: import fails until the implementation lands — expected TDD red phase.
from deep_worker_cli.agent import _dedupe_tools_by_name_priority


def _names(tools: list[Any]) -> list[str]:
    return [getattr(t, "name", None) for t in tools]


def _make_tool(tool_name: str):
    @tool(tool_name)
    def _t(x: str) -> str:
        """A trivial test tool."""
        return x

    return _t


class TestDedupe:
    def test_no_duplicates_unchanged(self) -> None:
        a, b, c = _make_tool("read_file"), _make_tool("execute"), _make_tool("grep")
        out = _dedupe_tools_by_name_priority([a, b, c])
        assert _names(out) == ["read_file", "execute", "grep"]
        assert out == [a, b, c]

    def test_keep_first_drops_later_duplicate(self) -> None:
        first = _make_tool("web_search")   # base/proxy (assembled first)
        other = _make_tool("read_file")
        native = _make_tool("web_search")  # native passthrough (appended last)
        out = _dedupe_tools_by_name_priority([first, other, native])
        # first web_search survives, the later duplicate is dropped.
        assert _names(out) == ["web_search", "read_file"]
        assert out[0] is first
        assert native not in out

    def test_order_preserved(self) -> None:
        t1, t2, t3, t4 = (
            _make_tool("alpha"),
            _make_tool("beta"),
            _make_tool("alpha"),  # dup of t1
            _make_tool("gamma"),
        )
        out = _dedupe_tools_by_name_priority([t1, t2, t3, t4])
        assert _names(out) == ["alpha", "beta", "gamma"]
        assert out == [t1, t2, t4]

    def test_real_collision_web_search_base_wins_native_dropped(self) -> None:
        """Models the actual bound set: base web_search first, several native
        tools after (incl. native web_search + non-colliding apply_patch)."""
        base_ws = _make_tool("web_search")            # base/proxy
        base_read = _make_tool("read_file")
        native_cmd = _make_tool("command_execution")  # native-only, kept
        native_ws = _make_tool("web_search")          # native dup, dropped
        native_patch = _make_tool("apply_patch")      # native-only, kept

        out = _dedupe_tools_by_name_priority(
            [base_ws, base_read, native_cmd, native_ws, native_patch]
        )
        assert _names(out) == [
            "web_search",
            "read_file",
            "command_execution",
            "apply_patch",
        ]
        # the surviving web_search is the BASE one (Q3: base wins).
        assert out[0] is base_ws
        assert native_ws not in out
        # non-colliding native tools are retained.
        assert native_cmd in out and native_patch in out

    def test_tools_without_name_passed_through(self) -> None:
        class _NoName:
            pass

        n1, n2 = _NoName(), _NoName()
        named = _make_tool("read_file")
        out = _dedupe_tools_by_name_priority([n1, named, n2])
        # objects without a .name are never deduped away.
        assert n1 in out and n2 in out and named in out
        assert len(out) == 3

    def test_empty_list(self) -> None:
        assert _dedupe_tools_by_name_priority([]) == []
