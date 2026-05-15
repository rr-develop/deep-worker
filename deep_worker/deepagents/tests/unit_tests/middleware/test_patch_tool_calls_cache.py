"""Tests for PatchToolCallsMiddleware (DESIGN_CONTEXT_DELTA_CACHE.md §4.6).

Behaviours under test:

    * No dangling tool calls → middleware returns None, the state is
      not rewritten and delta-checkpointing can append normally.
    * Dangling tool call present → middleware returns the Overwrite
      with the patched list (current behaviour preserved).
    * Messages already scanned once are NOT re-scanned in a second
      before_agent call on the same instance with the same history.

Tests construct a minimal state dict directly — no LangGraph, no
backend.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _load_patch_tool_calls_module() -> Any:
    """Load ``patch_tool_calls`` without running the parent package's
    __init__, which imports ProjectMemoryMiddleware → semantic_search →
    llama_index. Not every test environment has llama_index; this test
    only needs the narrow middleware class."""
    module_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "deepagents" / "middleware" / "patch_tool_calls.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_isolated_patch_tool_calls", str(module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_isolated_patch_tool_calls"] = mod
    spec.loader.exec_module(mod)
    return mod


_PTC = _load_patch_tool_calls_module()
PatchToolCallsMiddleware = _PTC.PatchToolCallsMiddleware


class _FakeRuntime:
    """Minimal stand-in for langgraph.runtime.Runtime."""

    context: Any = None


def _state(messages: list[Any]) -> dict[str, Any]:
    return {"messages": messages}


class TestPatchToolCallsCache:
    def test_no_dangling_returns_none(self) -> None:
        """When every tool call has a matching ToolMessage, the middleware
        MUST NOT rewrite the message list. Returning a full Overwrite here
        invalidates the delta-checkpointing append assumption."""
        mw = PatchToolCallsMiddleware()
        ai = AIMessage(
            content="",
            id="a1",
            tool_calls=[{"id": "c1", "name": "grep", "args": {}}],
        )
        tm = ToolMessage(content="ok", tool_call_id="c1", name="grep", id="t1")
        msgs = [HumanMessage(content="go", id="h1"), ai, tm]

        out = mw.before_agent(_state(msgs), _FakeRuntime())  # type: ignore[arg-type]

        assert out is None, (
            "clean history must not trigger a full-list Overwrite"
        )

    def test_dangling_tool_call_patched(self) -> None:
        """Dangling tool call still produces a patched list."""
        mw = PatchToolCallsMiddleware()
        ai = AIMessage(
            content="",
            id="a1",
            tool_calls=[{"id": "c1", "name": "grep", "args": {}}],
        )
        # No ToolMessage for c1.
        msgs = [HumanMessage(content="go", id="h1"), ai]

        out = mw.before_agent(_state(msgs), _FakeRuntime())  # type: ignore[arg-type]

        assert out is not None
        assert "messages" in out
        # langgraph.types.Overwrite stores its payload in .value
        patched = list(out["messages"].value)
        assert len(patched) == 3  # original 2 + 1 injected ToolMessage
        # The injected one is a ToolMessage with tool_call_id == c1.
        injected = patched[-1]
        assert isinstance(injected, ToolMessage)
        assert injected.tool_call_id == "c1"

    def test_second_call_with_unchanged_history_is_noop(self) -> None:
        """Re-running before_agent on the same (already-clean) history
        produces None both times. No work accumulates."""
        mw = PatchToolCallsMiddleware()
        ai = AIMessage(
            content="", id="a1",
            tool_calls=[{"id": "c1", "name": "grep", "args": {}}],
        )
        tm = ToolMessage(content="ok", tool_call_id="c1", name="grep", id="t1")
        msgs = [HumanMessage(content="go", id="h1"), ai, tm]

        first = mw.before_agent(_state(msgs), _FakeRuntime())  # type: ignore[arg-type]
        second = mw.before_agent(_state(msgs), _FakeRuntime())  # type: ignore[arg-type]

        assert first is None
        assert second is None

    def test_empty_history_returns_none(self) -> None:
        mw = PatchToolCallsMiddleware()
        assert mw.before_agent(_state([]), _FakeRuntime()) is None  # type: ignore[arg-type]
