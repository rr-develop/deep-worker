"""Regression: empty bind_tools() must NEVER strip previously-bound tools.

This guards against incident 2026-06-12: after LangGraph summarization /
session rotation, ``bind_tools([])`` was being invoked on the model
(observed 321 times in a single session via ``[ClaudeCodeChatModel.bind_tools]
Called with 0 tools``). The bound result silently went on to create a
``ProxyMCPServer`` with **zero** tools, so the Claude CLI ended up with no
``mcp__deepworker__*`` tools — the agent lost ``execute``/``read_file``/etc.

Root cause:
- ``bind_tools()`` creates a *new* ``ClaudeCodeChatModel`` instance.
- ``__init__`` defaults ``_tools_metadata = []``.
- An empty rebind therefore wiped the tools without warning.

Contract enforced here:
1. ``bind_tools([])`` on an already-tooled model is a **no-op** — it returns
   ``self`` and tools_metadata is preserved.
2. ``bind_tools([])`` on a fresh untooled model is also a no-op (still
   returns self, still has zero tools — symmetric).
3. ``bind_tools([t1])`` -> ``bind_tools([])`` -> the resulting model still
   exposes ``t1`` (the "after-rotation" path).
4. Replacing tools with a non-empty list still works (no regression of
   bind_tools' real purpose).
5. The original model is unchanged after any of the above (LangChain
   immutability invariant must hold).
"""
from __future__ import annotations

from langchain_core.tools import tool as langchain_tool

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel


@langchain_tool
def _t_alpha(x: int) -> int:
    """Alpha tool."""
    return x


@langchain_tool
def _t_beta(s: str) -> str:
    """Beta tool."""
    return s


class TestEmptyRebindNeverStripsTools:
    def test_empty_rebind_on_tooled_model_preserves_tools(self) -> None:
        """The reported bug: bind_tools([]) on a tooled model wiped tools."""
        model = ClaudeCodeChatModel(base_options=None)
        bound = model.bind_tools([_t_alpha, _t_beta])
        assert len(bound.tools_metadata) == 2, "precondition"

        rebound = bound.bind_tools([])

        # Tools survive — this is the contract.
        names = [m["name"] for m in rebound.tools_metadata]
        assert "_t_alpha" in names
        assert "_t_beta" in names
        assert len(rebound.tools_metadata) == 2

    def test_empty_rebind_returns_same_instance_on_tooled_model(self) -> None:
        """No-op should not allocate a new model — it's free."""
        model = ClaudeCodeChatModel(base_options=None)
        bound = model.bind_tools([_t_alpha])

        rebound = bound.bind_tools([])

        assert rebound is bound

    def test_empty_rebind_on_fresh_model_is_noop(self) -> None:
        """Symmetric: empty bind on a fresh model is also a no-op."""
        model = ClaudeCodeChatModel(base_options=None)
        rebound = model.bind_tools([])
        assert rebound is model
        assert rebound.tools_metadata == []

    def test_nonempty_rebind_still_replaces_tools(self) -> None:
        """Non-empty bind_tools must still work — no regression of real use."""
        model = ClaudeCodeChatModel(base_options=None)
        bound_a = model.bind_tools([_t_alpha])
        bound_b = bound_a.bind_tools([_t_beta])

        # New non-empty bind replaces — current public contract.
        names = [m["name"] for m in bound_b.tools_metadata]
        assert names == ["_t_beta"]
        # Original model still untooled (immutability).
        assert model.tools_metadata == []
        # Intermediate model still has only alpha (immutability).
        assert [m["name"] for m in bound_a.tools_metadata] == ["_t_alpha"]

    def test_rotation_path_simulation(self) -> None:
        """End-to-end the exact path that caused the incident.

        Real path:  factory.bind_tools([N tools]) -> ainvoke -> LangGraph
        internally calls model.bind_tools([]) before the rotated session's
        first call -> ProxyMCPServer(tools_metadata=[]) -> CLI loses tools.

        After this fix the rotated path still sees N tools.
        """
        base = ClaudeCodeChatModel(base_options=None)
        tooled = base.bind_tools([_t_alpha, _t_beta])
        # Simulate the internal empty rebind that happened 321× in the log.
        after_rotation = tooled.bind_tools([])
        # The CLI now creates ProxyMCPServer(tools_metadata=after_rotation.tools_metadata).
        assert len(after_rotation.tools_metadata) == 2
