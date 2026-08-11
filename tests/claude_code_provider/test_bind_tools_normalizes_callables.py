"""Regression: ``ClaudeCodeChatModel.bind_tools`` must normalize ``Callable``
to ``StructuredTool`` (NOT silently skip), and ``raise TypeError`` on any
non-tool input.

Background — a related internal incident analysis (not included here)
=====================================================================
The pre-fix ``bind_tools`` body had this silent-skip:

    for tool_item in tools:
        if not isinstance(tool_item, BaseTool):
            continue            # ← SILENT SKIP
        ...

That guard was a defensive reflex against ``AttributeError`` on raw
callables and dict-specs, but it broke the contract in a way that
caused production failures four times in three weeks:

- The CLI's `process/agent_process.py:1033` constructs
  ``web_tools`` from a mix of ``BaseTool`` instances AND **plain
  ``def`` functions** (``http_request``, ``fetch_url``, ``read_image``
  in the CLI's `tools.py:69, 198, 261` — none decorated with
  ``@tool``).
- LangGraph ``ToolNode`` wraps such callables internally with
  ``StructuredTool.from_function`` — so they DO work on the graph
  side.
- But our ``bind_tools`` skipped them silently → they never made it
  into ``_tools_metadata`` → never made it into ``ProxyMCPServer``
  → Claude CLI's MCP tool list was missing them.
- On the restart path that mismatch caused the agent to lose its
  filesystem/execution tools after restart (see a related internal
  incident analysis (not included here)).

Contract enforced here
======================
1. A plain ``Callable`` (function decorated with nothing, or a bare
   ``def``) MUST be normalized via ``StructuredTool.from_function`` and
   contribute to ``tools_metadata`` (no silent skip).
2. Mixed input (BaseTool + Callable + Callable) MUST yield ALL of them
   in ``tools_metadata``. This is the exact production shape of
   ``web_tools``.
3. A non-callable, non-BaseTool input (e.g. ``dict``, ``int``, an
   arbitrary class) MUST raise ``TypeError`` with a message that
   explicitly mentions "did you forget @tool" so the operator knows
   what went wrong.
4. ``bind_tools([])`` MUST remain a no-op (returns ``self``) — this
   preserves the earlier fix contract for LangGraph's internal
   empty-rebind during summarization.

⚠ DO NOT delete or weaken these tests without a corresponding incident
post-mortem that explicitly supersedes the related internal incident
analysis (not included here).
The silent-skip is the **single** behaviour that, four times running,
has caused agents to lose their tools in production after a restart.
"""
from __future__ import annotations

import pytest
from langchain_core.tools import tool as langchain_tool

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel


# ─── Fixture tools ───────────────────────────────────────────────────────────

@langchain_tool
def _decorated_alpha(x: int) -> int:
    """Decorated alpha — already a BaseTool."""
    return x


def _bare_beta(y: str) -> str:
    """Bare beta — a plain def. THIS is the shape that was silently
    dropped by the pre-fix ``bind_tools``.
    """
    return y.upper()


def _bare_gamma(z: float, scale: float = 1.0) -> float:
    """Bare gamma with a default — exercises sig-with-defaults path."""
    return z * scale


# ─── Contract tests ──────────────────────────────────────────────────────────

class TestBindToolsNormalizesCallables:
    """``bind_tools`` MUST treat plain Callables as first-class tools."""

    def test_bare_function_becomes_tool_metadata(self) -> None:
        """A bare ``def`` (no @tool) must end up in tools_metadata.

        This is the EXACT failure mode of ``http_request``/``fetch_url``/
        ``read_image`` in production — they are plain defs and used to
        be silently dropped.
        """
        model = ClaudeCodeChatModel(base_options=None)
        bound = model.bind_tools([_bare_beta])

        names = [m["name"] for m in bound.tools_metadata]
        assert "_bare_beta" in names, (
            f"Bare def was silently dropped — regression. Got: {names!r}"
        )

    def test_mixed_basetool_and_callables_all_preserved(self) -> None:
        """Production-shape input: BaseTool + bare def + bare def.

        This is precisely ``web_tools`` (after rationalising
        WebSearchDelegationTool to a BaseTool). All three MUST be in
        ``tools_metadata`` after a single ``bind_tools`` call.
        """
        model = ClaudeCodeChatModel(base_options=None)
        bound = model.bind_tools([_decorated_alpha, _bare_beta, _bare_gamma])

        names = sorted(m["name"] for m in bound.tools_metadata)
        assert names == ["_bare_beta", "_bare_gamma", "_decorated_alpha"], (
            f"Mixed BaseTool+Callable input lost entries — got {names!r}. "
            f"This is the regression shape."
        )

    def test_bare_function_schema_has_named_params(self) -> None:
        """The synthesized schema must carry the function's parameters.

        Without parameters in input_schema, Claude CLI cannot generate
        correct tool calls — even if the tool is *listed*.
        """
        model = ClaudeCodeChatModel(base_options=None)
        bound = model.bind_tools([_bare_gamma])

        md = next(m for m in bound.tools_metadata if m["name"] == "_bare_gamma")
        props = md["input_schema"]["properties"]
        assert "z" in props and "scale" in props, (
            f"input_schema missing function params — got {props!r}"
        )

    def test_non_callable_non_basetool_raises_typeerror(self) -> None:
        """A dict or int MUST raise — silent skip is forbidden.

        The pre-fix code returned silently with that tool dropped from
        ``_tools_metadata``. The new contract is: loud failure at startup
        beats silent failure four hours into a session.
        """
        model = ClaudeCodeChatModel(base_options=None)

        with pytest.raises(TypeError, match=r"@tool"):
            model.bind_tools([_decorated_alpha, {"type": "function"}])  # type: ignore[list-item]

        with pytest.raises(TypeError, match=r"@tool"):
            model.bind_tools([42])  # type: ignore[list-item]

    def test_empty_list_is_still_noop(self) -> None:
        """The earlier fix contract MUST survive.

        ``bind_tools([])`` is called 100s of times per session by
        LangGraph's internal summarization-rebind path. It MUST remain a
        no-op returning ``self`` — otherwise the empty rebind wipes
        previously-bound tools.

        Co-existence proof: nothing in the new code prevents the
        earlier guard from running first.
        """
        model = ClaudeCodeChatModel(base_options=None)
        previously_bound = model.bind_tools([_decorated_alpha])
        rebound = previously_bound.bind_tools([])

        assert rebound is previously_bound, (
            "bind_tools([]) MUST return self (the earlier fix contract)"
        )
        assert len(rebound.tools_metadata) == 1
