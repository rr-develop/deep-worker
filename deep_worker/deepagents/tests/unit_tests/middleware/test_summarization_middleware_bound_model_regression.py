"""Regression test for the 2026-06-13 incident: SummarizationMiddleware captures
the **unbound** chat-model reference instead of the bind_tools-result.

Root cause (see
a related internal incident analysis (not included here)):

    SummarizationMiddleware(model=model, ...)  # <-- raw, pre-bind_tools model

is built in ``graph.py`` BEFORE ``create_agent(model, tools=tools, ...)``
calls ``model.bind_tools(tools)`` to produce the bound instance that the
turn-node actually invokes. LangChain's ``bind_tools`` contract is to
return a NEW model instance; the original ``model`` is left with
``_tools_metadata=[]``. When ``SummarizationMiddleware._acreate_summary``
calls ``self.model.ainvoke(prompt)`` it therefore goes through the
unbound instance — and with the global ``ClientManager`` singleton
shared across instances (``ClaudeCodeChatModel._client_manager``) that
empty rebind silently propagates ``tools_metadata=[]`` into the next
``ProxyMCPServer`` rebuild on session rotation. The CLI subprocess ends
up with **zero** ``mcp__deepworker__*`` tools.

This test reproduces the exact failure mode by spying on every
``SummarizationMiddleware`` constructed via ``create_deep_agent`` and
asserting that the captured model carries the SAME tools_metadata as
the user passed to ``create_deep_agent(tools=[...])``.

RED on current ``graph.py`` (where middleware sees the unbound model with
``tools_metadata == []``).
GREEN once ``graph.py`` passes the bind_tools-result instead.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as langchain_tool

from deepagents.graph import create_deep_agent
from deepagents.middleware.summarization import SummarizationMiddleware


# ---------------------------------------------------------------------------
# Fake model that mimics ClaudeCodeChatModel.bind_tools contract:
#   - holds ``_tools_metadata`` (mirrors ClaudeCodeChatModel._tools_metadata)
#   - ``bind_tools(tools)`` returns a NEW instance with the metadata copied;
#     ``bind_tools([])`` returns self (matches the 2026-06-12 fix invariant).
# This is the minimum reproduction surface for the unbound-capture bug —
# no real Claude SDK, no proxy MCP server, no langgraph runtime needed.
# ---------------------------------------------------------------------------
class _FakeBindToolsModel(BaseChatModel):
    """Stand-in for ``ClaudeCodeChatModel`` that captures tools on bind_tools.

    Mirrors only the surface our regression depends on:
    - ``bind_tools(tools)`` is the **only** way ``_tools_metadata`` is set
    - ``bind_tools(tools)`` returns a NEW instance — the LangChain
      immutability contract that is the root of the 2026-06-13 incident.
    """

    _tools_metadata_storage: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:  # pragma: no cover - required by BaseChatModel
        return "fake-bind-tools"

    @property
    def tools_metadata(self) -> list[dict[str, Any]]:
        # Snapshot the storage so callers see a stable list per-instance.
        return list(self._tools_metadata_storage)

    def bind_tools(
        self,
        tools: Any,
        **_kwargs: Any,
    ) -> "_FakeBindToolsModel":
        # Empty rebind is a no-op (mirrors the 2026-06-12 ClaudeCodeChatModel
        # invariant). Without this guard nothing else here changes — the
        # 2026-06-13 bug is independent and happens BEFORE any rebind.
        if not tools:
            return self
        new_instance = _FakeBindToolsModel()
        new_instance._tools_metadata_storage = [
            {"name": getattr(t, "name", str(t))} for t in tools
        ]
        return new_instance

    # BaseChatModel abstract glue — never actually called in this test.
    def _generate(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        return AIMessage(content="")

    async def _agenerate(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        return AIMessage(content="")


# Two distinct LangChain tools (N >= 2 as the spec requires).
@langchain_tool
def _t_alpha(x: int) -> int:
    """Alpha tool."""
    return x


@langchain_tool
def _t_beta(s: str) -> str:
    """Beta tool."""
    return s


@langchain_tool
def _t_gamma(b: bool) -> bool:
    """Gamma tool."""
    return b


# ---------------------------------------------------------------------------
# The spy: record every SummarizationMiddleware ever instantiated AND the
# model it captured. We patch the constructor at module-level so it fires
# for every call site in graph.py (gp_middleware, subagent_middleware,
# deepagent_middleware — all 3 sites must pass the bound model).
# ---------------------------------------------------------------------------
@pytest.fixture
def summarization_spies(monkeypatch: pytest.MonkeyPatch) -> list[SummarizationMiddleware]:
    """Collect every ``SummarizationMiddleware`` constructed during a test."""
    spies: list[SummarizationMiddleware] = []
    real_init = SummarizationMiddleware.__init__

    def spying_init(self: SummarizationMiddleware, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        spies.append(self)

    # Patch on the deepagents-side wrapper class (the one ``graph.py`` actually
    # instantiates). The base class lives in langchain.agents.middleware.
    monkeypatch.setattr(SummarizationMiddleware, "__init__", spying_init)
    return spies


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSummarizationMiddlewareSeesBoundModel:
    """The middleware MUST see the bind_tools-result, NOT the unbound model.

    These tests are the RED step of the 2026-06-13 incident. They fail on the
    current ``graph.py`` (lines 184, 227, 272) which capture the unbound
    ``model`` reference in ``SummarizationMiddleware(model=model, ...)``.
    They go GREEN once those sites are rewritten to pass the bind_tools
    result (or a late-bound getter that resolves to it at summary time).
    """

    def test_main_agent_middleware_sees_tools(
        self, summarization_spies: list[SummarizationMiddleware]
    ) -> None:
        """``create_deep_agent`` produces a deepagent_middleware stack whose
        SummarizationMiddleware MUST carry a model with the user's tools.

        Pre-fix: the captured model is the unbound input → tools_metadata=[].
        Post-fix: it is the bind_tools-result → tools_metadata=[alpha, beta].
        """
        model = _FakeBindToolsModel()
        tools = [_t_alpha, _t_beta]

        # We don't need to .invoke() — just instantiating the agent runs
        # the middleware construction code in graph.py.
        create_deep_agent(model=model, tools=tools)

        # Every SummarizationMiddleware constructed during create_deep_agent
        # must reference a model whose tools_metadata matches user tools.
        # There can be more than one (gp_middleware + deepagent_middleware +
        # per-subagent), but ALL of them MUST have N tools, not 0.
        assert summarization_spies, (
            "create_deep_agent did not construct any SummarizationMiddleware "
            "instance — wiring change broke the test, not the production code"
        )
        for idx, sm in enumerate(summarization_spies):
            captured_metadata = getattr(sm.model, "tools_metadata", None)
            assert captured_metadata is not None, (
                f"SummarizationMiddleware #{idx} captured a model with no "
                f"tools_metadata attribute at all (model={sm.model!r}). "
                "Test target shape changed."
            )
            captured_names = sorted(m["name"] for m in captured_metadata)
            expected_names = sorted(t.name for t in tools)
            assert captured_names == expected_names, (
                f"SummarizationMiddleware #{idx} captured a model with "
                f"tools_metadata={captured_names!r}, expected {expected_names!r}. "
                "The 2026-06-13 incident: middleware was built with the UNBOUND "
                "model reference (before bind_tools), so the captured instance "
                "has empty tools_metadata. On session rotation that empty "
                "list gets propagated into ProxyMCPServer and the Claude CLI "
                "loses every mcp__deepworker__* tool."
            )

    @pytest.mark.parametrize("n_tools", [2, 3, 5])
    def test_parametrized_tool_counts_are_all_preserved(
        self,
        summarization_spies: list[SummarizationMiddleware],
        n_tools: int,
    ) -> None:
        """Guard: every tool the caller passes ends up in middleware.model.

        Parametrized over a few sizes so a future regression that off-by-one
        truncates the bound list (e.g. ``tools[:-1]``) cannot pass by luck.
        """
        # Pool of distinct named tools — slice to the requested count.
        pool = [_t_alpha, _t_beta, _t_gamma]
        while len(pool) < n_tools:
            # Build extra tools with unique names so the comparison is exact.
            unique_name = f"_t_extra_{len(pool)}"

            def _make(name: str) -> Any:  # noqa: ANN401
                @langchain_tool(name)
                def _extra(x: int) -> int:
                    """extra."""
                    return x

                return _extra

            pool.append(_make(unique_name))
        tools = pool[:n_tools]

        model = _FakeBindToolsModel()
        create_deep_agent(model=model, tools=tools)

        assert summarization_spies, "no middleware captured"
        for sm in summarization_spies:
            captured = getattr(sm.model, "tools_metadata", [])
            assert len(captured) == n_tools, (
                f"middleware saw {len(captured)} tools, expected {n_tools} "
                f"(captured names={[m['name'] for m in captured]!r})"
            )

    def test_subagent_middleware_sees_tools(
        self, summarization_spies: list[SummarizationMiddleware]
    ) -> None:
        """When the user passes a sub-agent spec, its middleware stack ALSO
        has a SummarizationMiddleware (graph.py:227). That site must also
        see the bind_tools-result.

        Pre-fix: subagent_middleware captures the unbound ``subagent_model``.
        Post-fix: it captures the bind_tools-result for the subagent's tools.
        """
        model = _FakeBindToolsModel()
        tools = [_t_alpha, _t_beta, _t_gamma]
        # Minimal SubAgent spec — name/description/prompt are required by the
        # SubAgent TypedDict; model defaults to the parent model so the
        # graph.py:227 SummarizationMiddleware site fires for it.
        subagents = [
            {
                "name": "researcher",
                "description": "does research",
                "system_prompt": "research subagent prompt",
            }
        ]

        create_deep_agent(model=model, tools=tools, subagents=subagents)  # type: ignore[arg-type]

        assert len(summarization_spies) >= 2, (
            f"Expected at least 2 SummarizationMiddleware instances "
            f"(main + subagent), got {len(summarization_spies)}"
        )
        for idx, sm in enumerate(summarization_spies):
            captured_metadata = getattr(sm.model, "tools_metadata", None)
            assert captured_metadata is not None, (
                f"SummarizationMiddleware #{idx}.model has no tools_metadata"
            )
            assert len(captured_metadata) >= 1, (
                f"SummarizationMiddleware #{idx}.model.tools_metadata is "
                f"empty (got {captured_metadata!r}). The 2026-06-13 incident: "
                "the subagent middleware site (graph.py:227) was also "
                "passing the unbound model reference, so summarization on "
                "the subagent silently strips its MCP toolset on rotation."
            )


# ---------------------------------------------------------------------------
# Structural guard — fail-fast at import-time if someone re-introduces a
# raw ``model=model`` / ``model=subagent_model`` keyword on any
# ``SummarizationMiddleware(...)`` call site inside ``graph.py``. This is a
# belt-and-suspenders complement to the behavioural tests above: behaviour
# tests catch the bug end-to-end, the AST guard pinpoints it at the source
# of truth and surfaces the exact line number that introduced the regress.
# ---------------------------------------------------------------------------
class TestGraphSourceForbidsRawModelOnSummarizationMiddleware:
    """Static AST scan of ``deepagents/graph.py``.

    The contract: ``SummarizationMiddleware(model=<expr>, ...)`` MUST NOT
    use ``model`` or ``subagent_model`` as the bare keyword argument. Those
    raw references are exactly what triggered the 2026-06-13 incident. The
    allowed shape is either ``summarization_model`` (a name bound to the
    ``bind_tools``-result) or any expression that includes a ``bind_tools``
    call. Anything else fails this guard.

    If you're reading this because the guard fired:
    - Read a related internal incident analysis (not included here).
    - Bind tools BEFORE building the middleware and pass the bound result.
    - DO NOT just rename the variable to bypass the guard.
    """

    FORBIDDEN_RAW_NAMES = frozenset({"model", "subagent_model"})

    def test_summarization_middleware_never_takes_raw_model_keyword(self) -> None:
        import ast
        from pathlib import Path

        graph_py = Path(__file__).resolve().parents[3] / "deepagents" / "graph.py"
        assert graph_py.is_file(), f"graph.py not found at {graph_py}"
        source = graph_py.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(graph_py))

        offenders: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # We match by the callable name, regardless of how it's imported.
            func = node.func
            callee_name: str | None = None
            if isinstance(func, ast.Name):
                callee_name = func.id
            elif isinstance(func, ast.Attribute):
                callee_name = func.attr
            if callee_name != "SummarizationMiddleware":
                continue
            for kw in node.keywords:
                if kw.arg != "model":
                    continue
                value = kw.value
                if isinstance(value, ast.Name) and value.id in self.FORBIDDEN_RAW_NAMES:
                    offenders.append((value.lineno, value.id))

        assert not offenders, (
            "deepagents/graph.py is re-introducing the 2026-06-13 incident: "
            "SummarizationMiddleware(model=...) is being constructed with a "
            "raw, pre-bind_tools model reference. Offending lines: "
            f"{offenders!r}. Bind tools first (see "
            "_bind_tools_for_middleware) and pass the bound result. "
            "Documentation: a related internal incident analysis "
            "(not included here)"
        )
