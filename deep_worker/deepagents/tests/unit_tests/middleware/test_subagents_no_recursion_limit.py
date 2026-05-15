"""Subagents must run without an artificial step cap.

Background
----------
``langchain_core/runnables/config.py:143`` defines
``DEFAULT_RECURSION_LIMIT = 25`` for any ``Runnable.invoke`` /
``ainvoke`` call that does not pass an explicit value. ``langgraph``
overrides this to ``10_000`` for its own pregel loop, but a subagent
invoked via ``subagent.ainvoke(state, config={...})`` here goes
through a path that lands at the langchain_core default unless we
pass an explicit override.

That default of 25 is what blew up demo_agent on 2026-04-25 12:13:23:
a subagent doing a content audit on RU-locale screenshots of a help
site needed dozens of file lookups + image reads, well past 25
pregel ticks, and ``subagents.py:479 atask`` raised
``GraphRecursionError`` straight up to the caller.

The codebase has an explicit project-level rule:
``deep_worker_cli/config.py:292``:

    # CRITICAL: NO RECURSION LIMIT ON AGENTS!
    # recursion_limit is STRICTLY FORBIDDEN. Agents must be able
    # to perform unlimited tool calls.

That rule is enforced for the main agent by
``deep_worker_cli/tests/unit_tests/test_no_recursion_limit.py`` —
but the rule was NOT extended to subagents, and the omission is
exactly the bug we hit.

Contract pinned by these tests
------------------------------

1. Both ``task`` (sync) and ``atask`` (async) MUST pass an explicit
   ``recursion_limit`` of ``sys.maxsize`` (the documented "unlimited"
   sentinel for langgraph-style RunnableConfig). Anything lower
   reintroduces the cap.

2. ``subagents.py`` source must not contain a literal ``recursion_limit``
   set to a small integer (≤ 1000) — guards against a future PR
   "reasonably" lowering it. The only acceptable form is the
   sentinel ``sys.maxsize`` (or an equivalent multi-billion constant).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import deepagents.middleware.subagents as subagents_mod


SUBAGENTS_SRC_PATH = Path(subagents_mod.__file__)


# ---------------------------------------------------------------------------
# 1. Source-level invariants (cheap, fast)
# ---------------------------------------------------------------------------


def _find_low_recursion_limits(source: str) -> list[str]:
    """Return any line in source that sets ``recursion_limit`` to a
    bounded number (anything other than ``sys.maxsize``).

    Skips comments and docstrings.
    """
    offending: list[str] = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        ):
            continue
        if "recursion_limit" not in stripped:
            continue
        # Allow the unlimited sentinel forms.
        if "sys.maxsize" in stripped:
            continue
        # Allow type annotations in function signatures
        # (e.g. ``recursion_limit: int | None = None`` as a parameter).
        if re.match(r".*recursion_limit\s*:\s*[A-Za-z|]+\s*=\s*None", stripped):
            continue
        # Disallow setting it to any literal int.
        if re.search(r"recursion_limit['\"\s]*[:=]\s*\d", stripped):
            offending.append(f"  line {i}: {stripped}")
    return offending


def test_subagents_module_has_no_low_recursion_limit() -> None:
    """``subagents.py`` must not set ``recursion_limit`` to any
    bounded integer literal — agents and subagents must run with
    no artificial step cap.

    This test deliberately permits ``sys.maxsize`` as a sentinel,
    because LangGraph's RunnableConfig requires an int (no
    ``inf``-style sentinel exists). See module docstring for
    rationale.
    """
    source = SUBAGENTS_SRC_PATH.read_text()
    offending = _find_low_recursion_limits(source)
    assert not offending, (
        "subagents.py must NOT set recursion_limit to a bounded "
        "literal — that reintroduces GraphRecursionError on any "
        "non-trivial subagent task. Use sys.maxsize for the "
        "'unlimited' sentinel.\n" + "\n".join(offending)
    )


def test_subagents_module_explicitly_uses_unlimited_sentinel() -> None:
    """Belt-and-braces: the module MUST contain the literal
    ``sys.maxsize`` somewhere in a context that wires it into a
    config. Without this positive check, deleting the override
    entirely would silently revert to the langchain_core default
    of 25 and the source-level scan above would still pass.
    """
    source = SUBAGENTS_SRC_PATH.read_text()
    assert "sys.maxsize" in source, (
        "subagents.py must explicitly pass recursion_limit=sys.maxsize "
        "in the subagent.invoke / ainvoke configs"
    )
    # And there must be at least one assignment of recursion_limit
    # to that sentinel. Accept either direct reference to
    # ``sys.maxsize`` or a module-level constant defined as
    # ``= sys.maxsize`` and used by the recursion_limit field.
    direct = re.compile(
        r"recursion_limit['\"]?\s*[:=]\s*sys\.maxsize"
    )
    if direct.search(source):
        return
    # Indirect form: a UPPER_SNAKE_CASE constant initialised from
    # ``sys.maxsize`` AND that exact constant used as the value of
    # a ``recursion_limit`` field.
    const_def = re.compile(
        r"^([A-Z_][A-Z0-9_]+)\s*=\s*sys\.maxsize",
        re.MULTILINE,
    )
    match = const_def.search(source)
    assert match is not None, (
        "subagents.py mentions sys.maxsize but does not wire it "
        "into a recursion_limit config field. The unlimited "
        "sentinel must actually reach Runnable.invoke."
    )
    const_name = match.group(1)
    used_pattern = re.compile(
        rf"recursion_limit['\"]?\s*[:=]\s*{re.escape(const_name)}"
    )
    assert used_pattern.search(source), (
        f"subagents.py defines {const_name} = sys.maxsize but does "
        f"not pass it as recursion_limit anywhere. The constant must "
        f"reach Runnable.invoke."
    )


# ---------------------------------------------------------------------------
# 2. Runtime invariants — exercise the ``atask`` / ``task`` codepaths
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_invoke_config() -> dict:
    """Inject a fake subagent that records the config dict passed to
    its ``invoke`` / ``ainvoke``. Returns a dict the test populates."""
    return {}


def _build_test_task_tool(captured: dict):
    """Construct a ``task`` tool whose ``subagents`` list points to a
    fake subagent that captures the invoke config. Returns the
    StructuredTool object — the sync callable is ``tool.func``, the
    async one is ``tool.coroutine``."""
    fake_subagent = MagicMock()

    async def _ainvoke(state, config):
        captured["config_async"] = config
        return {"messages": [_FakeMessage("ok")]}

    def _invoke(state, config):
        captured["config_sync"] = config
        return {"messages": [_FakeMessage("ok")]}

    fake_subagent.ainvoke = _ainvoke
    fake_subagent.invoke = _invoke

    spec = {
        "name": "my_sub",
        "description": "fake subagent for tests",
        "runnable": fake_subagent,
    }
    return subagents_mod._build_task_tool([spec])


class _FakeMessage:
    """Minimal stand-in for an AIMessage with a ``text`` attribute —
    enough for ``_return_command_with_state_update`` to work without
    pulling in the real langchain_core message types."""

    def __init__(self, text: str) -> None:
        self.text = text


def _make_runtime() -> MagicMock:
    """Build a minimal ``ToolRuntime`` that ``_validate_and_prepare_state``
    can chew on. The state needs an iterable of message-like dicts."""
    runtime = MagicMock()
    runtime.state = {"messages": []}
    runtime.config = {"configurable": {"thread_id": "agent_main"}}
    runtime.tool_call_id = "call_xyz"
    return runtime


def test_atask_passes_unlimited_recursion_limit(
    captured_invoke_config,
) -> None:
    """``atask`` (async) must pass ``recursion_limit=sys.maxsize`` so
    the subagent can run for as many tool calls as needed."""
    import asyncio

    tool = _build_test_task_tool(captured_invoke_config)
    runtime = _make_runtime()
    asyncio.run(
        tool.coroutine(
            description="anything",
            subagent_type="my_sub",
            runtime=runtime,
        )
    )

    config = captured_invoke_config["config_async"]
    assert "recursion_limit" in config, (
        "atask must pass recursion_limit explicitly — the langchain_core "
        "default is 25 and that crashes any non-trivial subagent."
    )
    assert config["recursion_limit"] == sys.maxsize, (
        f"atask must pass recursion_limit=sys.maxsize "
        f"(got {config['recursion_limit']}). Any bounded limit will "
        f"reintroduce GraphRecursionError on real workloads."
    )


def test_task_passes_unlimited_recursion_limit(
    captured_invoke_config,
) -> None:
    """``task`` (sync) must pass ``recursion_limit=sys.maxsize`` for
    the same reason as the async variant."""
    tool = _build_test_task_tool(captured_invoke_config)
    runtime = _make_runtime()
    tool.func(
        description="anything",
        subagent_type="my_sub",
        runtime=runtime,
    )

    config = captured_invoke_config["config_sync"]
    assert "recursion_limit" in config
    assert config["recursion_limit"] == sys.maxsize


# ---------------------------------------------------------------------------
# 3. Cross-check: the existing main-agent rule still holds for subagents
# ---------------------------------------------------------------------------


def test_subagents_does_not_assign_default_recursion_limit() -> None:
    """The string ``DEFAULT_RECURSION_LIMIT`` must not appear as a
    bound value in subagents.py — it would imply we deferred to the
    langchain_core / langgraph default, which is what we are
    explicitly overriding."""
    source = SUBAGENTS_SRC_PATH.read_text()
    # Allow the symbol in comments / docstrings; ban it in any
    # active expression.
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""'):
            continue
        if "DEFAULT_RECURSION_LIMIT" in stripped:
            pytest.fail(
                f"line {i}: subagents.py must not bind to "
                f"DEFAULT_RECURSION_LIMIT — that defers to a 25-step "
                f"cap and reproduces the original bug:\n  {stripped}"
            )


# ---------------------------------------------------------------------------
# 4. Runtime backstop — _build_subagent_config asserts the sentinel
# ---------------------------------------------------------------------------


def test_build_subagent_config_returns_unlimited_sentinel() -> None:
    """The single source of truth for subagent config must include
    ``recursion_limit`` set to the unlimited sentinel."""
    config = subagents_mod._build_subagent_config("test_thread_id")
    assert "recursion_limit" in config, (
        "_build_subagent_config must include recursion_limit — "
        "without it the langchain_core default of 25 silently "
        "applies."
    )
    assert config["recursion_limit"] == sys.maxsize
    assert config["configurable"]["thread_id"] == "test_thread_id"


def test_build_subagent_config_runtime_guard_fires_on_downgrade(
    monkeypatch,
) -> None:
    """If a future patch breaks the constant — points it at a small
    integer instead of ``sys.maxsize`` — the runtime backstop
    inside ``_build_subagent_config`` must raise immediately on
    the first subagent call.

    This is the guarantee that "recursion_limit was secretly
    downgraded" cannot become a silent regression.
    """
    # Simulate a regression: someone changes the constant to a
    # bounded value. ``_build_subagent_config`` reads the constant
    # at call time, so monkey-patching is enough to trigger the
    # guard without editing the source.
    monkeypatch.setattr(
        subagents_mod,
        "_SUBAGENT_RECURSION_LIMIT_UNLIMITED",
        25,  # the old, broken default
    )
    with pytest.raises(RuntimeError, match=r"recursion_limit was downgraded"):
        subagents_mod._build_subagent_config("any_thread")
