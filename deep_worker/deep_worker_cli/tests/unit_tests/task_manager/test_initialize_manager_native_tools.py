"""Tests for ``TaskManager._initialize_manager`` provider-specific tool wiring.

When ``config.provider == "codex"`` the manager agent runs against Codex CLI
just like the subordinate. Codex hard-codes a handful of native tools
(``apply_patch``, ``update_plan``, ``view_image``, ``request_user_input``,
``list_mcp_resources``…) into every API request — no ``--disable`` flag
strips them. ``codex_provider.event_pump`` records any native tool call
into LangGraph as an ``AIMessage.tool_calls`` referencing a BaseTool by
the Codex native name.

If those BaseTool objects are NOT in the manager agent's bound tool set,
LangGraph's ToolNode raises ``Tool '<name>' not found`` and the manager
turn dies. The fix here is to mirror what
``deep_worker_cli/process/agent_process.py:475-477`` does for the main
agent — extend the manager-bound tool list with ``NATIVE_TOOLS`` when
the provider is ``codex``.

These tests patch ``create_cli_agent`` and ``create_model`` so they run
fast and provider-agnostic; we only need to inspect the ``tools=``
kwarg actually delivered to ``create_cli_agent``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the source ``codex_provider`` package importable without installing it.
# The dev tree puts the provider at /path/to/project/codex_provider.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_worker_cli.task_manager.task_manager import TaskManager  # noqa: E402


def _stub_config(provider: str) -> MagicMock:
    cfg = MagicMock()
    cfg.provider = provider
    cfg.model = None
    cfg.working_dir = Path("/tmp")
    cfg.web_search_policy = "auto"
    return cfg


def _stub_session_manager() -> MagicMock:
    return MagicMock()


def _make_tm(provider: str) -> TaskManager:
    return TaskManager(
        agent_name="t1",
        config=_stub_config(provider),
        session_manager=_stub_session_manager(),
    )


@pytest.mark.asyncio
async def test_initialize_manager_for_codex_injects_native_tools() -> None:
    """When ``config.provider == "codex"`` the manager agent MUST be
    constructed with the Codex passthrough BaseTools in its ``tools=``
    list. Otherwise the manager turn cannot survive a native tool call
    from Codex (apply_patch, update_plan, etc.) — ToolNode would raise
    ``Tool not found`` and the manager would die.
    """
    from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS

    tm = _make_tm("codex")

    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())  # (manager_agent, manager_backend)

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    assert mock_create.called, "create_cli_agent must be invoked"
    kwargs = mock_create.call_args.kwargs
    bound_tools = kwargs.get("tools") or []

    bound_tool_names = {getattr(t, "name", None) for t in bound_tools}
    expected_names = {t.name for t in NATIVE_TOOLS}
    missing = expected_names - bound_tool_names
    assert not missing, (
        f"manager bound tools must include every Codex native passthrough. "
        f"Missing: {missing}. Bound: {bound_tool_names}."
    )


@pytest.mark.asyncio
async def test_initialize_manager_for_non_codex_does_not_inject_native_tools() -> None:
    """For non-codex providers the manager must NOT carry Codex-specific
    BaseTools — they are useless noise and would confuse provider-side
    tool routing. Symmetric to the main-agent wiring in
    ``agent_process.py``.

    Checked by object identity, not by name: the universal
    ``web_search_delegation_tool`` (passed to every provider's manager)
    happens to use the LangChain tool name ``"web_search"``, which is
    also the name of one Codex native passthrough; that collision is a
    pre-existing concern already present in ``agent_process.py`` for
    the main agent, and not what this test is trying to catch. What
    matters here is that none of the codex_provider.native_tools
    BaseTool *objects* leak into a non-codex manager.
    """
    from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS

    tm = _make_tm("deepseek")

    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    kwargs = mock_create.call_args.kwargs
    bound_tools = kwargs.get("tools") or []
    native_ids = {id(nt) for nt in NATIVE_TOOLS}
    leaked = [t for t in bound_tools if id(t) in native_ids]
    assert not leaked, (
        f"non-codex manager must NOT carry Codex native passthrough "
        f"BaseTool objects. Leaked: "
        f"{[getattr(t, 'name', None) for t in leaked]}."
    )


@pytest.mark.asyncio
async def test_initialize_manager_keeps_existing_kwargs_intact() -> None:
    """Regression guard: adding the codex tools must not alter the
    other kwargs the manager relies on (assistant_id with the ':tm'
    suffix, ``enable_shell=False``, ``enable_memory=False``,
    ``enable_skills=False``, custom system_prompt, project_memory_dir).
    """
    tm = _make_tm("codex")

    fake_model = MagicMock()
    fake_agent = (MagicMock(), MagicMock())

    with patch(
        "deep_worker_cli.config.create_model", return_value=fake_model,
    ), patch(
        "deep_worker_cli.agent.create_cli_agent", return_value=fake_agent,
    ) as mock_create:
        await tm._initialize_manager()

    kwargs = mock_create.call_args.kwargs
    assert kwargs.get("assistant_id") == "t1:tm"
    assert kwargs.get("enable_shell") is False
    assert kwargs.get("enable_memory") is False
    assert kwargs.get("enable_skills") is False
    assert kwargs.get("auto_approve") is True
    assert kwargs.get("project_memory_dir") == Path("/tmp")
    assert isinstance(kwargs.get("system_prompt"), str) and kwargs["system_prompt"]


# Silence unused-import warnings — AsyncMock is imported for parity
# with other unit tests in this directory but not used directly here.
_ = AsyncMock
