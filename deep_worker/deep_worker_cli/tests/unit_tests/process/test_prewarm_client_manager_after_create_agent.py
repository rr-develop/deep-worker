"""Regression: ``AgentProcess.start()`` must deterministically prewarm the
``ClientManager`` with the FINAL tool set right after ``create_cli_agent``.

Background — incident 2026-06-13
=====================================
The pre-fix path used ``_wait_for_mcp_tools_ready`` — a 30-second
polling barrier that watched ``ClaudeCodeChatModel._tools_metadata``
for the FIRST non-empty list. That was wrong on two counts:

1. **Polling at all is a code smell** — we already have the full
   ``tool_node.tools_by_name`` available SYNCHRONOUSLY at the moment
   ``create_agent(...)`` returns (see
   ``langchain/agents/factory.py:880-905``). Waiting for a side effect
   of someone else's ``bind_tools`` to land was always racy.

2. **Polling for "any non-empty set" is even worse** — on restart the
   FIRST bind on the model happened to be ``_bind_tools_for_middleware``
   (deepagents/graph.py:252) with the ORIGINAL caller-supplied tools
   list (7 ``web_tools`` only, NOT the final 23 that include
   filesystem/skills/todo middlware tools). The barrier returned True
   on 7 → ``_handle_invoke`` fired → first ``ainvoke`` cached a
   ``ProxyMCPServer`` with the wrong (smaller) tool set in
   ``ClientManager.clients[name]``. From that point on the CLI session
   was permanently missing 16 tools, including ``read_file`` /
   ``write_file`` / ``execute``.

Contract enforced here
======================
After ``create_cli_agent`` returns, ``AgentProcess.start()`` MUST:

A) Read the FINAL tool list out of the compiled agent graph (the
   ``tool_node`` LangGraph attaches as the ``tools`` node), NOT the
   ``web_tools`` argument it passed in. The final list is the one the
   model_node will use on every turn.

B) Call ``ClientManager.prepare_proxy_only(thread_id=self.name,
   tools_metadata=<from final tool list>, working_dir=...)`` BEFORE
   any code path that can trigger ``model.ainvoke`` (notably
   ``_auto_continue_after_restart``).

This makes the post-restart cache state deterministic:
``ClientManager._proxy_only[self.name]`` holds a ``ProxyMCPServer``
built from the FINAL tool inventory, ready to be ADOPTED by the first
``get_or_create_client`` call. Subsequent ``ainvoke`` calls hit the
cache (``client_manager.py``) and inherit the correct toolset.

⚠ DO NOT replace prewarm with another polling barrier. The deterministic
sync extraction from the compiled graph is the WHOLE point. See
incident 2026-06-13 post-mortem.

⚠ DO NOT switch this back to ``get_or_create_client``. That call opens
the Claude CLI session immediately and side-steps the history-restore
block in ``ClaudeCodeChatModel.ainvoke`` (gated on
``not has_client(thread_id)``), so the agent forgets every prior turn
on every restart. See the 2026-06-15 incident (an observed live hang) and
``claude_code_provider/tests/test_client_manager_prewarm_no_connect.py``.
"""
from __future__ import annotations

import builtins
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.provider_paths import ProviderOrigin, ProviderResolution


# ─── Fake tool node + agent ─────────────────────────────────────────────────

def _make_basetool(name: str) -> StructuredTool:
    """Build a real LangChain ``StructuredTool`` for the given name.

    We use a real BaseTool here (rather than a MagicMock) so the
    extraction probe goes through the same ``ClaudeCodeChatModel.
    bind_tools`` path real code does. A mock would silently dodge the
    new "BaseTool-or-Callable only" contract we are explicitly
    testing in test_bind_tools_normalizes_callables.py.
    """
    def _impl(value: str = "") -> str:
        return value

    return StructuredTool.from_function(
        _impl,
        name=name,
        description=f"desc:{name}",
    )


def _build_fake_agent(tool_names: list[str]) -> MagicMock:
    """Build a MagicMock that mimics the compiled LangGraph agent enough
    for the prewarm extraction to find ``tool_node.tools_by_name``.

    The real ``langchain.agents.factory`` puts a ``ToolNode`` instance at
    ``graph.add_node("tools", tool_node)``. After ``compile()`` it lives
    at ``compiled_graph.nodes["tools"]`` (LangGraph stores it inside a
    PregelNode wrapper). The prewarm code MUST be robust to that wrapper
    structure — see ``AgentProcess._extract_tool_node`` for the actual
    extraction logic.
    """
    fake_tool_node = MagicMock()
    fake_tool_node.tools_by_name = {n: _make_basetool(n) for n in tool_names}

    fake_agent = MagicMock()
    # Real attribute path LangGraph exposes on a compiled StateGraph
    fake_agent.nodes = {"tools": fake_tool_node}
    return fake_agent


# ─── Fake ClientManager that records the prewarm call ───────────────────────

class _RecordingClientManager:
    """Records the (thread_id, tools_metadata, working_dir) of every
    ``prepare_proxy_only`` call so the test can assert exactly one
    prewarm with the FINAL tool set.

    ⚠ NEW CONTRACT — the 2026-06-15 incident.
    Prewarm must NOT call ``get_or_create_client`` — that spawned a
    Claude CLI subprocess BEFORE history-restore could run, and the
    agent forgot its conversation on every restart. Prewarm now calls
    ``prepare_proxy_only`` which only reserves the ProxyMCPServer cache
    slot. The CLI client is opened lazily on the first ``ainvoke`` so
    ``resume_session_id`` can be set correctly. See
    ``claude_code_provider/tests/test_client_manager_prewarm_no_connect.py``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        # Defensive guard: if anything tries to open a CLI client during
        # prewarm we want the test to fail loudly, not silently pass.
        self._fake_proxy = MagicMock()

    async def prepare_proxy_only(
        self,
        thread_id: str,
        tools_metadata: list[dict[str, Any]],
        working_dir: str | None = None,
        **_: Any,
    ) -> Any:
        self.calls.append(
            {
                "thread_id": thread_id,
                "tool_names": [m["name"] for m in tools_metadata],
                "working_dir": working_dir,
            }
        )
        return self._fake_proxy

    async def get_or_create_client(
        self,
        thread_id: str,
        tools_metadata: list[dict[str, Any]],
        working_dir: str | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        # If prewarm ever calls this again, the test must FAIL —
        # opening the CLI client during prewarm is the regression we
        # are guarding against.
        raise AssertionError(
            "Prewarm must NOT call get_or_create_client — that opens "
            "the Claude CLI session before history-restore can run "
            "and re-introduces the 2026-06-15 incident."
        )


# ─── Contract tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prewarm_called_after_create_cli_agent_with_final_tools() -> None:
    """Prewarm MUST land with the FULL final tool inventory, not the
    caller's web_tools subset.

    Production shape: web_tools has 7, after middleware the final tool
    inventory is 23. We must prewarm with all 23 so the cached
    ``ProxyMCPServer`` matches what every subsequent turn will bind.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    final_tools = [
        # web_tools subset (7 of these come from agent_process.start)
        "http_request", "fetch_url", "web_search", "read_image",
        "restart_agent", "restart_self", "send_message_to_agent",
        # middleware-injected (16 more, from FilesystemMiddleware etc.)
        "read_file", "write_file", "edit_file", "ls", "grep", "glob",
        "execute", "wait_for_job", "kill_job", "write_todos",
        "task", "memory_write", "memory_search", "skill",
        "history_range", "history_at",
    ]
    fake_agent = _build_fake_agent(final_tools)
    fake_cm = _RecordingClientManager()

    extracted = AgentProcess._extract_final_tool_metadata(
        fake_agent, provider="claude-code"
    )
    await AgentProcess._prewarm_client_manager(
        client_manager=fake_cm,
        thread_id="test_agent_main",
        tools_metadata=extracted,
        working_dir="/path/to/job",
    )

    assert len(fake_cm.calls) == 1, (
        f"Expected exactly one prewarm call, got {len(fake_cm.calls)}"
    )
    call = fake_cm.calls[0]
    assert call["thread_id"] == "test_agent_main"
    assert call["working_dir"] == "/path/to/job"
    assert sorted(call["tool_names"]) == sorted(final_tools), (
        f"Prewarm landed with the WRONG tool set — this is the exact "
        f"shape that caused the 2026-06-13 incident. "
        f"Got {sorted(call['tool_names'])!r}, expected "
        f"{sorted(final_tools)!r}."
    )


def test_extract_tool_metadata_handles_missing_tool_node() -> None:
    """Some agents (no tools at all) have no 'tools' node. Extraction
    must return an empty list, NEVER raise — start() must still complete.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_agent = MagicMock()
    fake_agent.nodes = {}  # No 'tools' node at all

    extracted = AgentProcess._extract_final_tool_metadata(
        fake_agent, provider="claude-code"
    )
    assert extracted == [], (
        f"Missing tool_node must yield empty list, got {extracted!r}"
    )


def test_extract_tool_metadata_preserves_tool_order_and_schema() -> None:
    """Sanity: extracted metadata carries (name, description, input_schema)
    triples — enough for ProxyMCPServer to advertise the tools correctly.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_agent = _build_fake_agent(["alpha", "beta"])
    extracted = AgentProcess._extract_final_tool_metadata(
        fake_agent, provider="claude-code"
    )

    assert [m["name"] for m in extracted] == ["alpha", "beta"]
    for m in extracted:
        assert "description" in m
        assert "input_schema" in m


@pytest.mark.asyncio
async def test_prewarm_swallows_client_manager_errors() -> None:
    """Prewarm MUST be best-effort.

    If ``prepare_proxy_only`` raises (ProxyMCPServer construction error,
    tool-metadata schema drift, etc.) prewarm logs and proceeds. The agent
    still starts; the first real ainvoke will build the proxy from scratch.
    NEVER let prewarm crash AgentProcess.start().
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_cm = MagicMock()
    fake_cm.prepare_proxy_only = AsyncMock(side_effect=RuntimeError("simulated"))

    # Must not raise
    await AgentProcess._prewarm_client_manager(
        client_manager=fake_cm,
        thread_id="test",
        tools_metadata=[{"name": "x", "description": "", "input_schema": {}}],
        working_dir="/tmp",
    )


# ─── §3.7 structural import invariant (DESIGN_PROVIDERS_FROM_WHEEL) ─────────

def _unavailable_resolution(pkg: str, remedy: str | None = None) -> ProviderResolution:
    """A resolution whose ``unavailable`` field is set (provider absent)."""
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="not found",
        unavailable="provider package not installed",
        remedy=remedy,
    )


def _available_resolution(pkg: str) -> ProviderResolution:
    """A resolution whose ``unavailable`` is None (provider importable)."""
    return ProviderResolution(
        pkg=pkg,
        mode="source",
        origin=ProviderOrigin.SOURCE,
        roots=("/tmp/example-tree/deep_worker",),
        reason="source root found",
        unavailable=None,
        remedy=None,
    )


def test_non_claude_provider_never_imports_claude_code_provider(monkeypatch) -> None:
    """§3.7 — non-Claude paths MUST NEVER import
    ``deep_worker.providers.claude_code_provider``.

    A wheel without the Claude SDK fork / claude CLI must still start API
    providers (deepseek, codex, ...). The guard on
    ``provider == "claude-code"`` happens BEFORE the import, so a
    non-Claude provider returns the raw tool inventory without ever
    touching the claude_code_provider package.

    We assert this two ways:
      1. the module is not present in ``sys.modules`` after the call;
      2. a spy on ``builtins.__import__`` records no import attempt for
         the claude_code_provider package during the call.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_agent = _build_fake_agent(["alpha", "beta"])
    real_import = builtins.__import__
    attempted: list[str] = []

    def _spy_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deep_worker.providers.claude_code_provider" or name.startswith(
            "deep_worker.providers.claude_code_provider."
        ):
            attempted.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _spy_import)
    # Drop the package from sys.modules so we can prove the call does not
    # (re)load it. If some earlier test already imported it, dropping the
    # entry lets us still verify no NEW load is triggered by this path.
    monkeypatch.delitem(
        sys.modules,
        "deep_worker.providers.claude_code_provider",
        raising=False,
    )

    extracted = AgentProcess._extract_final_tool_metadata(
        fake_agent, provider="deepseek"
    )

    assert attempted == [], (
        f"Non-Claude path MUST NOT import claude_code_provider, attempted "
        f"{attempted!r}"
    )
    assert "deep_worker.providers.claude_code_provider" not in sys.modules
    assert [m["name"] for m in extracted] == ["alpha", "beta"]
    assert extracted[0]["description"] == "desc:alpha"


def test_claude_path_missing_capability_raises_capability_error(monkeypatch) -> None:
    """§3.7 — the Claude path only surfaces a CLASSIFIED missing-capability
    (actionable ``CapabilityError``), never a raw ModuleNotFoundError.

    When ``ensure_provider_importable`` reports the provider unavailable,
    ``_extract_final_tool_metadata(provider="claude-code")`` MUST raise
    ``CapabilityError`` with capability ``"claude-code"`` and the remedy —
    so start() fails loudly and actionably instead of crashing on an
    unhelpful import error.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_agent = _build_fake_agent(["alpha"])
    resolution = _unavailable_resolution(
        "deep_worker.providers.claude_code_provider",
        remedy="pip install deep-worker[claude-code]",
    )

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ) as mock_resolve:
        with pytest.raises(CapabilityError) as exc_info:
            AgentProcess._extract_final_tool_metadata(
                fake_agent, provider="claude-code"
            )

    mock_resolve.assert_called_once_with("deep_worker.providers.claude_code_provider")
    assert exc_info.value.capability == "claude-code"
    assert exc_info.value.remedy == "pip install deep-worker[claude-code]"
    assert "pip install" in str(exc_info.value)


def test_claude_path_internal_defect_reraises(monkeypatch) -> None:
    """§3.7 — the Claude path re-raises INTERNAL defects instead of
    misclassifying them as missing capabilities.

    If ``ensure_provider_importable`` reports the provider AVAILABLE but
    the actual ``deep_worker.providers.claude_code_provider`` import fails
    for an unrelated reason (broken submodule, runtime import error), the
    original exception MUST propagate — it must NOT be converted to a
    ``CapabilityError`` (that would lie about the cause) and must NOT be
    swallowed.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    fake_agent = _build_fake_agent(["alpha"])
    resolution = _available_resolution("deep_worker.providers.claude_code_provider")
    real_import = builtins.__import__

    def _broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deep_worker.providers.claude_code_provider" or name.startswith(
            "deep_worker.providers.claude_code_provider."
        ):
            raise ImportError("simulated internal defect in claude provider")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    monkeypatch.delitem(
        sys.modules,
        "deep_worker.providers.claude_code_provider",
        raising=False,
    )

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=resolution,
    ):
        with pytest.raises(ImportError, match="simulated internal defect"):
            AgentProcess._extract_final_tool_metadata(
                fake_agent, provider="claude-code"
            )
