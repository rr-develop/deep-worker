"""Agent-side handler tests for the two Full-Dialog introspection modes
(§9 of deep_worker_web/docs/DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md):

- ``AgentProcess._handle_get_system_prompt`` returns the INVARIANT (startup)
  system prompt captured at build time, with a recompute fallback.
- ``AgentProcess._handle_get_tools`` returns the provider-dependent tool list
  reduced to name/description.

These stub ``AgentProcess`` via ``__new__`` (same pattern as
``test_get_history_no_limit.py``) so no RPyC / LangGraph compile is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _make_proc(
    *,
    invariant_prompt: str | None,
    provider: str = "claude-code",
    model: str = "opus",
    agent: Any = None,
) -> Any:
    from deep_worker_cli.process.agent_process import AgentProcess

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "test-agent"
    proc.config = SimpleNamespace(
        provider=provider, model=model, working_dir="/tmp/test-agent"
    )
    proc.agent = agent
    if invariant_prompt is not None:
        proc._invariant_system_prompt = invariant_prompt
    return proc


class TestGetSystemPrompt:
    @pytest.mark.asyncio
    async def test_returns_captured_invariant(self) -> None:
        proc = _make_proc(invariant_prompt="INVARIANT PROMPT TEXT")
        resp = await proc._handle_get_system_prompt()
        assert resp["success"] is True
        assert resp["system_prompt"] == "INVARIANT PROMPT TEXT"
        assert resp["provider"] == "claude-code"
        assert resp["model"] == "opus"

    @pytest.mark.asyncio
    async def test_recompute_fallback_when_attr_missing(self) -> None:
        # No _invariant_system_prompt attribute → must recompute, not crash.
        proc = _make_proc(invariant_prompt=None)
        resp = await proc._handle_get_system_prompt()
        assert resp["success"] is True
        # Recomputed prompt must be a non-empty string mentioning the agent name
        # (get_system_prompt embeds the assistant_id identity section).
        assert isinstance(resp["system_prompt"], str)
        assert len(resp["system_prompt"]) > 0
        assert "test-agent" in resp["system_prompt"]


class TestGetTools:
    @pytest.mark.asyncio
    async def test_reduces_metadata_to_name_description(self, monkeypatch) -> None:
        proc = _make_proc(invariant_prompt="x")
        # Patch the metadata extractor to a fixed inventory.
        fake_meta = [
            {"name": "execute", "description": "Run a shell command", "input_schema": {}},
            {"name": "read_file", "description": "Read a file", "input_schema": {}},
        ]
        monkeypatch.setattr(
            type(proc),
            "_extract_final_tool_metadata",
            staticmethod(lambda agent, provider=None: fake_meta),
        )
        resp = await proc._handle_get_tools()
        assert resp["success"] is True
        assert resp["provider"] == "claude-code"
        assert resp["tools"] == [
            {"name": "execute", "description": "Run a shell command"},
            {"name": "read_file", "description": "Read a file"},
        ]

    @pytest.mark.asyncio
    async def test_tool_less_agent_returns_empty(self, monkeypatch) -> None:
        proc = _make_proc(invariant_prompt="x")
        monkeypatch.setattr(
            type(proc),
            "_extract_final_tool_metadata",
            staticmethod(lambda agent, provider=None: []),
        )
        resp = await proc._handle_get_tools()
        assert resp["success"] is True
        assert resp["tools"] == []

    @pytest.mark.asyncio
    async def test_metadata_objects_with_attributes(self, monkeypatch) -> None:
        # Metadata entries may be objects (not dicts) — extractor returns
        # ToolMetadata-like objects with .name/.description attributes.
        proc = _make_proc(invariant_prompt="x")
        meta = [
            SimpleNamespace(name="web_search", description="Search the web"),
        ]
        monkeypatch.setattr(
            type(proc),
            "_extract_final_tool_metadata",
            staticmethod(lambda agent, provider=None: meta),
        )
        resp = await proc._handle_get_tools()
        assert resp["success"] is True
        assert resp["tools"] == [{"name": "web_search", "description": "Search the web"}]
