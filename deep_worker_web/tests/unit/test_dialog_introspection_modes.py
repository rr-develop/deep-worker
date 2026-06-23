"""Unit tests for the two introspection Full-Dialog modes (§9 of
DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md):

- ``system_prompt`` — the invariant (startup) system prompt.
- ``tools`` — the provider-dependent available-tool list.

These are pure-function tests for the new renderers in
``app.services.dialog_formatter``; no RPyC, no real agent.
"""

from __future__ import annotations

import pytest

from app.services.dialog_formatter import render_system_prompt, render_tools


class TestRenderSystemPrompt:
    def test_basic_header_and_body(self) -> None:
        out = render_system_prompt(
            "You are agent X.\nBe concise.",
            provider="claude-code",
            model="opus",
        )
        assert "--- System Prompt (invariant, provider=claude-code, model=opus) ---" in out
        assert "You are agent X." in out
        assert "Be concise." in out

    def test_header_precedes_body(self) -> None:
        out = render_system_prompt("BODYTEXT", provider="deepseek", model="dsv2")
        assert out.index("---") < out.index("BODYTEXT")

    def test_empty_prompt_has_placeholder(self) -> None:
        out = render_system_prompt("", provider="claude-code", model="opus")
        assert "(no system prompt)" in out

    def test_none_provider_model_tolerated(self) -> None:
        out = render_system_prompt("hi", provider=None, model=None)
        # Must not crash and must still show the body.
        assert "hi" in out
        assert "provider=" in out

    def test_cyrillic_preserved(self) -> None:
        out = render_system_prompt("You are agent Ro.", provider="claude-code", model="opus")
        assert "You are agent Ro." in out


class TestRenderTools:
    def test_lists_name_and_description(self) -> None:
        tools = [
            {"name": "execute", "description": "Run a shell command"},
            {"name": "read_file", "description": "Read a file"},
        ]
        out = render_tools(tools, provider="claude-code")
        assert "--- Available Tools (provider=claude-code, count=2) ---" in out
        assert "- execute: Run a shell command" in out
        assert "- read_file: Read a file" in out

    def test_empty_tool_list(self) -> None:
        out = render_tools([], provider="claude-code")
        assert "count=0" in out
        assert "(no tools)" in out

    def test_missing_description_tolerated(self) -> None:
        out = render_tools([{"name": "foo"}], provider="grok")
        assert "- foo" in out

    def test_multiline_description_first_line_only(self) -> None:
        tools = [{"name": "bar", "description": "First line\nSecond line"}]
        out = render_tools(tools, provider="claude-code")
        assert "- bar: First line" in out
        assert "Second line" not in out

    def test_none_provider_tolerated(self) -> None:
        out = render_tools([{"name": "x", "description": "y"}], provider=None)
        assert "- x: y" in out
