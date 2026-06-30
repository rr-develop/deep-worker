"""Unit tests for provider-dependent proxy-tool invocation hints.

See design: docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md

The base deep-worker tools (send_message_to_agent, restart_self, …) are
exposed to the model through an MCP-proxy whose tool-name FORM depends on
the provider:

* claude-code → ``mcp__deepworker__<tool>`` (static server name)
* codex       → ``mcp__deepworker_<pid>__<tool>`` (server name carries PID)
* grok        → bare tool name; the grok CLI routes it as an MCP tool on the
                ``deep_worker_proxy_<pid>`` server (no string prefix)
* others / no-proxy (deepseek/openai/…) → bare name (native LangGraph tools)

``proxy_tool_form`` is the single pure source of truth for that mapping;
``provider_tool_invocation_note`` renders the full system-prompt note.
"""

from __future__ import annotations

import os

from deep_worker_cli.prompts.proxy_tool_hints import (
    provider_tool_invocation_note,
    proxy_tool_form,
)


class TestProxyToolForm:
    def test_claude_code_prefix(self) -> None:
        assert (
            proxy_tool_form("claude-code", "send_message_to_agent")
            == "mcp__deepworker__send_message_to_agent"
        )

    def test_codex_prefix_with_explicit_pid(self) -> None:
        assert (
            proxy_tool_form("codex", "send_message_to_agent", pid=1234)
            == "mcp__deepworker_1234__send_message_to_agent"
        )

    def test_codex_prefix_defaults_to_current_pid(self) -> None:
        out = proxy_tool_form("codex", "restart_self")
        assert out == f"mcp__deepworker_{os.getpid()}__restart_self"

    def test_grok_is_bare_name(self) -> None:
        # grok has no string prefix — CLI resolves the tool as an MCP tool.
        assert proxy_tool_form("grok", "send_message_to_agent") == (
            "send_message_to_agent"
        )
        assert "mcp__" not in proxy_tool_form("grok", "send_message_to_agent")

    def test_deepseek_falls_back_to_bare(self) -> None:
        assert (
            proxy_tool_form("deepseek", "send_message_to_agent")
            == "send_message_to_agent"
        )

    def test_unknown_provider_falls_back_to_bare(self) -> None:
        assert (
            proxy_tool_form("totally-unknown", "send_message_to_agent")
            == "send_message_to_agent"
        )

    def test_none_provider_falls_back_to_bare(self) -> None:
        assert (
            proxy_tool_form(None, "send_message_to_agent")
            == "send_message_to_agent"
        )


class TestProviderToolInvocationNote:
    """The rendered system-prompt note must be provider-truthful."""

    def test_claude_code_note_mentions_mcp_prefix_and_command_tools(self) -> None:
        note = provider_tool_invocation_note("claude-code")
        assert "mcp__deepworker__" in note
        for tool in (
            "send_message_to_agent",
            "restart_self",
            "restart_agent",
        ):
            assert tool in note
        # claude form is conditional ("if short name does not resolve").
        assert "mcp__deepworker__send_message_to_agent" in note

    def test_codex_note_mentions_pid_prefix_and_reservation(self) -> None:
        note = provider_tool_invocation_note("codex", pid=4242)
        assert "mcp__deepworker_4242__" in note
        # The note must explain WHY: short names are reserved for Codex
        # native tools.
        assert "Codex" in note or "native" in note.lower()

    def test_grok_note_disambiguates_mcp_vs_native_and_forbids_disk(self) -> None:
        note = provider_tool_invocation_note("grok")
        # grok note must NOT advertise an mcp__ prefix.
        assert "mcp__" not in note
        # It must say these are MCP tools, not native grok built-ins, and
        # forbid searching the schema on disk.
        lowered = note.lower()
        assert "mcp" in lowered
        assert "not" in lowered  # "NOT native built-ins" style phrasing
        assert "disk" in lowered or "mcps/" in note

    def test_no_proxy_providers_get_empty_note(self) -> None:
        for provider in ("deepseek", "openai", "anthropic", "google", None):
            assert provider_tool_invocation_note(provider) == ""

    def test_unknown_provider_gets_empty_note(self) -> None:
        assert provider_tool_invocation_note("totally-unknown") == ""
