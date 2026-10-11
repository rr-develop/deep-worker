"""Unit tests for provider-dependent proxy-tool invocation hints.

See design: docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md

The base deep-worker tools (send_message_to_agent, restart_self, …) are
exposed to the model through an MCP-proxy whose tool-name FORM depends on
the provider:

* claude-code-sdk → ``mcp__deepworker__<tool>`` (static server name)
* codex → ``mcp__deepworker_<pid>__<tool>`` (server name carries PID)
* claude-code → ``mcp__deep_worker_acp_proxy_<pid>__<tool>`` (shared ACP
  layer proxy)
* grok → MANGLED name ``deep_worker_acp_proxy_<pid>__<tool>`` in the CLI's
  own tool list (grok 1.0.5 dispatches through its ``use_tool`` built-in;
  see a related internal incident analysis, not included here)
* others / no-proxy (deepseek/openai/anthropic/google/gemini/vertexai/minimax/
  any OpenAI-compatible provider from providers.json) → bare name
  (native LangGraph tools)

``proxy_tool_form`` is the single pure source of truth for that mapping;
``provider_tool_invocation_note`` renders the full system-prompt note.

Owner directive 2026-08-18: every built-in provider MUST have a
non-empty note that carries a CRITICAL emphasis on
``send_message_to_agent``. Unknown provider names log a ``WARNING``
and fall back to the stateless-API default — silent regressions when
a new provider is added are unacceptable, so this is enforced both by
the warning and by the test below.
"""

from __future__ import annotations

import logging
import os

import pytest

from deep_worker_cli.prompts.proxy_tool_hints import (
    ACP_PROXY_MANGLED,
    ACP_PROXY_MCP_PREFIX,
    ACP_PROXY_SERVER_ROOT,
    ACP_PROXY_STATIC,
    ACP_PROXY_WITH_PID,
    STATELESS_API,
    critical_send_message_to_agent_block,
    provider_tool_invocation_note,
    proxy_tool_form,
)

# Every provider that ships with deep-worker MUST be covered. The
# "automatic" path: stateless-API providers all share one note (no
# per-provider entry needed); ACP providers go through ACP_PROXY_*.
# This set is the regression-detection net — if a new provider lands
# without being added to the right category, the warning in
# provider_tool_invocation_note fires AND this test fails.
ALL_BUILTIN_PROVIDERS: tuple[str, ...] = (
    "deepseek",
    "openai",
    "anthropic",
    "google",
    "gemini",
    "vertexai",
    "minimax",
    "claude-code-sdk",
    "claude-code",
    "codex",
    "grok",
)


class TestProxyToolForm:
    def test_claude_code_prefix(self) -> None:
        assert (
            proxy_tool_form("claude-code-sdk", "send_message_to_agent")
            == "mcp__deepworker__send_message_to_agent"
        )

    def test_codex_prefix_with_explicit_pid(self) -> None:
        assert (
            proxy_tool_form("codex", "send_message_to_agent", pid=1234)
            == "mcp__deepworker_1234__send_message_to_agent"
        )

    def test_claude_code_prefix_with_explicit_pid(self) -> None:
        # claude-code rides the SHARED ACP layer: its proxy server is
        # deep_worker_acp_proxy_<pid>, not the codex deepworker_<pid>.
        assert (
            proxy_tool_form("claude-code", "send_message_to_agent", pid=5678)
            == "mcp__deep_worker_acp_proxy_5678__send_message_to_agent"
        )

    def test_codex_prefix_defaults_to_current_pid(self) -> None:
        out = proxy_tool_form("codex", "restart_self")
        assert out == f"mcp__deepworker_{os.getpid()}__restart_self"

    def test_grok_is_mangled_server_prefixed_name(self) -> None:
        """grok 1.0.5 lists our tools under ``<server>__<tool>`` (no
        ``mcp__`` prefix) — the bare name is a NATIVE built-in at best
        (a related internal incident analysis, not included here)."""
        assert (
            proxy_tool_form("grok", "send_message_to_agent", pid=777)
            == "deep_worker_acp_proxy_777__send_message_to_agent"
        )
        out = proxy_tool_form("grok", "send_message_to_agent")
        assert out == (
            f"deep_worker_acp_proxy_{os.getpid()}__send_message_to_agent"
        )
        assert "mcp__" not in out

    def test_grok_live_server_name_wins_over_pid(self) -> None:
        assert (
            proxy_tool_form(
                "grok",
                "send_message_to_agent",
                pid=777,
                server_name="deep_worker_acp_proxy_42",
            )
            == "deep_worker_acp_proxy_42__send_message_to_agent"
        )

    def test_acp_proxy_root_matches_provider_source(self) -> None:
        """The literal root in this module must equal the server name the
        shared ACP proxy actually constructs — drift here re-creates the
        stale-hint half of a past incident."""
        import inspect

        from deep_worker.providers.acp import proxy_mcp_server

        src = inspect.getsource(proxy_mcp_server)
        assert f'"{ACP_PROXY_SERVER_ROOT}' + '{os.getpid()}"' in src

    def test_deepseek_falls_back_to_bare(self) -> None:
        assert (
            proxy_tool_form("deepseek", "send_message_to_agent")
            == "send_message_to_agent"
        )

    def test_minimax_falls_back_to_bare(self) -> None:
        assert (
            proxy_tool_form("minimax", "send_message_to_agent")
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
    """The rendered system-prompt note must be provider-truthful AND
    always carry the CRITICAL emphasis on ``send_message_to_agent``."""

    def test_claude_code_note_mentions_mcp_prefix_and_command_tools(self) -> None:
        note = provider_tool_invocation_note("claude-code-sdk")
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

    def test_claude_code_note_mentions_pid_prefix(self) -> None:
        # claude-code rides the shared ACP layer: the prefix is the layer's
        # proxy server name (deep_worker_acp_proxy_<pid>), NOT the codex
        # deepworker_<pid> naming this note used to (wrongly) claim.
        note = provider_tool_invocation_note("claude-code", pid=9999)
        assert (
            "mcp__deep_worker_acp_proxy_9999__send_message_to_agent" in note
        )
        assert "mcp__deep_worker_acp_proxy_9999__restart_self" in note
        assert "mcp__deepworker_9999__" not in note

    def test_grok_note_names_the_mangled_form_and_forbids_disk(self) -> None:
        note = provider_tool_invocation_note("grok", pid=777)
        # grok 1.0.5 lists our tools MANGLED, without an mcp__ prefix.
        assert "deep_worker_acp_proxy_777__send_message_to_agent" in note
        assert "mcp__" not in note
        # The pre-3fe834d server root and the ≤1.0.4 wrapper must be gone —
        # both were the stale-hint half of a past incident.
        assert "deep_worker_proxy_<pid>" not in note
        assert "CallMcpTool" not in note
        # It must still say these are MCP tools, not native grok built-ins,
        # and forbid searching the schema on disk.
        lowered = note.lower()
        assert "mcp" in lowered
        assert "not" in lowered  # "NOT native built-ins" style phrasing
        assert "disk" in lowered or "mcps/" in note

    def test_stateless_api_providers_get_specific_note(self) -> None:
        """Every stateless-API provider must receive a non-empty note that
        uses the provider-correct (bare) invocation form.

        This is the regression-detection net for ``deepseek``, ``minimax``
        and other stateless-API providers that previously got an empty
        string from this function.
        """
        for provider in STATELESS_API:
            note = provider_tool_invocation_note(provider)
            assert note, f"provider {provider!r} got empty note"
            assert "send_message_to_agent" in note
            assert "CRITICAL" in note

    def test_minimax_gets_specific_note(self) -> None:
        """Owner directive 2026-08-18: explicitly check minimax, since it
        is the most recent config-driven provider and would have been
        silent-falling-back to ``""`` under the old implementation.
        """
        note = provider_tool_invocation_note("minimax")
        assert note, "minimax got empty note (silent regression)"
        assert "send_message_to_agent" in note
        assert "CRITICAL" in note

    def test_unknown_provider_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown provider names must log a WARNING (so silent
        regressions show up in operator logs) AND still receive a
        strong generic-but-non-empty note."""
        with caplog.at_level(logging.WARNING, logger="deep_worker_cli.prompts.proxy_tool_hints"):
            note = provider_tool_invocation_note("totally-unknown")
        assert note, "unknown provider got empty note"
        assert "CRITICAL" in note
        assert any(
            "totally-unknown" in record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ), "no WARNING logged for unknown provider"

    def test_none_provider_gets_default_note(self) -> None:
        note = provider_tool_invocation_note(None)
        assert note
        assert "CRITICAL" in note

    @pytest.mark.parametrize("provider", ALL_BUILTIN_PROVIDERS)
    def test_every_builtin_provider_has_critical_block(
        self, provider: str
    ) -> None:
        """Regression-detection net: every built-in provider MUST
        receive a non-empty note that carries the CRITICAL
        ``send_message_to_agent`` emphasis. If a new provider is added
        but not classified, this test fails immediately."""
        note = provider_tool_invocation_note(provider)
        assert note, f"{provider}: empty note"
        assert "CRITICAL" in note, f"{provider}: no CRITICAL emphasis"
        assert "send_message_to_agent" in note, (
            f"{provider}: send_message_to_agent not mentioned"
        )
        # The CRITICAL block must spell out the invocation form the model
        # should use. For ACP providers this is the provider-specific
        # mangled/prefixed form; for stateless-API providers it stays bare.
        if provider in ACP_PROXY_STATIC:
            assert "mcp__deepworker__send_message_to_agent" in note
        elif provider in ACP_PROXY_WITH_PID:
            assert "mcp__deepworker_" in note
        elif provider in ACP_PROXY_MCP_PREFIX:
            assert "mcp__deep_worker_acp_proxy_" in note
        elif provider in ACP_PROXY_MANGLED:
            assert "deep_worker_acp_proxy_" in note
            assert "mcp__" not in note


class TestCriticalBlock:
    """The reusable ``critical_send_message_to_agent_block`` helper used
    by ACP-layer / provider-specific ``_build_tool_contract`` functions."""

    def test_returns_provider_correct_form(self) -> None:
        assert (
            "mcp__deepworker__send_message_to_agent"
            in critical_send_message_to_agent_block("claude-code-sdk")
        )
        assert (
            "mcp__deepworker_4242__send_message_to_agent"
            in critical_send_message_to_agent_block("codex", pid=4242)
        )
        # grok: MANGLED server-prefixed name, no mcp__ prefix.
        grok_block = critical_send_message_to_agent_block("grok", pid=777)
        assert "deep_worker_acp_proxy_777__send_message_to_agent" in grok_block
        assert "mcp__" not in grok_block
        # Stateless-API: bare name.
        block = critical_send_message_to_agent_block("deepseek")
        assert "send_message_to_agent" in block
        # The bare-name form must appear without an mcp__ prefix when
        # the provider routes via bare name.
        assert "mcp__" not in block

    def test_live_server_name_wins_in_critical_block(self) -> None:
        """The ACP layer passes the LIVE proxy server name — it must win
        over the PID fallback so a contract rendered for a resumed session
        never names a dead PID."""
        block = critical_send_message_to_agent_block(
            "grok", pid=1, server_name="deep_worker_acp_proxy_31337"
        )
        assert (
            "deep_worker_acp_proxy_31337__send_message_to_agent" in block
        )

    def test_invariant_policy_text(self) -> None:
        """The policy (MUST reply via send_message_to_agent) is the
        SAME across all providers — only the invocation form changes.
        Owner directive 2026-08-18: the CRITICAL block must be invariant
        even when the surface transport differs (e.g. ``System:>`` on
        ACP providers)."""
        for provider in ALL_BUILTIN_PROVIDERS + (None,):
            block = critical_send_message_to_agent_block(provider)
            assert "MUST reply using" in block, provider
            assert "ONLY way to respond to a peer" in block, provider
            assert "free-text reply" in block, provider
            assert "System:>" in block, (
                f"{provider}: CRITICAL block must mention the System:> "
                f"transport detail so the model doesn't mistake peer "
                f"messages for user turns"
            )