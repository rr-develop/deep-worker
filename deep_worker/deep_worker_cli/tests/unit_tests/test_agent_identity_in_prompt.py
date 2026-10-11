"""Tests for agent identity injection into the system prompt.

The agent must know its own registry name so that it understands:
  * which part of an incoming `<sender> -> <receiver> :> <msg>` marker
    refers to it (the receiver side),
  * which name will be prepended as sender when it calls
    send_message_to_agent.

Previously the prompt mentioned inter-agent messaging but never told the
LLM what its OWN name was — it had to guess from context, which
sometimes failed in multi-agent setups.
"""

from __future__ import annotations

from deep_worker_cli.agent import get_system_prompt


class TestAgentIdentityInPrompt:
    """get_system_prompt(assistant_id=...) must render the agent's own name
    into the resulting prompt text so the LLM sees it."""

    def test_prompt_contains_agent_identity_section(self) -> None:
        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
        )
        # The prompt must have an explicit "Your Agent Identity" style
        # section (or similar) that embeds the name. We accept any
        # heading form but require the name to appear in a sentence
        # like "Your name is `adam`" / "You are the agent named 'adam'"
        # so the LLM cannot miss it.
        assert "adam" in prompt, "agent name must be embedded in the prompt"
        # Be stricter: the marker line must literally reference the name
        # as THIS agent's identity, not just any mention.
        lowered = prompt.lower()
        assert (
            "your name is" in lowered
            or "your agent name is" in lowered
            or "you are the agent" in lowered
        ), (
            "Prompt must contain a sentence explicitly stating the agent's "
            "own name (e.g. 'Your agent name is `adam`'). Got snippet: "
            f"{prompt[:500]!r}"
        )

    def test_two_different_agents_get_distinct_identities(self) -> None:
        """Rendering for two different assistant_ids must produce two
        distinct prompts, each correctly identifying its own agent."""
        prompt_adam = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
        )
        prompt_eva = get_system_prompt(
            assistant_id="eva",
            sandbox_type=None,
            project_memory_dir=None,
        )

        assert "adam" in prompt_adam
        assert "eva" in prompt_eva
        # And — crucially — the OTHER agent's name must NOT appear as
        # the identity claim in each prompt.
        assert "Your agent name is `eva`" not in prompt_adam
        assert "Your agent name is `adam`" not in prompt_eva

    def test_prompt_describes_new_prefix_format(self) -> None:
        """Prompt must teach the LLM the new '<sender> -> <receiver> :>'
        marker (with the sender/receiver meanings), so it understands
        both outgoing echo and incoming messages from peers."""
        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
        )
        # Structural marker: literally contains the skeleton.
        # We accept either the exact form or a close variant that shows
        # sender/receiver placeholders.
        assert "->" in prompt and ":>" in prompt, (
            "Prompt must describe the '<sender> -> <receiver> :>' marker."
        )


class TestProviderToolInvocationNoteInPrompt:
    """get_system_prompt must embed a PROVIDER-DEPENDENT note teaching the
    model the correct MCP-proxy invocation form for base tools.

    See docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md (RC-A).
    """

    def test_claude_code_prompt_parametrizes_invocation_form(self) -> None:
        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
            provider="claude-code-sdk",
        )
        # The MCP form must appear, not just the bare name, so the model
        # uses the resolvable invocation name.
        assert "mcp__deepworker__send_message_to_agent" in prompt

    def test_codex_prompt_contains_pid_prefixed_form(self) -> None:
        import os

        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
            provider="codex",
        )
        assert f"mcp__deepworker_{os.getpid()}__send_message_to_agent" in prompt

    def test_grok_prompt_has_mcp_tool_note_without_prefix(self) -> None:
        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
            provider="grok",
        )
        # grok: no string prefix advertised; instead a disambiguating note.
        assert "mcp__deepworker__send_message_to_agent" not in prompt
        lowered = prompt.lower()
        assert "mcp" in lowered
        # forbids disk schema hunting
        assert "disk" in lowered or "mcps/" in prompt

    def test_deepseek_prompt_has_no_provider_invocation_note(self) -> None:
        prompt = get_system_prompt(
            assistant_id="adam",
            sandbox_type=None,
            project_memory_dir=None,
            provider="deepseek",
        )
        # No MCP-proxy prefix leaks into a native-tool provider prompt.
        assert "mcp__deepworker__" not in prompt
        assert "mcp__deepworker_" not in prompt
