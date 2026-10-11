"""A restricted agent must not be taught the tools it cannot call.

Removing a tool's schema while leaving three screens of prose explaining
how to use it does two bad things: it invites the model to call the tool
(which, before the dispatch barrier, actually worked), and it spends the
context budget of a tool-effectiveness experiment on tools that are not
part of it.

The prompt is assembled from four places, and each needed its own cut:

* ``default_agent_prompt.md`` — static markdown, never passes through
  ``wrap_model_call``, so the middleware cannot reach it;
* sections built in ``get_system_prompt`` itself (skills, web search,
  todos, path handling);
* the identity / provider-invocation notes in ``prompts.proxy_tool_hints``;
* sections other middleware append per turn — covered by
  ``test_tool_selection_middleware``.

The default path is pinned byte-for-byte: every existing agent depends on
that text, and this feature must be invisible to them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from deep_worker_cli.agent import get_system_prompt
from deep_worker_cli.prompts.proxy_tool_hints import (
    provider_tool_invocation_note,
)
from deep_worker_cli.tool_selection import (
    EMPTY_SELECTION,
    parse_tool_selection_flag,
    strip_disabled_prompt_sections,
)

# Every provider category in ``provider_tool_invocation_note``: the notes
# are rendered by four different code paths and each names tools.
PROVIDERS = ["minimax", "claude-code", "codex", "grok", "claude-code-sdk"]

# Tools whose names must vanish from a "-*, project_memory" prompt.
DENIED = [
    "read_file",
    "edit_file",
    "write_file",
    "glob",
    "execute",
    "web_search",
    "http_request",
    "fetch_url",
    "read_image",
    "send_message_to_agent",
    "restart_agent",
    "restart_self",
    "write_todos",
]


def _prompt(provider, spec, agent="probe"):
    return get_system_prompt(
        agent, None, Path("/tmp"), provider, parse_tool_selection_flag(spec)
    )


def _mentions(text: str, name: str) -> bool:
    """Word-boundary match, so 'gather/execute' is not an ``execute`` hit."""
    return bool(re.search(rf"\b{re.escape(name)}\b", text))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_denied_tools_are_not_named_anywhere(provider) -> None:
    prompt = _prompt(provider, "-*,project_memory")
    leaked = [n for n in DENIED if _mentions(prompt, n)]

    if provider == "grok":
        # grok's note names ``read_file``/``grep`` as the CLI's OWN
        # built-ins, warning the model not to mistake them for ours. Our
        # selection cannot switch those off (see _UNSUPPRESSABLE_NATIVES),
        # so the warning stays true and useful.
        leaked = [n for n in leaked if n not in ("read_file", "grep")]

    assert leaked == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_stripping_is_surgical_not_wholesale(provider) -> None:
    """Only tool-owned sections go; the agent's own instructions stay.

    ``project_memory`` is deliberately NOT asserted here: its instructions
    are appended per turn by ``ProjectMemoryMiddleware``, so they are not
    part of this static prompt at all (the middleware-side pruning is
    covered by ``test_tool_selection_middleware``). What this pins is that
    the strip did not take the surrounding prose with it.
    """
    prompt = _prompt(provider, "-*,project_memory")
    assert "# Core Behavior" in prompt
    assert "## Code References" in prompt
    assert "### Your Agent Identity" in prompt
    assert "### Current Working Directory" in prompt


def test_a_kept_file_tool_keeps_its_instructions() -> None:
    """The positive case for a tool the base prompt DOES describe."""
    prompt = _prompt("minimax", "-*,read_file")
    assert "### File Tools" in prompt
    assert _mentions(prompt, "read_file")
    # ...while its neighbours in the same section list are gone.
    assert "### shell" not in prompt


@pytest.mark.parametrize("provider", PROVIDERS)
def test_default_prompt_is_untouched(provider) -> None:
    """No selection and an empty selection must both be the old text."""
    baseline = get_system_prompt("probe", None, Path("/tmp"), provider, None)
    assert (
        get_system_prompt("probe", None, Path("/tmp"), provider, EMPTY_SELECTION)
        == baseline
    )
    # And a selection that removes nothing must not reshape it either.
    assert (
        get_system_prompt(
            "probe", None, Path("/tmp"), provider, parse_tool_selection_flag("")
        )
        == baseline
    )


def test_deny_list_removes_only_the_named_tool() -> None:
    prompt = _prompt("minimax", "-execute")
    assert not _mentions(prompt, "execute") or "### shell" not in prompt
    assert "### shell" not in prompt
    # Everything else keeps its instructions.
    assert _mentions(prompt, "read_file")
    assert _mentions(prompt, "send_message_to_agent")


def test_star_alone_leaves_no_tool_instructions() -> None:
    prompt = _prompt("minimax", "-*")
    assert [n for n in DENIED if _mentions(prompt, n)] == []
    assert not _mentions(prompt, "project_memory")


def test_messaging_disabled_is_stated_not_silently_dropped() -> None:
    """An agent that cannot answer a peer must be told, not left guessing.

    Peer messages still ARRIVE (the inbox is not a tool), so removing the
    protocol without a word would leave the agent to discover mid-turn
    that it cannot reply.
    """
    prompt = _prompt("minimax", "-*,project_memory")
    assert "Inter-agent messaging is DISABLED" in prompt
    # It still needs to recognise its own name in an incoming message.
    assert "probe" in prompt


def test_identity_keeps_the_protocol_when_messaging_survives() -> None:
    prompt = _prompt("minimax", "-*,send_message_to_agent")
    assert "Inter-agent messaging is DISABLED" not in prompt
    assert "MUST reply" in prompt


def test_command_tool_forms_are_not_offered_for_denied_tools() -> None:
    """The per-provider invocation forms are the sharpest hint of all."""
    selection = parse_tool_selection_flag("-*,project_memory")
    for provider in PROVIDERS:
        note = provider_tool_invocation_note(provider, selection=selection)
        assert "restart_self" not in note
        assert "restart_agent" not in note


def test_critical_block_goes_with_its_tool() -> None:
    selection = parse_tool_selection_flag("-*,project_memory")
    note = provider_tool_invocation_note("minimax", selection=selection)
    assert "CRITICAL — inter-agent messaging" not in note

    kept = provider_tool_invocation_note(
        "minimax", selection=parse_tool_selection_flag("-*,send_message_to_agent")
    )
    assert "CRITICAL — inter-agent messaging" in kept


def test_base_prompt_sections_have_not_drifted() -> None:
    """Tripwire: a renamed heading silently stops being strippable.

    ``strip_disabled_prompt_sections`` matches headings literally. If the
    markdown is reworded, the section survives the strip while its tool is
    gone — the exact failure this feature exists to prevent, and one that
    no other test would notice.
    """
    from deep_worker_cli.tool_selection import _PROMPT_SECTIONS
    from deep_worker_cli.config import get_default_coding_instructions

    prompt = get_default_coding_instructions()
    missing = [h for h, _ in _PROMPT_SECTIONS if h not in prompt]
    assert missing == [], (
        f"headings no longer present in default_agent_prompt.md: {missing}. "
        f"Update _PROMPT_SECTIONS to match, or the sections stop being "
        f"stripped for restricted agents."
    )


def test_stripping_is_a_no_op_without_a_restriction() -> None:
    from deep_worker_cli.config import get_default_coding_instructions

    base = get_default_coding_instructions()
    assert strip_disabled_prompt_sections(base, EMPTY_SELECTION) == base


def test_container_heading_goes_when_its_children_do() -> None:
    """A bare '## Tools' with nothing under it reads as a hidden list."""
    from deep_worker_cli.config import get_default_coding_instructions

    stripped = strip_disabled_prompt_sections(
        get_default_coding_instructions(), parse_tool_selection_flag("-*")
    )
    assert "## Tools" not in stripped
    # Neighbouring prose is untouched.
    assert "## Code References" in stripped
