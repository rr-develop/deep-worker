"""Unit tests for the prompt fragment (M2 reviewer note, design §4.4).

Key requirements:
* the fragment is built from the ACTUAL (mangled) tool names — the
  agent sees ``serena__find_symbol``, not ``serena.find_symbol``;
* no serena tools → empty string (optionality: the prompt does not
  change by a single byte);
* a tool absent from the list → no line about it (do not promise what
  does not exist).
"""
from __future__ import annotations

from deep_worker_cli.lsp.prompt import lsp_prompt_fragment

# The real set from a Serena 1.6.1 --context ide measurement (design §11),
# after mangling ``.`` → ``__``.
_REAL_TOOLS = [
    "serena__find_symbol",
    "serena__find_referencing_symbols",
    "serena__find_declaration",
    "serena__find_implementations",
    "serena__get_symbols_overview",
    "serena__get_diagnostics_for_file",
    "serena__rename_symbol",
    "serena__replace_symbol_body",
]


def test_no_tools_yields_empty_fragment() -> None:
    """Optionality: without LSP the prompt does not change."""
    assert lsp_prompt_fragment([]) == ""


def test_unrelated_tools_yield_empty_fragment() -> None:
    """Presence of OTHER MCP tools must not enable the LSP fragment."""
    assert lsp_prompt_fragment(["github__create_issue", "read_file"]) == ""


def test_fragment_mentions_mangled_names_verbatim() -> None:
    frag = lsp_prompt_fragment(_REAL_TOOLS)
    assert frag
    assert "serena__find_symbol" in frag
    assert "serena__find_referencing_symbols" in frag


def test_fragment_never_advertises_bare_canonical_names() -> None:
    """The bare ``serena.find_symbol`` cannot be called by the agent — no
    such tool name exists. The prompt must not mention it."""
    frag = lsp_prompt_fragment(_REAL_TOOLS)
    assert "serena.find_symbol" not in frag
    assert "serena.find_referencing_symbols" not in frag


def test_absent_tool_produces_no_line() -> None:
    """If the server did not provide rename_symbol, do not promise it to the agent."""
    subset = ["serena__find_symbol", "serena__get_symbols_overview"]
    frag = lsp_prompt_fragment(subset)
    assert "serena__find_symbol" in frag
    assert "rename_symbol" not in frag
    assert "find_referencing_symbols" not in frag


def test_fragment_is_present_even_for_minimal_toolset() -> None:
    frag = lsp_prompt_fragment(["serena__find_symbol"])
    assert "serena__find_symbol" in frag


def test_fragment_steers_away_from_grep() -> None:
    """The point of the feature: the agent must not grep where symbolic search exists."""
    frag = lsp_prompt_fragment(_REAL_TOOLS).lower()
    assert "grep" in frag


def test_fragment_mentions_warmup_without_promising_slow_first_call() -> None:
    """§4.4/§7.1: measurements show the first symbolic call is CHEAP (~1 s) —
    what is expensive is agent startup. So we tell the agent about a possible
    warm-up delay, but do NOT promise "the first call will be slow"."""
    frag = lsp_prompt_fragment(_REAL_TOOLS).lower()
    assert "delay" in frag
    assert "do not treat it as a hang" in frag
    assert "first call" not in frag


def test_custom_server_prefix_is_respected() -> None:
    """The server name in mcp.json may differ from 'serena'."""
    frag = lsp_prompt_fragment(["lsp__find_symbol"], server_name="lsp")
    assert "lsp__find_symbol" in frag


def test_fragment_tells_agent_to_omit_optional_fields() -> None:
    """Top-quality UX: the agent should by default NOT pass optional fields
    and override them only deliberately.

    Why the instruction was added: a Claude model, seeing fields with
    ``default`` in a tool's JSON schema, tends to proactively pass them all
    ("just in case"). Every extra key is tokens and cognitive load. The wire
    contract is already satisfied (the server applies its own defaults when a
    key is omitted — proven by the integration test
    ``test_optional_fields_are_fully_omissible_on_wire``); here we close the
    UX contract on the prompt side.

    We check CONTENT (by keywords), not the exact wording, so the text can
    be improved without rewriting the test.
    """
    frag = lsp_prompt_fragment(["serena__find_symbol"])
    lower = frag.lower()
    # Instruction about "pass only the required fields".
    assert "required" in lower, frag
    # Explicit mention that optional fields may be omitted.
    assert "optional" in lower, frag
    # And the note that the server applies its own defaults.
    assert "default" in lower, frag


def test_omit_optional_instruction_absent_when_no_lsp_tools() -> None:
    """Prompt optionality: if there are no LSP tools, no text (including
    about optional fields) is mixed into the prompt."""
    assert lsp_prompt_fragment([]) == ""
    assert lsp_prompt_fragment(["github__create_issue"]) == ""
