"""Prompt fragment steering the agent toward symbolic tools instead of grep.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md`` §4.4 (note M2).

Why this is needed at all: upstream Serena explicitly documents that
clients with strong built-in tools systematically ignore MCP tools
(which is why they ship reminder hooks). We have no hook mechanism,
so the only available lever is the system prompt.

Two hard requirements:

1. The names in the text are EXACTLY the ones the agent sees in tool calls,
   i.e. already mangled (``serena.find_symbol`` → ``serena__find_symbol``).
   The fragment is therefore built from the actual ``tool.name`` values,
   not from hardcoded literals.
2. No LSP tools → empty string. The prompt of a non-LSP project does not
   change by a single byte (optionality).
"""
from __future__ import annotations

from collections.abc import Sequence

from deep_worker_cli.lsp.serena_config import SERENA_SERVER_NAME

#: bare tool name → hint string for the agent. Order = display order.
#: Keep formulations short: a long prompt gets pushed out of attention.
_TOOL_HINTS: tuple[tuple[str, str], ...] = (
    ("find_symbol", "find a symbol by name (class/function/method) anywhere in the project"),
    (
        "find_referencing_symbols",
        "find all usages of a symbol (Show usages / Find references)",
    ),
    ("find_declaration", "jump to the declaration (Go to definition)"),
    ("find_implementations", "find implementations of an interface/abstract method"),
    ("get_symbols_overview", "quick overview of a file's symbols (outline) without reading it fully"),
    ("get_diagnostics_for_file", "errors/warnings for a file"),
    ("rename_symbol", "rename a symbol everywhere at once"),
    ("replace_symbol_body", "replace the whole body of a symbol"),
    ("insert_after_symbol", "insert code after a symbol"),
    ("insert_before_symbol", "insert code before a symbol"),
    ("safe_delete_symbol", "safely delete a symbol"),
)


def lsp_prompt_fragment(
    tool_names: Sequence[str],
    *,
    server_name: str = SERENA_SERVER_NAME,
) -> str:
    """Build a system-prompt fragment from the ACTUAL tool names.

    Args:
        tool_names: tool names as the agent sees them (mangled).
        server_name: MCP server name in ``mcp.json`` (name prefix).

    Returns:
        The ready fragment, or ``""`` if there are no LSP tools.
    """
    available = set(tool_names)
    prefix = f"{server_name}__"

    lines: list[str] = []
    for bare, hint in _TOOL_HINTS:
        mangled = f"{prefix}{bare}"
        # Only tools that are actually present: do not promise the agent
        # what the server did not provide.
        if mangled in available:
            lines.append(f"- `{mangled}` — {hint}")

    if not lines:
        return ""

    return (
        "## Code navigation via LSP (symbolic tools)\n"
        "\n"
        "Precise symbolic navigation is available for this project. "
        "Prefer it over textual search:\n"
        "\n" + "\n".join(lines) + "\n"
        "\n"
        "Rules:\n"
        "- Looking for where a symbol is declared or who uses it — take a "
        "symbolic tool, NOT `grep`. One call instead of a grep→read→grep chain.\n"
        "- `grep` remains appropriate for text patterns, non-code, "
        "configs, logs, and when the symbol name is not known in advance.\n"
        "- Make edits to existing code with symbolic tools when they are "
        "available: more precise than editing by line numbers.\n"
        "- A one-time delay while the language server warms up is possible — "
        "that is normal, do not treat it as a hang and do not repeat the request.\n"
        "- Pass ONLY the required fields of the tool. Optional fields "
        "(with a `default` in the schema) are filled in by the server "
        "itself — override them only when you deliberately want a "
        "different value. Fewer fields — fewer tokens, faster call.\n"
    )
