"""MCP stdio transport for the bridge (DESIGN §15 commit 5c).

5b gave us :class:`~deep_worker_cli.lsp.bridge_loop.BridgeSession` — the
gRPC proxy to the shared service. This module is the transport around
it: it advertises the tool surface over MCP and forwards each call.

The invariant carried through the whole design applies here too, at the
last hop before the model sees anything: **a failure is reported as an
MCP error, never as a successful empty result.** If a transport fault
came back as an empty list, the model would read "no matches" and
conclude the symbol does not exist — a confident wrong answer, which is
strictly worse than a visible error.

This module holds no project state and spawns nothing. Only
``deep_worker_cli.lsp.service*`` may spawn the backend.
"""
from __future__ import annotations

import json
from typing import Any, Final, Protocol

import mcp.types as types
from mcp.server import Server

from deep_worker_cli.lsp.bridge_loop import tool_names

_SERVER_NAME: Final = "serena"

#: Per-tool description + the arguments the model must supply. Kept
#: close to the Serena tool surface so prompts that already reference
#: these names and parameters keep working across the migration.
_TOOL_SPECS: Final[dict[str, tuple[str, dict[str, Any], list[str]]]] = {
    "find_symbol": (
        "Find a symbol (class/function/method) by name path.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
            "depth": {"type": "integer"},
            "include_body": {"type": "boolean"},
        },
        ["name_path"],
    ),
    "find_referencing_symbols": (
        "Find all references to a symbol (show usages).",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
        },
        ["name_path", "relative_path"],
    ),
    "find_declaration": (
        "Go to the declaration of a symbol.",
        {
            "relative_path": {"type": "string"},
            "regex": {"type": "string"},
        },
        ["relative_path", "regex"],
    ),
    "find_implementations": (
        "Find implementations of an interface or abstract method.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
        },
        ["name_path", "relative_path"],
    ),
    "get_symbols_overview": (
        "Outline of the symbols defined in a file.",
        {"relative_path": {"type": "string"}},
        ["relative_path"],
    ),
    "get_diagnostics_for_file": (
        "Errors and warnings reported for a file.",
        {"relative_path": {"type": "string"}},
        ["relative_path"],
    ),
    "rename_symbol": (
        "Rename a symbol across the whole codebase.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
            "new_name": {"type": "string"},
        },
        ["name_path", "relative_path", "new_name"],
    ),
    "replace_symbol_body": (
        "Replace the body of a symbol.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
            "body": {"type": "string"},
        },
        ["name_path", "relative_path", "body"],
    ),
    "insert_before_symbol": (
        "Insert content immediately before a symbol definition.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
            "body": {"type": "string"},
        },
        ["name_path", "relative_path", "body"],
    ),
    "insert_after_symbol": (
        "Insert content immediately after a symbol definition.",
        {
            "name_path": {"type": "string"},
            "relative_path": {"type": "string"},
            "body": {"type": "string"},
        },
        ["name_path", "relative_path", "body"],
    ),
    "safe_delete_symbol": (
        "Delete a symbol if nothing references it.",
        {
            "name_path_pattern": {"type": "string"},
            "relative_path": {"type": "string"},
        },
        ["name_path_pattern", "relative_path"],
    ),
}


class _SessionLike(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


def tool_definitions() -> list[types.Tool]:
    """MCP tool descriptors for the whole service surface."""
    tools: list[types.Tool] = []
    for name in tool_names():
        description, properties, required = _TOOL_SPECS[name]
        tools.append(
            types.Tool(
                name=name,
                description=description,
                inputSchema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
        )
    return tools


def build_mcp_server(session: _SessionLike) -> Server:
    """Build an MCP server that forwards every tool call to ``session``."""
    server: Server = Server(_SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> Any:
        # Any exception raised here is turned into an MCP error result
        # by the framework, which is exactly what we want: the model
        # must see a failure as a failure, never as "no matches".
        result = await session.call_tool(name, arguments or {})
        if result is None:
            # Not wrapped on purpose: the SDK turns ``None`` into an
            # error result. A payload-shaped envelope here would turn a
            # visible failure into an empty success, and an empty string
            # cannot be told apart from a genuinely empty answer.
            return result
        # The backend already flattened Serena's blocks into one JSON
        # string, so the shape is a plain payload, not MCP content. It
        # has to be put back into an envelope: the SDK only builds one
        # for a ``dict``; every other iterable (a list of symbols, say)
        # is taken as ready-made content and fails validation.
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    result, indent=2, ensure_ascii=False, default=str
                ),
            )
        ]

    return server


async def serve_stdio(session: _SessionLike) -> None:
    """Run the MCP stdio loop until the client disconnects."""
    from mcp.server.stdio import stdio_server

    server = build_mcp_server(session)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
