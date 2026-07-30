"""Generation of the ``mcp.json`` entry for Serena + workspace root resolution.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md`` §4.1 / §4.3.

The module is pure functions with no side effects and no I/O on import.
The shape of the returned entry must be understood by the EXISTING loader
``external_mcp/config.py`` without a single change to it (the contract is
pinned by the test ``test_generated_entry_parses_via_existing_loader``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: Server name in ``mcp.json``. It is also the prefix of the tool names
#: the agent will see after mangling: ``serena.find_symbol`` →
#: ``serena__find_symbol``.
SERENA_SERVER_NAME = "serena"

#: Executable name. Verified on serena-agent 1.6.1:
#: ``uv tool install -p 3.13 serena-agent`` installs ``serena``,
#: ``serena-agent`` and ``serena-hooks`` (see §11 of the design).
SERENA_COMMAND = "serena"

#: Serena context for external agent harnesses: disables part of the
#: duplicate file tools, since we already have our own.
DEFAULT_CONTEXT = "ide"


def resolve_workspace_root(
    working_dir: Path | str,
    *,
    explicit: Path | str | None = None,
) -> Path:
    """Determine the workspace root for Serena.

    Priority (§4.3 of the design):
        1. ``explicit`` — if given explicitly;
        2. the git repository root (``find_project_root``);
        3. ``working_dir`` itself.

    For a git worktree (``.git`` is a FILE, not a directory) the root
    becomes the worktree itself — the existing ``find_project_root``
    already ensures this, and we intentionally do NOT change it.

    The result is always absolute and normalized: the cwd of the child
    process is not guaranteed, so a relative path is not acceptable.
    """
    if explicit is not None:
        return Path(explicit).resolve()

    start = Path(working_dir).resolve()

    # Local import: keep the module free of heavy dependencies at import
    # time.
    from deep_worker_cli.project_utils import find_project_root  # noqa: PLC0415

    return (find_project_root(start) or start).resolve()


def build_serena_mcp_entry(
    *,
    workspace_root: Path | str,
    context: str = DEFAULT_CONTEXT,
    auto_approve: bool = True,
    serena_command: str | None = None,
) -> dict[str, Any]:
    """Build the ``mcpServers.serena`` entry for ``mcp.json``.

    Args:
        workspace_root: project root that Serena will open as its project.
            Normalized to an absolute path.
        context: Serena context (``ide`` by default).
        auto_approve: mark the server as auto-approved. Defaults to
            ``True`` — the user's decision (write tools without HITL).
        serena_command: path/name of the binary, if Serena is not on PATH.

    Returns:
        A JSON-compatible dict in the form understood by
        ``external_mcp.config._parse_one``.
    """
    root = Path(workspace_root).resolve()
    return {
        "type": "stdio",
        "command": serena_command or SERENA_COMMAND,
        "args": [
            "start-mcp-server",
            "--context",
            context,
            "--transport",
            "stdio",
            # Explicit --project, NOT --project-from-cwd: the cwd of the
            # child process is not guaranteed and is not worktree-safe.
            "--project",
            str(root),
        ],
        "auto_approve": bool(auto_approve),
    }
