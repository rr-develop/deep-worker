"""``deep-worker lsp`` subcommand — manages the optional LSP navigation feature.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md`` §4.2 / §10 (decision #1).

    deep-worker lsp enable  [--scope user|project|agent:<name>] [--force]
    deep-worker lsp disable [--scope ...]
    deep-worker lsp status  [--scope ...]

The ``mcp.json`` file is created with mode ``0o600`` in a ``0o700`` directory.

A convenience flag at agent creation is intentionally NOT added in v1 —
enabling stays an explicit operation on the config.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from deep_worker_cli.lsp.enable import (
    SERENA_SERVER_NAME,
    disable_lsp_at_path,
    enable_lsp_at_path,
    is_lsp_enabled_at_path,
)
from deep_worker_cli.lsp.serena_config import SERENA_COMMAND, resolve_workspace_root

_INSTALL_HINT = "uv tool install -p 3.13 serena-agent"


def _scope_to_path(scope: str) -> Path:
    """The same scope contract as ``deep-worker mcp``."""
    home = Path.home()
    if scope == "user":
        return home / ".deepagents" / "mcp.json"
    if scope == "project":
        return resolve_workspace_root(Path.cwd()) / ".deepagents" / "mcp.json"
    if scope.startswith("agent:"):
        name = scope.split(":", 1)[1].strip()
        if not name:
            raise SystemExit("--scope agent:<name> requires a non-empty agent name")
        return home / ".deepagents" / name / "mcp.json"
    raise SystemExit(
        f"unknown --scope {scope!r}; expected user | project | agent:<name>"
    )


def _config_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "config", None)
    if explicit:
        return Path(explicit)
    return _scope_to_path(getattr(args, "scope", "project"))


def _workspace_root(args: argparse.Namespace) -> Path:
    explicit_root = getattr(args, "project_root", None)
    return resolve_workspace_root(Path.cwd(), explicit=explicit_root)


def setup_lsp_parser(subparsers, make_help_action=None) -> None:
    """Register the ``lsp`` group on the top-level parser."""
    lsp = subparsers.add_parser(
        "lsp", help="Manage optional LSP code-navigation (Serena MCP)"
    )
    lsp_sub = lsp.add_subparsers(dest="lsp_command")

    for name, help_text in (
        ("enable", "Enable LSP navigation for a scope"),
        ("disable", "Disable LSP navigation for a scope"),
        ("status", "Show whether LSP navigation is enabled"),
    ):
        p = lsp_sub.add_parser(name, help=help_text)
        p.add_argument(
            "--scope",
            default="project",
            help="project (default) | user | agent:<name>",
        )
        p.add_argument(
            "--config",
            default=None,
            help="explicit path to mcp.json (overrides --scope)",
        )
        if name == "enable":
            p.add_argument(
                "--project-root",
                default=None,
                help="workspace root Serena should open (default: git root of cwd)",
            )
            p.add_argument(
                "--force",
                action="store_true",
                help="overwrite an existing serena entry",
            )
            p.add_argument(
                "--no-auto-approve",
                action="store_true",
                help="require HITL approval for Serena tools (default: auto-approved)",
            )


def handle_lsp_command(args: argparse.Namespace) -> int:
    command = getattr(args, "lsp_command", None)
    if command == "enable":
        return _cmd_enable(args)
    if command == "disable":
        return _cmd_disable(args)
    if command == "status":
        return _cmd_status(args)
    print("Usage: deep-worker lsp {enable|disable|status} [--scope ...]")
    return 2


def _warn_if_serena_missing() -> bool:
    if shutil.which(SERENA_COMMAND):
        return True
    print(
        f"warning: '{SERENA_COMMAND}' not found in PATH — tools will not appear "
        f"until it is installed:\n    {_INSTALL_HINT}"
    )
    return False


def _cmd_enable(args: argparse.Namespace) -> int:
    path = _config_path(args)
    try:
        changed = enable_lsp_at_path(
            path,
            workspace_root=_workspace_root(args),
            force=bool(getattr(args, "force", False)),
            auto_approve=not bool(getattr(args, "no_auto_approve", False)),
        )
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 1

    if changed:
        print(f"LSP navigation enabled in {path}")
    else:
        print(
            f"LSP navigation already enabled in {path} (use --force to overwrite)"
        )
    _warn_if_serena_missing()
    print("Restart the agent for the change to take effect.")
    return 0


def _cmd_disable(args: argparse.Namespace) -> int:
    path = _config_path(args)
    try:
        removed = disable_lsp_at_path(path)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 1
    print(
        f"LSP navigation disabled in {path}"
        if removed
        else f"LSP navigation was not enabled in {path}"
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    path = _config_path(args)
    enabled = is_lsp_enabled_at_path(path)
    print(f"config:  {path}")
    print(f"status:  {'enabled' if enabled else 'disabled'}")
    binary = shutil.which(SERENA_COMMAND)
    print(
        f"backend: {SERENA_SERVER_NAME} ({SERENA_COMMAND}) "
        + (f"found at {binary}" if binary else f"not found in PATH — {_INSTALL_HINT}")
    )
    return 0
