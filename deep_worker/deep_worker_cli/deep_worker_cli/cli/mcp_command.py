"""``deep-worker mcp`` subcommand.

Manage the external MCP server config files. See
``docs/DESIGN_EXTERNAL_MCP_TOOLS.md`` §3.7.

Subcommands:
    deep-worker mcp list   [--scope user|project|agent:<n>|all]
    deep-worker mcp add    <name> --command "npx -y @x/y" [--env K=V]...
                            [--scope user|project|agent:<n>] [--auto-approve]
    deep-worker mcp remove <name> [--scope user|project|agent:<n>]

Files are created with mode ``0o600`` (they may contain tokens).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deep_worker_cli.cli.cwd import invocation_cwd
from deep_worker_cli.external_mcp.config import (
    ServerSpec,
    load_mcp_config,
)


def _scope_to_path(scope: str) -> Path:
    home = Path.home()
    if scope == "user":
        return home / ".deepagents" / "mcp.json"
    if scope == "project":
        from deep_worker_cli.project_utils import find_project_root

        # Starting point is the INVOCATION directory, not the process cwd:
        # the launcher does ``cd "$SCRIPT_DIR"``, which made project-scope
        # resolve to the install directory (writing into another repo). See cli/cwd.py.
        start = invocation_cwd()
        root = find_project_root(start) or start
        return root / ".deepagents" / "mcp.json"
    if scope.startswith("agent:"):
        name = scope.split(":", 1)[1].strip()
        if not name:
            raise SystemExit("--scope agent:<name> requires a non-empty agent name")
        return home / ".deepagents" / name / "mcp.json"
    raise SystemExit(
        f"unknown --scope {scope!r}; expected user | project | agent:<name>"
    )


def _read(path: Path) -> dict:
    if not path.exists():
        return {"mcpServers": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "mcpServers" not in data:
        # Migrate cursor-style flat → wrapped.
        data = {"mcpServers": dict(data)}
    return data


def _atomic_write_0600(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.chmod(0o600)
    tmp.replace(path)


def setup_mcp_parser(subparsers, make_help_action=None) -> None:
    """Register the ``mcp`` subcommand on the top-level parser."""
    mcp = subparsers.add_parser("mcp", help="Manage external MCP servers")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")

    list_p = mcp_sub.add_parser("list", help="List configured MCP servers")
    list_p.add_argument(
        "--scope",
        default="all",
        help="Limit to one scope: user | project | agent:<name> | all (default)",
    )

    add_p = mcp_sub.add_parser("add", help="Add or update an MCP server entry")
    add_p.add_argument("name", help="Server name (used as the tool-name prefix)")
    add_p.add_argument(
        "--command", required=False,
        help="stdio command to run (e.g. 'npx -y @modelcontextprotocol/server-github')",
    )
    add_p.add_argument(
        "--url", required=False,
        help="streamable-HTTP MCP URL (mutually exclusive with --command)",
    )
    add_p.add_argument(
        "--env", action="append", default=[],
        help="env var to pass, KEY=VALUE; repeatable. Use ${env:VAR} to defer.",
    )
    add_p.add_argument("--cwd", default=None, help="working dir for the stdio process")
    add_p.add_argument(
        "--scope", default="user",
        help="user (default) | project | agent:<name>",
    )
    add_p.add_argument(
        "--auto-approve", action="store_true",
        help="mark this server as auto-approved (no HITL even without -sp)",
    )
    add_p.add_argument(
        "--disabled", action="store_true",
        help="add the entry but disabled (won't start until you set disabled=false)",
    )

    rm_p = mcp_sub.add_parser("remove", help="Remove an MCP server entry")
    rm_p.add_argument("name")
    rm_p.add_argument("--scope", default="user")


def handle_mcp_command(args: argparse.Namespace) -> int:
    if args.mcp_command == "list":
        return _cmd_list(args.scope)
    if args.mcp_command == "add":
        return _cmd_add(args)
    if args.mcp_command == "remove":
        return _cmd_remove(args)
    print("Usage: deep-worker mcp {list|add|remove} ...")
    return 2


def _cmd_list(scope: str) -> int:
    home = Path.home()
    user_path = home / ".deepagents" / "mcp.json"
    project_path = _scope_to_path("project")
    # ``agent:<n>`` and ``all`` cases are folded into the same dump below by
    # walking ~/.deepagents/*/mcp.json explicitly.
    if scope in ("user", "project"):
        path = _scope_to_path(scope)
        if not path.exists():
            print(f"(empty) {path}")
            return 0
        print(f"# {path}")
        print(path.read_text(encoding="utf-8"))
        return 0
    if scope.startswith("agent:"):
        path = _scope_to_path(scope)
        if not path.exists():
            print(f"(empty) {path}")
            return 0
        print(f"# {path}")
        print(path.read_text(encoding="utf-8"))
        return 0
    # all: dump effective merge for the current cwd.
    try:
        merged = load_mcp_config(
            user_path=user_path,
            project_path=project_path,
            agent_path=None,
        )
    except Exception as e:
        print(f"error: {e}")
        return 1
    if not merged:
        print("(no MCP servers configured for this scope)")
        return 0
    for name, spec in merged.items():
        print(_format_spec(name, spec))
    return 0


def _format_spec(name: str, spec: ServerSpec) -> str:
    if spec.type == "stdio":
        cmd = spec.command or ""
        if spec.args:
            cmd = cmd + " " + " ".join(spec.args)
        extras = []
        if spec.env:
            extras.append("env=" + ",".join(spec.env.keys()))
        if spec.cwd:
            extras.append(f"cwd={spec.cwd}")
        if spec.auto_approve:
            extras.append("auto_approve")
        suffix = ("  [" + " ".join(extras) + "]") if extras else ""
        return f"  - {name}: stdio  {cmd}{suffix}"
    return f"  - {name}: http   {spec.url}"


def _cmd_add(args: argparse.Namespace) -> int:
    if bool(args.command) == bool(args.url):
        print("error: provide exactly one of --command or --url")
        return 2
    env: dict[str, str] = {}
    for kv in args.env or []:
        if "=" not in kv:
            print(f"error: --env requires KEY=VALUE, got {kv!r}")
            return 2
        k, _, v = kv.partition("=")
        env[k.strip()] = v
    entry: dict = {}
    if args.command:
        # Very common user mistake — quoting the entire command.
        parts = args.command.split()
        entry["command"] = parts[0]
        if len(parts) > 1:
            entry["args"] = parts[1:]
        if args.cwd:
            entry["cwd"] = args.cwd
    else:
        entry["type"] = "http"
        entry["url"] = args.url
    if env:
        entry["env"] = env
    if args.auto_approve:
        entry["auto_approve"] = True
    if args.disabled:
        entry["disabled"] = True

    path = _scope_to_path(args.scope)
    data = _read(path)
    data["mcpServers"][args.name] = entry
    _atomic_write_0600(path, data)
    print(f"wrote {path}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    path = _scope_to_path(args.scope)
    if not path.exists():
        print(f"(nothing to remove) {path}")
        return 0
    data = _read(path)
    if args.name in data["mcpServers"]:
        del data["mcpServers"][args.name]
        _atomic_write_0600(path, data)
        print(f"removed {args.name!r} from {path}")
    else:
        print(f"(not present) {args.name!r} in {path}")
    return 0
