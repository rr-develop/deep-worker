"""Migrate legacy per-agent Serena ``mcp.json`` entries to the bridge.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §15 commit 4 and
``docs/POLICY_NO_INPROCESS_HEAVY_MCP.md``.

The legacy entry made every agent exec the Serena binary itself, so N
agents on one project meant N Serena processes, N language servers and
N project models. This module rewrites such an entry so the agent
launches the thin bridge instead, and all agents share one service.

Safety properties:

* the original file is copied to ``<name>.bak`` before the first
  rewrite (and never overwritten by a later no-op run, so the very
  first pre-migration state stays recoverable);
* other MCP servers in the same file are preserved verbatim;
* the rewrite is atomic (tmp + ``os.replace``) and keeps 0600/0700, as
  the config may contain ``${env:...}`` secret references.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

from deep_worker_cli.lsp.bridge import BRIDGE_SERVER_NAME, build_bridge_mcp_entry

_CONFIG_MODE: Final = 0o600
_DIR_MODE: Final = 0o700
#: Marker of the legacy shape: the backend CLI's own subcommand.
_LEGACY_VERB: Final = "start-mcp-server"


def is_legacy_entry(entry: object) -> bool:
    """True if ``entry`` execs the backend binary directly."""
    if not isinstance(entry, dict):
        return False
    argv: list[object] = [entry.get("command")]
    args: object = entry.get("args")
    if isinstance(args, list):
        argv.extend(args)
    return any(str(a) == _LEGACY_VERB for a in argv)


def _project_root_of(entry: dict[str, Any], fallback: Path) -> Path:
    """Recover ``--project <root>`` from the legacy argv."""
    args = [str(a) for a in (entry.get("args") or [])]
    if "--project" in args:
        i = args.index("--project")
        if i + 1 < len(args):
            return Path(args[i + 1])
    return fallback


def migrate_mcp_config(
    path: Path,
    *,
    workspace_root: Path | None = None,
    follow_agent_cwd: bool = False,
) -> bool:
    """Rewrite a legacy Serena entry in ``path`` to launch the bridge.

    Returns ``True`` if the file was modified, ``False`` if there was
    nothing to do (missing file, no serena entry, already migrated).
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False

    wrapped = "mcpServers" in raw
    servers = raw.get("mcpServers") if wrapped else raw
    if not isinstance(servers, dict):
        return False

    entry = servers.get(BRIDGE_SERVER_NAME)
    if not is_legacy_entry(entry):
        return False
    assert isinstance(entry, dict)  # narrowed by is_legacy_entry

    # USER scope must not inherit the legacy literal: migrating a
    # ``--project /some/repo`` entry at user scope would recreate the
    # exact defect this placeholder exists to remove (one project
    # becoming every agent's navigation target). The no-``--project``
    # legacy form is worse still — its fallback is ``path.parent``,
    # i.e. ``~/.deep_worker`` for the user layer.
    servers[BRIDGE_SERVER_NAME] = build_bridge_mcp_entry(
        workspace_root=(
            None
            if follow_agent_cwd
            else _project_root_of(entry, workspace_root or path.parent)
        ),
        auto_approve=bool(entry.get("auto_approve", True)),
        follow_agent_cwd=follow_agent_cwd,
    )

    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        bak.chmod(_CONFIG_MODE)

    _atomic_write(path, raw if wrapped else servers)
    return True


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    path.parent.chmod(_DIR_MODE)
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _CONFIG_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        tmp.chmod(_CONFIG_MODE)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
