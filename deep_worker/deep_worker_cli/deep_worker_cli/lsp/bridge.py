"""Client-side bridge: per-agent stdio MCP front-end for the shared
per-project LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §6 (bridge) and
``docs/POLICY_NO_INPROCESS_HEAVY_MCP.md``.

Why this module exists
----------------------
Agents speak MCP over stdio. The heavy backend (Serena + a language
server + its project model) must be shared by ALL agents working on the
same project — one process per project, not one per agent. The bridge is
the cheap half of that split: it starts nothing heavy, holds no project
model, and forwards every tool call to the shared service over gRPC.

Hard invariants (pinned by ``tests/unit_tests/lsp/test_bridge_entry.py``)
------------------------------------------------------------------------
* **The bridge never spawns the backend.** Only the allowlisted package
  ``deep_worker_cli.lsp.service*`` may. The bridge asks the supervisor
  to ``ensure_started`` and the supervisor execs the *service* CLI —
  the backend is spawned by that service process, never here.
* **No stdio fallback.** If the shared service cannot be reached the
  bridge raises :class:`BridgeUnavailable` with an actionable message.
  Silently degrading to an in-process backend is exactly the design the
  policy forbids, so it does not exist even as a hidden flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from deep_worker_cli.external_mcp.config import AGENT_WORKING_DIR_PLACEHOLDER
from deep_worker_cli.lsp.service.registry import RegistryEntry, ServiceRegistry
from deep_worker_cli.lsp.service.service_cli import build_serve_argv
from deep_worker_cli.lsp.service.supervisor import (
    ServiceStartTimeout,
    ServiceSupervisor,
)

#: Name of the MCP server as the agent sees it. Kept identical to the
#: legacy value so tool names the model already knows
#: (``serena__find_symbol``) stay stable across the migration.
BRIDGE_SERVER_NAME: Final = "serena"

_DW_COMMAND: Final = "deep-worker"


class BridgeUnavailable(RuntimeError):
    """The shared LSP service could not be reached or started.

    Deliberately terminal: there is no degraded mode to fall back to.
    """


def connect_or_fail(
    project_root: Path,
    *,
    registry_dir: Path | None = None,
) -> RegistryEntry:
    """Ensure the shared service for ``project_root`` is live.

    Returns the registry entry describing where to reach it. Raises
    :class:`BridgeUnavailable` if it cannot be started — the caller is
    expected to exit non-zero so the agent sees a clear failure instead
    of a silently tool-less session.
    """
    root = Path(project_root).resolve()
    registry = ServiceRegistry(base_dir=registry_dir)
    supervisor = ServiceSupervisor(
        registry, lambda slug_dir: build_serve_argv(root, slug_dir)
    )
    try:
        return supervisor.ensure_started(root)
    except ServiceStartTimeout as e:
        raise BridgeUnavailable(
            f"shared LSP service for {root} is not available: {e}\n"
            f"Start it manually to see why:\n"
            f"    {_DW_COMMAND} lsp serve --project-root {root}\n"
            f"and inspect its state with:\n"
            f"    {_DW_COMMAND} lsp service-status --project-root {root}"
        ) from e


def build_bridge_mcp_entry(
    *,
    workspace_root: Path | str | None = None,
    auto_approve: bool = True,
    dw_command: str | None = None,
    follow_agent_cwd: bool = False,
) -> dict[str, Any]:
    """Build the ``mcpServers.<BRIDGE_SERVER_NAME>`` entry for the bridge.

    Replaces the legacy entry that exec'd the backend binary directly
    inside every agent. Shape stays compatible with the existing loader
    in ``external_mcp/config.py``.

    With ``follow_agent_cwd`` the root is written as the placeholder
    ``${agent:working_dir}`` instead of a literal path, and the loader
    substitutes each agent's own workspace root at start. That is the
    USER-scope form: a config shared by every agent must not name one
    project, or it silently becomes everybody's navigation target (see
    ``docs/design/lsp_user_scope_follows_agent_cwd.md``).

    Project- and agent-scope entries keep the literal root — there the
    whole point is to name one specific project.
    """
    if follow_agent_cwd:
        root_arg = AGENT_WORKING_DIR_PLACEHOLDER
    elif workspace_root is not None:
        root_arg = str(Path(workspace_root).resolve())
    else:
        raise ValueError(
            "build_bridge_mcp_entry requires workspace_root unless "
            "follow_agent_cwd is set"
        )
    return {
        "type": "stdio",
        "command": dw_command or _DW_COMMAND,
        "args": ["lsp", "bridge", "--project-root", root_arg],
        "auto_approve": bool(auto_approve),
    }
