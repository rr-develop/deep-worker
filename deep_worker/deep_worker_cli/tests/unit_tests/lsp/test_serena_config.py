"""Unit tests for the mcp.json entry generator for Serena.

Contract — see lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md §4.1.

The key "lock" here is test_generated_entry_parses_via_existing_loader:
the generated entry must be understood by the EXISTING loader in
external_mcp/config.py without a single change to it. If someone changes
the entry shape or the loader, the test will show the divergence at once.
"""
from __future__ import annotations

import json
from pathlib import Path

from deep_worker_cli.external_mcp.config import ServerSpec, load_mcp_config
from deep_worker_cli.lsp.serena_config import (
    SERENA_COMMAND,
    SERENA_SERVER_NAME,
    build_serena_mcp_entry,
)


def test_entry_has_stdio_shape(tmp_path: Path) -> None:
    entry = build_serena_mcp_entry(workspace_root=tmp_path)
    assert entry["type"] == "stdio"
    assert entry["command"] == SERENA_COMMAND == "serena"
    assert isinstance(entry["args"], list)
    assert all(isinstance(a, str) for a in entry["args"])


def test_entry_uses_start_mcp_server_with_ide_context(tmp_path: Path) -> None:
    args = build_serena_mcp_entry(workspace_root=tmp_path)["args"]
    assert args[0] == "start-mcp-server"
    # --context ide — the upstream-recommended context for external
    # agent harnesses (see design §4.1).
    assert "--context" in args
    assert args[args.index("--context") + 1] == "ide"
    assert "--transport" in args
    assert args[args.index("--transport") + 1] == "stdio"


def test_project_arg_is_absolute_resolved_path(tmp_path: Path) -> None:
    """`--project` must be ABSOLUTE: the subprocess cwd is not guaranteed."""
    nested = tmp_path / "sub"
    nested.mkdir()
    relative_ish = nested / ".." / "sub"
    args = build_serena_mcp_entry(workspace_root=relative_ish)["args"]
    project = args[args.index("--project") + 1]
    assert Path(project).is_absolute()
    assert ".." not in project
    assert Path(project) == nested.resolve()


def test_auto_approve_defaults_true(tmp_path: Path) -> None:
    """User decision #2: write tools go under auto_approve."""
    assert build_serena_mcp_entry(workspace_root=tmp_path)["auto_approve"] is True


def test_auto_approve_can_be_disabled(tmp_path: Path) -> None:
    entry = build_serena_mcp_entry(workspace_root=tmp_path, auto_approve=False)
    assert entry["auto_approve"] is False


def test_custom_command_override(tmp_path: Path) -> None:
    """Serena may not be on PATH — we allow an explicit path to the binary."""
    entry = build_serena_mcp_entry(
        workspace_root=tmp_path, serena_command="/opt/bin/serena"
    )
    assert entry["command"] == "/opt/bin/serena"


def test_generated_entry_parses_via_existing_loader(tmp_path: Path) -> None:
    """CONTRACT LOCK: our generator ↔ the existing load_mcp_config.

    Nothing in external_mcp/config.py needs to change — we verify this
    honestly by running the entry through the real loader.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {"mcpServers": {SERENA_SERVER_NAME: build_serena_mcp_entry(
                workspace_root=workspace
            )}}
        ),
        encoding="utf-8",
    )

    merged = load_mcp_config(user_path=cfg, project_path=None, agent_path=None)

    assert SERENA_SERVER_NAME in merged
    spec = merged[SERENA_SERVER_NAME]
    assert isinstance(spec, ServerSpec)
    assert spec.type == "stdio"
    assert spec.command == "serena"
    assert spec.auto_approve is True
    assert str(workspace.resolve()) in spec.args


def test_entry_layers_the_deep_worker_serena_mode(tmp_path: Path) -> None:
    """Commit 11 refresh-tool fix: without ``--add-mode`` pointing at
    our shipped YAML, Serena 1.6.x hides ``restart_language_server``
    and ``SerenaBackend.refresh`` has nothing to call — a defect the
    multi-agent stress test caught live.

    Pins two things:

    * ``args`` contains ``--add-mode`` immediately followed by an
      absolute path.
    * That path exists on disk and its YAML actually opts
      ``restart_language_server`` into ``included_optional_tools``.
    """
    from deep_worker_cli.lsp.resources import SERENA_MODE_YML

    args = build_serena_mcp_entry(workspace_root=tmp_path)["args"]
    assert "--add-mode" in args, (
        "regression: commit-11 refresh fix — build_serena_mcp_entry "
        "must layer the deep-worker mode YAML so "
        "restart_language_server is advertised"
    )
    mode_arg = args[args.index("--add-mode") + 1]
    mode_path = Path(mode_arg)
    assert mode_path.is_absolute()
    assert mode_path == SERENA_MODE_YML
    assert mode_path.is_file(), f"shipped mode YAML missing on disk: {mode_path}"
    text = mode_path.read_text(encoding="utf-8")
    assert "included_optional_tools" in text
    assert "restart_language_server" in text


def test_refresh_tool_candidates_prefer_restart_language_server() -> None:
    """Commit 11 refresh-tool fix: ``restart_language_server`` must
    come FIRST in the pick order — it's the only candidate Serena
    1.6.x actually advertises, and picking it over legacy names is
    what makes ``SerenaBackend.refresh`` work under load.
    """
    from deep_worker_cli.lsp.service.serena_backend import (
        _REFRESH_TOOL_CANDIDATES,
        pick_refresh_tool,
    )

    assert _REFRESH_TOOL_CANDIDATES[0] == "restart_language_server"
    # Sanity: when Serena advertises the full 1.6.x set (find_symbol
    # etc. + restart_language_server), pick() returns exactly that.
    advertised = [
        "find_symbol",
        "get_symbols_overview",
        "restart_language_server",
        "activate_project",  # legacy, must NOT win
    ]
    assert pick_refresh_tool(advertised) == "restart_language_server"
