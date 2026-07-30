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
