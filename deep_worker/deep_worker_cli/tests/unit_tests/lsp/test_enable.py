"""Unit tests for enabling the LSP option (writing the mcp.json layer).

Contract — see lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md §4.2:
* merge into an existing mcp.json; other entries must not be lost;
* idempotent (a repeated call neither duplicates nor overwrites);
* permissions: file 0o600, directory 0o700 (policy: no 777/666);
* atomic write.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

from deep_worker_cli.external_mcp.config import load_mcp_config
from deep_worker_cli.lsp.enable import (
    SERENA_SERVER_NAME,
    disable_lsp_at_path,
    enable_lsp_at_path,
    is_lsp_enabled_at_path,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_creates_config_with_serena_entry(tmp_path: Path) -> None:
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    data = _read(cfg)
    assert SERENA_SERVER_NAME in data["mcpServers"]
    # §14: the entry launches the thin bridge, not the backend binary —
    # one shared service per project instead of one per agent.
    entry = data["mcpServers"][SERENA_SERVER_NAME]
    assert entry["command"] == "deep-worker"
    assert entry["args"][:2] == ["lsp", "bridge"]


def test_file_mode_is_0600_and_dir_0700(tmp_path: Path) -> None:
    """Anti-regression against 666/777 (project policy + daemon umask incident)."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.parent.stat().st_mode) == 0o700


def test_preserves_other_servers(tmp_path: Path) -> None:
    """Other MCP servers must survive enabling LSP."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx", "args": ["gh"]}}}),
        encoding="utf-8",
    )

    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    servers = _read(cfg)["mcpServers"]
    assert "github" in servers
    assert servers["github"]["command"] == "npx"
    assert SERENA_SERVER_NAME in servers


def test_is_idempotent_and_does_not_clobber(tmp_path: Path) -> None:
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)
    # The user manually tweaked the entry:
    data = _read(cfg)
    data["mcpServers"][SERENA_SERVER_NAME]["args"].append("--log-level=DEBUG")
    cfg.write_text(json.dumps(data), encoding="utf-8")

    changed = enable_lsp_at_path(cfg, workspace_root=tmp_path)

    assert changed is False, "re-enabling must not change anything"
    assert "--log-level=DEBUG" in _read(cfg)["mcpServers"][SERENA_SERVER_NAME]["args"]


def test_force_overwrites_existing_entry(tmp_path: Path) -> None:
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)
    data = _read(cfg)
    data["mcpServers"][SERENA_SERVER_NAME]["args"] = ["bogus"]
    cfg.write_text(json.dumps(data), encoding="utf-8")

    changed = enable_lsp_at_path(cfg, workspace_root=tmp_path, force=True)

    assert changed is True
    assert _read(cfg)["mcpServers"][SERENA_SERVER_NAME]["args"][:2] == [
        "lsp",
        "bridge",
    ]


def test_flat_cursor_style_config_is_migrated_not_lost(tmp_path: Path) -> None:
    """A flat root (cursor-style) is supported by the loader — when
    merging we must not lose existing entries."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"github": {"command": "npx"}}), encoding="utf-8")

    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    merged = load_mcp_config(user_path=cfg, project_path=None, agent_path=None)
    assert "github" in merged
    assert SERENA_SERVER_NAME in merged


def test_enabled_predicate_and_disable_roundtrip(tmp_path: Path) -> None:
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    assert is_lsp_enabled_at_path(cfg) is False

    enable_lsp_at_path(cfg, workspace_root=tmp_path)
    assert is_lsp_enabled_at_path(cfg) is True

    assert disable_lsp_at_path(cfg) is True
    assert is_lsp_enabled_at_path(cfg) is False
    # A repeated disable is a no-op and must not fail.
    assert disable_lsp_at_path(cfg) is False


def test_disable_keeps_other_servers(tmp_path: Path) -> None:
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx"}}}), encoding="utf-8"
    )
    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    disable_lsp_at_path(cfg)

    assert "github" in _read(cfg)["mcpServers"]


def test_no_temp_file_left_behind(tmp_path: Path) -> None:
    """Atomic writes must not leave .tmp litter behind."""
    cfg = tmp_path / ".deep_worker" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    leftovers = [p.name for p in cfg.parent.iterdir() if p.name != "mcp.json"]
    assert leftovers == []
