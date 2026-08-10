"""Unit tests for migrating legacy per-agent Serena entries to the bridge.

DESIGN §15 commit 4: every existing ``mcp.json`` that execs the backend
binary directly (one heavy process per agent — the exact shape the
policy forbids) must be rewritten to launch the thin bridge instead.

Pinned behaviour:

* a legacy entry is rewritten in place and a ``.bak`` is left behind;
* an already-migrated entry is a no-op (idempotent) and writes no
  second ``.bak``;
* unrelated MCP servers in the same file are preserved byte-for-byte;
* a file with no serena entry at all is untouched;
* file/dir permissions stay 0600/0700 (config may hold secrets).
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

from deep_worker_cli.lsp.migrate import migrate_mcp_config

_LEGACY_ARGS: list[str] = [
    "start-mcp-server",
    "--context",
    "ide",
    "--transport",
    "stdio",
    "--project",
    "/tmp/proj",
]

_LEGACY: dict[str, object] = {
    "type": "stdio",
    "command": "serena",
    "args": list(_LEGACY_ARGS),
    "auto_approve": True,
}


def _write(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps({"mcpServers": servers}, indent=2))
    path.chmod(0o600)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())["mcpServers"]


def test_legacy_entry_is_rewritten_to_bridge(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    _write(cfg, {"serena": dict(_LEGACY)})

    assert migrate_mcp_config(cfg) is True

    entry = _read(cfg)["serena"]
    joined = " ".join([entry["command"], *entry["args"]])
    assert "bridge" in joined
    assert "start-mcp-server" not in joined
    assert entry["command"] != "serena"


def test_migration_leaves_a_backup(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    _write(cfg, {"serena": dict(_LEGACY)})
    migrate_mcp_config(cfg)

    bak = cfg.with_suffix(cfg.suffix + ".bak")
    assert bak.exists()
    assert json.loads(bak.read_text())["mcpServers"]["serena"] == _LEGACY


def test_migration_is_idempotent(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    _write(cfg, {"serena": dict(_LEGACY)})

    assert migrate_mcp_config(cfg) is True
    after_first = _read(cfg)
    bak = cfg.with_suffix(cfg.suffix + ".bak")

    assert migrate_mcp_config(cfg) is False
    assert _read(cfg) == after_first
    # The backup must still hold the ORIGINAL pre-migration entry: if a
    # later run overwrote it with the already-migrated file, the true
    # rollback target would be lost forever.
    assert json.loads(bak.read_text())["mcpServers"]["serena"] == _LEGACY


def test_backup_is_never_overwritten_by_a_later_migration(tmp_path: Path):
    """The .bak must always be the FIRST pre-migration state.

    A user can re-add a legacy entry by hand (or an older tool can
    rewrite it). Migrating again must not clobber the original backup
    with this newer, less-original content — otherwise the true
    rollback target is silently lost.
    """
    cfg = tmp_path / ".deepagents" / "mcp.json"
    _write(cfg, {"serena": dict(_LEGACY), "keepme": {"command": "x", "args": []}})
    assert migrate_mcp_config(cfg) is True

    bak = cfg.with_suffix(cfg.suffix + ".bak")
    original_backup = bak.read_text()

    # A legacy entry reappears with DIFFERENT content.
    second = dict(_LEGACY)
    second["args"] = [*_LEGACY_ARGS[:-1], "/tmp/other-proj"]
    _write(cfg, {"serena": second})
    assert migrate_mcp_config(cfg) is True

    assert bak.read_text() == original_backup
    assert "keepme" in json.loads(bak.read_text())["mcpServers"]


def test_other_servers_are_preserved(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    other = {"type": "stdio", "command": "some-other", "args": ["--x"]}
    _write(cfg, {"serena": dict(_LEGACY), "other": other})

    migrate_mcp_config(cfg)

    servers = _read(cfg)
    assert servers["other"] == other


def test_config_without_serena_is_untouched(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    other = {"type": "stdio", "command": "some-other", "args": []}
    _write(cfg, {"other": other})
    before = cfg.read_text()

    assert migrate_mcp_config(cfg) is False
    assert cfg.read_text() == before
    assert not cfg.with_suffix(cfg.suffix + ".bak").exists()


def test_missing_config_is_a_noop(tmp_path: Path):
    assert migrate_mcp_config(tmp_path / "nope" / "mcp.json") is False


def test_permissions_are_preserved(tmp_path: Path):
    cfg = tmp_path / ".deepagents" / "mcp.json"
    _write(cfg, {"serena": dict(_LEGACY)})

    migrate_mcp_config(cfg)

    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.parent.stat().st_mode) == 0o700
    bak = cfg.with_suffix(cfg.suffix + ".bak")
    assert stat.S_IMODE(bak.stat().st_mode) == 0o600
