"""Unit tests for the ``deep-worker lsp enable|status`` CLI group (§10 decision #1).

No convenience flag is added for agent creation in v1 — only the group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli.lsp_command import (
    handle_lsp_command,
    setup_lsp_parser,
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_lsp_parser(subparsers)
    return parser.parse_args(argv)


def test_parser_registers_enable_and_status() -> None:
    assert _parse(["lsp", "enable"]).lsp_command == "enable"
    assert _parse(["lsp", "status"]).lsp_command == "status"


def test_parser_supports_scope_and_force() -> None:
    args = _parse(["lsp", "enable", "--scope", "project", "--force"])
    assert args.scope == "project"
    assert args.force is True


def test_enable_writes_config_to_explicit_path(tmp_path: Path) -> None:
    cfg = tmp_path / ".deepagents" / "mcp.json"
    args = _parse(
        ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)]
    )

    rc = handle_lsp_command(args)

    assert rc == 0
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "serena" in data["mcpServers"]


def test_enable_is_idempotent_via_cli(tmp_path: Path) -> None:
    cfg = tmp_path / ".deepagents" / "mcp.json"
    argv = ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)]

    assert handle_lsp_command(_parse(argv)) == 0
    assert handle_lsp_command(_parse(argv)) == 0  # repeat — not an error


def test_status_reports_disabled_then_enabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / ".deepagents" / "mcp.json"
    status_argv = ["lsp", "status", "--config", str(cfg)]

    handle_lsp_command(_parse(status_argv))
    assert "disabled" in capsys.readouterr().out.lower()

    handle_lsp_command(
        _parse(["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)])
    )
    handle_lsp_command(_parse(status_argv))
    assert "enabled" in capsys.readouterr().out.lower()


def test_status_reports_serena_binary_presence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-blocking reviewer request: status must check the `serena` binary."""
    cfg = tmp_path / ".deepagents" / "mcp.json"
    handle_lsp_command(_parse(["lsp", "status", "--config", str(cfg)]))
    out = capsys.readouterr().out.lower()
    assert "serena" in out
    # We explicitly report whether the binary was found or not — without
    # this, the user would not understand why the tools did not appear.
    assert ("not found" in out) or ("found" in out)


def test_unknown_subcommand_returns_usage_code() -> None:
    args = argparse.Namespace(lsp_command=None)
    assert handle_lsp_command(args) == 2
