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
from deep_worker_cli.lsp.gradle_init import GRADLE_INIT_SCRIPT_NAME


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


# --------------------------------------------------------------------------
# T8 — Gradle init script for Kotlin/Android (§7.1, §10 of the design)
#
# The script is installed BY DEFAULT on `lsp enable`: without it, cross-file
# navigation of Kotlin LSP on AGP projects is silently empty, and the user
# would not guess such a patch is needed. Opt-out — `--no-gradle-init`.
#
# The tests use GRADLE_USER_HOME (the standard Gradle variable) so they do
# not touch the developer's real ~/.gradle.
# --------------------------------------------------------------------------


def test_enable_installs_gradle_init_script_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"

    rc = handle_lsp_command(
        _parse(["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)])
    )

    assert rc == 0
    assert (gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME).exists()


def test_enable_respects_no_gradle_init_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"

    rc = handle_lsp_command(
        _parse(
            [
                "lsp",
                "enable",
                "--config",
                str(cfg),
                "--project-root",
                str(tmp_path),
                "--no-gradle-init",
            ]
        )
    )

    assert rc == 0
    assert not (gradle_home / "init.d").exists()


def test_enable_reports_global_scope_of_the_init_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The file affects ALL of the user's builds — this must not be hidden."""
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"

    handle_lsp_command(
        _parse(["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)])
    )

    out = capsys.readouterr().out
    assert str(gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME) in out


def test_disable_removes_gradle_init_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"
    target = gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME

    handle_lsp_command(
        _parse(["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)])
    )
    assert target.exists()

    assert handle_lsp_command(_parse(["lsp", "disable", "--config", str(cfg)])) == 0
    assert not target.exists()


def test_disable_keeps_foreign_init_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with the same name but without our marker is foreign."""
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    init_d = gradle_home / "init.d"
    init_d.mkdir(parents=True)
    target = init_d / GRADLE_INIT_SCRIPT_NAME
    target.write_text("// somebody else\n", encoding="utf-8")

    cfg = tmp_path / ".deepagents" / "mcp.json"
    handle_lsp_command(_parse(["lsp", "disable", "--config", str(cfg)]))

    assert target.read_text(encoding="utf-8") == "// somebody else\n"


def test_status_reports_gradle_init_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"

    handle_lsp_command(_parse(["lsp", "status", "--config", str(cfg)]))
    assert "gradle init" in capsys.readouterr().out.lower()

    handle_lsp_command(
        _parse(["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)])
    )
    capsys.readouterr()
    handle_lsp_command(_parse(["lsp", "status", "--config", str(cfg)]))
    out = capsys.readouterr().out
    assert str(gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME) in out


def test_enable_reports_outdated_script_instead_of_already_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reviewer note: "already installed" with differing content is a
    misleading message. The user must learn they have an OLD (or edited) copy
    and that it is fixed with --force."""
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"
    argv = ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)]

    handle_lsp_command(_parse(argv))
    target = gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME
    marker = target.read_text(encoding="utf-8").splitlines()[0]
    target.write_text(marker + "\n// stale v0 body\n", encoding="utf-8")
    capsys.readouterr()

    handle_lsp_command(_parse(argv))

    out = capsys.readouterr().out
    assert "already up to date" not in out, out
    assert "differs from the packaged version" in out, out
    assert "--force" in out, out
    # Without --force the content stays untouched.
    assert "// stale v0 body" in target.read_text(encoding="utf-8")


def test_enable_force_updates_outdated_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gradle_home = tmp_path / "gradle-home"
    monkeypatch.setenv("GRADLE_USER_HOME", str(gradle_home))
    cfg = tmp_path / ".deepagents" / "mcp.json"
    argv = ["lsp", "enable", "--config", str(cfg), "--project-root", str(tmp_path)]

    handle_lsp_command(_parse(argv))
    target = gradle_home / "init.d" / GRADLE_INIT_SCRIPT_NAME
    packaged = target.read_text(encoding="utf-8")
    target.write_text(packaged.splitlines()[0] + "\n// stale\n", encoding="utf-8")

    handle_lsp_command(_parse([*argv, "--force"]))

    assert target.read_text(encoding="utf-8") == packaged
