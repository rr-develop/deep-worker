"""CLI layer for managing languages: ``deep-worker lsp languages``.

See ``docs/DESIGN_LSP_LANGUAGES.md`` §5 and amendments A4/A5/A7.

Here the command behavior is pinned, not the YAML parsing (that lives in
``test_lsp_languages.py``): flag application order, refusal on conflict,
strict read-only semantics without flags, and the fact that a repeated
``enable`` really reconciles languages — without this, auto-detection would
never work for already-enabled projects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from deep_worker_cli.cli.lsp_command import handle_lsp_command, setup_lsp_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_lsp_parser(sub)
    return parser.parse_args(argv)


_CONFIG = """\
project_name: "demo"

# languages the servers are started for
languages:
- kotlin

encoding: "utf-8"
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".serena").mkdir(mode=0o700)
    cfg = tmp_path / ".serena" / "project.yml"
    cfg.write_text(_CONFIG)
    cfg.chmod(0o600)
    return tmp_path


@pytest.fixture
def fake_facts(monkeypatch: pytest.MonkeyPatch):
    """Stub out the Serena probe — unit tests must not depend on the install."""
    import deep_worker_cli.cli.lsp_command as cli
    import deep_worker_cli.lsp.languages as mod

    state = {
        "composition": {"kotlin": 65.0, "python": 32.0, "bash": 2.0},
        "known": ["kotlin", "python", "go", "bash"],
        "priority": {"kotlin": 1, "python": 1, "bash": 1},
    }

    def _facts(_root):
        return state

    monkeypatch.setattr(mod, "language_facts", _facts)
    monkeypatch.setattr(cli, "_workspace_root", lambda args: Path(args.project_root))
    return state


def test_languages_subcommand_registered() -> None:
    args = _parse(["lsp", "languages", "--add", "python"])
    assert args.lsp_command == "languages"
    assert args.add == ["python"]


def test_detect_adds_missing_language(project: Path, fake_facts, capsys) -> None:
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project), "--detect"])
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    from deep_worker_cli.lsp.languages import read_languages

    assert read_languages(project / ".serena" / "project.yml") == [
        "kotlin",
        "python",
    ]
    # A7: the restart hint is printed ONLY on a real change.
    assert "deep-worker lsp stop" in out


def test_no_restart_hint_on_noop(project: Path, fake_facts, capsys) -> None:
    """A7: a spurious hint would restart the service for nothing."""
    handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project), "--detect"])
    )
    capsys.readouterr()
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project), "--detect"])
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "already up to date" in out
    assert "deep-worker lsp stop" not in out


def test_add_and_remove_same_language_conflicts(
    project: Path, fake_facts, capsys
) -> None:
    """A5: a hidden priority would hide the user's typo."""
    rc = handle_lsp_command(
        _parse(
            [
                "lsp", "languages", "--project-root", str(project),
                "--add", "go", "--remove", "go",
            ]
        )
    )
    assert rc == 1
    assert "both --add and --remove" in capsys.readouterr().out


def test_unknown_language_rejected(project: Path, fake_facts, capsys) -> None:
    before = (project / ".serena" / "project.yml").read_bytes()
    rc = handle_lsp_command(
        _parse(
            ["lsp", "languages", "--project-root", str(project), "--add", "pyton"]
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "unknown language" in out
    assert (project / ".serena" / "project.yml").read_bytes() == before


def test_remove_last_language_refused(project: Path, fake_facts, capsys) -> None:
    before = (project / ".serena" / "project.yml").read_bytes()
    rc = handle_lsp_command(
        _parse(
            [
                "lsp", "languages", "--project-root", str(project),
                "--remove", "kotlin",
            ]
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "no languages" in out
    assert (project / ".serena" / "project.yml").read_bytes() == before


def test_flag_order_detect_then_add_then_remove(
    project: Path, fake_facts, capsys
) -> None:
    """A5: the order is fixed — detect, then add, then remove."""
    rc = handle_lsp_command(
        _parse(
            [
                "lsp", "languages", "--project-root", str(project),
                "--detect", "--add", "go", "--remove", "python",
            ]
        )
    )
    assert rc == 0, capsys.readouterr().out
    from deep_worker_cli.lsp.languages import read_languages

    # detect gave kotlin+python, add added go, remove dropped python.
    assert read_languages(project / ".serena" / "project.yml") == ["kotlin", "go"]


def test_no_flags_is_read_only(project: Path, fake_facts, capsys) -> None:
    """A5: without flags the command changes nothing."""
    cfg = project / ".serena" / "project.yml"
    before = cfg.read_bytes()
    before_mtime = cfg.stat().st_mtime_ns
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project)])
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert cfg.read_bytes() == before
    assert cfg.stat().st_mtime_ns == before_mtime
    assert "% of recognized source files" in out, out
    assert "above threshold but NOT configured: python" in out, out


def test_no_flags_does_not_create_config(tmp_path: Path, fake_facts, capsys) -> None:
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(tmp_path)])
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert not (tmp_path / ".serena" / "project.yml").exists()
    assert "not configured" in out


def test_explicit_detect_fails_loud_without_serena(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A4: an explicit --detect must fail loudly, not report 'success with no changes'."""
    import deep_worker_cli.cli.lsp_command as cli
    import deep_worker_cli.lsp.languages as mod

    def _boom(_root):
        raise mod.LanguageConfigError("serena is not installed")

    monkeypatch.setattr(mod, "language_facts", _boom)
    monkeypatch.setattr(cli, "_workspace_root", lambda args: Path(args.project_root))
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project), "--detect"])
    )
    assert rc == 1
    assert "serena is not installed" in capsys.readouterr().out


def test_empty_composition_detect_exits_one(
    project: Path, fake_facts, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A6: the contract of an empty composition for an explicit --detect."""
    import deep_worker_cli.lsp.languages as mod

    monkeypatch.setattr(
        mod, "language_facts", lambda _r: {"composition": {}, "known": [], "priority": {}}
    )
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(project), "--detect"])
    )
    assert rc == 1
    assert "no supported source files" in capsys.readouterr().out


# ── A4: repeated enable reconciles languages ───────────────────────────────


def test_repeat_enable_reconciles_languages(
    project: Path, fake_facts, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A4: reconciliation runs ALWAYS, not only when mcp.json changed.

    That is exactly how Python fell out of a project's languages: the project
    had long been enabled, ``enable_lsp_at_path`` returns a no-op, and if
    detection were tied to that flag, auto-detection would never work for
    existing installs.
    """
    import deep_worker_cli.cli.lsp_command as cli

    # enable is a no-op: the MCP entry is already in place.
    monkeypatch.setattr(cli, "enable_lsp_at_path", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_install_gradle_init", lambda **k: None)
    monkeypatch.setattr(cli, "_warn_if_serena_missing", lambda: True)
    monkeypatch.setattr(cli, "_config_path", lambda args: tmp_path / "mcp.json")

    from deep_worker_cli.lsp.languages import read_languages

    assert read_languages(project / ".serena" / "project.yml") == ["kotlin"]
    rc = handle_lsp_command(
        _parse(["lsp", "enable", "--project-root", str(project)])
    )
    assert rc == 0
    assert read_languages(project / ".serena" / "project.yml") == [
        "kotlin",
        "python",
    ], "repeated enable must add the missing language"


def test_enable_survives_detection_failure(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """A4: during enable, a detection failure is a warning, not a failure.

    Languages are not the main goal of the command; failing LSP enablement
    because of them would be worse than enabling without them.
    """
    import deep_worker_cli.cli.lsp_command as cli
    import deep_worker_cli.lsp.languages as mod

    def _boom(_root):
        raise mod.LanguageConfigError("serena is not installed")

    monkeypatch.setattr(mod, "language_facts", _boom)
    monkeypatch.setattr(cli, "_workspace_root", lambda args: Path(args.project_root))
    monkeypatch.setattr(cli, "enable_lsp_at_path", lambda *a, **k: True)
    monkeypatch.setattr(cli, "_install_gradle_init", lambda **k: None)
    monkeypatch.setattr(cli, "_warn_if_serena_missing", lambda: True)
    monkeypatch.setattr(cli, "_config_path", lambda args: tmp_path / "mcp.json")

    rc = handle_lsp_command(
        _parse(["lsp", "enable", "--project-root", str(project)])
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "warning: language detection skipped" in out
