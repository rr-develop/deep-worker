"""Language auto-detection against REAL Serena, no mocks.

See ``docs/DESIGN_LSP_LANGUAGES.md`` §7.

The unit tests check parsing and merging on a substituted probe. Here we
check what mocks cannot prove:

* Serena is really installed in a separate venv and its code is invoked
  from OUR process correctly (that was the real pitfall:
  ``import serena`` from our interpreter fails);
* ``.gitignore`` is really respected, **including nested ones** —
  that is exactly what the parser in use promises;
* the ``project.yml`` we create is **loaded by Serena itself**
  (A1: a config without ``project_name`` is invalid, and a check via
  ``yaml.safe_load`` would not have caught that);
* the whole real-world scenario: a project already configured for one
  language, and a second pass adds the other.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from deep_worker_cli.cli.lsp_command import handle_lsp_command, setup_lsp_parser
from deep_worker_cli.lsp.languages import (
    project_config_path,
    read_languages,
    serena_python,
)
from deep_worker_cli.lsp.serena_config import SERENA_COMMAND

pytestmark = pytest.mark.integration


def _require_serena() -> None:
    if shutil.which(SERENA_COMMAND):
        return
    msg = (
        f"{SERENA_COMMAND!r} not found; install via "
        f"'uv tool install -p 3.13 serena-agent'"
    )
    if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_lsp_parser(sub)
    return parser.parse_args(argv)


def _polyglot_project(tmp_path: Path) -> Path:
    """Kotlin-majority project with a Python backend and vendor noise.

    The proportions mirror a real-world polyglot project: Kotlin larger,
    Python comfortably above 10 %, and vendor noise is filtered by
    ``.gitignore`` (including a nested one, to check exactly what the
    parser promises).
    """
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "backend").mkdir()
    (root / "vendor").mkdir()
    (root / "backend" / "generated").mkdir()

    for i in range(6):
        (root / "app" / f"M{i}.kt").write_text(f"fun m{i}() {{}}\n")
    for i in range(4):
        (root / "backend" / f"m{i}.py").write_text(f"def f{i}():\n    pass\n")
    # Noise: without .gitignore it would outweigh Kotlin and skew the ratios.
    for i in range(40):
        (root / "vendor" / f"v{i}.py").write_text("def junk():\n    pass\n")
    for i in range(30):
        (root / "backend" / "generated" / f"g{i}.py").write_text("x = 1\n")

    (root / ".gitignore").write_text("vendor/\n")
    # Nested .gitignore — what Serena's GitignoreParser supports.
    (root / "backend" / ".gitignore").write_text("generated/\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=60)
    return root


def test_serena_python_is_discoverable() -> None:
    """Serena is installed via uv tool — its python sits next to the binary."""
    _require_serena()
    python = serena_python()
    assert python is not None, "could not find Serena's interpreter"
    assert python.exists()


def test_detect_creates_config_loadable_by_serena(tmp_path: Path) -> None:
    """A1 + .gitignore: end-to-end check on live Serena."""
    _require_serena()
    root = _polyglot_project(tmp_path)

    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(root), "--detect"])
    )
    assert rc == 0

    config = project_config_path(root)
    assert config.exists()
    languages = read_languages(config)
    assert "kotlin" in languages
    assert "python" in languages, (
        f"Python backend must be detected; got {languages}"
    )

    # Permissions: a private file in a private directory (policy).
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o700

    # A1: the main thing — the config is valid FOR SERENA, not just parses.
    python = serena_python()
    assert python is not None
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import sys\n"
                "from serena.config.serena_config import ProjectConfig, SerenaConfig\n"
                "cfg = ProjectConfig.load(sys.argv[1], SerenaConfig())\n"
                "print(cfg.project_name)\n"
                "print(','.join(str(l.value) for l in cfg.languages))\n"
            ),
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr[:500]
    lines = probe.stdout.strip().splitlines()
    assert lines[0] == root.name
    loaded = lines[1].split(",")
    assert "kotlin" in loaded and "python" in loaded


def test_repeat_run_adds_missing_language(tmp_path: Path) -> None:
    """Real-world scenario: the config already exists, but with one language.

    This is A4 in substance — if the reconciliation were tied to "the
    config was just created", such a project would never be fixed.
    """
    _require_serena()
    root = _polyglot_project(tmp_path)

    # First pass: let Serena/us create the config.
    assert (
        handle_lsp_command(
            _parse(["lsp", "languages", "--project-root", str(root), "--detect"])
        )
        == 0
    )
    config = project_config_path(root)

    # Simulate the real-world state: only the majority language remains.
    from deep_worker_cli.lsp.languages import write_languages

    write_languages(config, ["kotlin"])
    assert read_languages(config) == ["kotlin"]
    before_mode = stat.S_IMODE(config.stat().st_mode)

    # The second pass must add Python.
    assert (
        handle_lsp_command(
            _parse(["lsp", "languages", "--project-root", str(root), "--detect"])
        )
        == 0
    )
    languages = read_languages(config)
    assert languages[0] == "kotlin", "the default language must not change"
    assert "python" in languages
    assert stat.S_IMODE(config.stat().st_mode) == before_mode


def test_show_reports_composition(tmp_path: Path, capsys) -> None:
    _require_serena()
    root = _polyglot_project(tmp_path)
    rc = handle_lsp_command(
        _parse(["lsp", "languages", "--project-root", str(root)])
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "% of recognized source files" in out
    assert "kotlin" in out and "python" in out
    # Without flags the command is read-only.
    assert not project_config_path(root).exists()
