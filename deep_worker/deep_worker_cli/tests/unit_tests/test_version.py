"""Tests for version-related functionality."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from deep_worker_cli._version import __version__


def _subprocess_env() -> dict[str, str]:
    """Return an env for subprocess that keeps local `deepagents`
    importable — mirrors conftest.py's sys.path injection.
    Without this, `python -m deep_worker_cli.main` in a
    child process fails with `ModuleNotFoundError: No module named
    'deepagents'` because no site-package install exists (policy:
    tests always run against source, see conftest.py header).
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join([
        str(Path(__file__).parent.parent.parent.parent / "deepagents"),
        # deep_worker_cli source dir — so `deep_worker_cli.main` resolves:
        str(Path(__file__).parent.parent.parent),
    ])
    env["PYTHONPATH"] = prefix + (os.pathsep + existing if existing else "")
    return env


def test_version_matches_pyproject() -> None:
    """Verify that __version__ in _version.py matches version in pyproject.toml."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    # Read the version from pyproject.toml
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_version = pyproject_data["project"]["version"]

    # Compare versions
    assert __version__ == pyproject_version, (
        f"Version mismatch: _version.py has '{__version__}' "
        f"but pyproject.toml has '{pyproject_version}'"
    )


def test_cli_version_flag() -> None:
    """Verify that --version flag outputs the correct version."""
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "--version"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    # argparse exits with 0 for --version
    assert result.returncode == 0
    assert f"deepagents-cli {__version__}" in result.stdout


def test_version_slash_command_message_format() -> None:
    """Verify the /version slash command message format matches expected output."""
    # This tests the exact message format used in app.py's _handle_command for /version
    expected_message = f"deepagents version: {__version__}"
    assert "deepagents version:" in expected_message
    assert __version__ in expected_message


def test_help_mentions_version_flag() -> None:
    """Verify that the CLI help text mentions --version."""
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "help"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    # Help command should succeed
    assert result.returncode == 0
    # Help output should mention --version
    assert "--version" in result.stdout


def test_cli_help_flag() -> None:
    """Verify that --help flag shows help and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "--help"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    # --help should exit with 0
    assert result.returncode == 0
    # Help output should mention key options
    assert "--version" in result.stdout
    assert "--agent" in result.stdout


def test_cli_help_flag_short() -> None:
    """Verify that -h flag shows help and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "-h"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    # -h should exit with 0
    assert result.returncode == 0
    # Help output should mention key options
    assert "--version" in result.stdout
    assert "--agent" in result.stdout


def test_help_excludes_interactive_features() -> None:
    """Verify that --help does not contain Interactive Features section."""
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "--help"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    # Help should succeed
    assert result.returncode == 0
    # Help should NOT contain Interactive Features section
    assert "Interactive Features" not in result.stdout
