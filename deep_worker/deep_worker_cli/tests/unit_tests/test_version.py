"""Tests for version-related functionality."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from deep_worker_cli._version import __version__


def _subprocess_env() -> dict[str, str]:
    """Return an env for subprocess that keeps local `deepagents` and
    `a local rpyc-async checkout` importable — mirrors conftest.py's sys.path
    injection. Without this, `python -m deep_worker_cli.main` in a
    child process fails with `ModuleNotFoundError: No module named
    'deepagents'` because no site-package install exists (policy:
    tests always run against source, see conftest.py header).
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join([
        str(Path(__file__).parent.parent.parent.parent / "deepagents"),
        "a local rpyc-async checkout",
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
    assert f"deep-worker {__version__}" in result.stdout
    # The ancestor's product name must not reach the user.
    assert "deepagents" not in result.stdout.lower()


def test_version_slash_command_names_this_product() -> None:
    """The ``/version`` reply in the TUI names deep-worker, not the ancestor.

    Reads the literal out of ``app.py``'s source rather than rebuilding it
    here: a test that formats the expected string itself passes no matter
    what the product prints.
    """
    app_src = (
        Path(__file__).parent.parent.parent / "deep_worker_cli" / "app.py"
    ).read_text()
    assert 'AppMessage(f"deep-worker version: {__version__}")' in app_src
    assert "deepagents version:" not in app_src


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
