"""Unit tests for resolving the invocation directory of the ``lsp`` command.

Bug (reported by a user)
------------------------------------
The user was standing in a git worktree and ran ``deep-worker lsp enable``.
The config was written into the MAIN repository's
``.deep_worker/mcp.json`` — that is, into the main repo, and with
``--project <repo>`` inside.

Cause: the bash launcher does ``cd "$SCRIPT_DIR"`` before running Python
(the directory of the launcher's physical file); the user's real directory
is stored in ``DEEP_WORKER_ORIGINAL_CWD``. The command read ``Path.cwd()``
and therefore saw the INSTALL directory, not the user's directory.

``DEEP_WORKER_ORIGINAL_CWD`` is the standard mechanism: ``cli/agent_command.py``
already uses it to determine the agent's ``working_dir``.

Contract pinned by the tests:
* if the variable is set — it wins over ``Path.cwd()``;
* if not set — fall back to ``Path.cwd()`` (direct invocation without the launcher);
* ``--project-root`` / ``--config`` still win over everything.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from deep_worker_cli.cli.lsp_command import (
    _config_path,
    _workspace_root,
    handle_lsp_command,
    setup_lsp_parser,
)

_ENV = "DEEP_WORKER_ORIGINAL_CWD"


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """Tests must not depend on the environment pytest runs in."""
    saved = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = saved


def _parse(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_lsp_parser(subparsers)
    return parser.parse_args(argv)


def _make_worktree(root: Path) -> Path:
    """A directory where ``.git`` is a FILE (as in a git worktree)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    return root


def test_workspace_root_uses_original_cwd_env(tmp_path: Path) -> None:
    """The main bug case: the user's directory is taken from the env."""
    user_dir = _make_worktree(tmp_path / "worktree")
    os.environ[_ENV] = str(user_dir)

    assert _workspace_root(_parse(["lsp", "enable"])) == user_dir


def test_config_path_uses_original_cwd_env(tmp_path: Path) -> None:
    """The config must land in the user's directory, not the install one."""
    user_dir = _make_worktree(tmp_path / "worktree")
    os.environ[_ENV] = str(user_dir)

    got = _config_path(_parse(["lsp", "enable", "--scope", "project"]))

    assert got == user_dir / ".deep_worker" / "mcp.json"


def test_explicit_project_root_still_wins(tmp_path: Path) -> None:
    user_dir = _make_worktree(tmp_path / "worktree")
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    os.environ[_ENV] = str(user_dir)

    args = _parse(["lsp", "enable", "--project-root", str(explicit)])

    assert _workspace_root(args) == explicit.resolve()


def test_explicit_config_still_wins(tmp_path: Path) -> None:
    user_dir = _make_worktree(tmp_path / "worktree")
    explicit_cfg = tmp_path / "custom" / "mcp.json"
    os.environ[_ENV] = str(user_dir)

    args = _parse(["lsp", "status", "--config", str(explicit_cfg)])

    assert _config_path(args) == explicit_cfg


def test_falls_back_to_cwd_without_env(tmp_path: Path, monkeypatch) -> None:
    """Direct invocation without the launcher: no env → Path.cwd() works."""
    user_dir = _make_worktree(tmp_path / "worktree")
    monkeypatch.chdir(user_dir)

    assert _workspace_root(_parse(["lsp", "enable"])) == user_dir


def test_enable_writes_into_user_directory_not_install_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end reproduction of the bug at the handler level.

    We simulate the launcher: cwd = "install directory", env = the user's
    directory. The write must land in the USER's directory.
    """
    install_dir = _make_worktree(tmp_path / "install")
    user_dir = _make_worktree(tmp_path / "user_worktree")

    monkeypatch.chdir(install_dir)
    os.environ[_ENV] = str(user_dir)

    rc = handle_lsp_command(_parse(["lsp", "enable"]))
    assert rc == 0

    user_cfg = user_dir / ".deep_worker" / "mcp.json"
    install_cfg = install_dir / ".deep_worker" / "mcp.json"

    assert user_cfg.exists(), "config must be written into the user's directory"
    assert not install_cfg.exists(), "config must NOT leak into the install dir"

    entry = json.loads(user_cfg.read_text(encoding="utf-8"))["mcpServers"]["serena"]
    assert str(user_dir) in entry["args"], entry["args"]
    assert str(install_dir) not in entry["args"], entry["args"]
