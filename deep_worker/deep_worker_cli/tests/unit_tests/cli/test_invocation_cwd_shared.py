"""The shared ``invocation_cwd()`` — the single source of the launch directory.

Contract (see ``cli/cwd.py``):

* if the launcher set ``DEEP_WORKER_ORIGINAL_CWD`` — return it;
* otherwise fall back to ``Path.cwd()``;
* ALL scope-by-cwd commands (``lsp``, ``mcp``) must go through it,
  not through ``Path.cwd()`` directly.

The last point is checked structurally: a regression like "someone wrote
``Path.cwd()`` in the project-scope resolver again" must fail here, not
surface to the user by writing into someone else's repository.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deep_worker_cli.cli.cwd import invocation_cwd

_CLI_DIR = Path(__file__).resolve().parents[3] / "deep_worker_cli" / "cli"


def test_returns_env_value_when_launcher_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEP_WORKER_ORIGINAL_CWD", str(tmp_path))
    assert invocation_cwd() == tmp_path


def test_falls_back_to_cwd_without_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEP_WORKER_ORIGINAL_CWD", raising=False)
    monkeypatch.chdir(tmp_path)
    assert invocation_cwd().resolve() == tmp_path.resolve()


def test_env_wins_over_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is exactly the case that catches the bug: the process already
    chdir'ed into the install directory, but the user launched the command from somewhere else."""
    install_dir = tmp_path / "install"
    user_dir = tmp_path / "user"
    install_dir.mkdir()
    user_dir.mkdir()

    monkeypatch.chdir(install_dir)
    monkeypatch.setenv("DEEP_WORKER_ORIGINAL_CWD", str(user_dir))

    assert invocation_cwd() == user_dir


@pytest.mark.parametrize("module_name", ["lsp_command.py", "mcp_command.py"])
def test_scope_resolvers_do_not_use_bare_path_cwd(module_name: str) -> None:
    """Structural lock: no ``Path.cwd()`` in the project-scope resolver.

    ``mcp_command`` already broke exactly this way (``find_project_root(Path.cwd())``),
    writing the project config into the install directory.
    """
    source = (_CLI_DIR / module_name).read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if "scope" not in node.name and "path" not in node.name:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "cwd"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "Path"
            ):
                offenders.append(f"{module_name}:{node.name}:{inner.lineno}")

    assert not offenders, (
        "scope resolver uses Path.cwd() instead of invocation_cwd(): "
        f"{offenders}"
    )


@pytest.mark.parametrize("module_name", ["lsp_command.py", "mcp_command.py"])
def test_scope_commands_import_shared_helper(module_name: str) -> None:
    """Both modules take the directory from the shared source, without local copies."""
    source = (_CLI_DIR / module_name).read_text(encoding="utf-8")
    assert "invocation_cwd" in source, (
        f"{module_name} does not use the shared invocation_cwd()"
    )
    assert "DEEP_WORKER_ORIGINAL_CWD" not in source, (
        f"{module_name} keeps a local copy of the launcher-cwd logic; "
        "the source of truth is cli/cwd.py"
    )
