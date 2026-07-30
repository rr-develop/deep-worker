"""Contract test for workspace root resolution (design §4.3).

IMPORTANT: `find_project_root` is NOT changed. Here we merely FREEZE its
behavior as the contract LSP depends on: in a git worktree the `.git`
file is a FILE, not a directory, and the root must be the worktree
itself (e.g. a local development checkout), not the main repository.

The test does not rely on any real checkout path — it builds an
equivalent structure in tmp_path, so it passes in any CI.
"""
from __future__ import annotations

from pathlib import Path

from deep_worker_cli.lsp.serena_config import resolve_workspace_root
from deep_worker_cli.project_utils import find_project_root


def _make_worktree_like(root: Path) -> Path:
    """A directory where .git is a FILE (as in a git worktree)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text(
        "gitdir: /path/to/main/.git/worktrees/wt\n", encoding="utf-8"
    )
    return root


def test_find_project_root_treats_git_file_as_root(tmp_path: Path) -> None:
    """Lock on the existing behavior: .git-as-file → the root is found."""
    wt = _make_worktree_like(tmp_path / "worktree")
    nested = wt / "pkg" / "sub"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == wt


def test_resolve_workspace_root_prefers_git_root(tmp_path: Path) -> None:
    wt = _make_worktree_like(tmp_path / "worktree")
    nested = wt / "pkg" / "sub"
    nested.mkdir(parents=True)

    assert resolve_workspace_root(nested) == wt


def test_resolve_workspace_root_explicit_wins(tmp_path: Path) -> None:
    """An explicitly given --project takes priority over auto-detection."""
    wt = _make_worktree_like(tmp_path / "worktree")
    other = tmp_path / "elsewhere"
    other.mkdir()

    assert resolve_workspace_root(wt, explicit=other) == other.resolve()


def test_resolve_workspace_root_falls_back_to_working_dir(tmp_path: Path) -> None:
    """No .git anywhere above → the working directory itself."""
    plain = tmp_path / "no_git_here"
    plain.mkdir()

    # tmp_path is usually not inside a git repository; if it happens to be,
    # the "return the found root" contract is also valid, so we check
    # leniently: the result is absolute and contains our directory.
    got = resolve_workspace_root(plain)
    assert got.is_absolute()
    assert got == plain.resolve() or plain.resolve().is_relative_to(got)


def test_resolve_workspace_root_returns_absolute(tmp_path: Path) -> None:
    wt = _make_worktree_like(tmp_path / "worktree")
    sub = wt / "a" / ".." / "a"
    (wt / "a").mkdir(parents=True)

    got = resolve_workspace_root(sub)
    assert got.is_absolute()
    assert ".." not in str(got)
