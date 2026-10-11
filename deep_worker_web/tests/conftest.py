"""Pytest configuration and shared fixtures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add deep_worker_web to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _resolve_rpyc_async() -> Path:
    """Pick the rpyc_async checkout this tree should test against.

    rpyc_async is a SEPARATE repository with a single production checkout,
    referenced by a fixed path in app code, so it has no dev isolation of its
    own: an edit there is visible to every tree at once.
    A dev tree must therefore test against the rpyc_async *dev worktree*, not
    prod — otherwise dev silently validates production code.

    The dev worktree is discovered by ASKING GIT which worktree holds branch
    ``dev`` (no hard-coded worktree path, so renaming or recreating it does
    not break the tests). Falls back to prod when there is no dev worktree.
    Override explicitly with DEEP_WORKER_RPYC_ASYNC=<path>.
    """
    override = os.environ.get("DEEP_WORKER_RPYC_ASYNC")
    if override:
        return Path(override)

    prod = Path("rpyc_async")
    if "_dev" not in str(Path(__file__).resolve()):
        return prod

    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=prod, capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return prod

    current = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree "):]
        elif line.strip() == "branch refs/heads/dev" and current:
            return Path(current)
    return prod


rpyc_async_path = _resolve_rpyc_async()
if rpyc_async_path.exists():
    sys.path.insert(0, str(rpyc_async_path))

# Add deep_worker_cli and deepagents from THE SAME TREE as deep_worker_web.
# This file lives at deep_worker_web/tests/conftest.py, so parents
# resolve as follows:
#   parent     = deep_worker_web/tests
#   parent x2  = deep_worker_web
#   parent x3  = <TREE>
# Using __file__-relative paths (instead of hard-coding an absolute checkout
# path) means tests in a dev checkout validate dev-tree code, and tests in a
# prod checkout validate prod-tree code — without cross-tree leakage.
# See docs/DESIGN_DEV_FLAG.md.
tree_root = Path(__file__).resolve().parent.parent.parent

deep_worker_path = tree_root / "deep_worker" / "deep_worker_cli"
if deep_worker_path.exists():
    sys.path.insert(0, str(deep_worker_path))

deepagents_path = tree_root / "deep_worker" / "deepagents"
if deepagents_path.exists():
    sys.path.insert(0, str(deepagents_path))


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test at `tests/test_*.py` (the UI / live-server
    Playwright suite that lives at the repository root) with
    `@pytest.mark.ui` so they are excluded by the default
    `-m "not ui and not integration"` filter configured in
    pytest.ini. Without this filter, `pytest tests/` would try to
    collect Playwright subprocess-server tests and hang on them."""
    root_tests_dir = Path(__file__).parent.resolve()
    ui_marker = pytest.mark.ui
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        # Only direct children of tests/ (NOT tests/unit/, tests/integration/).
        if item_path.parent == root_tests_dir and item_path.name.startswith("test_"):
            item.add_marker(ui_marker)


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use asyncio backend for async tests."""
    return "asyncio"
