"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add deep_worker_web to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add rpyc_async to path for RPyC AsyncioServer
rpyc_async_path = Path("rpyc_async")
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
