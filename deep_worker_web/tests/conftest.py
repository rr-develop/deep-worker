"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add deep_worker_web to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the local rpyc_async source checkout to path for RPyC AsyncioServer
rpyc_async_path = Path("rpyc_async")
if rpyc_async_path.exists():
    sys.path.insert(0, str(rpyc_async_path))

# Add the CLI package to path for ProcessRegistry and other utilities
dw_cli_source_path = project_root.parent / "deep_worker" / "deep_worker_cli"
if dw_cli_source_path.exists():
    sys.path.insert(0, str(dw_cli_source_path))

# Add deepagents to path (required by deep_worker_cli)
deepagents_path = project_root.parent / "deep_worker" / "deepagents"
if deepagents_path.exists():
    sys.path.insert(0, str(deepagents_path))


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use asyncio backend for async tests."""
    return "asyncio"
