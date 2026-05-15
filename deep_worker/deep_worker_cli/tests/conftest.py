"""Global pytest configuration for deep_worker_cli tests.

CRITICAL REQUIREMENT: ALWAYS TEST LOCAL REPOSITORY CODE
========================================================

This conftest.py ensures that tests ALWAYS use the local repository code
from deep_worker/deepagents, NOT installed packages
from site-packages.

WHY THIS IS CRITICAL:
---------------------
- Tests MUST validate the actual code in THIS repository
- Testing installed packages is COMPLETELY UNACCEPTABLE
- Using site-packages versions leads to:
  * Testing outdated/different code
  * False test results
  * Wasted debugging time
  * Production bugs that tests don't catch

NEVER REMOVE OR DISABLE THIS PATH MANIPULATION!
------------------------------------------------
If you think you need to remove this, you are WRONG.
The correct solution is to fix the code, not the test configuration.

This module is executed BEFORE any test code runs, ensuring correct
import paths from the very beginning.
"""

import sys
from pathlib import Path

# ============================================================================
# CRITICAL: Force local deepagents to be first in sys.path
# ============================================================================
# This MUST happen before any imports from deepagents.
# DO NOT use editable install or pip install - those are NOT the repository code.

# Path to local deepagents (sibling to deep_worker_cli)
LOCAL_DEEPAGENTS = Path(__file__).parent.parent.parent / "deepagents"

if not LOCAL_DEEPAGENTS.exists():
    msg = (
        f"CRITICAL ERROR: Local deepagents not found at {LOCAL_DEEPAGENTS}\n"
        "Tests REQUIRE local repository code, not installed packages.\n"
        "Check your repository structure."
    )
    raise RuntimeError(msg)

# FORCE local deepagents to be FIRST (before site-packages)
deepagents_str = str(LOCAL_DEEPAGENTS)
if deepagents_str in sys.path:
    sys.path.remove(deepagents_str)
sys.path.insert(0, deepagents_str)

# Verify that we're using the correct deepagents
try:
    import deepagents
    actual_path = Path(deepagents.__file__).parent.parent
    if actual_path != LOCAL_DEEPAGENTS:
        msg = (
            f"CRITICAL ERROR: Wrong deepagents is being imported!\n"
            f"Expected: {LOCAL_DEEPAGENTS}\n"
            f"Actual:   {actual_path}\n"
            f"This means tests are running against WRONG CODE.\n"
            f"Check sys.path configuration."
        )
        raise RuntimeError(msg)
except ImportError as e:
    msg = (
        f"CRITICAL ERROR: Cannot import deepagents from {LOCAL_DEEPAGENTS}\n"
        f"Error: {e}\n"
        f"Tests CANNOT run without local deepagents."
    )
    raise RuntimeError(msg) from e

# Verify ProjectMemoryMiddleware is available
try:
    from deepagents.middleware.project_memory import ProjectMemoryMiddleware
    # Success - we have the real local version
except ImportError:
    # This should NOT happen if we're using local code
    msg = (
        "CRITICAL ERROR: ProjectMemoryMiddleware not found in local deepagents.\n"
        f"Path: {LOCAL_DEEPAGENTS / 'deepagents' / 'middleware' / 'project_memory.py'}\n"
        "This indicates the local deepagents is incomplete or corrupted."
    )
    raise RuntimeError(msg)

print(f"✓ Tests will use LOCAL deepagents from: {LOCAL_DEEPAGENTS}")
print(f"✓ ProjectMemoryMiddleware available from local code")


def pytest_collection_modifyitems(config, items):
    """Auto-mark everything under `tests/integration/` and
    `tests/integration_ai/` as `@pytest.mark.integration`.

    The default `-m 'not integration'` filter in pyproject.toml
    deselects those suites because they either spin up a real
    deep-worker agent or run `AsyncioServer + rpyc client` in the
    same process (forbidden by the rpyc-async project's own policy on
    same-process server/client tests).
    Running them deliberately with `pytest -m integration` or
    pointing at the directory still works.
    """
    import pytest as _pytest
    tests_dir = Path(__file__).parent.resolve()
    integration_dirs = {
        tests_dir / "integration",
        tests_dir / "integration_ai",
    }
    marker = _pytest.mark.integration
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        for d in integration_dirs:
            try:
                item_path.relative_to(d)
            except ValueError:
                continue
            else:
                item.add_marker(marker)
                break
