"""Utility functions for deep-worker CLI."""

import os
import sys


def is_running_tests() -> bool:
    """Detect if running in test environment.

    Checks multiple indicators with priority:
    1. DEEP_WORKER_FORCE_PRODUCTION=1 - override, force production mode (return False)
    2. DEEP_WORKER_TEST_MODE=1 - explicit test mode (return True)
    3. PYTEST_CURRENT_TEST - pytest sets this automatically
    4. pytest in sys.modules - pytest is loaded

    Returns:
        True if running in test environment, False otherwise
    """
    # Allow explicit override to force production mode in tests
    if os.environ.get("DEEP_WORKER_FORCE_PRODUCTION") == "1":
        return False

    # Explicit test mode
    if os.environ.get("DEEP_WORKER_TEST_MODE") == "1":
        return True

    # Pytest environment variable
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    # Pytest module loaded
    if "pytest" in sys.modules:
        return True

    return False
