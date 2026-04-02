"""Test that interactive.py module can be imported without errors.

This test ensures that all imports in interactive.py work correctly,
particularly the rpyc module import which requires sys.path setup.
"""

import sys
from pathlib import Path

import pytest


def test_interactive_module_can_be_imported():
    """Test that interactive.py can be imported without ModuleNotFoundError.

    This test verifies that the rpyc module is properly accessible
    when importing the interactive module.
    """
    # This test should pass after we fix the import issue in interactive.py
    try:
        from deep_worker_cli.cli.interactive import InteractiveApp

        # Verify the class is actually imported
        assert InteractiveApp is not None
        assert hasattr(InteractiveApp, '__name__')
        assert InteractiveApp.__name__ == 'InteractiveApp'
    except ModuleNotFoundError as e:
        # This will fail before the fix
        pytest.fail(f"Failed to import interactive module: {e}")


def test_rpyc_is_importable_from_interactive():
    """Test that rpyc module is accessible when interactive.py imports it."""
    # Import the interactive module
    from deep_worker_cli.cli.interactive import InteractiveApp

    # Try to import rpyc directly
    try:
        import rpyc
        assert rpyc is not None
    except ModuleNotFoundError:
        pytest.fail("rpyc module not found - sys.path not properly configured")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
