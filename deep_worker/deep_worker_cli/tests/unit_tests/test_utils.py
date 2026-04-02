"""Unit tests for utility functions."""

import sys
from unittest.mock import patch

import pytest


class TestIsRunningTests:
    """Tests for is_running_tests() function."""

    def test_force_production_overrides_all(self, monkeypatch):
        """Test DEEP_WORKER_FORCE_PRODUCTION=1 returns False (production mode)."""
        from deep_worker_cli.utils import is_running_tests

        # Set all test indicators
        monkeypatch.setenv("DEEP_WORKER_FORCE_PRODUCTION", "1")
        monkeypatch.setenv("DEEP_WORKER_TEST_MODE", "1")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_something")

        # Should return False because FORCE_PRODUCTION overrides
        assert is_running_tests() is False

    def test_explicit_test_mode(self, monkeypatch):
        """Test DEEP_WORKER_TEST_MODE=1 returns True."""
        from deep_worker_cli.utils import is_running_tests

        monkeypatch.setenv("DEEP_WORKER_TEST_MODE", "1")
        # Clear other indicators
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        assert is_running_tests() is True

    def test_pytest_current_test_env(self, monkeypatch):
        """Test PYTEST_CURRENT_TEST env var returns True."""
        from deep_worker_cli.utils import is_running_tests

        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_something.py::test_func")
        monkeypatch.delenv("DEEP_WORKER_TEST_MODE", raising=False)
        monkeypatch.delenv("DEEP_WORKER_FORCE_PRODUCTION", raising=False)

        assert is_running_tests() is True

    def test_pytest_in_sys_modules(self, monkeypatch):
        """Test pytest in sys.modules returns True."""
        from deep_worker_cli.utils import is_running_tests

        # Clear env vars
        monkeypatch.delenv("DEEP_WORKER_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("DEEP_WORKER_FORCE_PRODUCTION", raising=False)

        # pytest should already be in sys.modules when running under pytest
        # This test verifies the detection works
        assert is_running_tests() is True

    def test_no_test_indicators(self, monkeypatch):
        """Test returns False when no test indicators present."""
        from deep_worker_cli.utils import is_running_tests

        # Clear all env vars
        monkeypatch.delenv("DEEP_WORKER_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("DEEP_WORKER_FORCE_PRODUCTION", raising=False)

        # Remove pytest from sys.modules temporarily
        pytest_module = sys.modules.get("pytest")
        if pytest_module:
            del sys.modules["pytest"]

        try:
            result = is_running_tests()
            assert result is False
        finally:
            # Restore pytest module
            if pytest_module:
                sys.modules["pytest"] = pytest_module

    def test_force_production_with_zero_value(self, monkeypatch):
        """Test DEEP_WORKER_FORCE_PRODUCTION=0 does not override."""
        from deep_worker_cli.utils import is_running_tests

        monkeypatch.setenv("DEEP_WORKER_FORCE_PRODUCTION", "0")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

        # Should detect as test because FORCE_PRODUCTION != "1"
        assert is_running_tests() is True

    def test_force_production_empty_string(self, monkeypatch):
        """Test DEEP_WORKER_FORCE_PRODUCTION='' does not override."""
        from deep_worker_cli.utils import is_running_tests

        monkeypatch.setenv("DEEP_WORKER_FORCE_PRODUCTION", "")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

        # Should detect as test because FORCE_PRODUCTION != "1"
        assert is_running_tests() is True
