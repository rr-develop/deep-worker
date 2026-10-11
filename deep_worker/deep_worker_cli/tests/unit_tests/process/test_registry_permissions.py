"""Tests for ProcessRegistry file permission handling.

This test suite verifies that ProcessRegistry correctly handles permission issues
when the registry file is owned by root or another user.
"""

import os
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.exceptions import RegistryLockError
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo
from datetime import datetime


class TestRegistryPermissions:
    """Test ProcessRegistry permission handling."""

    def test_registry_creates_file_with_correct_permissions(self, tmp_path: Path):
        """Test that registry creates files with 0o666 permissions."""
        registry_file = tmp_path / "registry.json"
        registry = ProcessRegistry(
            registry_file=registry_file, lock_file=tmp_path / "registry.lock"
        )

        # Register agent
        agent_info = AgentProcessInfo(
            name="test_agent",
            pid=12345,
            pid_file=tmp_path / "test.pid",
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=18861,
        )
        registry.register(agent_info)

        # Verify file exists and has correct permissions
        assert registry_file.exists()
        file_perms = os.stat(registry_file).st_mode & 0o777
        assert file_perms == 0o666, f"Expected 0o666, got {oct(file_perms)}"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permission bits, so a read-only file stays writable",
)
class TestRegistryOwnershipValidation:
    """Test pre-startup validation of registry file ownership."""

    def test_can_check_registry_file_ownership(self, tmp_path: Path):
        """Test that we can detect if registry file is writable."""
        registry_file = tmp_path / "registry.json"

        # File doesn't exist - should be writable (can create)
        assert registry_file.parent.exists()
        assert os.access(registry_file.parent, os.W_OK)

        # Create file
        registry_file.write_text("{}")
        assert registry_file.exists()

        # File is writable by current user
        assert os.access(registry_file, os.W_OK)

        # Make file read-only
        os.chmod(registry_file, 0o444)

        # File is NOT writable
        assert not os.access(registry_file, os.W_OK)

    def test_validate_registry_writable_before_daemonize(self, tmp_path: Path):
        """Test validation function that should be called BEFORE daemonize().

        This test verifies the fix for the bug where errors during agent
        startup are not visible to the user because the process has already
        daemonized.
        """
        from deep_worker_cli.process.registry import ProcessRegistry

        # This should be a new method in ProcessRegistry: validate_writable()
        registry_file = tmp_path / "registry.json"
        registry = ProcessRegistry(
            registry_file=registry_file, lock_file=tmp_path / "registry.lock"
        )

        # Case 1: File doesn't exist - should be OK if parent is writable
        result = registry.validate_writable()
        assert result["writable"] is True

        # Case 2: File exists and is writable
        registry_file.write_text("{}")
        os.chmod(registry_file, 0o666)
        result = registry.validate_writable()
        assert result["writable"] is True

        # Case 3: File exists but is read-only (simulate root ownership)
        os.chmod(registry_file, 0o444)
        result = registry.validate_writable()
        assert result["writable"] is False
        assert "error" in result
        assert "permission" in result["error"].lower() or "read-only" in result[
            "error"
        ].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
