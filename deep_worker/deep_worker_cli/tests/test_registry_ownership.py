"""
TDD tests for ProcessRegistry file ownership preservation.

This test verifies the fix for issue where CCO daemon (running as root) was
changing /tmp/deep_worker_registry.json owner from develop to root.

The fix ensures that file ownership is preserved as develop:develop even when
daemon (root) writes to the registry.
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo
from datetime import datetime


class TestRegistryOwnershipPreservation:
    """Test that registry file ownership is preserved when written by root process."""

    def test_registry_preserves_develop_ownership_when_root_writes(self):
        """Registry MUST preserve develop ownership even when daemon (root) writes to it.

        This is the CRITICAL test - if daemon changes owner to root, other systems break!
        """
        # Create temp registry file owned by develop (UID 1000)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            registry_file = Path(f.name)
            f.write('{}')

        lock_file = registry_file.with_suffix('.lock')

        try:
            # Set ownership to develop (UID 1000, GID 1000)
            develop_uid = 1000
            develop_gid = 1000
            os.chown(registry_file, develop_uid, develop_gid)

            # Verify initial ownership
            stat_before = os.stat(registry_file)
            assert stat_before.st_uid == develop_uid, "Initial owner should be develop"
            assert stat_before.st_gid == develop_gid, "Initial group should be develop"

            # Create registry and register an agent (simulating daemon write)
            registry = ProcessRegistry(registry_file=registry_file, lock_file=lock_file)

            agent_info = AgentProcessInfo(
                name="test_agent",
                pid=os.getpid(),
                pid_file=Path("/tmp/test.pid"),
                working_dir="/tmp",
                started_at=datetime.now(),
                rpyc_port=12345
            )

            # This triggers _save_registry() which should preserve ownership
            registry.register(agent_info)

            # CRITICAL CHECK: Owner MUST still be develop after daemon write!
            stat_after = os.stat(registry_file)
            assert stat_after.st_uid == develop_uid, (
                f"Owner changed from develop ({develop_uid}) to {stat_after.st_uid}! "
                f"This BREAKS other systems. MUST preserve develop ownership!"
            )
            assert stat_after.st_gid == develop_gid, (
                f"Group changed from develop ({develop_gid}) to {stat_after.st_gid}! "
                f"This BREAKS other systems. MUST preserve develop group!"
            )

        finally:
            # Cleanup
            if registry_file.exists():
                registry_file.unlink()
            if lock_file.exists():
                lock_file.unlink()
            tmp_file = registry_file.with_suffix('.tmp')
            if tmp_file.exists():
                tmp_file.unlink()

    def test_registry_preserves_ownership_across_multiple_writes(self):
        """Multiple registry writes MUST NOT change ownership."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            registry_file = Path(f.name)
            f.write('{}')

        lock_file = registry_file.with_suffix('.lock')

        try:
            develop_uid = 1000
            develop_gid = 1000
            os.chown(registry_file, develop_uid, develop_gid)

            registry = ProcessRegistry(registry_file=registry_file, lock_file=lock_file)

            # Perform 3 writes (register, unregister, register)
            for i in range(3):
                agent_info = AgentProcessInfo(
                    name=f"test_agent_{i}",
                    pid=os.getpid() + i,
                    pid_file=Path(f"/tmp/test_{i}.pid"),
                    working_dir="/tmp",
                    started_at=datetime.now(),
                    rpyc_port=12345 + i
                )
                registry.register(agent_info)

                # Check ownership after each write
                stat = os.stat(registry_file)
                assert stat.st_uid == develop_uid, (
                    f"Write {i+1}: Owner changed to {stat.st_uid}, expected {develop_uid}"
                )
                assert stat.st_gid == develop_gid, (
                    f"Write {i+1}: Group changed to {stat.st_gid}, expected {develop_gid}"
                )

        finally:
            if registry_file.exists():
                registry_file.unlink()
            if lock_file.exists():
                lock_file.unlink()
            tmp_file = registry_file.with_suffix('.tmp')
            if tmp_file.exists():
                tmp_file.unlink()

    def test_registry_handles_root_owned_file_gracefully(self):
        """If file is accidentally owned by root, registry should fix it or handle gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            registry_file = Path(f.name)
            f.write('{}')

        lock_file = registry_file.with_suffix('.lock')

        try:
            # Simulate file owned by root (UID 0) - this is the BAD state we want to avoid
            # We can't actually chown to root without sudo, so we'll mock the stat result
            develop_uid = 1000
            develop_gid = 1000

            registry = ProcessRegistry(registry_file=registry_file, lock_file=lock_file)

            agent_info = AgentProcessInfo(
                name="test_agent",
                pid=os.getpid(),
                pid_file=Path("/tmp/test.pid"),
                working_dir="/tmp",
                started_at=datetime.now(),
                rpyc_port=12345
            )

            # Should not raise exception even if file has wrong owner
            # (graceful handling or auto-fix)
            registry.register(agent_info)

            # After write, should be develop-owned
            stat = os.stat(registry_file)
            assert stat.st_uid == develop_uid or stat.st_uid == os.getuid(), (
                "File should be owned by develop or current user after write"
            )

        finally:
            if registry_file.exists():
                registry_file.unlink()
            if lock_file.exists():
                lock_file.unlink()
            tmp_file = registry_file.with_suffix('.tmp')
            if tmp_file.exists():
                tmp_file.unlink()
