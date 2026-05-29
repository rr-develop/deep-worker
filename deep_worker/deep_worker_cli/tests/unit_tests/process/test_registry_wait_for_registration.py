"""Tests for ProcessRegistry.wait_for_registration() — event-driven, no polling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.types import AgentProcessInfo


class TestWaitForRegistration:
    """Tests for the event-driven wait_for_registration method."""

    async def test_returns_immediately_when_already_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Agent is already in the registry — no waiting needed."""
        from deep_worker_cli.process.registry import ProcessRegistry

        # Skip the alive check — we know this PID is fake, but the
        # registry entry exists, which is all we need to test.
        monkeypatch.setattr(
            "deep_worker_cli.process.registry.ProcessRegistry._is_process_alive",
            lambda self, pid: True,
        )

        registry_file = tmp_path / "registry.json"
        registry = ProcessRegistry(registry_file=registry_file)

        # Pre-populate the registry
        entry = {
            "test_agent": {
                "pid": 99999,
                "pid_file": "/tmp/test.pid",
                "working_dir": "/tmp",
                "started_at": "2026-01-01T00:00:00",
                "last_activity": "2026-01-01T00:00:00",
                "rpyc_port": 18861,
            }
        }
        registry_file.write_text(json.dumps(entry))

        # Agent is already in registry — wait_for_registration returns immediately
        result = await registry.wait_for_registration("test_agent", timeout=5)
        assert result is not None
        assert isinstance(result, AgentProcessInfo)
        assert result.name == "test_agent"
        assert result.pid == 99999

    async def test_inotify_unavailable_fallback_returns_none(
        self, tmp_path: Path
    ) -> None:
        """When inotify is unavailable (non-Linux), returns None gracefully."""
        import ctypes.util

        from deep_worker_cli.process.registry import ProcessRegistry

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({}))
        registry = ProcessRegistry(registry_file=registry_file)

        # Simulate inotify unavailable: ctypes.util.find_library returns None
        with patch.object(ctypes.util, "find_library", return_value=None):
            result = await registry.wait_for_registration(
                "nonexistent", timeout=0.5
            )

        # No agent, inotify unavailable — returns None, no crash
        assert result is None

    async def test_no_sleep_or_polling_in_wait_for_registration(
        self, tmp_path: Path
    ) -> None:
        """Verify wait_for_registration contains no time.sleep or loop+sleep."""
        from deep_worker_cli.process.registry import ProcessRegistry

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({}))
        registry = ProcessRegistry(registry_file=registry_file)

        source = type(registry).wait_for_registration.__code__
        import dis

        instructions = list(dis.get_instructions(source))

        # Collect all bytecode instruction names
        opnames = [instr.opname for instr in instructions]

        # Should NOT contain LOAD_GLOBAL/METHOD for 'sleep' or 'monotonic'
        for bad_name in ("sleep", "monotonic"):
            assert bad_name not in opnames, (
                f"wait_for_registration must not call time.{bad_name} — "
                f"found in bytecode: {opnames}"
            )

    async def test_registry_module_no_time_import(
        self, tmp_path: Path
    ) -> None:
        """The registry module itself must not import 'time'."""
        import deep_worker_cli.process.registry as reg

        with open(reg.__file__) as f:
            source = f.read()

        # 'import time' or 'from time import' must not be at module level
        import_lines = [
            line for line in source.splitlines()
            if "import time" in line.replace(" ", "").lower()
        ]
        assert not import_lines, (
            f"registry.py must not import time (polling indicator): "
            f"{import_lines}"
        )
