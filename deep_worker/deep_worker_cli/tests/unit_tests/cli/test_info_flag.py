"""Tests for --info flag: show_agent_info()."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_worker_cli.types import AgentProcessInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


LAUNCH_CONFIG = {
    "provider": "claude-code",
    "model": "opus",
    "working_dir": "/path/to/dev",
    "tree_root": "/path/to/dev",
    "dev_tree": True,
    "skip_permissions": True,
    "task_manager": True,
    "history_memory": True,
    "memory_only": False,
    "use_sqlite": False,
    "auto_approve_send_message": False,
    "clean": False,
    "started_at": "2026-06-05T20:14:29",
    "pid": 1842836,
    "cli_args": "--restart myagent -m",
}


def _write_full_metadata(data_dir: Path, launch_config: dict, *,
                         status: str = "idle",
                         last_activity: str = "2026-06-05T20:18:42"):
    """Write a complete session_metadata.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": status,
        "tasks_completed": 0,
        "message_count": 1,
        "error_message": None,
        "last_activity": last_activity,
        "storage_version": 5,
        "launch_config": launch_config,
    }
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2))


# ===========================================================================
# show_agent_info — no metadata
# ===========================================================================


class TestNoMetadata:
    def test_missing_metadata_file_returns_error(self, temp_data_dir):
        """show_agent_info returns 1 when session_metadata.json is missing."""
        from deep_worker_cli.cli.agent_command import show_agent_info

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=temp_data_dir / "nonexistent"):
            rc = show_agent_info("test_agent")

        assert rc == 1

    def test_missing_metadata_file_prints_error(self, temp_data_dir, capsys):
        """show_agent_info prints error message when file is missing."""
        from deep_worker_cli.cli.agent_command import show_agent_info

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=temp_data_dir / "nobody"):
            show_agent_info("nobody")

        captured = capsys.readouterr()
        assert "No session metadata found" in captured.err
        assert "nobody" in captured.err

    def test_corrupt_metadata_returns_error(self, temp_data_dir):
        """Corrupt JSON file returns exit code 1."""
        agent_dir = temp_data_dir / "corrupt_agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "session_metadata.json").write_text("not json {{{")

        from deep_worker_cli.cli.agent_command import show_agent_info

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=agent_dir):
            rc = show_agent_info("corrupt_agent")

        assert rc == 1


# ===========================================================================
# show_agent_info — happy path
# ===========================================================================


class TestHappyPath:
    def test_all_required_fields_appear(self, temp_data_dir, capsys):
        """Output contains all expected fields from the design spec."""
        agent_dir = temp_data_dir / "myagent"
        _write_full_metadata(agent_dir, LAUNCH_CONFIG)

        from deep_worker_cli.cli.agent_command import show_agent_info

        with (
            patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                  return_value=agent_dir),
            patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_reg,
        ):
            mock_reg().lookup.return_value = None
            rc = show_agent_info("myagent")

        assert rc == 0
        out = capsys.readouterr().out

        assert "agent: myagent" in out
        assert "provider: claude-code" in out
        assert "model: opus" in out
        assert "working_dir: /path/to/dev" in out
        assert "tree_root:" in out
        assert "dev_tree=true" in out
        assert "started_at: 2026-06-05T20:14:29" in out
        assert "pid: 1842836" in out
        assert "flags:" in out
        assert "cli_args: --restart myagent -m" in out
        assert "status: idle" in out
        assert "last_activity: 2026-06-05T20:18:42" in out

    def test_preferred_provider_shown(self, temp_data_dir, capsys):
        """When preferred_provider is set, it appears in provider line."""
        cfg = dict(LAUNCH_CONFIG)
        cfg["preferred_provider"] = "codex"
        agent_dir = temp_data_dir / "pref_agent"
        _write_full_metadata(agent_dir, cfg)

        from deep_worker_cli.cli.agent_command import show_agent_info

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=agent_dir):
            show_agent_info("pref_agent")

        out = capsys.readouterr().out
        assert "preferred: codex" in out

    def test_flag_values_lowercase(self, temp_data_dir, capsys):
        """Boolean flags are printed as lowercase true/false."""
        agent_dir = temp_data_dir / "flaggy"
        _write_full_metadata(agent_dir, LAUNCH_CONFIG)

        from deep_worker_cli.cli.agent_command import show_agent_info

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=agent_dir):
            show_agent_info("flaggy")

        out = capsys.readouterr().out

        # Extract the flags section
        flags_section = out.split("flags:")[1].split("cli_args:")[0]
        assert "true" in flags_section
        assert "false" in flags_section


# ===========================================================================
# show_agent_info — running / not running
# ===========================================================================


class TestRunningStatus:
    def test_shows_running_when_in_registry(self, temp_data_dir, capsys):
        """When ProcessRegistry.lookup returns info, show (running)."""
        agent_dir = temp_data_dir / "runner"
        _write_full_metadata(agent_dir, LAUNCH_CONFIG)

        process_info = AgentProcessInfo(
            name="runner",
            pid=99999,
            pid_file=agent_dir / "agent.pid",
            working_dir="/path/to/dev",
            started_at="2026-06-05T20:14:29",
            rpyc_port=12345,
        )

        from deep_worker_cli.cli.agent_command import show_agent_info

        with (
            patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                  return_value=agent_dir),
            patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_reg_cls,
        ):
            mock_reg_cls().lookup.return_value = process_info
            rc = show_agent_info("runner")

        assert rc == 0
        out = capsys.readouterr().out
        assert "(running)" in out
        assert "pid: 99999" in out  # Uses process_info.pid

    def test_shows_not_running_when_not_in_registry(self, temp_data_dir, capsys):
        """When ProcessRegistry.lookup returns None, show (NOT RUNNING)."""
        agent_dir = temp_data_dir / "stopped"
        _write_full_metadata(agent_dir, LAUNCH_CONFIG)

        from deep_worker_cli.cli.agent_command import show_agent_info

        pid = LAUNCH_CONFIG["pid"]
        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                   return_value=agent_dir):
            # Patch ProcessRegistry to return None, and patch os.path.isdir
            # to return False for /proc/<pid> to ensure "NOT RUNNING"
            with (
                patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_reg,
                patch("deep_worker_cli.cli.agent_command.os.path.isdir",
                      return_value=False),
            ):
                mock_reg().lookup.return_value = None
                rc = show_agent_info("stopped")

        assert rc == 0
        out = capsys.readouterr().out
        assert "(NOT RUNNING)" in out
        assert f"pid: {pid}" in out

    def test_shows_running_via_proc_when_no_registry(self, temp_data_dir, capsys):
        """Fallback: if registry has no entry but /proc/<pid> exists, show running."""
        agent_dir = temp_data_dir / "proc_alive"
        cfg = dict(LAUNCH_CONFIG)
        cfg["pid"] = 1  # PID 1 always exists
        _write_full_metadata(agent_dir, cfg)

        from deep_worker_cli.cli.agent_command import show_agent_info

        with (
            patch("deep_worker_cli.cli.agent_command._get_agent_data_dir",
                  return_value=agent_dir),
            patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_reg,
        ):
            mock_reg().lookup.return_value = None
            # os.path.isdir(f"/proc/{pid}") will be called with pid=1 → True
            # This is a real /proc/1 check so we don't need to mock it
            rc = show_agent_info("proc_alive")

        assert rc == 0
        out = capsys.readouterr().out
        assert "(running)" in out


# ===========================================================================
# create_parser — --info integration
# ===========================================================================


class TestParser:
    def test_info_flag_accepted_with_name(self):
        """--info accepts a positional name."""
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["--info", "myagent"])

        assert args.info is True
        assert args.name == "myagent"

    def test_info_without_name_sets_name_to_none(self):
        """--info without a name leaves name as None (enforced in main())."""
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        args = parser.parse_args(["--info"])

        assert args.info is True
        assert args.name is None
