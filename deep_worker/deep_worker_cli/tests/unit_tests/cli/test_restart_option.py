"""Tests for --restart option: merge_restart_config, stop_agent, restart flow."""

import argparse
import json
import os
import signal
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.types import AgentProcessInfo, AgentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def saved_launch_config():
    """A realistic saved launch_config."""
    return {
        "provider": "claude-code",
        "model": "opus",
        "working_dir": "/path/to/project",
        "skip_permissions": True,
        "task_manager": True,
        "memory_only": False,
        "use_sqlite": False,
        "clean": True,  # was True in the original launch
        "started_at": "2026-04-11T16:59:41.692129",
        "pid": 3726921,
        "cli_args": "myagent -sp -tm --dir /path/to/project --provider claude-code --model opus --clean",
    }


@pytest.fixture
def default_args():
    """Argparse namespace with parser defaults (nothing explicitly passed)."""
    return argparse.Namespace(
        name="myagent",
        dir=None,
        model=None,
        provider="deepseek",  # parser default
        skip_permissions=False,
        task_manager=False,
        clean=False,
        memory_only=False,
        use_sqlite=False,
        config=None,
        restart=True,
        list_threads=False,
        remove_thread=None,
        show_history=None,
        remove_before=None,
        dry_run=False,
        format="text",
        with_counts=False,
        agent=None,
        sort="thread_id",
        reverse=False,
    )


def _write_metadata(data_dir: Path, launch_config: dict | None, status: str = "idle"):
    """Write a session_metadata.json with optional launch_config."""
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": status,
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": datetime.now().isoformat(),
    }
    if launch_config is not None:
        metadata["launch_config"] = launch_config
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2))


# ===========================================================================
# merge_restart_config tests
# ===========================================================================

class TestMergeRestartConfig:
    """Tests for merge_restart_config()."""

    def test_saved_values_used_when_no_overrides(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        result = merge_restart_config(saved_launch_config, default_args)

        assert result.provider == "claude-code"
        assert result.model == "opus"
        assert result.dir == Path("/path/to/project")
        assert result.skip_permissions is True
        assert result.task_manager is True
        assert result.memory_only is False
        assert result.use_sqlite is False

    def test_clean_never_inherited(self, saved_launch_config, default_args):
        """--clean from saved config must NOT be inherited."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        assert saved_launch_config["clean"] is True  # was True in original
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.clean is False  # must be False

    def test_explicit_clean_override(self, saved_launch_config, default_args):
        """--clean passed explicitly on CLI is honored."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.clean = True
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.clean is True

    def test_provider_override(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.provider = "gemini"
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.provider == "gemini"

    def test_model_override(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.model = "sonnet"
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.model == "sonnet"

    def test_dir_override(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.dir = Path("/path/to/other")
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.dir == Path("/path/to/other")

    def test_boolean_flag_override_sp(self, saved_launch_config, default_args):
        """Explicitly passing -sp overrides saved False."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        saved_launch_config["skip_permissions"] = False
        default_args.skip_permissions = True
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.skip_permissions is True

    def test_boolean_flag_override_tm(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        saved_launch_config["task_manager"] = False
        default_args.task_manager = True
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.task_manager is True

    def test_memory_only_override(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.memory_only = True
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.memory_only is True

    def test_use_sqlite_override(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.use_sqlite = True
        result = merge_restart_config(saved_launch_config, default_args)
        assert result.use_sqlite is True

    def test_name_preserved(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config

        result = merge_restart_config(saved_launch_config, default_args)
        assert result.name == "myagent"

    def test_provider_default_not_treated_as_override(self, saved_launch_config, default_args):
        """When provider is parser-default 'deepseek' and NOT explicitly passed,
        the saved value wins (this is the no-override restart case)."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        assert default_args.provider == "deepseek"  # parser default
        # No explicit flags passed on the CLI.
        result = merge_restart_config(saved_launch_config, default_args, explicit=set())
        assert result.provider == "claude-code"  # from saved


# ===========================================================================
# Provider-switch regression tests (incident observed 2026-05-29)
#
# `deep-worker --restart <agent> --provider deepseek` was silently restarting
# with the PREVIOUS provider/model. Two root causes in merge_restart_config:
#   1. Override detection compared args.provider against _PARSER_DEFAULTS.
#      Since the parser default IS "deepseek", an explicit `--provider deepseek`
#      compared equal to the default and was dropped -> saved provider kept.
#   2. When the provider changed but --model was not given, the saved model of
#      the OLD provider leaked through, so the new provider booted on a stale,
#      incompatible model.
# ===========================================================================

def _parse(argv):
    """Parse argv with the real CLI parser (mirrors the production path)."""
    from deep_worker_cli.cli.agent_command import create_parser
    return create_parser().parse_args(argv)


class TestProviderSwitchRegression:
    """Explicit --provider must win and must not drag the old model along."""

    def test_explicit_provider_equal_to_parser_default_is_honored(self, saved_launch_config):
        """`--restart a --provider deepseek` over a saved claude-code config.

        The reported bug: deepseek equals the parser default, so it was treated
        as 'not passed' and the saved claude-code provider was kept.
        """
        from deep_worker_cli.cli.agent_command import merge_restart_config

        assert saved_launch_config["provider"] == "claude-code"
        args = _parse(["myagent", "--restart", "--provider", "deepseek"])
        explicit = {"provider"}

        result = merge_restart_config(saved_launch_config, args, explicit=explicit)

        assert result.provider == "deepseek"

    def test_provider_switch_drops_stale_saved_model(self, saved_launch_config):
        """Switching provider without --model must NOT inherit the old model.

        saved = claude-code/opus ; restart with --provider deepseek (no --model).
        The opus model is a claude model and must not leak onto deepseek.
        """
        from deep_worker_cli.cli.agent_command import merge_restart_config

        assert saved_launch_config["model"] == "opus"
        args = _parse(["myagent", "--restart", "--provider", "deepseek"])
        explicit = {"provider"}

        result = merge_restart_config(saved_launch_config, args, explicit=explicit)

        assert result.provider == "deepseek"
        assert result.model != "opus", (
            "Stale claude-code model 'opus' leaked onto the new deepseek provider"
        )

    @pytest.mark.parametrize(
        "argv, exp_provider, exp_model",
        [
            # No override at all: saved provider/model both win.
            (["myagent", "--restart"], "claude-code", "opus"),
            # provider only (== parser default): new provider, model reset.
            (["myagent", "--restart", "--provider", "deepseek"], "deepseek", None),
            # provider only (!= parser default): new provider, model reset.
            (["myagent", "--restart", "--provider", "gemini"], "gemini", None),
            # provider + model: both explicit values win.
            (["myagent", "--restart", "--provider", "deepseek", "--model", "deepseek-chat"],
             "deepseek", "deepseek-chat"),
            # model only: saved provider kept, explicit model wins.
            (["myagent", "--restart", "--model", "sonnet"], "claude-code", "sonnet"),
            # same provider re-passed explicitly + no model: keep saved model
            # (provider did NOT actually change, so nothing stale to drop).
            (["myagent", "--restart", "--provider", "claude-code"], "claude-code", "opus"),
        ],
    )
    def test_override_matrix(self, saved_launch_config, argv, exp_provider, exp_model):
        """Exhaustive override combinations, derived from the real parser.

        Guards against a future refactor silently swallowing an explicit flag
        again — especially `--provider <parser-default>`.
        """
        from deep_worker_cli.cli.agent_command import (
            _explicit_cli_flags,
            merge_restart_config,
        )

        args = _parse(argv)
        explicit = _explicit_cli_flags(argv[1:])  # strip leading "myagent"
        result = merge_restart_config(saved_launch_config, args, explicit=explicit)

        assert result.provider == exp_provider
        assert result.model == exp_model


class TestExplicitCliFlags:
    """_explicit_cli_flags must report presence, not value."""

    def test_provider_equal_to_default_still_detected(self):
        from deep_worker_cli.cli.agent_command import _explicit_cli_flags

        # deepseek IS the parser default, yet it was explicitly passed.
        assert "provider" in _explicit_cli_flags(["--restart", "--provider", "deepseek"])

    def test_absent_flags_not_reported(self):
        from deep_worker_cli.cli.agent_command import _explicit_cli_flags

        flags = _explicit_cli_flags(["myagent", "--restart"])
        assert "provider" not in flags
        assert "model" not in flags

    def test_short_and_long_boolean_flags_detected(self):
        from deep_worker_cli.cli.agent_command import _explicit_cli_flags

        flags = _explicit_cli_flags(["a", "--restart", "-sp", "--model", "x"])
        assert "skip_permissions" in flags
        assert "model" in flags


# ===========================================================================
# stop_agent tests
# ===========================================================================

class TestStopAgent:
    """Tests for stop_agent()."""

    @pytest.mark.asyncio
    async def test_stop_idle_agent(self, temp_data_dir, saved_launch_config):
        """Idle agent should be stopped with SIGTERM."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="idle")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="myagent",
            pid=99999,
            pid_file=Path("/tmp/deep_worker_myagent.pid"),
            working_dir="/path/to/test",
            started_at=datetime.now(),
            rpyc_port=18861,
        )

        with patch("deep_worker_cli.process.agent_stop.os.kill") as mock_kill, \
             patch("deep_worker_cli.process.agent_stop._is_process_alive") as mock_alive:
            # Process dies after SIGTERM
            mock_alive.side_effect = [True, False]
            result = await stop_agent("myagent", registry, data_dir=temp_data_dir)

        assert result is True
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_stop_waiting_input_agent(self, temp_data_dir, saved_launch_config):
        """Agent in waiting_input status should be stoppable."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="waiting_input")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="myagent", pid=99999, pid_file=Path("/tmp/x.pid"),
            working_dir="/path/to/test", started_at=datetime.now(), rpyc_port=18861,
        )

        with patch("deep_worker_cli.process.agent_stop.os.kill"), \
             patch("deep_worker_cli.process.agent_stop._is_process_alive", return_value=False):
            result = await stop_agent("myagent", registry, data_dir=temp_data_dir)

        assert result is True

    @pytest.mark.asyncio
    async def test_error_on_actively_running_agent(self, temp_data_dir, saved_launch_config):
        """Actively running agent (status=running) should raise RuntimeError."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="running")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="myagent", pid=99999, pid_file=Path("/tmp/x.pid"),
            working_dir="/path/to/test", started_at=datetime.now(), rpyc_port=18861,
        )

        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent("myagent", registry, data_dir=temp_data_dir)

    @pytest.mark.asyncio
    async def test_not_running_returns_true(self, temp_data_dir, saved_launch_config):
        """If agent is not in registry, return True (nothing to stop)."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="idle")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = None

        result = await stop_agent("myagent", registry, data_dir=temp_data_dir)
        assert result is True

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self, temp_data_dir, saved_launch_config):
        """If process doesn't die within timeout, raise TimeoutError."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="idle")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="myagent", pid=99999, pid_file=Path("/tmp/x.pid"),
            working_dir="/path/to/test", started_at=datetime.now(), rpyc_port=18861,
        )

        with patch("deep_worker_cli.process.agent_stop.os.kill"), \
             patch("deep_worker_cli.process.agent_stop._is_process_alive", return_value=True), \
             pytest.raises(TimeoutError, match="did not stop"):
            await stop_agent("myagent", registry, data_dir=temp_data_dir, timeout=0.5)

    @pytest.mark.asyncio
    async def test_stop_error_status_agent(self, temp_data_dir, saved_launch_config):
        """Agent in error status should be stoppable."""
        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        _write_metadata(temp_data_dir, saved_launch_config, status="error")

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="myagent", pid=99999, pid_file=Path("/tmp/x.pid"),
            working_dir="/path/to/test", started_at=datetime.now(), rpyc_port=18861,
        )

        with patch("deep_worker_cli.process.agent_stop.os.kill"), \
             patch("deep_worker_cli.process.agent_stop._is_process_alive", return_value=False):
            result = await stop_agent("myagent", registry, data_dir=temp_data_dir)

        assert result is True


# ===========================================================================
# Restart flow validation tests
# ===========================================================================

class TestRestartValidation:
    """Tests for restart_agent() validation logic."""

    def test_no_metadata_file_returns_error(self, temp_data_dir):
        """If session_metadata.json doesn't exist, load_launch_config returns None."""
        from deep_worker_cli.cli.agent_command import load_launch_config

        result = load_launch_config("nonexistent_agent")
        assert result is None

    def test_no_launch_config_in_metadata(self, temp_data_dir):
        """If metadata exists but has no launch_config, returns None."""
        from deep_worker_cli.cli.agent_command import load_launch_config

        # Write old-format metadata with no launch_config
        data_dir = temp_data_dir / "myagent"
        _write_metadata(data_dir, launch_config=None)

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir", return_value=data_dir):
            result = load_launch_config("myagent")

        assert result is None

    def test_valid_launch_config_loaded(self, temp_data_dir, saved_launch_config):
        """Valid launch_config is loaded correctly."""
        from deep_worker_cli.cli.agent_command import load_launch_config

        data_dir = temp_data_dir / "myagent"
        _write_metadata(data_dir, saved_launch_config)

        with patch("deep_worker_cli.cli.agent_command._get_agent_data_dir", return_value=data_dir):
            result = load_launch_config("myagent")

        assert result is not None
        assert result["provider"] == "claude-code"
        assert result["model"] == "opus"


class TestRestartLaunchConfigPersistence:
    """Effective params on restart must be saved as new launch_config."""

    def test_overrides_persisted_in_new_launch_config(self, saved_launch_config, default_args):
        """When restart merges overrides, the new launch_config reflects them."""
        from deep_worker_cli.cli.agent_command import merge_restart_config, build_launch_config

        default_args.model = "sonnet"
        effective = merge_restart_config(saved_launch_config, default_args)

        launch_config = build_launch_config(effective)
        assert launch_config["model"] == "sonnet"
        assert launch_config["provider"] == "claude-code"
        assert launch_config["clean"] is False  # --clean not passed

    def test_clean_explicit_persisted(self, saved_launch_config, default_args):
        from deep_worker_cli.cli.agent_command import merge_restart_config, build_launch_config

        default_args.clean = True
        effective = merge_restart_config(saved_launch_config, default_args)

        launch_config = build_launch_config(effective)
        assert launch_config["clean"] is True

    def test_sequential_restarts_use_latest(self, saved_launch_config, default_args):
        """Simulate two restarts: first overrides model, second uses it."""
        from deep_worker_cli.cli.agent_command import merge_restart_config, build_launch_config

        # First restart: override model
        default_args.model = "sonnet"
        effective1 = merge_restart_config(saved_launch_config, default_args)
        config1 = build_launch_config(effective1)
        assert config1["model"] == "sonnet"

        # Second restart: no model override — should use sonnet from config1
        default_args2 = argparse.Namespace(**vars(default_args))
        default_args2.model = None  # not overridden
        effective2 = merge_restart_config(config1, default_args2)
        assert effective2.model == "sonnet"
