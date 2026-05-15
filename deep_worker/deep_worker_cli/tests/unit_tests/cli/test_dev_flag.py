"""Unit tests for the --dev flag in deep-worker.

The --dev flag selects the development codebase instead of the
production tree. The shell launcher
performs the actual tree switch BEFORE Python starts. These tests cover
the Python-side contract:

  1. argparse accepts --dev (so the dev tree's argparse does not reject it
     after the re-exec).
  2. build_launch_config() persists ``dev_tree`` so --restart can find it.
     The value comes from either the parsed flag OR (after re-exec, when
     bash strips the flag from argv) the DEEP_WORKER_DEV_TREE env var.
  3. merge_restart_config() inherits ``dev_tree`` from the saved config
     and honors an explicit --dev override, exactly like every other
     persisted option.

The bash dispatch itself is covered in tests/integration_tests/test_dev_launcher.py.

Design doc: docs/DESIGN_DEV_FLAG.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_launch_config_dev() -> dict:
    """A saved launch_config that was originally started with --dev."""
    return {
        "provider": "claude-code",
        "model": "opus",
        "working_dir": "/path/to/dev-tree",
        "skip_permissions": True,
        "task_manager": True,
        "memory_only": False,
        "use_sqlite": False,
        "clean": False,
        "dev_tree": True,
        "started_at": "2026-05-13T22:00:00.000000",
        "pid": 12345,
        "cli_args": "myagent --dev -sp -tm --provider claude-code --model opus",
    }


@pytest.fixture
def saved_launch_config_prod() -> dict:
    """A saved launch_config that was originally started WITHOUT --dev."""
    return {
        "provider": "claude-code",
        "model": "opus",
        "working_dir": "/path/to/production-tree",
        "skip_permissions": True,
        "task_manager": True,
        "memory_only": False,
        "use_sqlite": False,
        "clean": False,
        "dev_tree": False,
        "started_at": "2026-05-13T22:00:00.000000",
        "pid": 12345,
        "cli_args": "myagent -sp -tm --provider claude-code --model opus",
    }


@pytest.fixture
def default_args() -> argparse.Namespace:
    """argparse namespace with parser defaults."""
    return argparse.Namespace(
        name="myagent",
        dir=None,
        model=None,
        provider="deepseek",
        skip_permissions=False,
        task_manager=False,
        clean=False,
        memory_only=False,
        use_sqlite=False,
        auto_approve_send_message=False,
        dev_tree=False,
        config=None,
        restart=True,
        list_threads=False,
        remove_thread=None,
        show_history=None,
        remove_before=None,
        dry_run=False,
        force=False,
        list_summaries=False,
        format="text",
        with_counts=False,
        agent=None,
        sort="thread_id",
        reverse=False,
    )


# ===========================================================================
# argparse acceptance
# ===========================================================================


class TestArgparseAcceptsDev:
    """The parser must accept --dev so the dev-tree's argparse does not
    reject it after the bash re-exec. The flag has no behavior in
    Python — bash already acted on it — but it must be a known option.
    """

    def test_long_form_accepted(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        ns = parser.parse_args(["myagent", "--dev"])
        assert ns.dev_tree is True

    def test_default_is_false(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        ns = parser.parse_args(["myagent"])
        assert ns.dev_tree is False

    def test_position_independent_before_name(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        ns = parser.parse_args(["--dev", "myagent"])
        assert ns.dev_tree is True
        assert ns.name == "myagent"

    def test_combines_with_other_flags(self) -> None:
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()
        ns = parser.parse_args(
            ["myagent", "--dev", "-sp", "-tm",
             "--provider", "claude-code", "--model", "opus"]
        )
        assert ns.dev_tree is True
        assert ns.skip_permissions is True
        assert ns.task_manager is True
        assert ns.provider == "claude-code"
        assert ns.model == "opus"


# ===========================================================================
# build_launch_config: dev_tree persistence
# ===========================================================================


class TestBuildLaunchConfigDevTree:
    """build_launch_config() must persist dev_tree so that --restart
    can read it back and the bash auto-detect works.

    After the bash re-exec, the --dev flag is stripped from argv, so the
    Python process sees args.dev_tree == False. The env var
    DEEP_WORKER_DEV_TREE=1 carries the signal. The function must honor
    either source.
    """

    def test_persists_dev_tree_true_when_arg_set(self, default_args) -> None:
        from deep_worker_cli.cli.agent_command import build_launch_config

        default_args.dev_tree = True
        # Make sure env doesn't accidentally provide the signal.
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("DEEP_WORKER_DEV_TREE", None)
            cfg = build_launch_config(default_args)

        assert cfg["dev_tree"] is True

    def test_persists_dev_tree_true_when_env_set(self, default_args) -> None:
        """After bash strips --dev from argv and re-execs, the env var
        is the only source of truth. build_launch_config must read it.
        """
        from deep_worker_cli.cli.agent_command import build_launch_config

        default_args.dev_tree = False  # bash stripped the flag
        with patch.dict("os.environ", {"DEEP_WORKER_DEV_TREE": "1"}):
            cfg = build_launch_config(default_args)

        assert cfg["dev_tree"] is True

    def test_persists_dev_tree_false_by_default(self, default_args) -> None:
        from deep_worker_cli.cli.agent_command import build_launch_config

        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("DEEP_WORKER_DEV_TREE", None)
            cfg = build_launch_config(default_args)

        assert cfg["dev_tree"] is False

    def test_env_var_other_value_not_true(self, default_args) -> None:
        """Only DEEP_WORKER_DEV_TREE=1 means True. '0', 'false', '' don't."""
        from deep_worker_cli.cli.agent_command import build_launch_config

        for value in ("0", "false", "False", "", "no"):
            with patch.dict("os.environ", {"DEEP_WORKER_DEV_TREE": value}):
                cfg = build_launch_config(default_args)
            assert cfg["dev_tree"] is False, (
                f"DEEP_WORKER_DEV_TREE={value!r} must not be treated as True"
            )


# ===========================================================================
# merge_restart_config: dev_tree inheritance
# ===========================================================================


class TestMergeRestartConfigDevTree:
    """merge_restart_config() must treat dev_tree like other persisted
    options: inherit from saved config, override on explicit CLI flag.
    """

    def test_inherits_dev_tree_true(
        self, saved_launch_config_dev, default_args
    ) -> None:
        from deep_worker_cli.cli.agent_command import merge_restart_config

        result = merge_restart_config(saved_launch_config_dev, default_args)
        assert result.dev_tree is True

    def test_inherits_dev_tree_false(
        self, saved_launch_config_prod, default_args
    ) -> None:
        from deep_worker_cli.cli.agent_command import merge_restart_config

        result = merge_restart_config(saved_launch_config_prod, default_args)
        assert result.dev_tree is False

    def test_explicit_dev_overrides_saved_false(
        self, saved_launch_config_prod, default_args
    ) -> None:
        """``deep-worker foo --restart --dev`` on a previously prod
        launch must switch to dev tree."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        default_args.dev_tree = True
        result = merge_restart_config(saved_launch_config_prod, default_args)
        assert result.dev_tree is True

    def test_saved_config_without_dev_tree_key_defaults_false(
        self, default_args
    ) -> None:
        """Old session_metadata files predate --dev and lack the key.
        Default to False (= prod tree)."""
        from deep_worker_cli.cli.agent_command import merge_restart_config

        legacy_saved = {
            "provider": "claude-code",
            "model": "opus",
            "working_dir": "/path/to/production-tree",
            "skip_permissions": True,
            "task_manager": False,
            "memory_only": False,
            "use_sqlite": False,
            "clean": False,
            # NOTE: no "dev_tree" key — legacy
            "started_at": "2026-04-01T00:00:00",
            "pid": 1,
            "cli_args": "myagent -sp",
        }
        result = merge_restart_config(legacy_saved, default_args)
        assert result.dev_tree is False
