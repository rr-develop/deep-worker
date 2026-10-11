"""Tests for --no-* flags on --restart: paired disabling options.

Acceptance criteria:
1. For every "enabling" option there is a paired "disabling" option.
2. If a disabling option is given on restart, the corresponding flag
   is removed from the saved agent configuration.
3. The behavior is documented in deep-worker --help.
4. Covered by unit tests.
"""

import argparse

import pytest

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    create_parser,
    merge_restart_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def saved_with_all_true():
    """Saved launch_config where every boolean flag is True."""
    return {
        "provider": "claude-code-sdk",
        "model": "opus",
        "working_dir": "/path/to/production",
        "skip_permissions": True,
        "task_manager": True,
        "memory_only": True,
        "use_sqlite": True,
        "history_memory": True,
        "auto_approve_send_message": True,
        "dev_tree": True,
        "started_at": "2026-04-11T16:59:41.692129",
        "pid": 3726921,
        "cli_args": "myagent -sp -tm -asm --memory-only --use-sqlite -m --dev",
    }


def _parse(argv):
    """Parse argv with the real CLI parser."""
    return create_parser().parse_args(argv)


# ===========================================================================
# Parser: --no-* flags are accepted
# ===========================================================================

class TestParserAcceptsNoFlags:
    """argparse must accept --no-* variants without error."""

    def test_no_skip_permissions(self):
        args = _parse(["a", "--no-skip-permissions"])
        assert args.skip_permissions is False

    def test_no_task_manager(self):
        args = _parse(["a", "--no-task-manager"])
        assert args.task_manager is False

    def test_no_tm_short_alias(self):
        """--no-tm is an alias for --no-task-manager."""
        args = _parse(["a", "--no-tm"])
        assert args.task_manager is False

    def test_no_auto_approve_send_message(self):
        args = _parse(["a", "--no-auto-approve-send-message"])
        assert args.auto_approve_send_message is False

    def test_no_memory_only(self):
        args = _parse(["a", "--no-memory-only"])
        assert args.memory_only is False

    def test_no_use_sqlite(self):
        args = _parse(["a", "--no-use-sqlite"])
        assert args.use_sqlite is False

    def test_no_history_memory(self):
        args = _parse(["a", "--no-history-memory"])
        assert args.history_memory is False

    def test_no_m_short_alias(self):
        """--no-m is an alias for --no-history-memory."""
        args = _parse(["a", "--no-m"])
        assert args.history_memory is False

    def test_no_dev(self):
        args = _parse(["a", "--no-dev"])
        assert args.dev_tree is False

    def test_positive_and_no_are_independent_actions(self):
        """Both --sp and --no-skip-permissions can be present; last wins."""
        args = _parse(["a", "-sp", "--no-skip-permissions"])
        assert args.skip_permissions is False

        args = _parse(["a", "--no-skip-permissions", "-sp"])
        assert args.skip_permissions is True

    def test_memory_only_and_no_memory_only_not_mutually_exclusive(self):
        """--no-memory-only lives outside the mutually exclusive group."""
        args = _parse(["a", "--no-memory-only", "-sp"])
        assert args.memory_only is False


# ===========================================================================
# _explicit_cli_flags: --no-* flags are detected as explicit
# ===========================================================================

class TestExplicitCliFlagsDetectsNoFlags:
    """_explicit_cli_flags must report --no-* dests in the explicit set."""

    def test_no_skip_permissions_detected(self):
        flags = _explicit_cli_flags(["--no-skip-permissions"])
        assert "skip_permissions" in flags

    def test_no_task_manager_detected(self):
        flags = _explicit_cli_flags(["--no-task-manager"])
        assert "task_manager" in flags

    def test_no_tm_alias_detected(self):
        flags = _explicit_cli_flags(["--no-tm"])
        assert "task_manager" in flags

    def test_no_auto_approve_send_message_detected(self):
        flags = _explicit_cli_flags(["--no-auto-approve-send-message"])
        assert "auto_approve_send_message" in flags

    def test_no_memory_only_detected(self):
        flags = _explicit_cli_flags(["--no-memory-only"])
        assert "memory_only" in flags

    def test_no_use_sqlite_detected(self):
        flags = _explicit_cli_flags(["--no-use-sqlite"])
        assert "use_sqlite" in flags

    def test_no_history_memory_detected(self):
        flags = _explicit_cli_flags(["--no-history-memory"])
        assert "history_memory" in flags

    def test_no_m_alias_detected(self):
        flags = _explicit_cli_flags(["--no-m"])
        assert "history_memory" in flags

    def test_no_dev_detected(self):
        flags = _explicit_cli_flags(["--no-dev"])
        assert "dev_tree" in flags


# ===========================================================================
# merge_restart_config: --no-* overrides a saved True value
# ===========================================================================

class TestMergeRestartConfigNoFlags:
    """--no-* on --restart must disable a saved True flag."""

    def test_no_skip_permissions_disables_saved_true(self, saved_with_all_true):
        """Saved skip_permissions=True, --no-skip-permissions → False."""
        args = _parse(["a", "--restart", "--no-skip-permissions"])
        explicit = {"skip_permissions"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.skip_permissions is False, (
            "--no-skip-permissions should override saved True"
        )

    def test_no_task_manager_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-task-manager"])
        explicit = {"task_manager"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.task_manager is False

    def test_no_tm_alias_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-tm"])
        explicit = {"task_manager"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.task_manager is False

    def test_no_auto_approve_send_message_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-auto-approve-send-message"])
        explicit = {"auto_approve_send_message"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.auto_approve_send_message is False

    def test_no_memory_only_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-memory-only"])
        explicit = {"memory_only"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.memory_only is False

    def test_no_use_sqlite_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-use-sqlite"])
        explicit = {"use_sqlite"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.use_sqlite is False

    def test_no_history_memory_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-history-memory"])
        explicit = {"history_memory"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.history_memory is False

    def test_no_m_alias_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-m"])
        explicit = {"history_memory"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.history_memory is False

    def test_no_dev_disables_saved_true(self, saved_with_all_true):
        args = _parse(["a", "--restart", "--no-dev"])
        explicit = {"dev_tree"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)
        assert result.dev_tree is False

    def test_all_no_flags_together_disables_all(self, saved_with_all_true):
        """Test that all --no-* flags together correctly disable everything."""
        args = _parse([
            "a", "--restart",
            "--no-skip-permissions",
            "--no-task-manager",
            "--no-auto-approve-send-message",
            "--no-memory-only",
            "--no-use-sqlite",
            "--no-history-memory",
            "--no-dev",
        ])
        explicit = {
            "skip_permissions", "task_manager", "auto_approve_send_message",
            "memory_only", "use_sqlite", "history_memory", "dev_tree",
        }
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)

        assert result.skip_permissions is False
        assert result.task_manager is False
        assert result.auto_approve_send_message is False
        assert result.memory_only is False
        assert result.use_sqlite is False
        assert result.history_memory is False
        assert result.dev_tree is False

    def test_no_flag_only_overrides_saved_not_other_saved_values(self, saved_with_all_true):
        """--no-sp should only affect skip_permissions, keep other True values."""
        args = _parse(["a", "--restart", "--no-skip-permissions"])
        explicit = {"skip_permissions"}
        result = merge_restart_config(saved_with_all_true, args, explicit=explicit)

        assert result.skip_permissions is False
        # Everything else stays True from saved config
        assert result.task_manager is True
        assert result.auto_approve_send_message is True
        assert result.memory_only is True
        assert result.use_sqlite is True
        assert result.history_memory is True
        assert result.dev_tree is True


# ===========================================================================
# Integration: _explicit_cli_flags + merge_restart_config end-to-end
# ===========================================================================

class TestNoFlagsEndToEnd:
    """Full path: real CLI flags → _explicit_cli_flags → merge_restart_config."""

    def _merge(self, argv: list[str], saved: dict):
        """Parse argv, detect explicit flags, and merge with saved config."""
        from deep_worker_cli.cli.agent_command import (
            _explicit_cli_flags,
            merge_restart_config,
        )
        args = _parse(argv)
        explicit = _explicit_cli_flags(argv[1:])
        return merge_restart_config(saved, args, explicit=explicit)

    def test_no_skip_permissions_end_to_end(self, saved_with_all_true):
        result = self._merge(["a", "--restart", "--no-skip-permissions"], saved_with_all_true)
        assert result.skip_permissions is False
        assert result.task_manager is True  # rest preserved

    def test_no_dev_end_to_end(self, saved_with_all_true):
        result = self._merge(["a", "--restart", "--no-dev"], saved_with_all_true)
        assert result.dev_tree is False

    def test_positive_flag_still_works_end_to_end(self, saved_with_all_true):
        """-sp should still enable skip_permissions even when saved=False."""
        saved_false = {**saved_with_all_true, "skip_permissions": False}
        result = self._merge(["a", "--restart", "-sp"], saved_false)
        assert result.skip_permissions is True
