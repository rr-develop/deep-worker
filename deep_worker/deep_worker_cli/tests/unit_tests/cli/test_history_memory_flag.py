"""Unit tests for CLI flags related to history memory (--history-memory, -m)."""

from __future__ import annotations

import argparse

import pytest

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    create_parser,
)


class TestHistoryMemoryCliFlag:
    """Tests for the --history-memory / -m CLI flag parsing."""

    def test_long_flag_sets_history_memory(self) -> None:
        """--history-memory should set history_memory=True."""
        parser = create_parser()
        args = parser.parse_args(["--history-memory"])
        assert args.history_memory is True

    def test_short_flag_sets_history_memory(self) -> None:
        """-m should set history_memory=True."""
        parser = create_parser()
        args = parser.parse_args(["-m"])
        assert args.history_memory is True

    def test_both_flags_set_history_memory(self) -> None:
        """Both -m and --history-memory should work together."""
        parser = create_parser()
        args = parser.parse_args(["-m", "--history-memory"])
        assert args.history_memory is True

    def test_default_is_false(self) -> None:
        """Without the flag, history_memory should be False."""
        parser = create_parser()
        args = parser.parse_args(["agent_name"])
        assert args.history_memory is False

    def test_history_memory_does_not_conflict_with_name(self) -> None:
        """-m should not consume the positional name argument."""
        parser = create_parser()
        args = parser.parse_args(["my_agent", "-m"])
        assert args.name == "my_agent"
        assert args.history_memory is True

    def test_explicit_flags_detects_long(self) -> None:
        """_explicit_cli_flags should detect --history-memory."""
        flags = _explicit_cli_flags(["--history-memory"])
        assert "history_memory" in flags

    def test_explicit_flags_detects_short(self) -> None:
        """_explicit_cli_flags should detect -m."""
        flags = _explicit_cli_flags(["-m"])
        assert "history_memory" in flags

    def test_explicit_flags_empty_when_no_flag(self) -> None:
        """_explicit_cli_flags should not contain history_memory when not passed."""
        flags = _explicit_cli_flags(["agent_name"])
        assert "history_memory" not in flags
