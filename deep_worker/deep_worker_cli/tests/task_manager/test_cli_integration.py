"""Tests for TaskManager CLI flag integration."""

import pytest

from deep_worker_cli.cli.agent_command import create_parser


def test_task_manager_flag_default():
    """Test that -tm flag defaults to False."""
    parser = create_parser()
    args = parser.parse_args(["test_agent"])

    assert hasattr(args, "task_manager")
    assert args.task_manager is False


def test_task_manager_flag_short():
    """Test that -tm flag enables TaskManager."""
    parser = create_parser()
    args = parser.parse_args(["test_agent", "-tm"])

    assert hasattr(args, "task_manager")
    assert args.task_manager is True


def test_task_manager_flag_long():
    """Test that --task-manager flag enables TaskManager."""
    parser = create_parser()
    args = parser.parse_args(["test_agent", "--task-manager"])

    assert hasattr(args, "task_manager")
    assert args.task_manager is True


def test_task_manager_with_other_flags():
    """Test that -tm works with other flags."""
    parser = create_parser()
    args = parser.parse_args([
        "test_agent",
        "--dir", "/tmp/test",
        "-sp",
        "-tm",
        "--model", "gpt-4"
    ])

    assert args.task_manager is True
    assert args.skip_permissions is True
    assert args.model == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
