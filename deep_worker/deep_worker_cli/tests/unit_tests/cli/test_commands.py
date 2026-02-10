"""Tests for CLI commands."""

from pathlib import Path

import pytest

from deep_worker_cli.cli.agent_command import create_parser as create_agent_parser
from deep_worker_cli.cli.client_command import create_parser as create_client_parser


class TestAgentCommand:
    """Tests for deep-worker agent command."""

    def test_parse_minimal_args(self):
        """Test parsing minimal required arguments."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent"])

        assert args.name == "my-agent"
        assert args.model == "deepseek-chat"
        assert args.provider == "deepseek"
        assert args.dir is None  # Will be set to DEEP_WORKER_ORIGINAL_CWD at runtime
        assert args.config is None
        assert args.skip_permissions is False

    def test_parse_with_model(self):
        """Test parsing with custom model."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent", "--model", "gpt-4", "--provider", "openai"])

        assert args.name == "my-agent"
        assert args.model == "gpt-4"
        assert args.provider == "openai"

    def test_parse_with_dir_option(self):
        """Test parsing with --dir option (like cco)."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent", "--dir", "/tmp/test"])

        assert args.name == "my-agent"
        assert args.dir == Path("/tmp/test")

    def test_parse_with_skip_permissions(self):
        """Test parsing with -sp (skip permissions) flag."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent", "-sp"])

        assert args.name == "my-agent"
        assert args.skip_permissions is True

    def test_parse_skip_permissions_long_form(self):
        """Test parsing with --skip-permissions flag."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent", "--skip-permissions"])

        assert args.name == "my-agent"
        assert args.skip_permissions is True

    def test_parse_with_config(self):
        """Test parsing with config file."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent", "--config", "/path/to/config.yaml"])

        assert args.name == "my-agent"
        assert args.config == Path("/path/to/config.yaml")

    def test_parse_all_args(self):
        """Test parsing all arguments together."""
        parser = create_agent_parser()
        args = parser.parse_args([
            "my-agent",
            "--model", "gpt-4",
            "--provider", "openai",
            "--dir", "/tmp/work",
            "--config", "/tmp/config.yaml",
            "-sp",
        ])

        assert args.name == "my-agent"
        assert args.model == "gpt-4"
        assert args.provider == "openai"
        assert args.dir == Path("/tmp/work")
        assert args.config == Path("/tmp/config.yaml")
        assert args.skip_permissions is True

    def test_default_working_directory_is_none(self):
        """Test that default working directory is None (set at runtime from env)."""
        parser = create_agent_parser()
        args = parser.parse_args(["my-agent"])

        # Should be None so it can be replaced with DEEP_WORKER_ORIGINAL_CWD
        assert args.dir is None


class TestClientCommand:
    """Tests for deep-worker-cli client command."""

    def test_parse_agent_name(self):
        """Test parsing agent name."""
        parser = create_client_parser()
        args = parser.parse_args(["my-agent"])

        assert args.agent_name == "my-agent"
        assert not args.list
        assert args.status is None

    def test_parse_list_flag(self):
        """Test parsing --list flag."""
        parser = create_client_parser()
        args = parser.parse_args(["--list"])

        assert args.list is True
        assert args.agent_name is None

    def test_parse_status_flag(self):
        """Test parsing --status flag."""
        parser = create_client_parser()
        args = parser.parse_args(["--status", "my-agent"])

        assert args.status == "my-agent"
        assert not args.list
        assert args.stop is None

    def test_parse_stop_flag(self):
        """Test parsing --stop flag."""
        parser = create_client_parser()
        args = parser.parse_args(["--stop", "my-agent"])

        assert args.stop == "my-agent"
        assert not args.list
        assert args.status is None

    def test_parse_no_args(self):
        """Test parsing with no arguments."""
        parser = create_client_parser()
        args = parser.parse_args([])

        assert args.agent_name is None
        assert not args.list
        assert args.status is None
        assert args.stop is None
