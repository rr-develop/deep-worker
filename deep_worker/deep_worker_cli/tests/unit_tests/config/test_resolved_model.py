"""Tests for AgentConfig.resolved_model property."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig


class TestResolvedModel:
    """Tests for the resolved_model property."""

    def test_returns_explicit_model(self):
        """When model is set explicitly, resolved_model returns it."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            config = AgentConfig(working_dir=Path("/tmp"), model="deepseek-v3", provider="deepseek")
        assert config.resolved_model == "deepseek-v3"

    def test_returns_default_when_model_is_none(self):
        """When model is None, resolved_model returns provider default."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            os.environ.pop("DEEPSEEK_MODEL", None)
            config = AgentConfig(working_dir=Path("/tmp"), model=None, provider="deepseek")
        assert config.resolved_model == "deepseek-chat"

    def test_respects_env_override(self):
        """When model is None, resolved_model checks env var first."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "deepseek-custom"}):
            config = AgentConfig(working_dir=Path("/tmp"), model=None, provider="deepseek")
            assert config.resolved_model == "deepseek-custom"

    def test_claude_code_default(self):
        """Claude-code provider returns 'sonnet' as default."""
        config = AgentConfig(working_dir=Path("/tmp"), model=None, provider="claude-code")
        assert config.resolved_model == "sonnet"
