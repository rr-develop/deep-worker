"""Tests for AgentConfig."""

import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig


class TestAgentConfigCreation:
    """Tests for AgentConfig creation."""

    def test_create_with_minimal_params(self):
        """Test creating config with minimal required parameters."""
        config = AgentConfig(
            model="deepseek-chat",
            working_dir="/tmp/test",
            api_key="test-api-key",  # Provide test API key
        )

        assert config.model == "deepseek-chat"
        assert config.working_dir == Path("/tmp/test")
        assert config.provider == "deepseek"  # Default provider
        assert config.system_prompt is None
        assert config.tools == []
        assert config.backend_type == "state"
        assert config.api_key == "test-api-key"

    def test_create_with_all_params(self):
        """Test creating config with all parameters."""
        config = AgentConfig(
            model="deepseek-chat",
            working_dir="/tmp/test",
            provider="deepseek",
            system_prompt="Test prompt",
            tools=["tool1", "tool2"],
            backend_type="filesystem",
            max_tokens=10000,
            temperature=0.5,
            api_key="test-api-key",
        )

        assert config.model == "deepseek-chat"
        assert config.working_dir == Path("/tmp/test")
        assert config.provider == "deepseek"
        assert config.system_prompt == "Test prompt"
        assert config.tools == ["tool1", "tool2"]
        assert config.backend_type == "filesystem"
        assert config.max_tokens == 10000
        assert config.temperature == 0.5
        assert config.api_key == "test-api-key"

    def test_working_dir_converted_to_path(self):
        """Test that working_dir is converted to Path."""
        config = AgentConfig(
            model="test-model",
            working_dir="/tmp/test",
        )

        assert isinstance(config.working_dir, Path)
        assert str(config.working_dir) == "/tmp/test"


class TestAgentConfigFromFile:
    """Tests for loading AgentConfig from file."""

    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
model: claude-sonnet-4-5-20250929
working_dir: /tmp/test-project
system_prompt: |
  You are a helpful assistant.
  Help the user with their tasks.
max_tokens: 15000
temperature: 0.7
backend_type: filesystem
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            config = AgentConfig.from_file(config_path)

            assert config.model == "claude-sonnet-4-5-20250929"
            assert config.working_dir == Path("/tmp/test-project")
            assert "helpful assistant" in config.system_prompt
            assert config.max_tokens == 15000
            assert config.temperature == 0.7
            assert config.backend_type == "filesystem"

        finally:
            config_path.unlink()

    def test_load_with_tools(self):
        """Test loading config with tools list."""
        yaml_content = """
model: test-model
working_dir: /tmp/test
tools:
  - tool1
  - tool2
  - tool3
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            config = AgentConfig.from_file(config_path)

            assert config.tools == ["tool1", "tool2", "tool3"]

        finally:
            config_path.unlink()

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_file(Path("/nonexistent/config.yaml"))


class TestAgentConfigValidation:
    """Tests for AgentConfig validation."""

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature"):
            AgentConfig(
                model="test-model",
                working_dir="/tmp/test",
                temperature=2.0,  # Invalid: > 1.0
            )

    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ValueError, match="max_tokens"):
            AgentConfig(
                model="test-model",
                working_dir="/tmp/test",
                max_tokens=-100,  # Invalid: negative
            )

    def test_empty_model(self):
        """Test that empty model raises error."""
        with pytest.raises(ValueError, match="model"):
            AgentConfig(
                model="",  # Invalid: empty
                working_dir="/tmp/test",
            )


class TestAgentConfigDefaults:
    """Tests for AgentConfig default values."""

    def test_default_max_tokens(self):
        """Test default max_tokens value."""
        config = AgentConfig(
            model="test-model",
            working_dir="/tmp/test",
        )

        assert config.max_tokens == 20000

    def test_default_temperature(self):
        """Test default temperature value."""
        config = AgentConfig(
            model="test-model",
            working_dir="/tmp/test",
        )

        assert config.temperature == 1.0

    def test_default_backend_type(self):
        """Test default backend_type value."""
        config = AgentConfig(
            model="test-model",
            working_dir="/tmp/test",
        )

        assert config.backend_type == "state"
