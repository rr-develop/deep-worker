"""Tests for AgentConfig with claude-code provider support."""

from pathlib import Path

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig


class TestAgentConfigClaudeCode:
    """Test AgentConfig with claude-code provider."""

    def test_agent_config_accepts_claude_code_provider(self, tmp_path):
        """Should accept claude-code as valid provider."""
        config = AgentConfig(
            working_dir=tmp_path,
            model="sonnet",
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.model == "sonnet"
        assert config.api_key is None  # claude-code doesn't need API key

    def test_agent_config_claude_code_with_haiku_model(self, tmp_path):
        """Should accept haiku model with claude-code provider."""
        config = AgentConfig(
            model="haiku",
            working_dir=tmp_path,
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.model == "haiku"
        assert config.api_key is None

    def test_agent_config_claude_code_with_opus_model(self, tmp_path):
        """Should accept opus model with claude-code provider."""
        config = AgentConfig(
            model="opus",
            working_dir=tmp_path,
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.model == "opus"
        assert config.api_key is None

    def test_agent_config_claude_code_with_full_model_name(self, tmp_path):
        """Should accept full model name with claude-code provider."""
        config = AgentConfig(
            model="claude-sonnet-4-5-20250929",
            working_dir=tmp_path,
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.api_key is None

    def test_agent_config_claude_code_with_none_model(self, tmp_path):
        """Should accept None model with claude-code provider (will use default)."""
        config = AgentConfig(
            model=None,
            working_dir=tmp_path,
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.model is None  # Will use default "sonnet" in create_model()
        assert config.api_key is None

    def test_agent_config_claude_code_doesnt_require_api_key(self, tmp_path, monkeypatch):
        """Should not require API key for claude-code provider."""
        # Remove all API keys from environment
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Should succeed without API key
        config = AgentConfig(
            model="sonnet",
            working_dir=tmp_path,
            provider="claude-code"
        )

        assert config.provider == "claude-code"
        assert config.api_key is None

    def test_agent_config_rejects_invalid_provider(self, tmp_path):
        """Should reject invalid provider."""
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                model="some-model",
                working_dir=tmp_path,
                provider="invalid-provider"
            )

        assert "provider must be one of" in str(exc_info.value)
        assert "invalid-provider" in str(exc_info.value)

    def test_agent_config_deepseek_still_requires_api_key(self, tmp_path, monkeypatch):
        """Should still require API key for deepseek provider (default behavior unchanged)."""
        # Remove API key
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        # Should fail without API key
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                model="deepseek-chat",
                working_dir=tmp_path,
                provider="deepseek"
            )

        assert "DEEPSEEK_API_KEY" in str(exc_info.value)

    def test_agent_config_allows_empty_model_string(self, tmp_path):
        """Should reject empty model string."""
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                model="",  # Empty string
                working_dir=tmp_path,
                provider="claude-code"
            )

        assert "model must be a non-empty string or None" in str(exc_info.value)

    def test_agent_config_allows_whitespace_only_model(self, tmp_path):
        """Should reject whitespace-only model string."""
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                model="   ",  # Whitespace only
                working_dir=tmp_path,
                provider="claude-code"
            )

        assert "model must be a non-empty string or None" in str(exc_info.value)
