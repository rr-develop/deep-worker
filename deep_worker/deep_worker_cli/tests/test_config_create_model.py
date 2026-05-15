"""Tests for create_model() function with provider and model_name parameters.

This test suite ensures that:
1. create_model() accepts provider and model_name parameters
2. Explicit provider/model take precedence over auto-detection
3. Auto-detection still works when parameters not provided (backward compatibility)
4. model_name_override parameter works for backward compatibility
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCreateModelWithParameters:
    """Test create_model() with explicit provider and model_name parameters."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup clean environment for each test."""
        # Clear all API keys to avoid auto-detection interference
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    def test_create_model_with_explicit_deepseek(self, monkeypatch):
        """Should create deepseek model when provider and model_name are explicit.

        `create_model` reads the API key from the module-level
        `settings` singleton (`settings.deepseek_api_key`), which is
        populated on first import from `~/.deepagents/.env`. A plain
        `monkeypatch.setenv(...)` does NOT retroactively update a
        singleton that's already been built, so tests must patch
        `deep_worker_cli.config.settings` directly.
        """
        from deep_worker_cli.config import create_model

        with patch("deep_worker_cli.config.settings") as mock_settings:
            mock_settings.has_deepseek = True
            mock_settings.deepseek_api_key = "sk-test-deepseek"
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            mock_settings.google_api_key = None
            mock_settings.google_cloud_project = None
            mock_settings.model_name = None
            mock_settings.model_provider = None

            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                mock_model = MagicMock()
                mock_openai.return_value = mock_model

                result = create_model(provider="deepseek", model_name="deepseek-chat")

                # Should create ChatOpenAI with deepseek config
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["model"] == "deepseek-chat"
                assert call_kwargs["base_url"] == "https://api.deepseek.com"
                assert call_kwargs["api_key"] == "sk-test-deepseek"
                assert result == mock_model

    def test_create_model_with_explicit_openai(self, monkeypatch):
        """Should create openai model when provider and model_name are explicit."""
        from deep_worker_cli.config import create_model

        # Mock settings to have OpenAI API key
        with patch("deep_worker_cli.config.settings") as mock_settings:
            mock_settings.has_openai = True
            mock_settings.openai_api_key = "sk-test-openai"

            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                mock_model = MagicMock()
                mock_openai.return_value = mock_model

                result = create_model(provider="openai", model_name="gpt-4o")

                # Should create ChatOpenAI with openai config
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["model"] == "gpt-4o"
                # Note: OpenAI doesn't use base_url in the current implementation
                assert result == mock_model

    def test_create_model_with_explicit_anthropic(self, monkeypatch):
        """Should create anthropic model when provider and model_name are explicit."""
        from deep_worker_cli.config import create_model

        # Mock settings to have Anthropic API key
        with patch("deep_worker_cli.config.settings") as mock_settings:
            mock_settings.has_anthropic = True
            mock_settings.anthropic_api_key = "sk-test-anthropic"

            with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
                mock_model = MagicMock()
                mock_anthropic.return_value = mock_model

                result = create_model(provider="anthropic", model_name="claude-sonnet-4-5-20250929")

                # Should create ChatAnthropic
                mock_anthropic.assert_called_once()
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["model_name"] == "claude-sonnet-4-5-20250929"
                assert call_kwargs["max_tokens"] == 20_000
                assert result == mock_model

    def test_create_model_without_parameters_uses_autodetection(self, monkeypatch):
        """Should use auto-detection when provider and model_name not provided."""
        # Set API key for auto-detection
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

        from deep_worker_cli.config import create_model

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_model = MagicMock()
            mock_openai.return_value = mock_model

            # Call without parameters - should auto-detect deepseek
            result = create_model()

            # Should auto-detect and use deepseek
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "deepseek" in call_kwargs["model"]
            assert result == mock_model

    def test_create_model_explicit_takes_precedence_over_autodetection(self, monkeypatch):
        """Should use explicit provider even when auto-detection would choose different."""
        from deep_worker_cli.config import create_model

        # Mock settings to have multiple API keys
        with patch("deep_worker_cli.config.settings") as mock_settings:
            mock_settings.has_deepseek = True
            mock_settings.has_openai = True
            mock_settings.deepseek_api_key = "sk-test-deepseek"
            mock_settings.openai_api_key = "sk-test-openai"

            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                mock_model = MagicMock()
                mock_openai.return_value = mock_model

                # Explicitly request openai (auto-detection would choose deepseek)
                result = create_model(provider="openai", model_name="gpt-4o")

                # Should use openai, not deepseek
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["model"] == "gpt-4o"
                # Check it's not deepseek by verifying no base_url (or check it's standard openai)
                assert result == mock_model

    def test_create_model_backward_compatibility_model_name_override(self, monkeypatch):
        """Should support model_name_override parameter for backward compatibility."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

        from deep_worker_cli.config import create_model

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_model = MagicMock()
            mock_openai.return_value = mock_model

            # Use old parameter name
            result = create_model(model_name_override="deepseek-chat")

            # Should still work
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "deepseek-chat"
            assert result == mock_model

    def test_create_model_fails_when_explicit_provider_has_no_api_key(self, monkeypatch):
        """Should fail with clear error when explicit provider lacks API key."""
        from deep_worker_cli.config import create_model

        # Mock settings to NOT have API key
        with patch("deep_worker_cli.config.settings") as mock_settings:
            mock_settings.has_deepseek = False
            mock_settings.deepseek_api_key = None

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                create_model(provider="deepseek", model_name="deepseek-chat")

            assert exc_info.value.code == 1

    def test_create_model_model_name_takes_precedence_over_override(self, monkeypatch):
        """Should prefer model_name over model_name_override when both provided."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

        from deep_worker_cli.config import create_model

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_model = MagicMock()
            mock_openai.return_value = mock_model

            # Both parameters provided - model_name should win
            result = create_model(
                provider="deepseek",
                model_name="deepseek-chat",
                model_name_override="ignored-model"
            )

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "deepseek-chat"  # Not "ignored-model"
            assert result == mock_model


class TestCreateModelProviderValidation:
    """Test provider validation in create_model()."""

    def test_create_model_with_none_provider_uses_autodetection(self, monkeypatch):
        """Should auto-detect when provider=None."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        from deep_worker_cli.config import create_model

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # provider=None should trigger auto-detection
            create_model(provider=None, model_name="deepseek-chat")

            # Should auto-detect deepseek
            assert mock_openai.called

    def test_create_model_with_none_model_name_uses_env_default(self, monkeypatch):
        """Should use environment default when model_name=None."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")

        from deep_worker_cli.config import create_model

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # model_name=None should use DEEPSEEK_MODEL env var
            create_model(provider="deepseek", model_name=None)

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "deepseek-reasoner"
