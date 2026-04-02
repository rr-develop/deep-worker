"""Tests for create_model() with claude-code provider.

This test suite ensures that:
1. create_model() accepts provider="claude-code"
2. Simple model names (sonnet, haiku, opus) work correctly
3. Full model names are supported
4. Default model is "sonnet" when model_name=None
5. No API key is required for claude-code
6. ClaudeCodeChatModel is properly instantiated
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# CRITICAL: Add parent directory to sys.path to enable claude_code_provider import
# ============================================================================
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class TestCreateModelClaudeCode:
    """Test create_model() with claude-code provider."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup clean environment for each test."""
        # Clear all API keys to avoid auto-detection interference
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def test_create_model_with_claude_code_sonnet(self, monkeypatch):
        """Should create claude-code model with sonnet."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(provider="claude-code", model_name="sonnet")

            # Should create ClaudeCodeChatModel with sonnet
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "sonnet"
            assert result == mock_model

    def test_create_model_with_claude_code_haiku(self, monkeypatch):
        """Should create claude-code model with haiku."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(provider="claude-code", model_name="haiku")

            # Should create ClaudeCodeChatModel with haiku
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "haiku"
            assert result == mock_model

    def test_create_model_with_claude_code_opus(self, monkeypatch):
        """Should create claude-code model with opus."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(provider="claude-code", model_name="opus")

            # Should create ClaudeCodeChatModel with opus
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "opus"
            assert result == mock_model

    def test_create_model_with_claude_code_full_model_name(self, monkeypatch):
        """Should accept full model name with claude-code."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(
                provider="claude-code",
                model_name="claude-sonnet-4-5-20250929"
            )

            # Should create ClaudeCodeChatModel with full model name
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
            assert result == mock_model

    def test_create_model_claude_code_defaults_to_sonnet(self, monkeypatch):
        """Should default to sonnet when model_name=None with claude-code."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            # model_name=None should default to "sonnet"
            result = create_model(provider="claude-code", model_name=None)

            # Should use default "sonnet"
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "sonnet"
            assert result == mock_model

    def test_create_model_claude_code_no_api_key_required(self, monkeypatch):
        """Should not require API key for claude-code provider."""
        # Remove all API keys
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            # Should succeed without API key
            result = create_model(provider="claude-code", model_name="sonnet")

            # Should create model successfully
            mock_claude.assert_called_once()
            assert result == mock_model

    def test_create_model_claude_code_passes_max_tokens(self, monkeypatch):
        """Should pass max_tokens to ClaudeCodeChatModel."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(provider="claude-code", model_name="sonnet")

            # Should pass max_tokens
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert "max_tokens" in call_kwargs
            assert call_kwargs["max_tokens"] == 20_000  # Default from config
            assert result == mock_model

    def test_create_model_claude_code_inherit_model(self, monkeypatch):
        """Should support 'inherit' model name for claude-code."""
        from deep_worker_cli.config import create_model

        with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
            mock_model = MagicMock()
            mock_claude.return_value = mock_model

            result = create_model(provider="claude-code", model_name="inherit")

            # Should pass "inherit" as model
            mock_claude.assert_called_once()
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "inherit"
            assert result == mock_model

    def test_create_model_auto_detect_claude_code_from_model_name(self, monkeypatch):
        """Should auto-detect claude-code provider from simple model names (sonnet, haiku, opus)."""
        from deep_worker_cli.config import create_model

        # Only simple names should auto-detect as claude-code
        # Full names like "claude-sonnet-4-5-20250929" should detect as anthropic (requires ANTHROPIC_API_KEY)
        test_cases = [
            "sonnet",
            "haiku",
            "opus",
            "inherit",
        ]

        for model_name in test_cases:
            with patch("claude_code_provider.ClaudeCodeChatModel") as mock_claude:
                mock_model = MagicMock()
                mock_claude.return_value = mock_model

                # Only provide model_name, let it auto-detect provider
                result = create_model(provider=None, model_name=model_name)

                # Should auto-detect claude-code and create model
                mock_claude.assert_called_once()
                call_kwargs = mock_claude.call_args[1]
                assert call_kwargs["model"] == model_name
                assert result == mock_model
