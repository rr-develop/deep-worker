"""Tests for get_default_model_name() function.

TDD: Verifies that provider→default model mapping is correct
and respects environment variable overrides.
"""

import os
from unittest.mock import patch

import pytest

from deep_worker_cli.config import get_default_model_name


class TestGetDefaultModelName:
    """Tests for the centralized provider→model mapping."""

    def test_deepseek_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPSEEK_MODEL", None)
            assert get_default_model_name("deepseek") == "deepseek-v4-flash"

    def test_deepseek_env_override(self):
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "deepseek-v3"}):
            assert get_default_model_name("deepseek") == "deepseek-v3"

    def test_openai_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_MODEL", None)
            result = get_default_model_name("openai")
            assert result is not None
            assert len(result) > 0

    def test_openai_env_override(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}):
            assert get_default_model_name("openai") == "gpt-4o"

    def test_anthropic_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_MODEL", None)
            result = get_default_model_name("anthropic")
            assert result is not None
            assert "claude" in result

    def test_google_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_MODEL", None)
            result = get_default_model_name("google")
            assert result is not None
            assert "gemini" in result

    def test_claude_code_default(self):
        assert get_default_model_name("claude-code") == "sonnet"

    def test_vertexai_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERTEX_AI_MODEL", None)
            result = get_default_model_name("vertexai")
            assert result is not None
            assert "gemini" in result

    def test_unknown_provider(self):
        """Unknown provider returns None."""
        assert get_default_model_name("nonexistent") is None
