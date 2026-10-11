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

    def test_claude_code_default_is_none(self):
        """claude-code-sdk must NOT pin a model — the CLI resolves its own default.

        Regression (from a past incident): the hardcoded ``"sonnet"``
        default was wrong. Per
        code.claude.com/docs/en/model-config#default-model-setting the real
        default depends on account type (Opus 5 for Anthropic API / Max /
        Team Premium / Enterprise PAYG / Bedrock / GCP; Sonnet 5 for Pro /
        Team Standard / Enterprise seats; Sonnet 4.5 on Microsoft Foundry).
        deep-worker cannot know the account type, so it must pass no
        ``--model`` at all and let ``claude`` decide — exactly what codex
        and grok already do.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_CODE_SDK_MODEL", None)
            assert get_default_model_name("claude-code-sdk") is None

    def test_claude_code_env_override(self):
        """CLAUDE_CODE_SDK_MODEL pins the model when the operator wants it."""
        with patch.dict(os.environ, {"CLAUDE_CODE_SDK_MODEL": "fable"}):
            assert get_default_model_name("claude-code-sdk") == "fable"

    def test_claude_code_matches_codex_grok_shape(self):
        """claude-code-sdk is a CLI wrapper like codex/grok: no hardcoded pin."""
        with patch.dict(os.environ, {}, clear=False):
            for var in ("CLAUDE_CODE_SDK_MODEL", "CODEX_MODEL", "GROK_MODEL"):
                os.environ.pop(var, None)
            assert get_default_model_name("claude-code-sdk") is None
            assert get_default_model_name("codex") is None
            assert get_default_model_name("grok") is None

    def test_vertexai_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERTEX_AI_MODEL", None)
            result = get_default_model_name("vertexai")
            assert result is not None
            assert "gemini" in result

    def test_unknown_provider(self):
        """Unknown provider returns None."""
        assert get_default_model_name("nonexistent") is None
