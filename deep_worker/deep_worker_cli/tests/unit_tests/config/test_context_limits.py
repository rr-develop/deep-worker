"""Tests for get_model_context_limit with short model names.

TDD: Verifies that Claude Code short names (opus, sonnet, haiku) resolve
to correct context limits.
"""

import pytest

from deep_worker_cli.config import get_model_context_limit


class TestShortModelNames:
    """Short names used by claude-code provider must resolve correctly."""

    def test_opus_returns_1m(self):
        assert get_model_context_limit("opus", None) == 1_000_000

    def test_sonnet_returns_correct(self):
        result = get_model_context_limit("sonnet", None)
        assert result >= 200_000

    def test_haiku_returns_correct(self):
        result = get_model_context_limit("haiku", None)
        assert result >= 200_000


class TestFullModelNames:
    """Full model names must continue to work."""

    def test_claude_opus_4_6(self):
        assert get_model_context_limit("claude-opus-4.6", None) == 1_000_000

    def test_claude_sonnet_4_5(self):
        assert get_model_context_limit("claude-sonnet-4.5", None) == 200_000

    def test_deepseek_chat(self):
        assert get_model_context_limit("deepseek-chat", None) == 128_000

    def test_deepseek_v4_flash(self):
        assert get_model_context_limit("deepseek-v4-flash", None) == 1_000_000

    def test_deepseek_v4_pro(self):
        assert get_model_context_limit("deepseek-v4-pro", None) == 1_000_000

    def test_unknown_model_returns_default(self):
        assert get_model_context_limit("some-unknown-model", None) == 100_000

    def test_profile_takes_priority(self):
        """model_profile max_input_tokens overrides everything."""
        assert get_model_context_limit("opus", {"max_input_tokens": 500_000}) == 500_000
