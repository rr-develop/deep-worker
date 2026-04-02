"""Unit tests for summarization configuration logic.

This test suite validates the automatic configuration of SummarizationMiddleware
based on model context limits for different AI providers.

Following TDD approach: tests written first (RED), then implementation (GREEN).
"""

from __future__ import annotations

import pytest

from deep_worker_cli.config import (
    PROVIDER_CONTEXT_LIMITS,
    SummarizationConfig,
    get_model_context_limit,
    get_summarization_config,
)


class TestProviderContextLimits:
    """Tests for PROVIDER_CONTEXT_LIMITS mapping."""

    def test_claude_opus_4_6_limit(self) -> None:
        """Test that Claude Opus 4.6 has 1M context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-opus-4.6"] == 1_000_000

    def test_claude_sonnet_4_6_limit(self) -> None:
        """Test that Claude Sonnet 4.6 has 1M context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-sonnet-4.6"] == 1_000_000

    def test_claude_sonnet_4_5_limit(self) -> None:
        """Test that Claude Sonnet 4.5 has 200K context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-sonnet-4.5"] == 200_000

    def test_claude_sonnet_4_limit(self) -> None:
        """Test that Claude Sonnet 4 has 200K context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-sonnet-4"] == 200_000

    def test_deepseek_chat_limit(self) -> None:
        """Test that DeepSeek chat (V3.2) has 128K context."""
        assert PROVIDER_CONTEXT_LIMITS["deepseek-chat"] == 128_000

    def test_deepseek_v3_limit(self) -> None:
        """Test that DeepSeek V3 has 128K context."""
        assert PROVIDER_CONTEXT_LIMITS["deepseek-v3"] == 128_000

    def test_deepseek_r1_limit(self) -> None:
        """Test that DeepSeek R1 has 64K context."""
        assert PROVIDER_CONTEXT_LIMITS["deepseek-r1"] == 64_000

    def test_gpt_4_limit(self) -> None:
        """Test that GPT-4 has 128K context."""
        assert PROVIDER_CONTEXT_LIMITS["gpt-4"] == 128_000

    def test_gpt_35_turbo_limit(self) -> None:
        """Test that GPT-3.5 Turbo has 16K context."""
        assert PROVIDER_CONTEXT_LIMITS["gpt-3.5-turbo"] == 16_385

    def test_default_fallback(self) -> None:
        """Test that default fallback is 100K."""
        assert PROVIDER_CONTEXT_LIMITS["default"] == 100_000


class TestGetModelContextLimit:
    """Tests for get_model_context_limit() function."""

    def test_from_model_profile_priority(self) -> None:
        """Test that model profile takes priority over known models."""
        profile = {"max_input_tokens": 150_000}
        limit = get_model_context_limit("claude-sonnet-4.5", profile)
        # Profile value (150K) overrides known value (200K)
        assert limit == 150_000

    def test_from_known_model_exact_match(self) -> None:
        """Test exact model name match."""
        limit = get_model_context_limit("claude-sonnet-4.5", None)
        assert limit == 200_000

    def test_from_known_model_partial_match(self) -> None:
        """Test partial model name match (provider prefix)."""
        limit = get_model_context_limit("anthropic:claude-sonnet-4.5", None)
        assert limit == 200_000

    def test_deepseek_with_provider_prefix(self) -> None:
        """Test DeepSeek model with provider prefix."""
        limit = get_model_context_limit("deepseek:deepseek-chat", None)
        assert limit == 128_000

    def test_fallback_unknown_model(self) -> None:
        """Test fallback for completely unknown model."""
        limit = get_model_context_limit("unknown-model-xyz", None)
        assert limit == 100_000

    def test_fallback_none_inputs(self) -> None:
        """Test fallback when both inputs are None."""
        limit = get_model_context_limit(None, None)
        assert limit == 100_000

    def test_openrouter_model_extraction(self) -> None:
        """Test extracting model name from OpenRouter format."""
        limit = get_model_context_limit("openrouter/anthropic/claude-sonnet-4.5", None)
        # Should find "claude-sonnet-4.5" and return 200K
        assert limit == 200_000


class TestGetSummarizationConfig:
    """Tests for get_summarization_config() function."""

    def test_1m_context_model(self) -> None:
        """Test config for 1M token context models (Claude Opus 4.6)."""
        config = get_summarization_config(1_000_000)
        assert config.trigger_fraction == 0.85
        assert config.keep_fraction == 0.10
        assert config.context_limit == 1_000_000
        assert config.trigger_tokens == 850_000
        assert config.keep_tokens == 100_000

    def test_400k_context_model(self) -> None:
        """Test config for 400K token context models (GPT-5)."""
        config = get_summarization_config(400_000)
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.12
        assert config.trigger_tokens == 320_000
        assert config.keep_tokens == 48_000

    def test_200k_context_model(self) -> None:
        """Test config for 200K token context models (Claude 4.5)."""
        config = get_summarization_config(200_000)
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.15
        assert config.trigger_tokens == 160_000
        assert config.keep_tokens == 30_000

    def test_128k_context_model_deepseek(self) -> None:
        """Test config for 128K token context models (DeepSeek V3)."""
        config = get_summarization_config(128_000)
        assert config.trigger_fraction == 0.75
        assert config.keep_fraction == 0.20
        assert config.trigger_tokens == 96_000
        assert config.keep_tokens == 25_600

    def test_64k_context_model(self) -> None:
        """Test config for 64K token context models (DeepSeek R1)."""
        config = get_summarization_config(64_000)
        assert config.trigger_fraction == 0.70
        assert config.keep_fraction == 0.25
        assert config.trigger_tokens == 44_800
        assert config.keep_tokens == 16_000

    def test_16k_context_model_gpt35(self) -> None:
        """Test config for small context models (GPT-3.5 Turbo)."""
        config = get_summarization_config(16_385)
        assert config.trigger_fraction == 0.60
        assert config.keep_fraction == 0.30
        # Check approximate values (rounding may occur)
        assert 9_800 <= config.trigger_tokens <= 9_900
        assert 4_900 <= config.keep_tokens <= 5_000

    def test_very_small_context_model(self) -> None:
        """Test config for very small context models (<16K)."""
        config = get_summarization_config(8_000)
        assert config.trigger_fraction == 0.50
        assert config.keep_fraction == 0.35
        assert config.trigger_tokens == 4_000
        assert config.keep_tokens == 2_800

    def test_override_trigger(self) -> None:
        """Test that override_trigger parameter works."""
        config = get_summarization_config(200_000, override_trigger=0.90)
        assert config.trigger_fraction == 0.90
        assert config.trigger_tokens == 180_000
        # Keep fraction not overridden
        assert config.keep_fraction == 0.15

    def test_override_keep(self) -> None:
        """Test that override_keep parameter works."""
        config = get_summarization_config(200_000, override_keep=0.25)
        # Trigger fraction not overridden
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.25
        assert config.keep_tokens == 50_000

    def test_override_both(self) -> None:
        """Test that both overrides work together."""
        config = get_summarization_config(
            128_000,
            override_trigger=0.85,
            override_keep=0.10,
        )
        assert config.trigger_fraction == 0.85
        assert config.keep_fraction == 0.10
        assert config.trigger_tokens == 108_800
        assert config.keep_tokens == 12_800

    def test_truncation_settings_large_context(self) -> None:
        """Test argument truncation settings for large context (1M+)."""
        config = get_summarization_config(1_000_000)
        assert config.truncate_after_messages == 20
        assert config.max_arg_length == 500

    def test_truncation_settings_medium_context(self) -> None:
        """Test argument truncation settings for medium context (64K-200K)."""
        config = get_summarization_config(128_000)
        assert config.truncate_after_messages == 15
        assert config.max_arg_length == 300

    def test_truncation_settings_small_context(self) -> None:
        """Test argument truncation settings for small context (<64K)."""
        config = get_summarization_config(32_000)
        assert config.truncate_after_messages == 10
        assert config.max_arg_length == 200

    def test_none_context_limit_fallback(self) -> None:
        """Test that None context limit uses fallback (100K)."""
        config = get_summarization_config(None)
        assert config.context_limit == 100_000
        # 100K falls into 64-128K range, so trigger=0.70, keep=0.25
        assert config.trigger_fraction == 0.70
        assert config.keep_fraction == 0.25
        assert config.trigger_tokens == 70_000
        assert config.keep_tokens == 25_000

    def test_model_name_parameter_logged(self) -> None:
        """Test that model_name parameter is accepted (for future use)."""
        # model_name is accepted but currently not used for logic
        # (logic based purely on context_limit)
        config = get_summarization_config(200_000, model_name="claude-sonnet-4.5")
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.15


class TestSummarizationConfig:
    """Tests for SummarizationConfig dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test that SummarizationConfig has all required fields."""
        config = SummarizationConfig(
            trigger_fraction=0.80,
            keep_fraction=0.15,
            truncate_after_messages=20,
            max_arg_length=500,
            context_limit=200_000,
            trigger_tokens=160_000,
            keep_tokens=30_000,
        )
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.15
        assert config.truncate_after_messages == 20
        assert config.max_arg_length == 500
        assert config.context_limit == 200_000
        assert config.trigger_tokens == 160_000
        assert config.keep_tokens == 30_000

    def test_dataclass_immutable(self) -> None:
        """Test that SummarizationConfig is frozen (immutable)."""
        config = SummarizationConfig(
            trigger_fraction=0.80,
            keep_fraction=0.15,
            truncate_after_messages=20,
            max_arg_length=500,
            context_limit=200_000,
            trigger_tokens=160_000,
            keep_tokens=30_000,
        )
        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(AttributeError):
            config.trigger_fraction = 0.90  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
