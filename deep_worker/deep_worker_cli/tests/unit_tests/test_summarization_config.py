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
    """Tests for PROVIDER_CONTEXT_LIMITS mapping.

    Architectural shift (rg1_worker 2026-08-18 → 2026-08-19):
    ---------------------------------------------------------
    As of the 2026-08-19 fix, PROVIDER_CONTEXT_LIMITS contains
    ONLY the ``default`` entry. Hand-curated values for specific
    models are NOT in this dict any more — they come from:

     1. ``model_profile["max_input_tokens"]`` (priority 1, the
        LangChain SDK's authoritative per-model value, kept
        current by the SDK maintainers on every release).
     2. The provider-keyed table in
        ``context_limit/provider_max.py`` (built from the same
        LangChain SDK profile data — never hand-curated).

    These tests pin the new architecture: PROVIDER_CONTEXT_LIMITS
    is a thin fallback shell, not a hand-curated model map.
    """

    def test_dict_only_has_default_fallback(self) -> None:
        """The dict contains ONLY the ``default`` key. Every other
        value the CLI uses is sourced from LangChain SDK profile
        data at runtime, not from a hand-curated entry here.
        """
        assert set(PROVIDER_CONTEXT_LIMITS.keys()) == {"default"}

    def test_default_fallback_is_100k(self) -> None:
        """Default fallback is 100K, so summarization does not
        trigger spuriously on long conversations for unknown
        models."""
        assert PROVIDER_CONTEXT_LIMITS["default"] == 100_000

    def test_no_family_aliases_in_dict(self) -> None:
        """Family aliases (fable, opus, sonnet, haiku, mythos) MUST
        NOT appear in this dict. Per the operator contract,
        family aliases refer to the latest available model in
        the family, which is dynamic — we cannot statically
        encode its context window.
        """
        for alias in ("fable", "opus", "sonnet", "haiku", "mythos"):
            assert alias not in PROVIDER_CONTEXT_LIMITS, (
                f"family alias {alias!r} in PROVIDER_CONTEXT_LIMITS. "
                f"Family aliases must resolve via model_profile (the "
                f"current generation's value), not via a static dict "
                f"entry. The rg1_worker 2026-08-18 lesson."
            )


class TestGetModelContextLimit:
    """Tests for get_model_context_limit() function."""

    def test_from_model_profile_priority(self) -> None:
        """Test that model profile takes priority over the static
        dict and the default fallback. The profile is the
        authoritative runtime source — it comes from the
        LangChain SDK's model introspection and is kept
        current on every SDK release.
        """
        profile = {"max_input_tokens": 150_000}
        limit = get_model_context_limit("claude-sonnet-4.5", profile)
        # Profile value (150K) wins over the default fallback (100K).
        assert limit == 150_000

    def test_profile_overrides_default_for_unknown_model(self) -> None:
        """An unknown model name with a profile → profile wins."""
        profile = {"max_input_tokens": 64_000}
        limit = get_model_context_limit("totally-unknown-xyz", profile)
        assert limit == 64_000

    def test_fallback_unknown_model(self) -> None:
        """Test fallback for completely unknown model (no profile)."""
        limit = get_model_context_limit("unknown-model-xyz", None)
        assert limit == 100_000

    def test_fallback_none_inputs(self) -> None:
        """Test fallback when both inputs are None."""
        limit = get_model_context_limit(None, None)
        assert limit == 100_000

    def test_empty_profile_does_not_override_default(self) -> None:
        """An empty profile dict (no ``max_input_tokens`` key) →
        falls through to the default fallback."""
        limit = get_model_context_limit("anything", {})
        assert limit == 100_000

    def test_profile_with_non_int_max_input_tokens_falls_through(self) -> None:
        """A profile with ``max_input_tokens`` set to a non-int
        (e.g. ``None``) falls through to the default. This is
        the rg1_worker 2026-08-18 belt-and-braces guard: the
        downstream code only accepts ints, so an unexpected
        type from a future SDK is silently ignored rather than
        corrupting the trigger calculation."""
        limit = get_model_context_limit("anything", {"max_input_tokens": None})
        assert limit == 100_000
        limit = get_model_context_limit("anything", {"max_input_tokens": "200000"})
        assert limit == 100_000


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
