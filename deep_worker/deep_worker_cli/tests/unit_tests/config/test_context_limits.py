"""Tests for the context-limit lookup helpers in ``deep_worker_cli.config``.

These pin the resolution order (agent_a 2026-08-18 lesson):

  1. ``model_profile["max_input_tokens"]`` (LangChain SDK introspected
     value, authoritative). This is the runtime source — it
     comes from the SDK's profile data which is kept current on
     every SDK release.
  2. Exact match against ``PROVIDER_CONTEXT_LIMITS`` (which only
     contains the ``default`` entry as of 2026-08-19; specific
     model values come from the SDK, not from a hand-curated dict).
  3. Partial match (substring, longest key wins) — currently a
     no-op since the dict only has ``default``.
  4. ``default`` (100k) — generic fallback.

Family aliases (fable, opus, sonnet, haiku, mythos) are NOT in
PROVIDER_CONTEXT_LIMITS and never resolve to a static value. They
come from the LangChain SDK profile data only when the operator
explicitly instantiates a model — at which point
``model.profile["max_input_tokens"]`` carries the current
generation's value.
"""

from __future__ import annotations

import pytest

from deep_worker_cli.config import (
    PROVIDER_CONTEXT_LIMITS,
    get_model_context_limit,
)


class TestProviderContextLimitsIsFallbackOnly:
    """As of 2026-08-19, PROVIDER_CONTEXT_LIMITS is a thin fallback
    shell with only the ``default`` key. All specific model
    values come from the LangChain SDK profile data, not from
    a hand-curated dict. The agent_a 2026-08-18 regression
    proved that hand-curated values go stale the moment a
    provider ships a new generation.
    """

    def test_dict_only_has_default(self) -> None:
        assert set(PROVIDER_CONTEXT_LIMITS.keys()) == {"default"}

    def test_default_is_100k(self) -> None:
        assert PROVIDER_CONTEXT_LIMITS["default"] == 100_000

    def test_no_family_aliases(self) -> None:
        """Family aliases (fable, opus, sonnet, haiku, mythos) MUST
        NOT be in this dict — they refer to the latest model in
        the family, which is dynamic. We cannot encode the
        current value here.
        """
        for alias in ("fable", "opus", "sonnet", "haiku", "mythos"):
            assert alias not in PROVIDER_CONTEXT_LIMITS


class TestUnknownModelReturnsDefault:
    """Without a model_profile, unknown models return the default."""

    def test_completely_unknown_model(self) -> None:
        assert get_model_context_limit("totally-unknown-xyz", None) == 100_000

    def test_none_model_name(self) -> None:
        assert get_model_context_limit(None, None) == 100_000

    @pytest.mark.parametrize(
        "alias", ["fable", "opus", "sonnet", "haiku", "mythos"]
    )
    def test_family_alias_returns_default(self, alias: str) -> None:
        """Family aliases without a profile → default fallback.

        The model_profile is what tells the CLI the current
        generation's context window; without it (e.g. during
        early boot before the model is instantiated), the 100k
        default is the safe fallback so summarization does not
        trigger spuriously on short conversations.
        """
        assert get_model_context_limit(alias, None) == 100_000


class TestProfilePriority:
    """The model_profile is the authoritative source."""

    def test_profile_overrides_default_for_family_alias(self) -> None:
        """Family alias with a profile → profile wins."""
        # Profile says 1M → 1M wins over the 100k default.
        assert get_model_context_limit(
            "fable", {"max_input_tokens": 1_000_000}
        ) == 1_000_000

    def test_profile_overrides_default_for_dated_id(self) -> None:
        """Dated id with a profile → profile wins."""
        assert get_model_context_limit(
            "claude-sonnet-4-5", {"max_input_tokens": 500_000}
        ) == 500_000

    def test_profile_overrides_default_for_unknown_model(self) -> None:
        """Even unknown models use the profile if provided."""
        assert get_model_context_limit(
            "totally-unknown-xyz", {"max_input_tokens": 64_000}
        ) == 64_000

    def test_empty_profile_does_not_override_default(self) -> None:
        """Empty profile dict → falls through to default."""
        assert get_model_context_limit("anything", {}) == 100_000

    def test_non_int_max_input_tokens_falls_through(self) -> None:
        """Defensive: a non-int profile value falls through to default."""
        assert get_model_context_limit(
            "anything", {"max_input_tokens": None}
        ) == 100_000
        assert get_model_context_limit(
            "anything", {"max_input_tokens": "200000"}
        ) == 100_000
