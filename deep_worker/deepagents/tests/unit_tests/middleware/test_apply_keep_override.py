"""``_apply_keep_override`` — the graph-side half of ``--summary-keep``.

The CLI has its own ``resolve_summary_keep``; this is the path that actually
reaches ``SummarizationMiddleware`` via ``create_deep_agent``. Both must obey
the same rule, so both need their own mutation-resistant tests.
"""

from __future__ import annotations

import pytest

from deepagents.middleware.summarization import _apply_keep_override


def _defaults(keep):
    return {
        "trigger": ("tokens", 340_000),
        "keep": keep,
        "truncate_args_settings": {
            "trigger": ("tokens", 340_000),
            "keep": ("tokens", 40_000),
        },
    }


class TestTokenCeiling:
    def test_ceiling_lowers_a_larger_default(self) -> None:
        out = _apply_keep_override(_defaults(("tokens", 40_000)), ("tokens", 10_000))
        assert out["keep"] == ("tokens", 10_000)

    def test_ceiling_never_raises_a_smaller_default(self) -> None:
        """min(), not max() — a ceiling is an upper bound only."""
        out = _apply_keep_override(_defaults(("tokens", 3_000)), ("tokens", 10_000))
        assert out["keep"] == ("tokens", 3_000)

    def test_equal_values_are_stable(self) -> None:
        out = _apply_keep_override(_defaults(("tokens", 10_000)), ("tokens", 10_000))
        assert out["keep"] == ("tokens", 10_000)

    def test_ceiling_against_fraction_default_passes_through(self) -> None:
        out = _apply_keep_override(_defaults(("fraction", 0.10)), ("tokens", 10_000))
        assert out["keep"] == ("tokens", 10_000)


class TestZeroIsExact:
    @pytest.mark.parametrize(
        "default_keep",
        [("tokens", 40_000), ("tokens", 3_000), ("fraction", 0.10), ("messages", 6)],
    )
    def test_zero_survives_every_default(self, default_keep) -> None:
        out = _apply_keep_override(_defaults(default_keep), ("tokens", 0))
        assert out["keep"] == ("tokens", 0)


class TestNonTokenSpecsReplace:
    def test_fraction_replaces_even_when_larger(self) -> None:
        out = _apply_keep_override(_defaults(("tokens", 3_000)), ("fraction", 0.50))
        assert out["keep"] == ("fraction", 0.50)

    def test_messages_replace_outright(self) -> None:
        out = _apply_keep_override(_defaults(("tokens", 3_000)), ("messages", 20))
        assert out["keep"] == ("messages", 20)


class TestUnrelatedFieldsUntouched:
    def test_trigger_is_preserved(self) -> None:
        out = _apply_keep_override(_defaults(("tokens", 40_000)), ("tokens", 0))
        assert out["trigger"] == ("tokens", 340_000)

    def test_truncate_args_keep_is_not_zeroed(self) -> None:
        """Arg-truncation cutoff is a different concept from the kept tail.

        Zeroing it would strip tool arguments from every message, not just
        the summarized prefix.
        """
        out = _apply_keep_override(_defaults(("tokens", 40_000)), ("tokens", 0))
        assert out["truncate_args_settings"]["keep"] == ("tokens", 40_000)

    def test_input_defaults_are_not_mutated_in_place(self) -> None:
        original = _defaults(("tokens", 40_000))
        _apply_keep_override(original, ("tokens", 0))
        assert original["keep"] == ("tokens", 40_000)
