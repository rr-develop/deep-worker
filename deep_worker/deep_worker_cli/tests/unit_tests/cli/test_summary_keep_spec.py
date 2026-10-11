"""``--summary-keep <spec>`` parsing and resolution.

Owner requirement (2026-08-20):
- token value is a CEILING: effective = min(default fraction of ``-l``, ceiling)
- explicit ``0`` means NO tail at all (never min()-ed)
- fraction form replaces the default outright (no min())
- message form kept for parity with the legacy ("messages", N) policy
"""

from __future__ import annotations

import argparse

import pytest

from deep_worker_cli.cli.agent_command import parse_summary_keep


class TestParseSummaryKeep:
    def test_zero_is_tokens_zero(self) -> None:
        assert parse_summary_keep("0") == ("tokens", 0)

    def test_bare_integer_is_tokens(self) -> None:
        assert parse_summary_keep("10000") == ("tokens", 10000)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("10k", 10_000), ("10K", 10_000), ("1k", 1_000)],
    )
    def test_k_suffix_is_tokens(self, text: str, expected: int) -> None:
        assert parse_summary_keep(text) == ("tokens", expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("10%", 0.10), ("50%", 0.50), ("0.10", 0.10), ("0.5", 0.5)],
    )
    def test_fraction_forms(self, text: str, expected: float) -> None:
        kind, value = parse_summary_keep(text)
        assert kind == "fraction"
        assert value == pytest.approx(expected)

    @pytest.mark.parametrize(("text", "expected"), [("20m", 20), ("6m", 6)])
    def test_message_suffix(self, text: str, expected: int) -> None:
        assert parse_summary_keep(text) == ("messages", expected)

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "abc", "-5", "-1k", "10x", "1.5", "150%", "-10%", "0m", "%"],
    )
    def test_rejects_invalid(self, text: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_summary_keep(text)

    def test_fraction_one_is_allowed_upper_bound(self) -> None:
        assert parse_summary_keep("100%") == ("fraction", 1.0)


class TestResolveSummaryKeep:
    """``resolve_summary_keep(spec, default_keep)`` applies the ceiling rule."""

    def test_none_spec_keeps_default(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        default = ("tokens", 40_000)
        assert resolve_summary_keep(None, default) == default

    def test_token_spec_is_a_ceiling_when_default_is_larger(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        # -l 400k -> default 10% = 40k; ceiling 10k wins
        assert resolve_summary_keep(("tokens", 10_000), ("tokens", 40_000)) == (
            "tokens",
            10_000,
        )

    def test_token_spec_does_not_raise_a_smaller_default(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        # -l 30k -> default 10% = 3k; ceiling 10k must NOT inflate it to 10k
        assert resolve_summary_keep(("tokens", 10_000), ("tokens", 3_000)) == (
            "tokens",
            3_000,
        )

    def test_zero_is_never_min_ed(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        assert resolve_summary_keep(("tokens", 0), ("tokens", 40_000)) == ("tokens", 0)
        assert resolve_summary_keep(("tokens", 0), ("fraction", 0.10)) == ("tokens", 0)
        assert resolve_summary_keep(("tokens", 0), ("messages", 6)) == ("tokens", 0)

    def test_fraction_spec_replaces_default_without_min(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        # explicit fraction wins outright, even if larger than the default
        assert resolve_summary_keep(("fraction", 0.50), ("tokens", 3_000)) == (
            "fraction",
            0.50,
        )

    def test_messages_spec_replaces_default_without_min(self) -> None:
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        assert resolve_summary_keep(("messages", 20), ("tokens", 3_000)) == (
            "messages",
            20,
        )

    def test_token_ceiling_against_fraction_default_is_deferred(self) -> None:
        """A ceiling cannot be compared to a fraction without ``-l``.

        Without an operator limit the default is a fraction of the model
        window, which is unknown at CLI time. The ceiling must be carried
        through so the middleware can apply it against the resolved window.
        """
        from deep_worker_cli.cli.agent_command import resolve_summary_keep

        assert resolve_summary_keep(("tokens", 10_000), ("fraction", 0.10)) == (
            "tokens",
            10_000,
        )
