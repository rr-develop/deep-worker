"""Unit tests for the ``-pp`` short alias of ``--preferred-provider``.

Both the production argparse parser (``create_parser``) and the override
detector (``_explicit_cli_flags``) must accept ``-pp <value>`` with the
exact same semantics as the long form, including being recognized as an
explicit override on restart.
"""

from __future__ import annotations

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    create_parser,
)


class TestPreferredProviderShortAlias:
    def test_pp_alias_parses_to_preferred_provider(self) -> None:
        args = create_parser().parse_args(
            ["my_agent", "-pp", "claude-code"]
        )
        assert args.preferred_provider == "claude-code"

    def test_long_form_still_works(self) -> None:
        args = create_parser().parse_args(
            ["my_agent", "--preferred-provider", "grok"]
        )
        assert args.preferred_provider == "grok"

    def test_pp_alias_marks_explicit_override(self) -> None:
        explicit = _explicit_cli_flags(
            ["my_agent", "-pp", "codex"]
        )
        assert "preferred_provider" in explicit

    def test_long_form_marks_explicit_override(self) -> None:
        explicit = _explicit_cli_flags(
            ["my_agent", "--preferred-provider", "codex"]
        )
        assert "preferred_provider" in explicit

    def test_absent_when_not_passed(self) -> None:
        explicit = _explicit_cli_flags(["my_agent"])
        assert "preferred_provider" not in explicit
