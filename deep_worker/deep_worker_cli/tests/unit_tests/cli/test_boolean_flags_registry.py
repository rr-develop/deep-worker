"""Property-level guards for the persisted-boolean-options registry.

These tests do NOT exercise behavior — they enforce a structural property:

  > Every boolean key in ``_PARSER_DEFAULTS`` is EITHER listed in
  > ``_BOOLEAN_FLAGS`` (preserved across restart) OR in
  > ``_NON_PRESERVED_BOOLEAN_OPTIONS`` (explicitly excluded).

Adding a new boolean to ``_PARSER_DEFAULTS`` without classifying it MUST
fail this suite. That is the regression guard. See
``deep_worker_web/docs/DESIGN_RESTART_OPTION_PRESERVATION.md`` (§3.3).
"""
from __future__ import annotations

import pytest

from deep_worker_cli.cli.agent_command import (
    _BOOLEAN_FLAGS,
    _NON_PRESERVED_BOOLEAN_OPTIONS,
    _PARSER_DEFAULTS,
    create_parser,
)


def test_every_boolean_default_is_classified() -> None:
    """Adding a new bool to _PARSER_DEFAULTS without classifying it MUST fail.

    The structural rule: every boolean key in _PARSER_DEFAULTS must appear
    EITHER in _BOOLEAN_FLAGS (preserved) OR in _NON_PRESERVED_BOOLEAN_OPTIONS
    (explicitly excluded). No third option. This is the regression guard
    that prevents silent restart-preservation bugs.
    """
    bool_keys = {k for k, v in _PARSER_DEFAULTS.items() if isinstance(v, bool)}
    preserved = {dest for dest, _t, _f in _BOOLEAN_FLAGS}
    classified = preserved | _NON_PRESERVED_BOOLEAN_OPTIONS
    missing = bool_keys - classified
    assert not missing, (
        f"Boolean option(s) {missing} added to _PARSER_DEFAULTS but neither "
        f"listed in _BOOLEAN_FLAGS (to preserve across restart) nor in "
        f"_NON_PRESERVED_BOOLEAN_OPTIONS (to explicitly exclude). "
        f"This will cause a silent restart-preservation regression. "
        f"See docs/DESIGN_RESTART_OPTION_PRESERVATION.md §3.3."
    )


def test_preserved_and_excluded_sets_are_disjoint() -> None:
    """Same dest must not appear in BOTH preserved and excluded sets."""
    preserved = {dest for dest, _t, _f in _BOOLEAN_FLAGS}
    overlap = preserved & _NON_PRESERVED_BOOLEAN_OPTIONS
    assert not overlap, (
        f"dest(s) {overlap} listed in BOTH _BOOLEAN_FLAGS (preserved) and "
        f"_NON_PRESERVED_BOOLEAN_OPTIONS (excluded). Pick one."
    )


def test_registry_dests_exist_in_parser_defaults() -> None:
    """Typo guard: every registry dest must be a real parser default key."""
    preserved = {dest for dest, _t, _f in _BOOLEAN_FLAGS}
    unknown = preserved - set(_PARSER_DEFAULTS.keys())
    assert not unknown, (
        f"_BOOLEAN_FLAGS contains dest(s) {unknown} that are not in "
        f"_PARSER_DEFAULTS — likely a typo."
    )


def test_excluded_dests_exist_in_parser_defaults() -> None:
    """Typo guard: every excluded dest must be a real parser default key."""
    unknown = _NON_PRESERVED_BOOLEAN_OPTIONS - set(_PARSER_DEFAULTS.keys())
    assert not unknown, (
        f"_NON_PRESERVED_BOOLEAN_OPTIONS contains dest(s) {unknown} that are "
        f"not in _PARSER_DEFAULTS — likely a typo."
    )


@pytest.mark.parametrize("dest, true_flag, _false_flag", _BOOLEAN_FLAGS)
def test_registry_true_flag_is_accepted_by_parser(
    dest: str, true_flag: str, _false_flag: str | None,
) -> None:
    """Every registry true_flag must be accepted by create_parser() and set
    dest=True. Catches a registry/parser drift (e.g. renamed --foo to --bar
    but forgot to update the registry)."""
    parser = create_parser()
    # parse a minimal valid invocation: positional name + the flag under test.
    args = parser.parse_args(["agentname", true_flag])
    assert getattr(args, dest) is True, (
        f"Flag {true_flag!r} for dest {dest!r} did not set the value to True "
        f"via create_parser(). Registry and parser are out of sync."
    )


@pytest.mark.parametrize(
    "dest, _true_flag, false_flag",
    [(d, t, f) for d, t, f in _BOOLEAN_FLAGS if f is not None],
)
def test_registry_false_flag_is_accepted_by_parser(
    dest: str, _true_flag: str, false_flag: str,
) -> None:
    """Every --no-* form listed in the registry must be accepted by
    create_parser() and set dest=False. Mirrors the True-flag check.
    """
    parser = create_parser()
    args = parser.parse_args(["agentname", false_flag])
    assert getattr(args, dest) is False, (
        f"Flag {false_flag!r} for dest {dest!r} did not set the value to "
        f"False via create_parser(). Registry and parser are out of sync."
    )
