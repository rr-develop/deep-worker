"""Incremental tool selection: --tools-add / --tools-remove.

See ``docs/DESIGN_TOOL_SELECTION.md`` §5. The point of these flags is that
they are *relative*: adding one tool to a restricted agent must not
require restating the tools it already has. Two properties carry the
feature and are pinned hardest here — the most specific rule wins, and the
stored spec survives a restart without re-applying itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deep_worker_cli.tool_selection import (
    ToolSelection,
    ToolSelectionError,
    describe_selection,
    is_pattern,
    merge_increments,
    parse_increment_spec,
    parse_tool_selection,
    parse_tool_selection_spec,
    specificity,
    validate_against_inventory,
)

TOOLS = [
    SimpleNamespace(name="read_file"),
    SimpleNamespace(name="execute"),
    SimpleNamespace(name="grep"),
    SimpleNamespace(name="serena__find_symbol"),
    SimpleNamespace(name="serena__rename_symbol"),
    SimpleNamespace(name="serena__get_diagnostics_for_file"),
]
NAMES = [t.name for t in TOOLS]


def _selection(spec: list[str]) -> ToolSelection:
    return parse_tool_selection(spec)


def _kept(spec: list[str]) -> list[str]:
    sel = _selection(spec)
    return [n for n in NAMES if sel.permits(n)]


class TestTheDefaultDoesNotMove:
    """No increments means the previous behaviour, bit for bit."""

    def test_no_rules_leaves_selection_unrestricted(self):
        assert parse_increment_spec(None, None) == ()
        assert not parse_tool_selection([]).restricts

    def test_no_rules_leaves_an_existing_spec_alone(self):
        assert merge_increments(["none", "read_file"], ()) == [
            "none",
            "read_file",
        ]

    def test_a_spec_without_increments_parses_as_before(self):
        sel = parse_tool_selection(["-execute"])
        assert sel.rules == ()
        assert sel.permits("read_file")
        assert not sel.permits("execute")


class TestTheMoreSpecificRuleWins:
    """The exception-to-a-family case the flags exist for."""

    def test_literal_beats_the_pattern_that_also_matches_it(self):
        rules = parse_increment_spec(["serena__find_symbol"], ["serena__*"])
        sel = ToolSelection(rules=rules)
        assert sel.permits("serena__find_symbol")
        assert not sel.permits("serena__rename_symbol")
        assert not sel.permits("serena__get_diagnostics_for_file")
        # Tools outside the family are untouched.
        assert sel.permits("read_file")

    def test_the_inverse_reads_the_same_way(self):
        rules = parse_increment_spec(["serena__*"], ["serena__find_symbol"])
        sel = ToolSelection(rules=rules)
        assert not sel.permits("serena__find_symbol")
        assert sel.permits("serena__rename_symbol")

    def test_a_longer_pattern_outranks_a_shorter_one(self):
        assert specificity("serena__find_symbol") > specificity("serena__f*")
        assert specificity("serena__f*") > specificity("serena__*")
        assert specificity("serena__*") > specificity("*")

    def test_a_literal_outranks_a_pattern_of_equal_literal_length(self):
        # "abc" and "abc*" share a 3-character literal part; the exact
        # name must still win, or the exception case becomes order-dependent.
        assert specificity("abc") > specificity("abc*")

    def test_rule_order_does_not_decide_the_outcome(self):
        forward = ToolSelection(
            rules=parse_increment_spec(["serena__find_symbol"], ["serena__*"])
        )
        backward = ToolSelection(
            rules=parse_increment_spec(["serena__find_symbol"], ["serena__*"])[
                ::-1
            ]
        )
        for name in NAMES:
            assert forward.permits(name) == backward.permits(name), name


class TestEquallySpecificOppositesAreRejected:
    def test_same_pattern_added_and_removed(self):
        with pytest.raises(ToolSelectionError, match="both added and removed"):
            parse_increment_spec(["serena__*"], ["serena__*"])

    def test_same_literal_added_and_removed(self):
        with pytest.raises(ToolSelectionError, match="both added and removed"):
            parse_increment_spec(["grep"], ["grep"])

    def test_the_rejection_survives_a_round_trip(self):
        with pytest.raises(ToolSelectionError, match="both added and removed"):
            parse_tool_selection(["+grep", "~grep"])


class TestTwoDifferentPatternsCanTieToo:
    """Equal specificity is reachable without equal pattern strings.

    ``serena__*`` and ``*d_symbol`` carry 8 literal characters each and
    both match ``serena__find_symbol``, so grouping by the pattern string
    — which is all the parser can do — cannot see the conflict. Order
    then decided it, which meant a rule could be silently overruled by an
    older one just for having been appended later.
    """

    OVERLAP = "serena__find_symbol"

    def test_the_two_patterns_really_are_equally_specific(self):
        assert specificity("serena__*") == specificity("*d_symbol")

    def test_order_does_not_flip_the_outcome(self):
        removed_first = parse_tool_selection(["~serena__*", "+*d_symbol"])
        added_first = parse_tool_selection(["+*d_symbol", "~serena__*"])
        assert removed_first.permits(self.OVERLAP) == added_first.permits(
            self.OVERLAP
        )

    def test_a_tie_resolves_to_deny(self):
        # Of the two directions only deny is safe to pick blind: guessing
        # wrong must not hand back a tool the agent was confined away from.
        for spec in (["~serena__*", "+*d_symbol"], ["+*d_symbol", "~serena__*"]):
            assert not parse_tool_selection(spec).permits(self.OVERLAP), spec

    def test_a_tie_is_rejected_once_the_inventory_is_known(self):
        sel = parse_tool_selection(["~serena__*", "+*d_symbol"])
        assert sel.ambiguous_rules(NAMES) == [("*d_symbol", "serena__*")]
        with pytest.raises(ToolSelectionError, match="equally specific"):
            validate_against_inventory(sel, NAMES)

    def test_patterns_that_never_overlap_are_not_a_conflict(self):
        # Equal specificity alone is not enough: these two match disjoint
        # sets, so there is nothing to arbitrate.
        sel = parse_tool_selection(["~serena__*", "+read_fil?"])
        assert sel.ambiguous_rules(NAMES) == []
        validate_against_inventory(sel, NAMES)

    def test_the_add_is_not_silently_lost_across_a_restart(self):
        # merge_increments appends, so before the tie-break the newer rule
        # lost to the older one by position and the flag did nothing at all.
        removed = merge_increments([], parse_increment_spec(None, ["serena__*"]))
        restored = merge_increments(
            removed, parse_increment_spec(["*d_symbol"], None)
        )
        assert restored == ["~serena__*", "+*d_symbol"]
        with pytest.raises(ToolSelectionError, match="equally specific"):
            validate_against_inventory(parse_tool_selection(restored), NAMES)


class TestIncrementsLayerOntoTheBaseSpec:
    def test_add_on_top_of_a_closed_allow_list(self):
        spec = merge_increments(
            parse_tool_selection_spec("none,read_file"),
            parse_increment_spec(["grep"], None),
        )
        assert _kept(spec) == ["read_file", "grep"]

    def test_remove_on_top_of_the_default(self):
        spec = merge_increments([], parse_increment_spec(None, ["execute"]))
        kept = _kept(spec)
        assert "execute" not in kept
        assert "read_file" in kept

    def test_a_family_removal_leaves_the_rest(self):
        spec = merge_increments([], parse_increment_spec(None, ["serena__*"]))
        assert _kept(spec) == ["read_file", "execute", "grep"]


class TestTheSpecSurvivesARestart:
    """The stored form is re-parsed on every start, so it must round-trip."""

    def test_increments_are_encoded_into_the_spec(self):
        spec = merge_increments(
            [], parse_increment_spec(["serena__find_symbol"], ["serena__*"])
        )
        assert spec == ["+serena__find_symbol", "~serena__*"]
        assert _kept(spec) == [
            "read_file",
            "execute",
            "grep",
            "serena__find_symbol",
        ]

    def test_reapplying_the_same_increment_is_idempotent(self):
        rules = parse_increment_spec(["grep"], None)
        once = merge_increments(["none", "read_file"], rules)
        twice = merge_increments(once, rules)
        assert once == twice

    def test_a_later_rule_supersedes_the_earlier_one(self):
        removed = merge_increments([], parse_increment_spec(None, ["grep"]))
        assert "grep" not in _kept(removed)
        restored = merge_increments(
            removed, parse_increment_spec(["grep"], None)
        )
        assert "grep" in _kept(restored)
        # The superseded rule is gone rather than merely outvoted, so the
        # spec cannot grow without bound across restarts.
        assert restored == ["+grep"]


class TestAPatternThatMatchesNothingIsAnError:
    def test_unmatched_pattern_is_reported(self):
        sel = ToolSelection(rules=parse_increment_spec(None, ["lsp_*"]))
        assert sel.unknown_names(NAMES) == ["lsp_*"]

    def test_the_shared_entry_point_raises_on_it(self):
        sel = ToolSelection(rules=parse_increment_spec(None, ["lsp_*"]))
        with pytest.raises(ToolSelectionError, match="unknown tool"):
            validate_against_inventory(sel, NAMES)

    def test_a_matching_pattern_is_accepted(self):
        sel = ToolSelection(rules=parse_increment_spec(None, ["serena__*"]))
        assert sel.unknown_names(NAMES) == []

    def test_an_unknown_literal_is_still_reported(self):
        sel = ToolSelection(rules=parse_increment_spec(["reed_file"], None))
        assert sel.unknown_names(NAMES) == ["reed_file"]


class TestTheBuiltConfigAndTheRestartConfigAgree:
    """Both must carry the increments, or validation cannot see them.

    Regression of a defect found on probe ``tsel_bad3`` (2026-08-31):
    the increments were folded into ``launch_config`` but NOT into the
    ``AgentConfig`` the process is built from, so
    ``deep-worker probe --tools-rm 'lsp_*'`` reached ``idle`` with a
    pattern that matched no tool. The unknown-name check runs against the
    AgentConfig spec, so it never saw the rule — the exact silent miss
    the check exists to prevent.
    """

    def _args(self, **kw):
        from argparse import Namespace

        base = dict(tools=None, tools_add=None, tools_remove=None)
        base.update(kw)
        return Namespace(**base)

    @staticmethod
    def _tool_spec_sources() -> dict[str, str]:
        """How every ``tools=`` / ``"tools":`` site builds its spec.

        Asserted structurally rather than behaviourally on purpose. The
        defect was a *call site* reading ``args.tools`` directly while the
        helper existed and was correct — so a test that only exercises the
        helper stays green through the bug (verified by mutation). What has
        to hold is that no site rebuilds the spec on its own.
        """
        import ast
        import inspect

        from deep_worker_cli.cli import agent_command

        tree = ast.parse(inspect.getsource(agent_command))
        found: dict[str, str] = {}
        for node in ast.walk(tree):
            value = None
            if isinstance(node, ast.keyword) and node.arg == "tools":
                value = node.value
                label = "kwarg tools="
            elif (
                isinstance(node, ast.Dict)
                and any(
                    isinstance(k, ast.Constant) and k.value == "tools"
                    for k in node.keys
                    if k is not None
                )
            ):
                for key, val in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "tools":
                        value = val
                        label = 'dict "tools":'
            if value is None:
                continue
            found[f"{label} line {value.lineno}"] = ast.dump(value)
        return found

    def test_every_tool_spec_site_goes_through_the_one_helper(self):
        sites = self._tool_spec_sources()
        assert sites, "the AST scan found no tools= sites — the scan broke"
        offenders = [
            where
            for where, dumped in sites.items()
            if "parse_tool_selection_spec" in dumped
            and "resolve_tool_spec" not in dumped
        ]
        assert not offenders, (
            f"these sites rebuild the tool spec instead of calling "
            f"resolve_tool_spec: {offenders}. The increments would reach "
            f"one of AgentConfig / launch_config but not the other, which "
            f"is how --tools-rm 'lsp_*' reached a running agent without "
            f"tripping the unknown-name check (probe tsel_bad3)."
        )

    def test_the_resolved_spec_still_fails_the_unknown_name_check(self):
        from deep_worker_cli.cli.agent_command import resolve_tool_spec

        spec = resolve_tool_spec(self._args(tools_remove=["lsp_*"]))
        selection = parse_tool_selection(spec)
        assert selection.unknown_names(NAMES) == ["lsp_*"]

    def test_resolve_tool_spec_layers_onto_an_explicit_tools_flag(self):
        from deep_worker_cli.cli.agent_command import resolve_tool_spec

        spec = resolve_tool_spec(
            self._args(tools="none,read_file", tools_add=["grep"])
        )
        selection = parse_tool_selection(spec)
        assert selection.permits("read_file")
        assert selection.permits("grep")
        assert not selection.permits("execute")

    def test_resolve_tool_spec_is_empty_without_any_flag(self):
        from deep_worker_cli.cli.agent_command import resolve_tool_spec

        assert resolve_tool_spec(self._args()) == []


class TestSurfaces:
    def test_is_pattern_recognises_the_glob_characters(self):
        assert is_pattern("serena__*")
        assert is_pattern("read_file?")
        assert is_pattern("[rw]ead")
        assert not is_pattern("read_file")

    def test_description_states_the_increments(self):
        sel = parse_tool_selection(["+serena__find_symbol", "~serena__*"])
        described = describe_selection(sel)
        assert "serena__*" in described
        assert "serena__find_symbol" in described

    def test_an_increment_alone_counts_as_a_restriction(self):
        # Otherwise create_cli_agent skips the middleware entirely and the
        # removal silently does nothing.
        assert parse_tool_selection(["~execute"]).restricts

    def test_a_bare_sign_is_rejected(self):
        with pytest.raises(ToolSelectionError, match="must be followed by"):
            parse_tool_selection(["+"])
