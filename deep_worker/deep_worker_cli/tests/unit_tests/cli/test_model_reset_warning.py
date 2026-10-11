"""A model reset must never be silent, and must tell the truth.

Incident (2026-07-28): ``--model fable`` was discarded and
replaced by ``sonnet`` with no warning whatsoever. The user's requirement is
verbatim: "if the model is not found and there is a fallback to something
else, a warning must be printed."

This module pins the WARNING contract of ``merge_restart_config``:

* a **reset** (a value is dropped) warns, naming the dropped model, the
  reason, and the *actual* replacement;
* an **explicit replacement** with a valid alias is honoured silently —
  warning on it would train operators to ignore warnings;
* exactly ONE stderr emission per reset (no logger duplicate);
* the reason must be accurate: ``stale-provider`` only when the saved
  ``model_provider`` genuinely differs from the target provider, otherwise
  ``unknown-alias``.

The preferred_model cascade matrix (C1-C6 in the design) is pinned here too,
including both ``--no-update-preferred`` paths, because a reset that leaves a
stale ``preferred_model`` behind re-poisons the next restart.

Design: docs/DESIGN_FIX_CLAUDE_CODE_SDK_MODEL_ALIASES_AND_DEFAULT.md
"""

from __future__ import annotations

import pytest

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    create_parser,
    merge_restart_config,
)
from deep_worker_cli.config import get_default_model_name


def _merge(argv: list[str], saved: dict):
    """Parse argv with the REAL parser and merge against saved config."""
    args = create_parser().parse_args(argv)
    explicit = _explicit_cli_flags(argv[1:])
    return merge_restart_config(saved, args, explicit=explicit)


def _merge_capturing(argv: list[str], saved: dict, capsys):
    """Merge and return (merged, stderr_text)."""
    merged = _merge(argv, saved)
    return merged, capsys.readouterr().err


@pytest.fixture
def saved_claude() -> dict:
    """A saved claude-code-sdk config exactly like the incident's."""
    return {
        "provider": "claude-code-sdk",
        "model": "sonnet",
        "model_provider": "claude-code-sdk",
        "working_dir": "/path/to/checkout",
        "skip_permissions": True,
        "preferred_provider": "claude-code-sdk",
        "preferred_model": "sonnet",
        "storage_version": 5,
    }


class TestValidAliasIsHonouredSilently:
    """(a) and (d): honouring a request is not a fallback."""

    def test_fable_survives_the_merge(self, saved_claude):
        """The core regression from the incident."""
        merged = _merge(["agent", "--restart", "--model", "fable"], saved_claude)
        assert merged.model == "fable", (
            f"--model fable must reach the launch config, got {merged.model!r}"
        )

    def test_fable_emits_no_warning(self, saved_claude, capsys):
        _, err = _merge_capturing(
            ["agent", "--restart", "--model", "fable"], saved_claude, capsys
        )
        assert "WARNING" not in err, (
            f"honouring a valid alias must be silent; stderr was: {err!r}"
        )

    def test_model_default_is_valid_and_silent(self, saved_claude, capsys):
        """(d) ``default`` is a documented alias — an explicit replacement,
        not a reset. It must not warn."""
        merged, err = _merge_capturing(
            ["agent", "--restart", "--model", "default"], saved_claude, capsys
        )
        assert merged.model == "default"
        assert "WARNING" not in err


class TestInvalidModelWarns:
    """(b): a dropped value must be reported."""

    def test_bogus_model_is_reset(self, saved_claude):
        merged = _merge(
            ["agent", "--restart", "--model", "bogusXYZ"], saved_claude
        )
        assert merged.model is None

    def test_bogus_model_warns_naming_the_token(self, saved_claude, capsys):
        _, err = _merge_capturing(
            ["agent", "--restart", "--model", "bogusXYZ"], saved_claude, capsys
        )
        assert "WARNING" in err
        assert "bogusXYZ" in err, (
            f"the warning must name the dropped model; stderr was: {err!r}"
        )

    def test_bogus_model_reason_is_unknown_alias_not_stale_provider(
        self, saved_claude, capsys
    ):
        """(g) The disambiguation bug.

        ``saved["model_provider"]`` is ``"claude-code-sdk"`` — truthy, and EQUAL
        to the target provider. A naive ``"stale-provider" if
        saved_model_provider else "unknown-alias"`` mislabels this rejection
        as a provider mismatch. It is not: the operator typed a token this
        provider does not recognise.
        """
        _, err = _merge_capturing(
            ["agent", "--restart", "--model", "bogusXYZ"], saved_claude, capsys
        )
        assert "unknown-alias" in err, (
            f"predicate rejection must report reason=unknown-alias; got: {err!r}"
        )
        assert "stale-provider" not in err, (
            "a truthy model_provider EQUAL to the current provider is not a "
            f"stale-provider mismatch; got: {err!r}"
        )


class TestStaleProviderReason:
    """The other half of (g): a genuine cross-provider mismatch."""

    def test_cross_provider_saved_model_reports_stale_provider(self, capsys):
        saved = {
            "provider": "claude-code-sdk",
            "model": "grok-composer-2.5-fast",
            "model_provider": "grok",
            "preferred_provider": "claude-code-sdk",
            "storage_version": 5,
        }
        merged, err = _merge_capturing(["a1", "--restart"], saved, capsys)
        assert merged.model is None
        assert "stale-provider" in err
        assert "grok-composer-2.5-fast" in err


class TestBareProviderDeclarationWarns:
    """(c): bare ``--provider`` drops the saved model."""

    def test_bare_provider_resets_and_warns(self, saved_claude, capsys):
        merged, err = _merge_capturing(
            ["agent", "--restart", "--provider", "claude-code-sdk"],
            saved_claude,
            capsys,
        )
        assert merged.model is None
        assert "WARNING" in err
        assert "sonnet" in err, (
            f"the warning must name the dropped 'sonnet'; stderr was: {err!r}"
        )
        assert "provider-declared" in err


class TestReplacementIsTruthful:
    """(e): the warning must name the model that will ACTUALLY be used."""

    def test_hardcoded_default_provider_names_concrete_model(self, capsys):
        """deepseek pins a concrete default — saying "resolved by the CLI"
        would be a lie in the very message meant to stop lying.

        The expectation is looked up at runtime: ``_PROVIDER_DEFAULT_MODELS``
        values drift (``deepseek-chat`` → ``deepseek-v4-flash``), and a
        hardcoded literal here would rot.
        """
        saved = {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "model_provider": "deepseek",
            "preferred_provider": "deepseek",
            "storage_version": 5,
        }
        _, err = _merge_capturing(
            ["a1", "--restart", "--model", "bogusXYZ"], saved, capsys
        )
        expected = get_default_model_name("deepseek")
        assert expected, "test precondition: deepseek must pin a default"
        assert expected in err, (
            f"warning must name the real replacement {expected!r}; got: {err!r}"
        )
        assert "resolved by the provider CLI" not in err, (
            "deepseek materialises a concrete default; claiming the CLI "
            f"resolves it is false. stderr: {err!r}"
        )

    def test_unpinned_provider_says_cli_resolves(self, saved_claude, capsys):
        """claude-code-sdk pins nothing, so the honest text IS "resolved by the
        provider CLI"."""
        _, err = _merge_capturing(
            ["agent", "--restart", "--model", "bogusXYZ"], saved_claude, capsys
        )
        assert "provider default" in err


class TestExactlyOneEmission:
    """(f): no print + logger.warning duplicate.

    The hazard being pinned is ONE reset being reported TWICE (the draft
    design emitted both ``print(stderr)`` and ``logger.warning``, and the root
    logger normally has a stderr handler). Distinct resets legitimately
    produce distinct lines — C1 drops both the active and the preferred model
    — so the assertion counts occurrences of a *specific dropped token*, not
    total warning lines.
    """

    def test_single_reset_reported_once(self, capsys):
        """A config with no preferred_model produces exactly one reset."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "sonnet",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": None,
            "storage_version": 5,
        }
        _, err = _merge_capturing(
            ["a1", "--restart", "--model", "bogusXYZ"], saved, capsys
        )
        assert err.count("was reset") == 1, (
            f"one reset must be reported exactly once; stderr was: {err!r}"
        )

    def test_each_dropped_token_named_once(self, saved_claude, capsys):
        """C1 drops two values; each is reported once, neither is duplicated."""
        _, err = _merge_capturing(
            ["agent", "--restart", "--model", "bogusXYZ"], saved_claude, capsys
        )
        assert err.count("'bogusXYZ' was reset") == 1, (
            f"the rejected explicit model must be reported once; got: {err!r}"
        )
        assert err.count("'sonnet' was reset") == 1, (
            f"the cleared preference must be reported once; got: {err!r}"
        )


class TestPreferredCascadeMatrix:
    """C1-C6 from design §4.5.3."""

    def test_c1_invalid_explicit_collapses_active_and_preferred(self, capsys):
        """C1: the old preference must NOT be silently resurrected."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "sonnet",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "opus",
            "storage_version": 5,
        }
        merged, err = _merge_capturing(
            ["a1", "--restart", "--model", "bogusXYZ"], saved, capsys
        )
        assert merged.model is None
        assert merged.preferred_model is None, (
            "an invalid explicit replacement must not leave the old "
            f"preference in place; got {merged.preferred_model!r}"
        )
        assert "preferred-reset-after-invalid-explicit" in err, (
            f"C1's second warning needs a truthful reason; got: {err!r}"
        )

    def test_c1_opus_not_labelled_unknown_alias(self, capsys):
        """``opus`` is a perfectly valid alias — it is cleared because the
        replacement request was invalid, not because opus is unrecognised.
        Mislabelling sends the operator debugging the wrong thing."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "sonnet",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "opus",
            "storage_version": 5,
        }
        _, err = _merge_capturing(
            ["a1", "--restart", "--model", "bogusXYZ"], saved, capsys
        )
        opus_lines = [ln for ln in err.splitlines() if "opus" in ln]
        assert opus_lines, f"expected a warning mentioning opus; got: {err!r}"
        for line in opus_lines:
            assert "unknown-alias" not in line, (
                f"'opus' is a valid alias and must not be called unknown: {line!r}"
            )

    def test_c2_valid_explicit_updates_both(self, saved_claude, capsys):
        """C2: a valid alias becomes both active and preferred, silently."""
        merged, err = _merge_capturing(
            ["agent", "--restart", "--model", "fable"], saved_claude, capsys
        )
        assert merged.model == "fable"
        assert merged.preferred_model == "fable"
        assert "WARNING" not in err

    def test_c3_failover_preserves_preferred_of_other_provider(self, capsys):
        """C3: a temporary failover must not destroy the operator's
        long-term preference on a DIFFERENT provider."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "sonnet",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "opus",
            "storage_version": 5,
        }
        merged, _ = _merge_capturing(
            ["a1", "--restart", "--provider", "codex", "--no-update-preferred"],
            saved,
            capsys,
        )
        assert merged.provider == "codex"
        assert merged.model is None, "the claude model is stale for codex"
        assert merged.preferred_provider == "claude-code-sdk"
        assert merged.preferred_model == "opus", (
            "failover must preserve the preferred provider's model; got "
            f"{merged.preferred_model!r}"
        )

    def test_c4_restore_on_idle_returns_to_preferred_model(self, capsys):
        """C4: returning to the preferred provider restores opus, silently
        (T3 is a RESTORE, not a reset)."""
        saved = {
            "provider": "codex",
            "model": None,
            "model_provider": "codex",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "opus",
            "storage_version": 5,
        }
        merged, err = _merge_capturing(
            ["a1", "--restart", "--provider", "claude-code-sdk", "--no-update-preferred"],
            saved,
            capsys,
        )
        assert merged.provider == "claude-code-sdk"
        assert merged.model == "opus", (
            f"restore-on-idle must bring back opus; got {merged.model!r}"
        )
        assert merged.preferred_model == "opus"
        assert "WARNING" not in err, (
            f"a restore drops nothing and must stay silent; got: {err!r}"
        )

    def test_c5_invalid_saved_preferred_is_dropped(self, capsys):
        """C5: a legacy-poisoned preferred_model must not survive."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "sonnet",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "grok-composer-2.5-fast",
            "storage_version": 5,
        }
        merged, _ = _merge_capturing(["a1", "--restart"], saved, capsys)
        assert merged.preferred_model != "grok-composer-2.5-fast"

    def test_c6_bare_provider_collapses_both(self, saved_claude, capsys):
        """C6: RC-1 — preferred must track the reset of active."""
        merged, err = _merge_capturing(
            ["agent", "--restart", "--provider", "claude-code-sdk"],
            saved_claude,
            capsys,
        )
        assert merged.model is None
        assert merged.preferred_model is None
        assert "provider-declared" in err


class TestNoRegressionOnExistingBehaviour:
    """The widened predicate must not resurrect the poison bugs it follows."""

    def test_unknown_string_still_rejected(self, capsys):
        """The literal ``unknown`` must never reach a launch config."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "unknown",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "storage_version": 5,
        }
        merged, _ = _merge_capturing(["a1", "--restart"], saved, capsys)
        assert merged.model is None

    def test_saved_fable_survives_plain_restart(self, capsys):
        """Once persisted, ``fable`` must survive a plain restart untouched."""
        saved = {
            "provider": "claude-code-sdk",
            "model": "fable",
            "model_provider": "claude-code-sdk",
            "preferred_provider": "claude-code-sdk",
            "preferred_model": "fable",
            "storage_version": 5,
        }
        merged, err = _merge_capturing(["a1", "--restart"], saved, capsys)
        assert merged.model == "fable"
        assert "WARNING" not in err
