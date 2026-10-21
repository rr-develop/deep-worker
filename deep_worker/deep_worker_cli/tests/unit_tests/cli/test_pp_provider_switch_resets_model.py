"""Regression: ``-pp <new-provider>`` on restart must drop the stale model.

Past incident: the agent was restarted with
``-pp codex`` and NO ``--model``. ``--preferred-provider`` flips the active
provider to codex, but ``merge_restart_config`` only reset the saved model
when ``--provider`` was used — not when ``-pp`` changed the provider. So the
saved grok model ``grok-composer-2.5-fast`` was carried forward and the agent
booted codex while still believing it was on the grok model (broken).

Contract: when the active provider changes (via ``--provider`` OR ``-pp``)
and no ``--model`` is explicitly given, the saved model belongs to the OLD
provider and MUST be dropped (``model = None``) so ``build_launch_config``
resolves the NEW provider's own default — "default = what the provider
considers default" (for codex/grok that is ``None`` → the CLI picks).

These use the REAL CLI parser + ``_explicit_cli_flags`` end-to-end (the same
pattern as ``test_no_flags.TestNoFlagsEndToEnd``), so they are immune to the
MagicMock/argv fragility of the older merge tests.
"""

from __future__ import annotations

import pytest

from deep_worker_cli.cli.agent_command import (
    _explicit_cli_flags,
    build_launch_config,
    create_parser,
    merge_restart_config,
)


def _merge(argv: list[str], saved: dict):
    """Parse argv with the real parser and merge against saved config."""
    args = create_parser().parse_args(argv)
    explicit = _explicit_cli_flags(argv[1:])
    return merge_restart_config(saved, args, explicit=explicit)


@pytest.fixture
def saved_grok() -> dict:
    """A saved launch_config exactly like the incident's: grok + grok model."""
    return {
        "provider": "grok",
        "model": "grok-composer-2.5-fast",
        "working_dir": "/tmp/agent_work",
        "skip_permissions": True,
        "task_manager": True,
        "history_memory": True,
        "preferred_provider": "grok",
        "preferred_model": "grok-composer-2.5-fast",
        "storage_version": 5,
    }


class TestPpProviderSwitchResetsModel:
    def test_pp_codex_drops_stale_grok_model(self, saved_grok):
        """The core bug: -pp codex must not keep the grok model."""
        merged = _merge(["agent_1", "--restart", "-pp", "codex"], saved_grok)

        assert merged.provider == "codex"
        assert merged.preferred_provider == "codex"
        assert merged.model is None, (
            "-pp switching providers without --model must drop the old "
            f"provider's model; got {merged.model!r}"
        )

    def test_pp_codex_preferred_model_not_stale(self, saved_grok):
        """The stale grok model must not leak into preferred_model either.

        For a -pp provider switch with no --model, preferred_model follows the
        (reset) active model → None.
        """
        merged = _merge(["agent_1", "--restart", "-pp", "codex"], saved_grok)
        assert merged.preferred_model is None
        assert merged.preferred_model != "grok-composer-2.5-fast"

    def test_pp_codex_persists_none_model(self, saved_grok):
        """End-to-end: the rebuilt launch_config persists model=None for codex
        so the Codex CLI picks its own default (not the grok string)."""
        merged = _merge(["agent_1", "--restart", "-pp", "codex"], saved_grok)
        cfg = build_launch_config(merged)
        assert cfg["provider"] == "codex"
        assert cfg["model"] is None, (
            f"codex with no --model must persist model=None; got {cfg['model']!r}"
        )

    def test_pp_same_provider_no_model_resets_to_default(self, saved_grok):
        """Operator rule (categorical): an explicit -pp declaration WITHOUT
        --model means "use that provider's default", even when -pp re-asserts
        the same provider. Supersedes the old assumption that same-provider -pp
        keeps the saved model.

        This is what heals the already-poisoned state: saved
        ``provider=codex, model=grok-composer-2.5-fast``; the operator types
        ``-pp codex`` (no --model) → we must reset model to None so the codex
        CLI picks its real default, not the leftover grok string.
        """
        # Simulate the poisoned state at provider=codex level too:
        poisoned_codex = dict(saved_grok, provider="codex", preferred_provider="codex")
        merged = _merge(["agent_1", "--restart", "-pp", "codex"], poisoned_codex)
        assert merged.provider == "codex"
        assert merged.model is None, (
            "explicit -pp without --model must use provider default (None), "
            f"not the saved foreign model; got {merged.model!r}"
        )

    def test_pp_same_provider_clears_preferred_model(self, saved_grok):
        """RC-1: preferred_model must also be cleared on any Fix1 gate
        firing — otherwise failover restore-on-idle would re-pin the stale
        foreign model later."""
        poisoned_codex = dict(saved_grok, provider="codex", preferred_provider="codex")
        merged = _merge(["agent_1", "--restart", "-pp", "codex"], poisoned_codex)
        assert merged.preferred_model is None, (
            f"preferred_model must track the model reset; got {merged.preferred_model!r}"
        )

    def test_plain_restart_keeps_saved_model(self, saved_grok):
        """A flat ``--restart`` with no provider/model flags is NOT an explicit
        provider declaration — Fix1 must not fire, the saved model is preserved.
        """
        merged = _merge(["agent_1", "--restart"], saved_grok)
        assert merged.provider == "grok"
        assert merged.model == "grok-composer-2.5-fast", (
            "plain --restart must preserve the saved deliberately-pinned model"
        )

    def test_plain_restart_drops_legacy_cross_provider_model(self, saved_grok):
        poisoned_codex = dict(
            saved_grok,
            provider="codex",
            model="opus",
            preferred_provider="codex",
            preferred_model="opus",
        )
        merged = _merge(["agent_2", "--restart"], poisoned_codex)

        assert merged.provider == "codex"
        assert merged.model is None
        assert merged.preferred_model is None

    def test_pp_switch_with_explicit_model_is_honored(self, saved_grok):
        """An explicit --model always wins, even on a -pp provider switch."""
        merged = _merge(
            ["agent_1", "--restart", "-pp", "codex", "--model", "gpt-5.5"],
            saved_grok,
        )
        assert merged.provider == "codex"
        assert merged.model == "gpt-5.5"

    def test_long_form_preferred_provider_also_resets(self, saved_grok):
        """The long ``--preferred-provider`` form behaves identically to -pp."""
        merged = _merge(
            ["agent_1", "--restart", "--preferred-provider", "codex"],
            saved_grok,
        )
        assert merged.provider == "codex"
        assert merged.model is None

    def test_provider_flag_still_resets_model(self, saved_grok):
        """Pre-existing --provider reset path stays intact (no regression)."""
        merged = _merge(
            ["agent_1", "--restart", "--provider", "codex"], saved_grok
        )
        assert merged.provider == "codex"
        assert merged.model is None
