"""Consumers of ``model_can_belong_to_provider`` must inherit the widened set.

The predicate is not only used to decide what gets launched — it also gates
what the operator is *told* is running. Two consumers filter display/metadata
through it:

* ``AgentConfig.set_dynamic_display_model`` (agent_config.py:123-131)
* ``AgentService`` metadata assembly (rpyc_service/agent_service.py:913-924)

Before the model-alias fix, a claude-code agent genuinely running ``fable``
had that value scrubbed by both, so the UI showed a blank or stale model.
Neither site needs a logic change — they import from ``deep_worker_cli.config``,
which now delegates to the leaf predicate — but the inherited behaviour change
is operator-visible, so it is pinned here.

Design: docs/DESIGN_FIX_CLAUDE_CODE_MODEL_ALIASES_AND_DEFAULT.md §5.1
"""

from __future__ import annotations

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig

CLAUDE_ALIASES = ["fable", "best", "default", "opusplan", "sonnet[1m]", "claude-fable-5"]


class TestDynamicDisplayModel:
    """``set_dynamic_display_model`` must keep valid claude-code aliases."""

    @pytest.mark.parametrize("alias", CLAUDE_ALIASES)
    def test_alias_is_kept(self, alias: str):
        config = AgentConfig(working_dir="/tmp", provider="claude-code")
        config.set_dynamic_display_model(alias)
        assert config.dynamic_display_model == alias, (
            f"a claude-code agent running {alias!r} must display it, not hide it"
        )

    def test_fable_regression(self):
        """The exact display symptom from that incident."""
        config = AgentConfig(working_dir="/tmp", provider="claude-code")
        config.set_dynamic_display_model("fable")
        assert config.dynamic_display_model == "fable"

    def test_foreign_model_still_rejected(self):
        """Widening must not make the filter useless."""
        config = AgentConfig(working_dir="/tmp", provider="claude-code")
        config.set_dynamic_display_model("grok-composer-2.5-fast")
        assert config.dynamic_display_model is None

    def test_unknown_sentinel_still_rejected(self):
        config = AgentConfig(working_dir="/tmp", provider="claude-code")
        config.set_dynamic_display_model("unknown")
        assert config.dynamic_display_model is None


class TestResolvedModelIsNoneForClaudeCode:
    """``resolved_model`` must not invent ``sonnet`` any more.

    With ``_PROVIDER_DEFAULT_MODELS["claude-code"] == ("CLAUDE_CODE_MODEL",
    None)``, an agent with no explicit model resolves to ``None`` and the
    ``claude`` CLI picks its own account-type default — the same contract
    codex and grok already have.
    """

    def test_no_model_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
        config = AgentConfig(working_dir="/tmp", provider="claude-code", model=None)
        assert config.resolved_model is None, (
            "claude-code must not pin a default; got "
            f"{config.resolved_model!r}"
        )

    def test_explicit_model_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
        config = AgentConfig(working_dir="/tmp", provider="claude-code", model="fable")
        assert config.resolved_model == "fable"

    def test_env_var_respected(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")
        config = AgentConfig(working_dir="/tmp", provider="claude-code", model=None)
        assert config.resolved_model == "opus"


# NOTE: metadata coverage for ``rpyc_service/agent_service.py:913-924`` is NOT
# here. An earlier version of this file reimplemented that filter chain locally
# and asserted against the copy — which would stay green even if the real
# consumer changed its precedence or stopped consulting the predicate at all.
# The real coverage calls ``AgentService.exposed_get_metadata()`` directly and
# lives in ``tests/unit_tests/rpyc_service/test_resolved_model_metadata.py``
# (see ``test_get_metadata_surfaces_claude_code_fable`` and siblings).
