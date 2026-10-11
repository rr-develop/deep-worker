"""Named, operator-configurable failover chains.

DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md. The owner directive these tests
protect: a chain is NEVER hard-coded in Python. Every test here writes its own
config under a ``tmp_path`` home and asserts behaviour follows the FILE — a
test that only exercised the seeded default would pass just as well against a
hard-coded chain, and prove nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.runtime.failover_chains import (
    DEFAULT_CHAIN_NAME,
    FailoverChain,
    FailoverChainConfigError,
    FailoverStep,
    chains_path,
    get_chain,
    load_chains,
    materialise_default,
    resolve_step_model,
)
from deep_worker_cli.runtime.provider_chain import failover_candidates


@pytest.fixture(autouse=True)
def config_driven_providers(tmp_path: Path) -> None:
    """Declare the config-driven providers the chain tests reference.

    Written into the SAME tmp home the chains use, never the operator's real
    ``~/.deep_worker`` (feedback_tests_never_touch_prod_state_2026_08_17).
    Without this, validating a chain that names ``or-glm-52`` would either
    fail, or — worse — silently pass by reading the developer's live
    ``providers.json``.
    """
    from deep_worker_cli.runtime.provider_registry import registry_path

    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": name,
                        "compatible_with": "openai",
                        "base_url": "https://example.invalid/v1",
                        "api_key_env": "DW_TEST_KEY",
                        "account_group": group,
                    }
                    for name, group in (
                        ("or-glm-52", "openrouter-account"),
                        ("or-laguna", "openrouter-account"),
                        ("minimax", "minimax-account"),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )


def _write(home: Path, doc: dict) -> Path:
    path = chains_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _names(steps) -> list[str]:
    return [s.provider for s in steps]


# ── Materialisation of the default chain (§3.3) ───────────────────────────


class TestMaterialisation:
    def test_absent_file_is_created_with_default_chain(self, tmp_path: Path):
        assert not chains_path(tmp_path).exists()

        chains = load_chains(tmp_path)

        assert chains_path(tmp_path).exists()
        assert _names(chains[DEFAULT_CHAIN_NAME].steps) == [
            "claude-code",
            "codex",
            "grok",
            "deepseek",
        ]

    def test_seeded_default_excludes_claude_code_sdk(self, tmp_path: Path):
        """The one deliberate behaviour change (design §8).

        ``claude-code`` and ``claude-code-sdk`` are two forms of one provider
        on one subscription, so a step between them lands in the same
        exhausted account (owner directive 2026-09-03).
        """
        chains = load_chains(tmp_path)
        assert "claude-code-sdk" not in _names(chains[DEFAULT_CHAIN_NAME].steps)

    def test_materialisation_is_idempotent(self, tmp_path: Path):
        materialise_default(tmp_path)
        chains_path(tmp_path).write_text(
            json.dumps({"version": 1, "chains": {"default": {"steps": ["codex"]}}}),
            encoding="utf-8",
        )

        materialise_default(tmp_path)

        # The operator owns the file once it exists — re-seeding would
        # silently revert their edit.
        assert _names(load_chains(tmp_path)[DEFAULT_CHAIN_NAME].steps) == ["codex"]

    def test_written_file_reloads_identically(self, tmp_path: Path):
        first = load_chains(tmp_path)
        second = load_chains(tmp_path)
        assert _names(first["default"].steps) == _names(second["default"].steps)


# ── The config is the source of truth, not Python ─────────────────────────


class TestConfigDrivesBehaviour:
    """Mutating the FILE must change failover. This is the anti-hardcode proof.

    Per feedback_ast_guard_proves_callers_not_helper_2026_08_31 these mutate
    both axes: the config document AND the call that consumes it.
    """

    def test_operator_order_is_followed(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["codex", "grok", "deepseek"]}}})

        chain = get_chain(home=tmp_path)

        assert _names(chain.steps) == ["codex", "grok", "deepseek"]
        assert _names(failover_candidates("codex", chain=chain, home=tmp_path)) == [
            "grok",
            "deepseek",
        ]

    def test_reordering_the_file_reorders_failover(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["codex", "grok"]}}})
        before = _names(
            failover_candidates("codex", chain=get_chain(home=tmp_path), home=tmp_path)
        )

        _write(tmp_path, {"chains": {"default": {"steps": ["grok", "codex"]}}})
        after = _names(
            failover_candidates("grok", chain=get_chain(home=tmp_path), home=tmp_path)
        )

        assert before == ["grok"]
        assert after == ["codex"]

    def test_no_chain_literal_remains_in_provider_chain_module(self):
        """The constant is gone, not merely unused."""
        import deep_worker_cli.runtime.provider_chain as pc

        assert not hasattr(pc, "PROVIDER_FAILOVER_CHAIN")


# ── Named chains (R2, R4) ─────────────────────────────────────────────────


class TestNamedChains:
    def test_agents_can_use_different_chains(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "chains": {
                    "default": {"steps": ["claude-code", "deepseek"]},
                    "cheap": {"steps": ["or-glm-52", "minimax"]},
                }
            },
        )

        assert _names(get_chain("default", tmp_path).steps) == ["claude-code", "deepseek"]
        assert _names(get_chain("cheap", tmp_path).steps) == ["or-glm-52", "minimax"]

    def test_absent_name_resolves_to_default(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["codex"]}}})

        assert get_chain(None, tmp_path).name == DEFAULT_CHAIN_NAME
        assert get_chain("", tmp_path).name == DEFAULT_CHAIN_NAME

    def test_unknown_chain_name_is_an_error(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["codex"]}}})

        with pytest.raises(FailoverChainConfigError, match="no chain named 'ghost'"):
            get_chain("ghost", tmp_path)

    def test_missing_default_is_an_error_not_a_fallback(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"other": {"steps": ["codex"]}}})

        with pytest.raises(FailoverChainConfigError, match="no 'default' chain"):
            load_chains(tmp_path)


# ── Composition is unrestricted; only existence is checked (R8, R9) ───────


class TestValidation:
    def test_config_driven_providers_are_first_class(self, tmp_path: Path):
        """The dead end this change exists to fix.

        Before chains were configurable, an off-chain provider got ``[]`` from
        ``failover_candidates`` and could neither leave nor be arrived at.
        """
        _write(
            tmp_path,
            {"chains": {"default": {"steps": ["or-glm-52", "minimax", "deepseek"]}}},
        )

        chain = get_chain(home=tmp_path)

        # Leaving a config-driven provider now works at all.
        assert _names(failover_candidates("or-glm-52", chain=chain, home=tmp_path)) == [
            "minimax",
            "deepseek",
        ]
        # And so does arriving at one from a built-in.
        assert "minimax" in _names(
            failover_candidates("claude-code", chain=chain, home=tmp_path)
        )

    def test_shared_account_group_is_honoured_for_config_driven(self, tmp_path: Path):
        """``or-*`` entries share one OpenRouter account, so one cap covers all."""
        _write(
            tmp_path,
            {"chains": {"default": {"steps": ["or-glm-52", "or-laguna", "deepseek"]}}},
        )

        got = failover_candidates("or-glm-52", chain=get_chain(home=tmp_path), home=tmp_path)

        assert _names(got) == ["deepseek"]

    def test_unknown_provider_is_rejected(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["ghost-provider"]}}})

        with pytest.raises(FailoverChainConfigError, match="unknown provider"):
            load_chains(tmp_path)

    def test_no_ordering_or_account_restriction_on_composition(self, tmp_path: Path):
        """An operator may write anything that exists — including siblings."""
        _write(
            tmp_path,
            {"chains": {"default": {"steps": ["claude-code-sdk", "claude-code", "codex"]}}},
        )

        assert _names(get_chain(home=tmp_path).steps) == [
            "claude-code-sdk",
            "claude-code",
            "codex",
        ]

    @pytest.mark.parametrize(
        "doc, match",
        [
            ({"chains": {}}, "non-empty object"),
            ({"chains": {"default": {"steps": []}}}, "non-empty 'steps'"),
            ({"chains": {"default": {"steps": [{}]}}}, "no 'provider'"),
            ({"chains": {"default": {"steps": [123]}}}, "must be a provider name"),
            ({"chains": {"default": "codex"}}, "must be a list of steps"),
        ],
    )
    def test_malformed_config_is_rejected_with_an_actionable_message(
        self, tmp_path: Path, doc: dict, match: str
    ):
        _write(tmp_path, doc)
        with pytest.raises(FailoverChainConfigError, match=match):
            load_chains(tmp_path)

    def test_invalid_json_is_rejected(self, tmp_path: Path):
        path = chains_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(FailoverChainConfigError, match="invalid JSON"):
            load_chains(tmp_path)

    def test_bare_string_step_is_sugar_for_provider(self, tmp_path: Path):
        _write(tmp_path, {"chains": {"default": {"steps": ["codex", {"provider": "grok"}]}}})

        steps = get_chain(home=tmp_path).steps
        assert steps == (FailoverStep("codex", None), FailoverStep("grok", None))


# ── Walking the chain (§4.5) ──────────────────────────────────────────────


class TestWalking:
    def test_candidates_are_steps_strictly_after_current(self, tmp_path: Path):
        chain = FailoverChain("c", (FailoverStep("codex"), FailoverStep("grok"), FailoverStep("deepseek")))

        assert _names(chain.candidates_after("grok")) == ["deepseek"]

    def test_last_step_yields_nothing(self, tmp_path: Path):
        chain = FailoverChain("c", (FailoverStep("codex"), FailoverStep("deepseek")))

        assert chain.candidates_after("deepseek") == ()

    def test_off_chain_provider_starts_at_step_zero(self, tmp_path: Path):
        """Review R3: 'after' is undefined when the provider isn't in the chain."""
        chain = FailoverChain("c", (FailoverStep("codex"), FailoverStep("deepseek")))

        assert _names(chain.candidates_after("openrouter")) == ["codex", "deepseek"]

    def test_off_chain_provider_still_skips_same_account_first_step(self, tmp_path: Path):
        """Review R3's edge case: start-at-zero must not defeat the account skip.

        ``claude-code-sdk`` is off-chain here, so the walk starts at step 0 —
        which draws on the same Claude subscription it just exhausted.
        """
        _write(tmp_path, {"chains": {"default": {"steps": ["claude-code", "codex"]}}})

        got = failover_candidates(
            "claude-code-sdk", chain=get_chain(home=tmp_path), home=tmp_path
        )

        assert _names(got) == ["codex"]

    def test_same_account_steps_are_skipped(self, tmp_path: Path):
        _write(
            tmp_path,
            {"chains": {"default": {"steps": ["claude-code", "claude-code-sdk", "codex"]}}},
        )

        got = failover_candidates("claude-code", chain=get_chain(home=tmp_path), home=tmp_path)

        assert _names(got) == ["codex"]


# ── Per-model steps (R5, R6, R7) ──────────────────────────────────────────


class TestModelSteps:
    def test_step_model_is_carried_to_the_caller(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "chains": {
                    "default": {
                        "steps": [
                            {"provider": "claude-code", "model": "opus"},
                            {"provider": "codex", "model": "gpt-5.2"},
                        ]
                    }
                }
            },
        )

        got = failover_candidates(
            "claude-code", chain=get_chain(home=tmp_path), model="opus", home=tmp_path
        )

        assert [(s.provider, s.model) for s in got] == [("codex", "gpt-5.2")]

    def test_per_model_failover_within_one_account(self, tmp_path: Path):
        """Owner directive: a model cap is not always an account cap.

        A step pinned to a DIFFERENT model survives the same-account filter —
        that pin is how the operator says the cap was per-model.
        """
        _write(
            tmp_path,
            {
                "chains": {
                    "default": {
                        "steps": [
                            {"provider": "claude-code", "model": "opus"},
                            {"provider": "claude-code", "model": "sonnet"},
                            {"provider": "codex"},
                        ]
                    }
                }
            },
        )

        got = failover_candidates(
            "claude-code", chain=get_chain(home=tmp_path), model="opus", home=tmp_path
        )

        assert [(s.provider, s.model) for s in got] == [
            ("claude-code", "sonnet"),
            ("codex", None),
        ]

    def test_unpinned_same_account_step_is_still_skipped(self, tmp_path: Path):
        """The exception is narrow: no model pinned means no per-model claim."""
        _write(
            tmp_path,
            {
                "chains": {
                    "default": {
                        "steps": [
                            {"provider": "claude-code", "model": "opus"},
                            {"provider": "claude-code"},
                            {"provider": "codex"},
                        ]
                    }
                }
            },
        )

        got = failover_candidates(
            "claude-code", chain=get_chain(home=tmp_path), model="opus", home=tmp_path
        )

        assert _names(got) == ["codex"]

    def test_position_matches_on_provider_and_model(self, tmp_path: Path):
        chain = FailoverChain(
            "tiers",
            (
                FailoverStep("claude-code", "opus"),
                FailoverStep("claude-code", "sonnet"),
                FailoverStep("codex"),
            ),
        )

        assert chain.index_of("claude-code", "sonnet") == 1
        # Unknown model falls back to the first step naming the provider.
        assert chain.index_of("claude-code", "haiku") == 0
        assert chain.index_of("claude-code") == 0

    def test_family_alias_is_passed_through_verbatim(self):
        """No family table in Python; the provider resolves it (design §4.2)."""
        assert resolve_step_model(FailoverStep("claude-code", "opus")) == "opus"
        assert resolve_step_model(FailoverStep("codex", "gpt-5.2")) == "gpt-5.2"

    def test_step_without_model_defers_to_the_provider(self):
        assert resolve_step_model(FailoverStep("codex")) is None


class TestNoProcessIsSpawned:
    """Resolving a step's model must never spawn a subprocess.

    Owner directive 2026-09-01: do not start a process to learn state we can
    read or forward. ``provider_max`` resolves family aliases by spawning a CLI;
    failover deliberately does not use it (design §4.2).
    """

    def test_resolve_step_model_spawns_nothing(self, monkeypatch: pytest.MonkeyPatch):
        import asyncio
        import subprocess

        def _boom(*args, **kwargs):
            raise AssertionError("failover must not spawn a process to resolve a model")

        monkeypatch.setattr(subprocess, "Popen", _boom)
        monkeypatch.setattr(subprocess, "run", _boom)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)

        assert resolve_step_model(FailoverStep("claude-code", "opus")) == "opus"
