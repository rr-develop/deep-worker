"""Claude Code model aliases must not be silently dropped.

Past incident: the operator ran

    deep-worker --restart <agent> --model fable

and the agent booted on ``sonnet`` without a single warning. Root cause:
``model_can_belong_to_provider`` used a four-token whitelist
(``sonnet``/``haiku``/``opus``/``inherit`` + substring ``claude``) that
predates the current Claude Code CLI. ``fable`` failed the predicate, so
``merge_restart_config`` nulled the model and ``build_launch_config``
substituted the hardcoded ``"sonnet"`` default.

The authoritative alias list is the "Model aliases" table in
https://code.claude.com/docs/en/model-config — ``default``, ``best``,
``fable``, ``sonnet``, ``opus``, ``haiku``, ``sonnet[1m]``, ``opus[1m]``,
``opusplan``.

These tests pin the predicate itself. The leaf module
``deep_worker_cli.model_predicate`` is the single source of truth shared by
``deep_worker_cli.config`` and the web layer, which historically kept a
divergent copy precisely because importing ``config`` drags in heavy deps.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deep_worker_cli.config import model_can_belong_to_provider
from deep_worker_cli.model_predicate import model_can_belong_to_provider as leaf_predicate

# Official short aliases (code.claude.com/docs/en/model-config §Model aliases)
DOCUMENTED_ALIASES = [
    "default",
    "best",
    "fable",
    "sonnet",
    "opus",
    "haiku",
    "sonnet[1m]",
    "opus[1m]",
    "opusplan",
]

FULL_MODEL_IDS = [
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5[1m]",
]


class TestClaudeCodeShortAliases:
    """The documented alias set must be accepted for claude-code."""

    @pytest.mark.parametrize("alias", DOCUMENTED_ALIASES)
    def test_documented_alias_accepted(self, alias: str):
        assert model_can_belong_to_provider("claude-code", alias) is True, (
            f"{alias!r} is a documented Claude Code alias and must not be "
            "dropped; dropping it silently substitutes the provider default"
        )

    def test_fable_accepted_regression(self):
        """The exact failure from that incident."""
        assert model_can_belong_to_provider("claude-code", "fable") is True

    @pytest.mark.parametrize("model_id", FULL_MODEL_IDS)
    def test_full_model_id_accepted(self, model_id: str):
        assert model_can_belong_to_provider("claude-code", model_id) is True

    def test_legacy_inherit_still_accepted(self):
        """Backward compat: older persisted configs may carry ``inherit``."""
        assert model_can_belong_to_provider("claude-code", "inherit") is True

    @pytest.mark.parametrize("alias", DOCUMENTED_ALIASES)
    def test_alias_case_insensitive(self, alias: str):
        assert model_can_belong_to_provider("claude-code", alias.upper()) is True

    @pytest.mark.parametrize(
        "foreign",
        ["gpt-5", "gpt-5.2", "grok-4", "grok-composer-2.5-fast", "deepseek-v4-flash",
         "gemini-3-pro-preview"],
    )
    def test_foreign_model_rejected(self, foreign: str):
        assert model_can_belong_to_provider("claude-code", foreign) is False

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_rejected(self, empty):
        assert model_can_belong_to_provider("claude-code", empty) is False

    def test_unknown_sentinel_rejected(self):
        """``unknown`` is the poison string from a past incident."""
        assert model_can_belong_to_provider("claude-code", "unknown") is False


class TestLeafDelegation:
    """``config`` must delegate to the leaf so the two cannot diverge."""

    @pytest.mark.parametrize("alias", DOCUMENTED_ALIASES + FULL_MODEL_IDS)
    def test_config_matches_leaf_for_claude_code(self, alias: str):
        assert model_can_belong_to_provider("claude-code", alias) is leaf_predicate(
            "claude-code", alias
        )

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("google", "gemini-3-pro-preview"),
            ("deepseek", "deepseek-v4-flash"),
            ("grok", "grok-4"),
            ("openai", "gpt-5.2"),
            ("codex", "gpt-5.2"),
            ("openai", "claude-sonnet-5"),
            ("grok", "gpt-5"),
        ],
    )
    def test_other_providers_unchanged(self, provider: str, model: str):
        """Widening claude-code must not perturb any other provider."""
        assert model_can_belong_to_provider(provider, model) is leaf_predicate(
            provider, model
        )

    def test_invalid_provider_rejected(self):
        assert leaf_predicate("", "fable") is False
        assert leaf_predicate(None, "fable") is False


class TestLeafHasNoHeavyDependencies:
    """The web layer imports the leaf; it must stay import-cheap.

    The web layer kept a divergent copy of this predicate specifically
    because importing ``deep_worker_cli.config`` pulls in langchain and
    friends, which breaks the web unit-test environment. The leaf only
    removes that duplication if it is genuinely dependency-free.
    """

    def test_import_pulls_no_langchain_or_transformers(self):
        code = (
            "import sys;"
            "import deep_worker_cli.model_predicate as m;"
            "assert m.model_can_belong_to_provider('claude-code', 'fable');"
            "heavy=[n for n in sys.modules"
            " if n.split('.')[0] in ('langchain','transformers','torch','deepagents')];"
            "print('HEAVY:'+','.join(sorted(heavy)))"
        )
        # Hermetic: the subprocess must not depend on an import path that
        # only exists because pytest happened to be invoked from inside the
        # package directory. Derive the package root from THIS file and put
        # it on PYTHONPATH explicitly, preserving any existing value.
        # (Without this the test passes from deep_worker_cli/ and fails from
        # the repo root — a false green that hid nothing but wasted a review
        # round.)
        pkg_root = Path(__file__).resolve().parents[3]
        assert (pkg_root / "deep_worker_cli" / "model_predicate.py").is_file(), (
            f"package root misresolved: {pkg_root}"
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{pkg_root}{os.pathsep}{existing}" if existing else str(pkg_root)
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(pkg_root),
        )
        assert result.returncode == 0, (
            f"leaf module failed to import standalone:\n{result.stderr}"
        )
        heavy_line = [
            ln for ln in result.stdout.splitlines() if ln.startswith("HEAVY:")
        ]
        assert heavy_line, f"probe produced no output: {result.stdout!r}"
        heavy = heavy_line[0][len("HEAVY:") :].strip()
        assert heavy == "", (
            f"model_predicate must be a leaf module, but importing it loaded: {heavy}"
        )
