"""Regression: an unspecified model must NOT be persisted / propagated
as the literal string ``"unknown"``.

Incident observed 2026-05-20: the agent was launched with
``--provider codex`` and no ``--model``. The codex provider is
designed to receive ``model=None`` so the Codex CLI picks its own
current default (see ``config.py:1688`` and
``DESIGN_CODEX_PROVIDER.md`` §15 Q2). Instead, the launch path turned
the missing model into the literal string ``"unknown"``:

    # cli/agent_command.py build_launch_config()
    model = get_default_model_name(args.provider) or "unknown"

For codex, ``get_default_model_name("codex")`` is ``None`` by design,
so ``model`` became ``"unknown"``. That string was then:

  1. persisted into ``session_metadata.json`` as ``"model": "unknown"``;
  2. read back on ``--restart`` (``merge_restart_config``) and passed
     as a real ``--model unknown`` CLI arg;
  3. forwarded to ``CodexChatModel(model="unknown")``, which the Codex
     API rejected with HTTP 400 ("The 'unknown' model is not
     supported when using Codex with a ChatGPT account").

The poison was self-propagating: once written to disk, every restart
re-read ``"unknown"`` and re-poisoned the next launch.

Contract under test: ``"unknown"`` is only ever an *informational*
placeholder for display/logging. It must NEVER reach the data path
(launch_config["model"], resolved_model, the value forwarded to
``create_model``). When a provider has no default model (codex),
the persisted / resolved model for an unspecified ``--model`` must be
``None``, not ``"unknown"``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.cli.agent_command import (
    build_launch_config,
    merge_restart_config,
    normalize_model_arg,
)


def _ns(**overrides: Any) -> argparse.Namespace:
    """Minimal argparse.Namespace as produced by the CLI parser."""
    base = dict(
        provider="codex",
        model=None,
        dir=Path("/tmp"),
        skip_permissions=True,
        task_manager=False,
        memory_only=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        dev_tree=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestBuildLaunchConfigCodexModel:
    def test_codex_without_model_persists_none_not_unknown(self) -> None:
        """codex + no --model → launch_config['model'] is None."""
        # Make sure no CODEX_MODEL env var leaks a value.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODEX_MODEL", None)
            cfg = build_launch_config(_ns(provider="codex", model=None))
        assert cfg["model"] is None, (
            f"codex with no --model must persist model=None so the Codex "
            f"CLI picks its default; got {cfg['model']!r}"
        )

    def test_codex_with_explicit_model_is_preserved(self) -> None:
        """An explicit --model is always honoured verbatim."""
        cfg = build_launch_config(_ns(provider="codex", model="gpt-5.5"))
        assert cfg["model"] == "gpt-5.5"

    def test_provider_with_real_default_still_resolves(self) -> None:
        """deepseek (which HAS a default) still gets it filled in."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPSEEK_MODEL", None)
            cfg = build_launch_config(_ns(provider="deepseek", model=None))
        assert cfg["model"] == "deepseek-v4-flash"

    def test_unknown_never_appears_as_model(self) -> None:
        """Belt-and-suspenders: 'unknown' must never be the model value."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODEX_MODEL", None)
            cfg = build_launch_config(_ns(provider="codex", model=None))
        assert cfg["model"] != "unknown"


class TestRestartConfigDoesNotPropagateUnknown:
    def test_restart_reads_none_model_not_unknown(self) -> None:
        """A clean (None) saved model stays None across restart."""
        saved = {
            "provider": "codex",
            "model": None,
            "working_dir": "/path/to/project",
            "skip_permissions": True,
        }
        args = _ns(provider="codex", model=None)
        effective = merge_restart_config(saved, args)
        assert effective.model is None

    def test_restart_heals_legacy_unknown_in_saved_config(self) -> None:
        """A legacy session_metadata.json poisoned with 'unknown' must
        be HEALED on restart, not re-propagated. After the fix, a saved
        ``"unknown"`` is treated as 'no model specified' (→ None) so the
        provider default kicks back in instead of forwarding the bad
        string to the API."""
        saved = {
            "provider": "codex",
            "model": "unknown",  # poisoned by the old bug
            "working_dir": "/path/to/project",
            "skip_permissions": True,
        }
        args = _ns(provider="codex", model=None)
        effective = merge_restart_config(saved, args)
        assert effective.model != "unknown", (
            "merge_restart_config must heal a legacy 'unknown' model "
            "rather than forward it as a real model name"
        )
        assert effective.model is None


class TestResolvedModelNoUnknownForCodex:
    def test_codex_resolved_model_is_none_not_unknown(self) -> None:
        """AgentConfig.resolved_model for codex+None must be None, so
        ``create_model`` forwards None to ``CodexChatModel`` and the CLI
        picks its default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CODEX_MODEL", None)
            config = AgentConfig(
                working_dir=Path("/tmp"), model=None, provider="codex"
            )
        assert config.resolved_model != "unknown", (
            "codex resolved_model must not be the literal 'unknown' — "
            "that string reaches the Codex API and is rejected with 400"
        )
        assert config.resolved_model is None

    def test_explicit_model_still_returned(self) -> None:
        config = AgentConfig(
            working_dir=Path("/tmp"), model="gpt-5.5", provider="codex"
        )
        assert config.resolved_model == "gpt-5.5"

    def test_deepseek_default_unaffected(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}, clear=False):
            os.environ.pop("DEEPSEEK_MODEL", None)
            config = AgentConfig(
                working_dir=Path("/tmp"), model=None, provider="deepseek"
            )
        assert config.resolved_model == "deepseek-v4-flash"


class TestNormalizeModelArg:
    """``normalize_model_arg`` is the single CLI-entry chokepoint that
    treats an explicit ``--model unknown`` as 'no model specified'.

    This catches the case the web restart path produced: the agent was
    relaunched DIRECTLY with ``--model unknown`` (copied from the
    poisoned cli_args), bypassing merge_restart_config's healing. The
    normalization runs right after ``parse_args`` so EVERY launch path —
    direct, --restart, or web-spawned — is covered."""

    def test_unknown_becomes_none(self) -> None:
        args = _ns(model="unknown")
        normalize_model_arg(args)
        assert args.model is None

    def test_case_insensitive_unknown_becomes_none(self) -> None:
        args = _ns(model="UNKNOWN")
        normalize_model_arg(args)
        assert args.model is None

    def test_real_model_preserved(self) -> None:
        args = _ns(model="gpt-5.5")
        normalize_model_arg(args)
        assert args.model == "gpt-5.5"

    def test_none_stays_none(self) -> None:
        args = _ns(model=None)
        normalize_model_arg(args)
        assert args.model is None

    def test_missing_model_attr_is_safe(self) -> None:
        """Defensive: a namespace without a model attr must not raise."""
        import argparse as _argparse
        args = _argparse.Namespace()
        normalize_model_arg(args)  # must not raise
