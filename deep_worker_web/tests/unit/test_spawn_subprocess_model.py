"""Regression: the web restart/spawn path must never relaunch an agent
with ``--model unknown``.

Past incident: ``_spawn_subprocess`` built the agent
command line as

    "--model", model or config.get("model", "unknown")

When the operator restarted that agent from the web UI, ``model`` was
None and ``config["model"]`` carried the legacy poison ``"unknown"``
(from a session_metadata.json written by the old build). The agent was
therefore relaunched with a literal ``--model unknown``, which Codex
rejected with HTTP 400 and the agent went straight to ERROR.

Contract under test:
  * A real model is passed through as ``--model <name>``.
  * ``"unknown"`` (from either the explicit ``model`` arg or the saved
    config) must NOT appear on the command line — the agent then picks
    its provider default.
  * A genuinely absent model must NOT emit a ``--model`` flag at all
    (no ``--model None`` / ``--model unknown`` fallback).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.routers.agents import _spawn_subprocess


def _capture_cmd(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch subprocess.Popen + file IO so ``_spawn_subprocess`` runs
    without launching anything, and capture the argv it built."""
    from deep_worker_cli.cli.agent_command import _BOOLEAN_FLAGS  # noqa: F401

    captured: list[list[str]] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        captured.append(cmd)
        return MagicMock()

    monkeypatch.setattr("app.routers.agents.subprocess.Popen", fake_popen)
    # Avoid touching the real log dir / opening files.
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return captured


def _model_value_after_flag(cmd: list[str]) -> str | None:
    """Return the value following ``--model`` in cmd, or None if absent."""
    if "--model" not in cmd:
        return None
    return cmd[cmd.index("--model") + 1]


BASE_CONFIG = {
    "provider": "codex",
    "working_dir": "/tmp/agent_work",
    "skip_permissions": True,
}


def test_real_model_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "gpt-5.5"}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) == "gpt-5.5"


def test_unknown_in_config_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The poisoned saved config must NOT yield --model unknown."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "unknown"}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    val = _model_value_after_flag(captured[0])
    assert val != "unknown", (
        f"web spawn must not relaunch with --model unknown; cmd was "
        f"{captured[0]}"
    )


def test_unknown_explicit_arg_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "unknown"}
    _spawn_subprocess("agent1", cfg, model="unknown", provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) != "unknown"


def test_absent_model_emits_no_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model anywhere → no --model flag, never --model None/unknown."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": None}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    cmd = captured[0]
    # A None/unknown model produces NO --model flag — the agent then
    # picks its provider default. (Crucially, never --model None, which
    # used to crash the spawn with a TypeError in the log line.)
    assert "--model" not in cmd, (
        f"absent model must not emit a --model flag; cmd was {cmd}"
    )


def test_provider_still_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: provider is unaffected by the model fix."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "unknown"}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    cmd = captured[0]
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "codex"


def test_provider_override_drops_saved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "provider": "claude-code-sdk", "model": "opus"}

    _spawn_subprocess("agent1", cfg, model=None, provider="codex", clean=False)

    cmd = captured[0]
    assert cmd[cmd.index("--provider") + 1] == "codex"
    assert "--model" not in cmd


# ---------------------------------------------------------------------------
# Past incident: Claude Code model aliases were silently dropped.
#
# ``_model_can_belong_to_provider`` in this module was a hand-copied mirror of
# the CLI predicate and carried the same stale four-token whitelist. A
# ``fable`` model in the saved config therefore failed the check and the web
# spawned the agent with NO ``--model`` at all, so the CLI fell back to the
# hardcoded ``sonnet`` default. The operator got a different model than the
# one their config said they were running, with nothing on screen.
#
# The predicate now lives in the leaf module ``deep_worker_cli.model_predicate``
# so the CLI and web cannot diverge again, and a drop emits a server-side
# warning log (NOT a browser-visible message — the endpoint response and UI
# are deliberately unchanged; see design §4.3.2).
# ---------------------------------------------------------------------------

CLAUDE_CONFIG = {
    "provider": "claude-code-sdk",
    "working_dir": "/tmp/agent_work",
    "skip_permissions": True,
}


@pytest.mark.parametrize(
    "alias",
    ["fable", "best", "default", "opusplan", "sonnet[1m]", "opus[1m]",
     "claude-fable-5", "claude-haiku-4-5-20251001"],
)
def test_claude_code_aliases_passed_through(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    """Every documented alias must survive the web spawn path."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**CLAUDE_CONFIG, "model": alias}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) == alias, (
        f"web spawn dropped the valid claude-code-sdk alias {alias!r}; "
        f"cmd was {captured[0]}"
    )


def test_fable_regression_vc15_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact incident: saved fable must not be silently omitted."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**CLAUDE_CONFIG, "model": "fable"}
    _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)
    assert "--model" in captured[0]
    assert _model_value_after_flag(captured[0]) == "fable"


def test_explicit_fable_arg_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**CLAUDE_CONFIG, "model": "sonnet"}
    _spawn_subprocess("agent1", cfg, model="fable", provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) == "fable"


def test_web_uses_shared_leaf_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The web must not keep a private copy of the predicate.

    The duplicate is what let the two implementations drift apart in the
    first place. Delegation is asserted structurally: the router's predicate
    must BE the leaf function, not merely agree with it today.
    """
    from deep_worker_cli.model_predicate import (
        model_can_belong_to_provider as leaf_predicate,
    )

    import app.routers.agents as agents_mod

    assert agents_mod._model_can_belong_to_provider is leaf_predicate, (
        "web router must delegate to deep_worker_cli.model_predicate rather "
        "than maintaining its own copy of the whitelist"
    )


def test_invalid_model_still_dropped_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A genuinely foreign model is still dropped — and now says so.

    The web contract is a SERVER-SIDE warning log; the HTTP response and the
    browser UI are intentionally not extended (design §4.3.2).
    """
    import logging

    captured = _capture_cmd(monkeypatch)
    cfg = {**CLAUDE_CONFIG, "model": "grok-composer-2.5-fast"}
    with caplog.at_level(logging.WARNING):
        _spawn_subprocess("agent1", cfg, model=None, provider=None, clean=False)

    assert "--model" not in captured[0], "foreign model must still be dropped"
    assert any(
        "grok-composer-2.5-fast" in rec.getMessage() for rec in caplog.records
    ), (
        "dropping a model must be reported in the server log; records were "
        f"{[r.getMessage() for r in caplog.records]}"
    )
