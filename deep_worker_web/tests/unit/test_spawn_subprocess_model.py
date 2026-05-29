"""Regression: the web restart/spawn path must never relaunch an agent
with ``--model unknown``.

Incident observed 2026-05-20: ``_spawn_subprocess`` built the agent
command line as

    "--model", model or config.get("model", "unknown")

When the operator restarted the agent from the web UI, ``model`` was
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
    "working_dir": "/path/to/other",
    "skip_permissions": True,
}


def test_real_model_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "gpt-5.5"}
    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) == "gpt-5.5"


def test_unknown_in_config_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The poisoned saved config must NOT yield --model unknown."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "unknown"}
    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)
    val = _model_value_after_flag(captured[0])
    assert val != "unknown", (
        f"web spawn must not relaunch with --model unknown; cmd was "
        f"{captured[0]}"
    )


def test_unknown_explicit_arg_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": "unknown"}
    _spawn_subprocess("agent_a", cfg, model="unknown", provider=None, clean=False)
    assert _model_value_after_flag(captured[0]) != "unknown"


def test_absent_model_emits_no_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model anywhere → no --model flag, never --model None/unknown."""
    captured = _capture_cmd(monkeypatch)
    cfg = {**BASE_CONFIG, "model": None}
    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)
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
    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)
    cmd = captured[0]
    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "codex"
