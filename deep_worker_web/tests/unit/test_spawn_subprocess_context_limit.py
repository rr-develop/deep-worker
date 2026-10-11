"""Web spawn must re-emit a persisted ``context_limit`` on the CLI line.

Web restart does not go through ``--restart`` / ``merge_restart_config`` —
it reconstructs a fresh argv from ``launch_config``. Omitting
``--context_limit`` would wipe a saved operator cap on every web restart.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.routers.agents import _spawn_subprocess


def _capture_cmd(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        captured.append(cmd)
        return MagicMock()

    monkeypatch.setattr("app.routers.agents.subprocess.Popen", fake_popen)
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return captured


def test_saved_context_limit_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": "/tmp",
        "context_limit": 400_000,
    }
    _spawn_subprocess("agent", cfg)
    cmd = captured[0]
    assert "--context_limit" in cmd
    assert cmd[cmd.index("--context_limit") + 1] == "400000"


def test_absent_context_limit_not_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_cmd(monkeypatch)
    cfg = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": "/tmp",
    }
    _spawn_subprocess("agent", cfg)
    assert "--context_limit" not in captured[0]
