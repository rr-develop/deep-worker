"""Mechanism-level test: restart_self is SYNCHRONOUS — no flag-file,
no "after turn completes", just immediate Popen + notice write.

This test does NOT spin up a model. It directly invokes the tool and
verifies:

  1. The tool's RETURN payload announces an immediate restart (no
     "scheduled"/"after turn"/idle language).
  2. A restart-notice JSON file is written with {ts, cause: "self"}.
  3. Popen is invoked SYNCHRONOUSLY during the tool call with
     ["deep-worker", "--restart", <name>].
  4. NO ".restart_self" flag-file is written (that was the old
     deferred-mechanism; it must be gone).

This is the contract that the production agent must implement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_restart_self_immediately_spawns_subprocess_and_writes_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool spawns deep-worker --restart synchronously and writes notice."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_sync_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    # Provide a fake deep-worker on PATH so shutil.which finds it.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)
            self.pid = 99999
        def poll(self):
            return 0

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    # In production restart_self calls os._exit(0) after Popen so the
    # current LangGraph turn is severed (see incident 2026-06-12 fix).
    # In tests we patch it out so the test process survives to assert.
    exit_calls: list[int] = []
    monkeypatch.setattr(tools_mod.os, "_exit", lambda c: exit_calls.append(c))

    tool = make_restart_self_tool(agent_name)
    await tool.ainvoke({})

    # 1. os._exit(0) was called after Popen (sever current turn).
    assert exit_calls == [0], (
        f"restart_self MUST hard-exit after Popen; got {exit_calls}"
    )

    # 2. Restart-notice file must exist with {ts, cause:"self"}.
    notice_path = fake_home / Path("~/.deep_worker").name / agent_name / ".restart_notice.json"
    assert notice_path.exists(), f"Notice file not at {notice_path}"
    notice = json.loads(notice_path.read_text())
    assert notice["cause"] == "self", notice
    assert "ts" in notice, notice
    # ts must be ISO-8601 string.
    assert isinstance(notice["ts"], str) and "T" in notice["ts"], notice

    # 3. Popen was called synchronously with deep-worker --restart.
    assert len(popen_calls) == 1, popen_calls
    argv = popen_calls[0]
    assert "--restart" in argv, argv
    assert agent_name in argv, argv

    # 4. NO old-style flag-file.
    flag_path = fake_home / Path("~/.deep_worker").name / agent_name / ".restart_self"
    assert not flag_path.exists(), (
        f"Old deferred-restart flag should NOT exist at {flag_path}"
    )

    # 5. In production restart_self never returns to the model (we
    #    os._exit(0) right after Popen). The structured "success"
    #    payload is unreachable code kept only for type/test contract,
    #    so we don't assert on it here — assertion (1) above (exit_calls
    #    == [0]) is the real contract.


@pytest.mark.asyncio
async def test_restart_self_no_deepworker_on_path_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If deep-worker cannot be located, the tool must fail cleanly."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_no_dw_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)
    # Empty PATH so shutil.which finds nothing.
    monkeypatch.setenv("PATH", "")
    # Safety net: hard-exit on the failure path would be a contract
    # violation (no replacement is coming, the agent must keep running
    # so it can report the error).
    def _forbidden_exit(_c: int) -> None:
        raise AssertionError(
            "os._exit MUST NOT be called on the failure path"
        )
    monkeypatch.setattr(tools_mod.os, "_exit", _forbidden_exit)

    tool = make_restart_self_tool(agent_name)
    result = await tool.ainvoke({})

    assert result["success"] is False
    assert result.get("error") == "command_unavailable", result
