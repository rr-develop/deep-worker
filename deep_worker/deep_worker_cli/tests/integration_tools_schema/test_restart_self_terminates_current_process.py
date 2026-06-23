"""Critical contract: restart_self MUST terminate the current process
*immediately* after spawning the replacement subprocess.

Why: the previous implementation simply called subprocess.Popen and then
``return {"success": True}``. The result was a race:
  - The new process started in parallel.
  - Meanwhile, the LangGraph turn in the *old* process kept running. The
    LLM finalised the assistant message, the turn ended, and the auto-
    continuation that the new process emitted afterwards was treated as a
    *separate* (idle) turn — exactly the failure mode the user reported.

The correct semantics: after Popen succeeds the tool MUST hard-exit the
current process (``os._exit``) so the current LangGraph turn is severed.
The checkpointer keeps the last persisted state (with the pending tool
call). The replacement process resumes from that state, consumes the
restart-notice, and its auto-continuation becomes the SAME logical turn
that was in flight.

These tests pin the contract:

  1. On successful Popen, the tool calls ``os._exit(0)``.
  2. Hard-exit happens AFTER Popen (subprocess must already be running).
  3. Failure paths (notice write error, deep-worker not on PATH,
     Popen OSError) DO NOT hard-exit — they return a structured error
     so the agent can react.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_restart_self_hard_exits_after_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool must call os._exit(0) right after spawning the replacement."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_hardexit_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    popen_calls: list[list] = []
    exit_calls: list[int] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)
            self.pid = 99999

    def _fake_exit(code: int) -> None:
        exit_calls.append(code)
        # We intentionally do NOT actually exit — we want the test to
        # observe that the tool called it.

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.os, "_exit", _fake_exit)

    tool = make_restart_self_tool(agent_name)
    await tool.ainvoke({})

    # Popen happened first.
    assert len(popen_calls) == 1, popen_calls
    # Hard-exit happened.
    assert exit_calls == [0], (
        f"restart_self MUST call os._exit(0) after Popen — got {exit_calls}"
    )


@pytest.mark.asyncio
async def test_restart_self_exit_happens_after_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequence must be Popen → then os._exit. Never the reverse."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_seq_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    events: list[str] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            events.append("popen")
            self.pid = 99999

    def _fake_exit(code: int) -> None:
        events.append(f"exit:{code}")

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.os, "_exit", _fake_exit)

    tool = make_restart_self_tool(agent_name)
    await tool.ainvoke({})

    assert events == ["popen", "exit:0"], (
        f"Sequence must be Popen→exit; got {events}"
    )


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_when_deep_worker_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If deep-worker is not on PATH the tool must NOT hard-exit — there
    is no replacement coming. Return a structured error instead."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_nodw_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    monkeypatch.setenv("PATH", "/nonexistent_directory")

    exit_calls: list[int] = []
    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)

    def _fake_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.os, "_exit", _fake_exit)

    tool = make_restart_self_tool(agent_name)
    result = await tool.ainvoke({})

    assert result["success"] is False
    assert result["error"] == "command_unavailable"
    # CRITICAL: do not exit — agent must keep running so it can report.
    assert exit_calls == [], (
        f"Must NOT hard-exit when deep-worker is missing; got {exit_calls}"
    )
    assert popen_calls == []


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_on_popen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Popen itself raises OSError the tool must NOT hard-exit."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_popenfail_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    exit_calls: list[int] = []

    def _failing_popen(*args, **kwargs):
        raise OSError("simulated spawn failure")

    def _fake_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _failing_popen)
    monkeypatch.setattr(tools_mod.os, "_exit", _fake_exit)

    tool = make_restart_self_tool(agent_name)
    result = await tool.ainvoke({})

    assert result["success"] is False
    assert result["error"] == "spawn_failed"
    assert exit_calls == [], (
        f"Must NOT hard-exit when Popen failed; got {exit_calls}"
    )


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_when_notice_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we cannot even write the notice file the tool must NOT hard-exit
    — the replacement process would have no way to know it was restarted."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_noticefail_{os.getpid()}"
    # Point HOME at a path we cannot write to.
    bad_home = tmp_path / "ro_home"
    bad_home.mkdir(mode=0o500)
    monkeypatch.setattr(Path, "home", lambda: bad_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: bad_home)

    exit_calls: list[int] = []
    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)

    def _fake_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.os, "_exit", _fake_exit)

    tool = make_restart_self_tool(agent_name)
    result = await tool.ainvoke({})

    # Restore perms so tmp_path can be cleaned up.
    bad_home.chmod(0o700)

    assert result["success"] is False
    assert result["error"] == "notice_write_error"
    assert exit_calls == [], exit_calls
    assert popen_calls == []
