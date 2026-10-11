"""Mechanism-level test: restart_self is SYNCHRONOUS — no flag-file,
no "after turn completes", just immediate Popen + notice write.

This test does NOT spin up a model. It directly invokes the tool and
verifies:

  1. The tool's RETURN payload announces an immediate restart (no
     "scheduled"/"after turn"/idle language).
  2. A restart-notice JSON file is written with {ts, cause: "self"}.
  3. Popen is invoked with ["deep-worker", "--restart", <name>] — after
     the tool result has been delivered, not during the call. The restart
     is still immediate from the agent's point of view (no "after the turn
     completes", no idle hop); what is ordered is the spawn against the
     result, so the transcript can never hold a dangling tool_use for the
     CLI to fabricate a user denial from.
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


from deep_worker_cli.storage_paths import agent_root, deep_worker_home
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

    from deep_worker_cli.process import self_restart_barrier as barrier

    barrier.reset_armed_self_restart_for_test()
    barrier.set_self_restart_delivery_available_for_test(True)
    monkeypatch.setattr(barrier.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)

    tool = make_restart_self_tool(agent_name)
    # Invoke as LangGraph does, so the InjectedToolCallId is populated: the
    # tool arms itself already bound to this id.
    result = await tool.ainvoke(
        {"args": {}, "id": "toolu_sync", "name": tool.name, "type": "tool_call"}
    )
    if hasattr(result, "content"):
        result = json.loads(result.content)

    # 1. The tool returns a real result instead of hard-exiting. The old
    #    os._exit(0) killed the in-process MCP server before the result
    #    reached the CLI, which then fabricated a "user rejected" denial
    #    (2026-08-28). Termination is the spawned driver's job now.
    assert result["success"] is True, result
    assert not hasattr(tools_mod, "os"), (
        "tools.py must not import os — its only use was the os._exit(0) "
        "that caused the fabricated user-denial"
    )

    # 2. Restart-notice file must exist with {ts, cause:"self"}.
    notice_path = agent_root(agent_name, home=fake_home, create=False) / ".restart_notice.json"
    assert notice_path.exists(), f"Notice file not at {notice_path}"
    notice = json.loads(notice_path.read_text())
    assert notice["cause"] == "self", notice
    assert "ts" in notice, notice
    # ts must be ISO-8601 string.
    assert isinstance(notice["ts"], str) and "T" in notice["ts"], notice

    # 3. The driver is spawned with deep-worker --restart — but only once
    #    the tool result has been recorded in the model's context. Spawning
    #    it during the call would let the driver SIGTERM this process while
    #    the result is still unrecorded, stranding the tool_use and
    #    resurrecting the fabricated denial through a narrower window.
    assert popen_calls == [], (
        "the driver must not be spawned while the result is unrecorded"
    )
    # Delivery is the checkpoint commit: the checkpointer's commit listener
    # hands the barrier the tool_call_ids that were committed.
    assert barrier.fire_armed_self_restart(frozenset({"toolu_sync"})) is True
    assert len(popen_calls) == 1, popen_calls
    argv = popen_calls[0]
    assert "--restart" in argv, argv
    assert agent_name in argv, argv

    # 4. NO old-style flag-file.
    flag_path = agent_root(agent_name, home=fake_home, create=False) / ".restart_self"
    assert not flag_path.exists(), (
        f"Old deferred-restart flag should NOT exist at {flag_path}"
    )

    # 5. The driver is told the stop is self-sanctioned, otherwise the
    #    "actively running" guard refuses it (the caller is mid-turn by
    #    construction) and no replacement is ever started. The flag must be
    #    backed by the one-time capability token — on its own it is a claim
    #    any external shell could make.
    assert "--self-requested" in argv, argv
    assert "--self-requested-token" in argv, argv

    from deep_worker_cli.process.self_restart_token import TOKEN_FILENAME

    token_path = (
        agent_root(agent_name, home=fake_home, create=False) / TOKEN_FILENAME
    )
    assert token_path.exists(), "restart_self must mint a capability token"
    on_disk = json.loads(token_path.read_text())["token"]
    assert argv[argv.index("--self-requested-token") + 1] == on_disk, argv


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
    from deep_worker_cli.process import self_restart_barrier as barrier

    barrier.set_self_restart_delivery_available_for_test(True)
    # No hard-exit is possible any more (tools.py does not import os), so
    # the agent is guaranteed to survive and report the error.
    tool = make_restart_self_tool(agent_name)
    result = await tool.ainvoke(
        {"args": {}, "id": "toolu_nodw", "name": tool.name, "type": "tool_call"}
    )
    if hasattr(result, "content"):
        result = json.loads(result.content)

    assert result["success"] is False
    assert result.get("error") == "command_unavailable", result
