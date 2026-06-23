"""Tests for restart_self tool: make_restart_self_tool factory.

The tool is SYNCHRONOUS and HARD-EXITS the current process after
spawning the replacement, so the in-flight LangGraph turn is severed
(see the 2026-06-12 incident). There is no flag-file, no
"already_restarting" state, and no "after the turn completes" semantics.

In production ``os._exit(0)`` is called right after ``subprocess.Popen``
and the return value below it is unreachable. In tests we patch
``os._exit`` so the test process survives to assert on side effects
(notice file, Popen args, exit_calls).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


async def _ainvoke(tool, **kwargs):
    return await tool.ainvoke(kwargs)


class TestRestartSelfFactory:
    def test_factory_returns_tool_with_expected_name(self):
        from deep_worker_cli.tools import make_restart_self_tool

        tool = make_restart_self_tool("alpha")
        assert tool.name == "restart_self"

    def test_tool_has_no_args_schema(self):
        from deep_worker_cli.tools import make_restart_self_tool

        tool = make_restart_self_tool("alpha")
        schema = tool.args_schema
        assert schema is None or len(getattr(schema, "model_fields", {})) == 0


class TestRestartSelfBehavior:
    def _setup(self, tmp_path, monkeypatch):
        """Make the tool see tmp_path as $HOME and find a fake deep-worker.

        Also patches ``os._exit`` so the test process survives the
        production hard-exit. Returns ``(popen_calls, exit_calls)``.
        """
        from deep_worker_cli import tools as tools_mod

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(tools_mod.Path, "home", lambda: tmp_path)

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir(exist_ok=True)
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

        exit_calls: list[int] = []
        monkeypatch.setattr(
            tools_mod.os, "_exit", lambda c: exit_calls.append(c)
        )
        return popen_calls, exit_calls

    async def test_restart_self_writes_notice_file(self, tmp_path, monkeypatch):
        """Tool writes .restart_notice.json in agent's data dir."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_self_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        notice_path = tmp_path / Path("~/.deep_worker").name / agent_name / ".restart_notice.json"
        assert notice_path.exists()
        notice = json.loads(notice_path.read_text())
        assert notice["cause"] == "self"
        assert "ts" in notice

    async def test_restart_self_spawns_deep_worker_synchronously(
        self, tmp_path, monkeypatch
    ):
        """Each tool call spawns ONE deep-worker --restart subprocess."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_spawn_{os.getpid()}"
        popen_calls, _exit_calls = self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        assert len(popen_calls) == 1
        argv = popen_calls[0]
        assert "--restart" in argv
        assert agent_name in argv

    async def test_restart_self_hard_exits_after_popen(
        self, tmp_path, monkeypatch
    ):
        """Tool calls os._exit(0) AFTER Popen — severs the current turn."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_exit_{os.getpid()}"
        popen_calls, exit_calls = self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        # The contract: exit(0) MUST be called, and AFTER Popen succeeded.
        assert exit_calls == [0]
        assert len(popen_calls) == 1

    async def test_restart_self_notice_written_with_private_perms(
        self, tmp_path, monkeypatch
    ):
        """Notice file is created with private permissions (0o600)."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_perm_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        notice_path = tmp_path / Path("~/.deep_worker").name / agent_name / ".restart_notice.json"
        assert notice_path.exists()
        mode = notice_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
