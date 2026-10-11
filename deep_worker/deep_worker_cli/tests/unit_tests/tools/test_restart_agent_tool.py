from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.types import AgentProcessInfo


async def _ainvoke(tool: Any, **kwargs: Any) -> Any:
    return await tool.ainvoke(kwargs)


def _process_info(name: str = "beta", pid: int = 12345) -> AgentProcessInfo:
    return AgentProcessInfo(
        name=name,
        pid=pid,
        pid_file=Path(f"/tmp/deep_worker_{name}.pid"),
        working_dir="/tmp",
        started_at=datetime.now(),
        rpyc_port=18861,
    )


class TestRestartAgentFactory:
    def test_factory_returns_tool_with_expected_name(self) -> None:
        from deep_worker_cli.tools import make_restart_agent_tool

        tool = make_restart_agent_tool("alpha")
        assert tool.name == "restart_agent"

    def test_schema_exposes_only_agent_name(self) -> None:
        from deep_worker_cli.tools import make_restart_agent_tool

        tool = make_restart_agent_tool("alpha")
        schema = tool.args_schema
        if hasattr(schema, "model_fields"):
            field_names = set(schema.model_fields.keys())
        else:
            field_names = set(schema.__fields__.keys())

        assert field_names == {"agent_name"}


class TestRestartAgentValidation:
    async def test_empty_agent_name_returns_invalid_argument(self) -> None:
        from deep_worker_cli.tools import make_restart_agent_tool

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name=" ")

        assert result["success"] is False
        assert result["error"] == "invalid_argument"

    async def test_self_restart_is_forbidden(self) -> None:
        from deep_worker_cli.tools import make_restart_agent_tool

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="alpha")

        assert result["success"] is False
        assert result["error"] == "self_restart_forbidden"


class TestRestartAgentBehavior:
    async def test_running_agent_is_actually_restarted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A live agent must be restarted, not reported back as "already running".

        The tool documents itself as the equivalent of
        ``deep-worker --restart <name>``, and that CLI path explicitly stops a
        running agent (``force=True``) before relaunching it — restarting a
        busy agent is the whole point of the destructive tool, and the only
        way to interrupt a stuck turn.

        The old code returned on ``registry.lookup() is not None`` and never
        reached the subprocess, so a running agent was never restarted; the
        tool silently degraded into "start if dead". It also returned
        ``success=True`` together with ``error="already_running"``, so the
        refusal read as success to every caller.
        """
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.tools import make_restart_agent_tool

        old = _process_info("beta", pid=12345)
        new = _process_info("beta", pid=67890)
        registry = MagicMock()
        registry.lookup.return_value = old
        registry.wait_for_registration = AsyncMock(return_value=new)
        run_mock = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))

        monkeypatch.setattr(tools_mod, "ProcessRegistry", MagicMock(return_value=registry))
        monkeypatch.setattr(tools_mod.shutil, "which", lambda _cmd: "/usr/bin/deep-worker")
        monkeypatch.setattr(tools_mod.subprocess, "run", run_mock)

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta")

        # The CLI was actually invoked for the running agent.
        run_mock.assert_called_once()
        assert run_mock.call_args[0][0] == ["/usr/bin/deep-worker", "--restart", "beta"]

        assert result["success"] is True
        assert result["already_running"] is False
        assert result.get("error") is None
        assert result["restarted_pid"] == 67890
        assert result["previous_pid"] == 12345

    async def test_running_agent_restart_requires_a_new_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seeing the *old* PID back in the registry is not a restart.

        ``wait_for_registration`` can resolve against the still-present entry
        of the process we meant to replace. Reporting that as success would
        claim new code is running while the old process still serves.
        """
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.tools import make_restart_agent_tool

        old = _process_info("beta", pid=12345)
        registry = MagicMock()
        registry.lookup.return_value = old
        registry.wait_for_registration = AsyncMock(return_value=_process_info("beta", pid=12345))
        run_mock = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))

        monkeypatch.setattr(tools_mod, "ProcessRegistry", MagicMock(return_value=registry))
        monkeypatch.setattr(tools_mod.shutil, "which", lambda _cmd: "/usr/bin/deep-worker")
        monkeypatch.setattr(tools_mod.subprocess, "run", run_mock)

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta")

        assert result["success"] is False
        assert result["error"] == "restart_timeout"

    async def test_missing_command_returns_command_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.tools import make_restart_agent_tool

        registry = MagicMock()
        registry.lookup.return_value = None
        monkeypatch.setattr(tools_mod, "ProcessRegistry", MagicMock(return_value=registry))
        monkeypatch.setattr(tools_mod.shutil, "which", lambda _cmd: None)

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta")

        assert result["success"] is False
        assert result["error"] == "command_unavailable"

    @pytest.mark.parametrize(
        ("stderr", "expected"),
        [
            ("no previous launch configuration found", "no_launch_config"),
            ("it is actively running (status: running)", "actively_running"),
            ("some other failure", "restart_failed"),
        ],
    )
    async def test_cli_failures_are_mapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stderr: str,
        expected: str,
    ) -> None:
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.tools import make_restart_agent_tool

        registry = MagicMock()
        registry.lookup.return_value = None
        completed = MagicMock(returncode=1, stdout="", stderr=stderr)

        monkeypatch.setattr(tools_mod, "ProcessRegistry", MagicMock(return_value=registry))
        monkeypatch.setattr(tools_mod.shutil, "which", lambda _cmd: "/usr/bin/deep-worker")
        monkeypatch.setattr(tools_mod.subprocess, "run", MagicMock(return_value=completed))

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta")

        assert result["success"] is False
        assert result["error"] == expected

    async def test_success_waits_for_live_registry_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.tools import make_restart_agent_tool
        from deep_worker_cli.types import AgentProcessInfo

        # Registry lookup returns None first (before subprocess), then
        # returns the agent info after the event-driven wait resolves.
        info = _process_info("beta")
        registry = MagicMock()
        registry.lookup.side_effect = [None, info]
        registry.wait_for_registration = AsyncMock(return_value=info)
        completed = MagicMock(returncode=0, stdout="starting", stderr="")

        monkeypatch.setattr(tools_mod, "ProcessRegistry", MagicMock(return_value=registry))
        monkeypatch.setattr(tools_mod.shutil, "which", lambda _cmd: "/usr/bin/deep-worker")
        monkeypatch.setattr(tools_mod.subprocess, "run", MagicMock(return_value=completed))

        tool = make_restart_agent_tool("alpha")
        result = await _ainvoke(tool, agent_name="beta")

        assert result["success"] is True
        assert result["already_running"] is False
        assert result["agent_name"] == "beta"
        # wait_for_registration must have been called (no sleep loop)
        assert registry.wait_for_registration.await_count == 1
        # lookup called only for the pre-subprocess check (no polling)
        assert registry.lookup.call_count == 1
