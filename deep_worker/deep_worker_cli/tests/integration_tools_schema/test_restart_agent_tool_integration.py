from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def preserve_registry():
    registry_file = Path("/tmp/deep_worker_registry.json")
    original = registry_file.read_text() if registry_file.exists() else None
    yield registry_file
    if original is None:
        registry_file.unlink(missing_ok=True)
    else:
        registry_file.write_text(original)


async def test_restart_agent_tool_invokes_restart_without_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_registry: Path,
) -> None:
    from deep_worker_cli.tools import make_restart_agent_tool

    agent_name = f"test_restart_tool_{os.getpid()}"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "argv.json"
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text(
        f"""#!/usr/bin/env python3
import json
import os
from datetime import datetime
from pathlib import Path

marker = Path({str(marker)!r})
marker.write_text(json.dumps({{"argv": __import__("sys").argv[1:]}}))
registry_file = Path("/tmp/deep_worker_registry.json")
try:
    registry = json.loads(registry_file.read_text())
except Exception:
    registry = {{}}
registry[{agent_name!r}] = {{
    "pid": os.getppid(),
    "pid_file": f"/tmp/deep_worker_{agent_name}.pid",
    "working_dir": "/tmp",
    "started_at": datetime.now().isoformat(),
    "last_activity": datetime.now().isoformat(),
    "rpyc_port": 18861,
}}
registry_file.write_text(json.dumps(registry))
print("Agent '{agent_name}' starting in background")
"""
    )
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    tool = make_restart_agent_tool("caller_agent")
    result = await tool.ainvoke({"agent_name": agent_name})

    assert result["success"] is True
    assert result["already_running"] is False
    assert json.loads(marker.read_text())["argv"] == ["--restart", agent_name]
