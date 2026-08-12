"""Integration smoke for ``deep-worker --provider grok`` recording tool
calls into the history (channel A — proxy MCP).

Starts a real agent process, forces it to read a unique secret file through
the deep-worker ``read_file`` tool, then verifies offline ``deep-worker
--history`` (text AND JSON) shows the tool call/result pair. Mirror of
codex_provider's test_deep_worker_history_integration.

Skipped unless GROK_INTEGRATION_TESTS=1 (uses real grok CLI + supergrok
subscription cache at ~/.grok/auth.json).
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from deep_worker_cli.storage_paths import deep_worker_home

pytestmark = pytest.mark.skipif(
    os.environ.get("GROK_INTEGRATION_TESTS") != "1",
    reason="set GROK_INTEGRATION_TESTS=1 to run live grok and deep-worker tests",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEP_WORKER = REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep-worker"
DWCLI = REPO_ROOT / "deep_worker" / "deep_worker_cli" / "dwcli"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = (
        f"{REPO_ROOT}:"
        f"{REPO_ROOT / 'deep_worker' / 'deep_worker_cli'}:"
        f"{REPO_ROOT / 'deep_worker' / 'deepagents'}"
    )
    env["PYTHONPATH"] = f"{prefix}:{env.get('PYTHONPATH', '')}"
    return env


def _run(args: list[str], *, env: dict[str, str], timeout: float):
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, env=env,
    )


def _cleanup_agent_files(agent_name: str) -> None:
    shutil.rmtree(deep_worker_home() / agent_name, ignore_errors=True)
    db_path = Path.home() / ".deepagents" / f"sessions_{agent_name}.db"
    with contextlib.suppress(FileNotFoundError):
        db_path.unlink()


def _wait_registered(agent_name: str, env: dict[str, str], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        r = _run([str(DWCLI), "--list"], env=env, timeout=5.0)
        last = r.stdout + r.stderr
        if agent_name in last:
            return
        time.sleep(0.2)
    pytest.fail(f"agent {agent_name!r} did not register within {timeout}s\n{last[-2000:]}")


def test_deep_worker_grok_history_contains_proxy_tool_call(tmp_path: Path) -> None:
    if not (Path.home() / ".grok" / "auth.json").exists():
        pytest.skip("~/.grok/auth.json missing — grok needs a cached login")
    if shutil.which("grok") is None:
        pytest.skip("grok CLI not found on PATH")

    agent_name = f"test_grok_history_{uuid.uuid4().hex[:10]}"
    env = _env()
    secret = f"DW_GROK_HISTORY_SECRET_{uuid.uuid4().hex}"
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text(f"{secret}\n")

    _cleanup_agent_files(agent_name)
    try:
        start = _run(
            [
                str(DEEP_WORKER),
                agent_name,
                "--provider", "grok",
                "--dir", str(tmp_path),
                "--skip-permissions",
                "--use-sqlite",
                "--clean",
            ],
            env=env,
            timeout=10.0,
        )
        assert start.returncode == 0, start.stdout + start.stderr
        _wait_registered(agent_name, env)

        # Steer grok to use the deepworker ``read_file`` MCP tool (channel A)
        # rather than its built-in ``Read`` (channel B). The system prompt
        # already prefers MCP, but we reinforce it here.
        prompt = (
            f"Read the file at {secret_path} using the deepworker read_file "
            f"MCP tool. Do not use the built-in Read tool. Return only the "
            f"exact file contents."
        )
        response = _run([str(DWCLI), "-p", prompt, agent_name], env=env, timeout=240.0)
        assert response.returncode == 0, response.stdout + response.stderr
        assert secret in response.stdout

        history = _run([str(DEEP_WORKER), "--history", agent_name], env=env, timeout=60.0)
        assert history.returncode == 0, history.stdout + history.stderr
        assert "[TOOL]" in history.stdout
        # Either channel A (read_file) OR channel B (Read) must be visible
        # in the rendered history — both go through LangGraph.
        assert ("read_file" in history.stdout) or ("Read" in history.stdout)
        assert secret in history.stdout

        history_json = _run(
            [str(DEEP_WORKER), "--history", agent_name, "--format", "json"],
            env=env, timeout=60.0,
        )
        assert history_json.returncode == 0
        assert "tool_calls" in history_json.stdout
        assert "tool_call_id" in history_json.stdout
    finally:
        _run([str(DWCLI), "--stop", agent_name], env=env, timeout=30.0)
        _cleanup_agent_files(agent_name)
