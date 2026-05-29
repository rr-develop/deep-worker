"""Live deep-worker integration for Codex tool-call history.

This test starts a real ``deep-worker --provider codex`` agent, forces it to
read a temporary secret file through the deep-worker tool layer, and verifies
that offline ``deep-worker --history`` contains the tool call/result.

It is skipped unless ``CODEX_INTEGRATION_TESTS=1`` is set because it consumes a
live Codex CLI request through the user's cached Codex login.
"""
from __future__ import annotations

import os
import contextlib
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CODEX_INTEGRATION_TESTS") != "1",
    reason="set CODEX_INTEGRATION_TESTS=1 to run live Codex integration tests",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEP_WORKER = REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep-worker"
DWCLI = REPO_ROOT / "deep_worker" / "deep_worker_cli" / "dwcli"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    # The launcher preserves existing PYTHONPATH after adding deep_worker paths.
    # Add the repo root so ``codex_provider`` is importable in the agent process.
    prefix = (
        f"{REPO_ROOT}:"
        f"{REPO_ROOT / 'deep_worker' / 'deep_worker_cli'}:"
        f"{REPO_ROOT / 'deep_worker' / 'deepagents'}"
    )
    env["PYTHONPATH"] = f"{prefix}:{env.get('PYTHONPATH', '')}"
    return env


def _run(
    args: list[str],
    *,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _cleanup_agent_files(agent_name: str) -> None:
    shutil.rmtree(Path("~/.deep_worker").expanduser() / agent_name, ignore_errors=True)
    db_path = Path.home() / ".deepagents" / f"sessions_{agent_name}.db"
    with contextlib.suppress(FileNotFoundError):
        db_path.unlink()


def _cleanup_codex_mcp_for_agent(agent_name: str, env: dict[str, str]) -> None:
    pid_file = Path(f"/tmp/deep_worker_{agent_name}.pid")
    with contextlib.suppress(FileNotFoundError, ValueError):
        pid = pid_file.read_text().strip()
        if pid:
            _run(["codex", "mcp", "remove", f"deepworker_{pid}"], env=env, timeout=10.0)


def _wait_registered(agent_name: str, env: dict[str, str], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_output = ""
    while time.monotonic() < deadline:
        result = _run([str(DWCLI), "--list"], env=env, timeout=5.0)
        last_output = result.stdout + result.stderr
        if agent_name in last_output:
            return
        time.sleep(0.2)
    out_log = Path("~/.deep_worker").expanduser() / "logs" / f"{agent_name}.out.log"
    err_log = Path("~/.deep_worker").expanduser() / "logs" / f"{agent_name}.err.log"
    pytest.fail(
        f"agent {agent_name!r} did not register within {timeout}s\n"
        f"last dwcli --list output:\n{last_output[-4000:]}\n"
        f"stdout log:\n{out_log.read_text()[-4000:] if out_log.exists() else '<missing>'}\n"
        f"stderr log:\n{err_log.read_text()[-4000:] if err_log.exists() else '<missing>'}"
    )


def test_dw_codex_history_contains_proxy_tool_call(tmp_path: Path) -> None:
    agent_name = f"test_codex_history_{uuid.uuid4().hex[:10]}"
    env = _env()
    secret = f"DW_CODEX_HISTORY_SECRET_{uuid.uuid4().hex}"
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text(f"{secret}\n")

    _cleanup_agent_files(agent_name)

    try:
        start = _run(
            [
                str(DEEP_WORKER),
                agent_name,
                "--provider",
                "codex",
                "--dir",
                str(tmp_path),
                "--skip-permissions",
                "--use-sqlite",
                "--clean",
            ],
            env=env,
            timeout=10.0,
        )
        assert start.returncode == 0, start.stdout + start.stderr
        _wait_registered(agent_name, env)

        prompt = (
            f"Read the file at {secret_path} using the read_file tool. "
            "Do not guess and do not use shell. Return only the exact file contents."
        )
        response = _run([str(DWCLI), "-p", prompt, agent_name], env=env, timeout=240.0)
        assert response.returncode == 0, response.stdout + response.stderr
        assert secret in response.stdout

        history = _run([str(DEEP_WORKER), "--history", agent_name], env=env, timeout=60.0)
        assert history.returncode == 0, history.stdout + history.stderr
        assert "[TOOL]" in history.stdout, history.stdout
        assert "read_file" in history.stdout, history.stdout
        assert secret in history.stdout, history.stdout

        history_json = _run(
            [str(DEEP_WORKER), "--history", agent_name, "--format", "json"],
            env=env,
            timeout=60.0,
        )
        assert history_json.returncode == 0, history_json.stdout + history_json.stderr
        assert "tool_calls" in history_json.stdout, history_json.stdout
        assert "tool_call_id" in history_json.stdout, history_json.stdout
        assert "read_file" in history_json.stdout, history_json.stdout
    finally:
        _cleanup_codex_mcp_for_agent(agent_name, env)
        _run([str(DWCLI), "--stop", agent_name], env=env, timeout=30.0)
        _cleanup_codex_mcp_for_agent(agent_name, env)
        _cleanup_agent_files(agent_name)
