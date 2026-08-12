from __future__ import annotations

import asyncio
import os
import shutil
import signal
import site
import sys
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.storage_paths import deep_worker_home
from deepagents.middleware.summarization_history import (
    deserialize_summarized_messages_jsonl,
)

pytestmark = pytest.mark.integration

_OVERALL_TIMEOUT = 180.0


def _require_integration() -> None:
    if os.environ.get("DEEP_WORKER_REQUIRE_INTEGRATION") != "1":
        pytest.skip("set DEEP_WORKER_REQUIRE_INTEGRATION=1 for the live provider test")
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.fail(
            "DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY is required for the live provider test"
        )


def _live_provider_and_model() -> tuple[str, str]:
    # Signed design prefers codex/gpt-5.6-sol. When that account is rate-limited,
    # deepseek is the project's default real provider with the same SQLite/RPyC path.
    override = os.environ.get("DEEP_WORKER_SUMMARY_INTEGRATION_PROVIDER")
    if override:
        model = os.environ.get("DEEP_WORKER_SUMMARY_INTEGRATION_MODEL")
        if not model:
            pytest.fail("DEEP_WORKER_SUMMARY_INTEGRATION_MODEL is required with provider override")
        return override, model
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek", "deepseek-v4-flash"
    return "codex", "gpt-5.6-sol"


async def _wait_pid_exit(pid: int, timeout: float) -> None:
    pidfd = os.pidfd_open(pid)
    loop = asyncio.get_running_loop()
    exited = asyncio.Event()
    loop.add_reader(pidfd, exited.set)
    try:
        await asyncio.wait_for(exited.wait(), timeout)
    finally:
        loop.remove_reader(pidfd)
        os.close(pidfd)


async def _run_live_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_integration()
    agent_name = f"requested_summary_{os.getpid()}"
    isolated_home = tmp_path / "home"
    workdir = tmp_path / "work"
    isolated_home.mkdir(mode=0o700)
    workdir.mkdir(mode=0o700)
    provider, model = _live_provider_and_model()
    if provider == "codex":
        source_auth = Path.home() / ".codex" / "auth.json"
        if not source_auth.is_file():
            pytest.fail("real Codex auth.json is required for the live provider test")
        isolated_codex = isolated_home / ".codex"
        isolated_codex.mkdir(mode=0o700)
        isolated_auth = isolated_codex / "auth.json"
        shutil.copyfile(source_auth, isolated_auth)
        isolated_auth.chmod(0o600)
    monkeypatch.setenv("HOME", str(isolated_home))

    cli_root = Path(__file__).resolve().parents[2]
    deepagents_root = cli_root.parent / "deepagents"
    env = os.environ.copy()
    env["HOME"] = str(isolated_home)
    env["DEEP_WORKER_ORIGINAL_CWD"] = str(workdir)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(cli_root), str(deepagents_root), site.getusersitepackages())
    )
    # Ensure provider keys survive isolated HOME.
    for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if key in os.environ:
            env[key] = os.environ[key]
    registry = ProcessRegistry()
    client = AgentClient(agent_name)
    daemon_pid: int | None = None

    try:
        launcher = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "deep_worker_cli.cli.agent_command",
            agent_name,
            "--provider",
            provider,
            "--model",
            model,
            "--use-sqlite",
            "--dir",
            str(workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(launcher.communicate(), 20.0)
        assert launcher.returncode == 0, (stdout + stderr).decode(errors="replace")

        info = await registry.wait_for_registration(agent_name, timeout=30.0)
        assert info is not None
        daemon_pid = info.pid
        await asyncio.wait_for(client.connect(), 15.0)
        below_threshold_context = "context " * 8_000
        for index in range(4):
            response = await asyncio.wait_for(
                client.invoke_agent(
                    f"Segment {index} contains REQUESTED_SUMMARY_ALPHA_{index}. "
                    f"{below_threshold_context} Reply exactly ACK{index} without using tools.",
                    stream=False,
                ),
                90.0,
            )
            assert response["success"] is True

        before = await asyncio.wait_for(client.get_history(limit=50), 10.0)
        before_messages = before["history"]
        assert len(before_messages) >= 2
        assert not any(
            "You are in the middle of a conversation that has been summarized."
            in str(message.get("content", ""))
            for message in before_messages
        )
        history_file = (
            deep_worker_home(isolated_home)
            / agent_name
            / "summaries"
            / f"{agent_name}.jsonl"
        )
        assert not history_file.exists()

        # Negative control: ordinary below-threshold turn must NOT compact.
        negative = await asyncio.wait_for(
            client.invoke_agent(
                "Negative control: reply exactly NEG_OK without using tools.",
                stream=False,
            ),
            90.0,
        )
        assert negative["success"] is True
        mid = await asyncio.wait_for(client.get_history(limit=50), 10.0)
        mid_messages = mid["history"]
        assert not any(
            "You are in the middle of a conversation that has been summarized."
            in str(message.get("content", ""))
            for message in mid_messages
        )
        assert not history_file.exists()

        summarized = await asyncio.wait_for(
            client._send_command({"command": "summarize", "prompt": None}),
            90.0,
        )
        assert summarized["success"] is True
        assert summarized["summarized"] is True

        after = await asyncio.wait_for(client.get_history(limit=50), 10.0)
        after_messages = after["history"]
        summary_messages = [
            message
            for message in after_messages
            if "You are in the middle of a conversation that has been summarized."
            in str(message.get("content", ""))
        ]
        assert len(summary_messages) == 1
        summary_content = str(summary_messages[0]["content"])
        assert "Segment 0" in summary_content
        assert "ACK0" in summary_content
        assert history_file.is_file()
        restored = deserialize_summarized_messages_jsonl(
            history_file.read_text(encoding="utf-8")
        )
        assert any("REQUESTED_SUMMARY_ALPHA_0" in str(message.content) for message in restored)
        assert any("ACK3" in str(message.get("content", "")) for message in after_messages[1:])
        assert any(
            "NEG_OK" in str(message.get("content", ""))
            for message in after_messages
        ) or "NEG_OK" in summary_content
    finally:
        await client.disconnect()
        if daemon_pid is not None:
            try:
                os.kill(daemon_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    await _wait_pid_exit(daemon_pid, 10.0)
                except TimeoutError:
                    os.kill(daemon_pid, signal.SIGKILL)
                    await _wait_pid_exit(daemon_pid, 5.0)


@pytest.mark.asyncio
async def test_requested_summarization_live_real_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await asyncio.wait_for(
        _run_live_scenario(tmp_path, monkeypatch),
        timeout=_OVERALL_TIMEOUT,
    )
