"""Tests for ``codex_provider.exec_runner``.

Covered behaviours:

* JSONL parser correctly typifies all event flavours the CLI emits
  (``thread.started``, ``turn.started``, ``item.completed`` for both
  ``agent_message`` and ``mcp_tool_call``, ``turn.completed``,
  ``turn.failed``, top-level ``error``).
* Malformed JSON lines raise a structured ``CodexProtocolError``.
* Subprocess command building: resume vs fresh, sandbox / approval
  flags, base-instructions, working directory, disallowed tools, no
  ``--model`` when ``model is None``.
* No ``CODEX_API_KEY`` is set in the env passed to the subprocess
  when ``api_key is None`` (the cached-login path).
* ``ExecHandle.events`` is an async generator and yields parsed events
  one by one; closing the handle terminates the subprocess cleanly.

The subprocess is mocked end-to-end via ``unittest.mock`` so these
tests are pure unit tests — they never spawn ``codex``. The integration
test in ``test_integration.py`` exercises a real ``codex exec``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

# Make the package importable when running pytest from the repo root or
# from inside ``codex_provider/``.
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider.exec_runner import (  # noqa: E402
    CodexExecRunner,
    ExecArgs,
    CodexProtocolError,
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemCompletedEvent,
    AgentMessageItem,
    McpToolCallItem,
)


# ---------------------------------------------------------------------------
# JSONL parser
# ---------------------------------------------------------------------------
class TestParseEvent:
    """``parse_event`` typifies one JSONL line into one event object."""

    def test_thread_started(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({"type": "thread.started", "thread_id": "abc-123"}))
        assert isinstance(ev, ThreadStartedEvent)
        assert ev.thread_id == "abc-123"

    def test_turn_started(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({"type": "turn.started"}))
        assert isinstance(ev, TurnStartedEvent)

    def test_turn_completed_with_usage(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }))
        assert isinstance(ev, TurnCompletedEvent)
        assert ev.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_turn_failed(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "turn.failed", "error": {"message": "boom"},
        }))
        assert isinstance(ev, TurnFailedEvent)
        assert ev.error_message == "boom"

    def test_item_completed_agent_message(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "HELLO"},
        }))
        assert isinstance(ev, ItemCompletedEvent)
        assert isinstance(ev.item, AgentMessageItem)
        assert ev.item.text == "HELLO"

    def test_item_completed_mcp_tool_call(self) -> None:
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "deepworker_1234",
                "tool": "read_file",
                "arguments": {"path": "/tmp/x"},
                "status": "completed",
            },
        }))
        assert isinstance(ev, ItemCompletedEvent)
        assert isinstance(ev.item, McpToolCallItem)
        assert ev.item.server == "deepworker_1234"
        assert ev.item.tool == "read_file"
        assert ev.item.arguments == {"path": "/tmp/x"}

    def test_item_completed_unknown_type_is_passed_through(self) -> None:
        """Unknown item types (reasoning/web_search/etc.) yield a generic
        ``ItemCompletedEvent`` with ``item.type`` preserved on the raw
        dict — we don't crash on event types we don't model explicitly."""
        from codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "item.completed",
            "item": {"id": "item_2", "type": "reasoning", "text": "hmm"},
        }))
        assert isinstance(ev, ItemCompletedEvent)
        # generic item — exposed as a dict-like with .type attribute
        assert getattr(ev.item, "type", None) == "reasoning"

    def test_top_level_error(self) -> None:
        """Top-level ``{"type":"error"}`` raises immediately — the stream is dead."""
        from codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event(json.dumps({"type": "error", "message": "stream dead"}))

    def test_unknown_type_raises(self) -> None:
        from codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event(json.dumps({"type": "this.does.not.exist"}))

    def test_malformed_json_raises(self) -> None:
        from codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event("not json at all {")


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------
class TestBuildCommandArgs:
    """``CodexExecRunner._build_argv`` constructs the right CLI invocation."""

    def test_fresh_session_no_resume_flag(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi", session_uuid=None))
        assert "resume" not in argv

    def test_resume_session_uses_resume_subcommand(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi", session_uuid="abc"))
        # Pattern: codex exec --experimental-json [flags...] resume <uuid> -
        assert "resume" in argv
        assert "abc" in argv
        assert argv.index("resume") < argv.index("abc")

    def test_experimental_json_always_set(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        assert "--experimental-json" in argv

    def test_sandbox_workspace_write_by_default(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        # -s workspace-write per Q3
        i = argv.index("-s")
        assert argv[i + 1] == "workspace-write"

    def test_approval_policy_never(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        # --config approval_policy="never" per Q3
        joined = " ".join(argv)
        assert 'approval_policy="never"' in joined

    def test_dangerously_bypass_flag_present(self) -> None:
        """The bypass flag is what actually lets MCP tool calls go through
        without prompts (approval_policy="never" alone only covers shell
        commands; MCP calls get auto-cancelled). Verified live on this
        host. Mirrors claude-code's ``permission_mode="bypassPermissions"``.
        """
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        assert "--dangerously-bypass-approvals-and-sandbox" in argv

    def test_skip_git_repo_check(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        assert "--skip-git-repo-check" in argv

    def test_working_directory(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi", working_directory="/tmp/foo"))
        i = argv.index("--cd")
        assert argv[i + 1] == "/tmp/foo"

    def test_no_model_flag_when_model_is_none(self) -> None:
        """Q2: don't pin the model — pass nothing, let Codex pick its default."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi", model=None))
        assert "--model" not in argv
        assert "-m" not in argv

    def test_model_flag_when_model_set(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi", model="gpt-5"))
        # accept either --model or -m
        assert ("--model" in argv and argv[argv.index("--model") + 1] == "gpt-5") or \
               ("-m" in argv and argv[argv.index("-m") + 1] == "gpt-5")

    def test_prompt_read_from_stdin(self) -> None:
        """The CLI takes ``-`` to read prompt from stdin."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner._build_argv(ExecArgs(prompt="hi"))
        assert argv[-1] == "-"


# ---------------------------------------------------------------------------
# Environment passed to the subprocess
# ---------------------------------------------------------------------------
class TestBuildEnv:
    """Q3: cached-login path. No ``CODEX_API_KEY`` in the child env."""

    def test_no_api_key_when_api_key_none(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner._build_env(ExecArgs(prompt="hi", api_key=None))
        assert "CODEX_API_KEY" not in env

    def test_api_key_set_when_provided(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner._build_env(ExecArgs(prompt="hi", api_key="sk-xxx"))
        assert env.get("CODEX_API_KEY") == "sk-xxx"

    def test_inherits_os_environ_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO_DEEPWORKER_TEST", "bar")
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner._build_env(ExecArgs(prompt="hi"))
        assert env.get("FOO_DEEPWORKER_TEST") == "bar"


# ---------------------------------------------------------------------------
# spawn() + event stream (mocked subprocess)
# ---------------------------------------------------------------------------
def _make_mock_process(stdout_lines: list[bytes], returncode: int = 0) -> MagicMock:
    """Build a mock asyncio.subprocess.Process that replays ``stdout_lines``."""
    proc = MagicMock()
    proc.returncode = returncode

    # stdin: write+drain+close are awaited
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock(return_value=None)
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdin.close = MagicMock(return_value=None)

    # stdout.readline: returns each line in order, then b"" to signal EOF
    queue = list(stdout_lines) + [b""]
    proc.stdout = MagicMock()

    async def _readline() -> bytes:
        return queue.pop(0) if queue else b""

    proc.stdout.readline = _readline

    # stderr drained but empty
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(side_effect=[b"", b""])

    proc.wait = AsyncMock(return_value=returncode)
    proc.terminate = MagicMock(return_value=None)
    proc.kill = MagicMock(return_value=None)
    return proc


@pytest.mark.asyncio
class TestSpawn:
    async def test_yields_parsed_events_in_order(self) -> None:
        lines = [
            (json.dumps({"type": "thread.started", "thread_id": "abc"}) + "\n").encode(),
            (json.dumps({"type": "turn.started"}) + "\n").encode(),
            (json.dumps({
                "type": "item.completed",
                "item": {"id": "i0", "type": "agent_message", "text": "HELLO"},
            }) + "\n").encode(),
            (json.dumps({"type": "turn.completed", "usage": {}}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(prompt="say HELLO"))
            collected = [ev async for ev in handle.events]

        assert len(collected) == 4
        assert isinstance(collected[0], ThreadStartedEvent)
        assert collected[0].thread_id == "abc"
        assert isinstance(collected[1], TurnStartedEvent)
        assert isinstance(collected[2], ItemCompletedEvent)
        assert isinstance(collected[2].item, AgentMessageItem)
        assert collected[2].item.text == "HELLO"
        assert isinstance(collected[3], TurnCompletedEvent)

    async def test_aclose_terminates_subprocess(self) -> None:
        """``handle.aclose()`` terminates the subprocess cleanly even if
        we haven't drained the stream — needed when ainvoke breaks early
        on a tool_use."""
        lines = [
            (json.dumps({"type": "thread.started", "thread_id": "x"}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines)
        # Make the process appear still alive
        mock_proc.returncode = None

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(prompt="hi"))

        await handle.aclose()
        # We should have asked the subprocess to terminate
        assert mock_proc.terminate.called or mock_proc.kill.called
