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

from deep_worker.providers.codex_provider.exec_runner import (  # noqa: E402
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
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({"type": "thread.started", "thread_id": "abc-123"}))
        assert isinstance(ev, ThreadStartedEvent)
        assert ev.thread_id == "abc-123"

    def test_turn_started(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({"type": "turn.started"}))
        assert isinstance(ev, TurnStartedEvent)

    def test_turn_completed_with_usage(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }))
        assert isinstance(ev, TurnCompletedEvent)
        assert ev.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_turn_failed(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "turn.failed", "error": {"message": "boom"},
        }))
        assert isinstance(ev, TurnFailedEvent)
        assert ev.error_message == "boom"

    def test_item_completed_agent_message(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "HELLO"},
        }))
        assert isinstance(ev, ItemCompletedEvent)
        assert isinstance(ev.item, AgentMessageItem)
        assert ev.item.text == "HELLO"

    def test_item_completed_mcp_tool_call(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
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
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        ev = parse_event(json.dumps({
            "type": "item.completed",
            "item": {"id": "item_2", "type": "reasoning", "text": "hmm"},
        }))
        assert isinstance(ev, ItemCompletedEvent)
        # generic item — exposed as a dict-like with .type attribute
        assert getattr(ev.item, "type", None) == "reasoning"

    def test_top_level_error(self) -> None:
        """Top-level ``{"type":"error"}`` raises immediately — the stream is dead."""
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event(json.dumps({"type": "error", "message": "stream dead"}))

    def test_error_with_nested_message_and_status(self) -> None:
        """Codex emits ``{"type":"error","status":400,...}`` with the
        message nested under ``error.message``, NOT at the top level.
        ``parse_event`` MUST extract both the nested message and the
        status code so the error reason (e.g. unsupported model) is not
        lost (issue #60)."""
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        raw = {
            "type": "error",
            "status": 400,
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "The 'gpt-5.2' model is not supported when "
                    "using Codex with a ChatGPT account."
                ),
            },
        }
        with pytest.raises(CodexProtocolError) as excinfo:
            parse_event(json.dumps(raw))
        msg = str(excinfo.value)
        assert "gpt-5.2" in msg, f"nested error message must be in output, got: {msg}"
        assert "400" in msg, f"status code must be in output, got: {msg}"

    def test_error_with_top_level_message_still_works(self) -> None:
        """A flat ``{"type":"error","message":"..."}`` (the old format)
        must still produce a sensible error."""
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError) as excinfo:
            parse_event(json.dumps({"type": "error", "message": "flat error"}))
        assert "flat error" in str(excinfo.value)

    def test_unknown_type_raises(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event(json.dumps({"type": "this.does.not.exist"}))

    def test_malformed_json_raises(self) -> None:
        from deep_worker.providers.codex_provider.exec_runner import parse_event
        with pytest.raises(CodexProtocolError):
            parse_event("not json at all {")


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------
class TestBuildCommandArgs:
    """``CodexExecRunner._build_argv`` constructs the right CLI invocation."""

    def test_fresh_session_no_resume_flag(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi", session_uuid=None))
        assert "resume" not in argv

    def test_resume_session_uses_resume_subcommand(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi", session_uuid="abc"))
        # Pattern: codex exec --experimental-json [flags...] resume <uuid> -
        assert "resume" in argv
        assert "abc" in argv
        assert argv.index("resume") < argv.index("abc")

    def test_experimental_json_always_set(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        assert "--experimental-json" in argv

    def test_sandbox_workspace_write_by_default(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        # -s workspace-write per Q3
        i = argv.index("-s")
        assert argv[i + 1] == "workspace-write"

    def test_approval_policy_never(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
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
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        assert "--dangerously-bypass-approvals-and-sandbox" in argv

    def test_skip_git_repo_check(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        assert "--skip-git-repo-check" in argv

    def test_working_directory(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi", working_directory="/tmp/foo"))
        i = argv.index("--cd")
        assert argv[i + 1] == "/tmp/foo"

    def test_no_model_flag_when_model_is_none(self) -> None:
        """Q2: don't pin the model — pass nothing, let Codex pick its default."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi", model=None))
        assert "--model" not in argv
        assert "-m" not in argv

    def test_model_flag_when_model_set(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi", model="gpt-5"))
        # accept either --model or -m
        assert ("--model" in argv and argv[argv.index("--model") + 1] == "gpt-5") or \
               ("-m" in argv and argv[argv.index("-m") + 1] == "gpt-5")

    def test_prompt_read_from_stdin(self) -> None:
        """The CLI takes ``-`` to read prompt from stdin."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        assert argv[-1] == "-"

    def test_disabled_features_become_disable_flags(self) -> None:
        """Codex CLI 0.130.0 disables native tools via per-feature flags,
        NOT via the ``disabled_tools=[...]`` config key.

        The old ``-c 'disabled_tools=[\"exec_command\", ...]'`` shape was
        empirically verified to be silently ignored: the resulting
        ``response.create`` websocket payload (see
        ``RUST_LOG=codex_api=trace``) still included every native tool
        verbatim. The working contract is to pass one ``--disable
        <feature>`` per feature whose tools we want stripped from the
        ``tools=[...]`` field of the API request.
        """
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(
            ExecArgs(prompt="hi", disabled_features=("shell_tool", "unified_exec"))
        )
        # Each feature gets its own --disable <name> argument pair.
        assert argv.count("--disable") == 2
        i = 0
        for feat in ("shell_tool", "unified_exec"):
            i = argv.index("--disable", i)
            assert argv[i + 1] == feat
            i += 2

    def test_disabled_features_argv_does_not_contain_dead_disabled_tools_config(
        self,
    ) -> None:
        """The dead ``--config disabled_tools=[...]`` path must be gone.

        Codex 0.130.0 silently ignored it; keeping it would add a large
        noisy noop argument to every codex exec.
        """
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(
            ExecArgs(prompt="hi", disabled_features=("shell_tool",))
        )
        joined = " ".join(argv)
        assert "disabled_tools=" not in joined

    def test_config_overrides_become_c_flags(self) -> None:
        """``ExecArgs.config_overrides`` is the general channel for
        top-level Codex config keys that are NOT feature flags.

        The canonical case is ``web_search="disabled"`` — Codex 0.130.0
        moved ``web_search`` out of ``[features]`` into a top-level
        string config (``"live"`` | ``"cached"`` | ``"disabled"``).
        Passing it via ``--disable web_search`` triggers a deprecation
        warning *and* still leaves the tool in the API.
        """
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(
            ExecArgs(
                prompt="hi",
                config_overrides=(("web_search", '"disabled"'),),
            )
        )
        # One -c override per pair.
        assert argv.count("--config") >= 1
        # The exact rendered shape Codex's TOML parser accepts.
        assert 'web_search="disabled"' in argv


class TestMcpServerIsolation:
    """Per-agent MCP server isolation (learned from a related internal incident).

    ROOT CAUSE of that incident: the provider used to register its proxy
    MCP server into the GLOBAL ``~/.codex/config.toml`` via
    ``codex mcp add deepworker_<pid>``. That file is shared by every
    codex process on the host and entries were never removed, so over
    time it accumulated dozens of ``[mcp_servers.deepworker_<dead_pid>]``
    entries. Every ``codex exec`` loaded ALL of them and tried to
    connect to each at startup; the dead ports hung for the full
    ``ProxyMCPServer`` timeout (120 s) per tool call before failing
    over. Agents appeared to "hang 10 minutes then suddenly work".

    THE FIX (and what these tests lock in): a codex agent must see ONLY
    its own MCP server, and must NOT touch the shared global config.
    Two argv guarantees enforce that:

      1. ``--ignore-user-config`` — codex does not load
         ``$CODEX_HOME/config.toml`` at all (auth still comes from
         CODEX_HOME, and sessions/resume are unaffected — those do not
         live in config.toml). This neutralises whatever junk other
         agents may have left behind.
      2. ``--config mcp_servers.<name>.url="http://127.0.0.1:<port>/mcp"``
         — the agent injects ITS OWN proxy server per-invocation, so
         the only MCP server codex knows about is this agent's.

    DO NOT regress to writing the global config. There is no sanctioned
    global MCP-server list; agents must be fully independent.
    """

    def test_ignore_user_config_flag_present(self) -> None:
        """Every codex exec must skip the shared global config.toml so it
        cannot inherit other agents' (stale) MCP servers."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        assert "--ignore-user-config" in argv, (
            "codex exec must pass --ignore-user-config so it does not load "
            "the shared ~/.codex/config.toml and inherit stale "
            "deepworker_<pid> MCP servers from other agents "
            "(observed live)"
        )

    def test_own_mcp_server_injected_when_provided(self) -> None:
        """When the agent has a proxy MCP server, its url is injected
        per-invocation as a --config override (NOT written to the
        global config)."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(
            ExecArgs(
                prompt="hi",
                mcp_server_name="deepworker_12345",
                mcp_server_port=54321,
            )
        )
        expected = 'mcp_servers.deepworker_12345.url="http://127.0.0.1:54321/mcp"'
        assert expected in argv, (
            f"expected per-invocation MCP injection {expected!r}; argv={argv}"
        )
        # It must be passed as a --config override.
        i = argv.index(expected)
        assert argv[i - 1] == "--config"

    def test_no_mcp_config_when_server_absent(self) -> None:
        """No proxy server configured → no mcp_servers override at all.
        (A turn with no tools bound never materialises a proxy.)"""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(prompt="hi"))
        joined = " ".join(argv)
        assert "mcp_servers." not in joined

    def test_only_own_server_injected_no_others(self) -> None:
        """The injected MCP config must reference ONLY this agent's
        server name — never a wildcard or a second server."""
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(
            ExecArgs(
                prompt="hi",
                mcp_server_name="deepworker_999",
                mcp_server_port=8080,
            )
        )
        mcp_configs = [a for a in argv if a.startswith("mcp_servers.")]
        assert len(mcp_configs) == 1, (
            f"exactly one MCP server may be injected (the agent's own); "
            f"got {mcp_configs}"
        )
        assert mcp_configs[0].startswith("mcp_servers.deepworker_999.")


# ---------------------------------------------------------------------------
# Environment passed to the subprocess
# ---------------------------------------------------------------------------
class TestBuildEnv:
    """Q3: cached-login path. No ``CODEX_API_KEY`` in the child env."""

    def test_no_api_key_when_api_key_none(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner.build_env_for_test(ExecArgs(prompt="hi", api_key=None))
        assert "CODEX_API_KEY" not in env

    def test_api_key_set_when_provided(self) -> None:
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner.build_env_for_test(ExecArgs(prompt="hi", api_key="sk-xxx"))
        assert env.get("CODEX_API_KEY") == "sk-xxx"

    def test_inherits_os_environ_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO_DEEPWORKER_TEST", "bar")
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        env = runner.build_env_for_test(ExecArgs(prompt="hi"))
        assert env.get("FOO_DEEPWORKER_TEST") == "bar"


# ---------------------------------------------------------------------------
# spawn() + event stream (mocked subprocess)
# ---------------------------------------------------------------------------
def _make_mock_process(stdout_lines: list[bytes], returncode: int | None = 0) -> MagicMock:
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

    async def test_nonzero_exit_after_eof_raises_error(self) -> None:
        """When the subprocess exits with a non-zero code (e.g. SIGKILL =
        exit code -9), the event generator should raise so the pump
        surfaces it as pump_error instead of silently returning partial
        data (issue #53)."""
        lines = [
            (json.dumps({"type": "turn.started"}) + "\n").encode(),
        ]
        # Process already exited with -9 (SIGKILL) when we hit EOF
        mock_proc = _make_mock_process(lines, returncode=-9)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(prompt="hi"))
            collected: list[Any] = []
            with pytest.raises(CodexProtocolError,
                               match="abnormal code -9.*SIGKILL"):
                async for ev in handle.events:
                    collected.append(ev)

        # We should have collected the first event before the abnormal exit
        assert len(collected) == 1
        assert isinstance(collected[0], TurnStartedEvent)

    async def test_exit_none_after_eof_does_not_raise(self) -> None:
        """When ``process.wait()`` returns ``None`` (process still alive),
        e.g. when aclose/cancel closes the pipe, we must NOT raise a false
        positive — the process is still running, EOF just means the pipe
        was closed."""
        lines = [
            (json.dumps({"type": "turn.started"}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines, returncode=0)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(prompt="hi"))
            collected = [ev async for ev in handle.events]

        assert len(collected) == 1
        assert isinstance(collected[0], TurnStartedEvent)


# ---------------------------------------------------------------------------
# AGENTS.override.md delivery (the real system-prompt path on Codex 0.130+)
# ---------------------------------------------------------------------------
# Codex CLI 0.130.0 silently ignores ``--config base_instructions=...`` —
# verified against the live ``session_meta.payload.base_instructions.text``
# (always shows the built-in default, never our value). The provider now
# drops an ``AGENTS.override.md`` into the working directory before each
# ``codex exec``; Codex prepends its contents as the first ``user``
# message of every turn (``core/src/agents_md.rs`` in the Codex binary).
@pytest.mark.asyncio
class TestAgentsOverrideDelivery:
    async def test_spawn_writes_override_file_with_marker(
        self, tmp_path: Path
    ) -> None:
        """``spawn`` must drop AGENTS.override.md in the working dir
        carrying our deepworker marker AND the full base_instructions."""
        from deep_worker.providers.codex_provider.exec_runner import (
            DEEPWORKER_OVERRIDE_FILENAME,
            DEEPWORKER_OVERRIDE_MARKER,
        )

        lines = [
            (json.dumps({"type": "turn.completed", "usage": {}}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        long_prompt = "You are a deep-worker agent.\n" * 200  # ~6 KB
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(
                prompt="hi",
                working_directory=str(tmp_path),
                base_instructions=long_prompt,
            ))

        target = tmp_path / DEEPWORKER_OVERRIDE_FILENAME
        assert target.exists(), "AGENTS.override.md must be created"
        body = target.read_text(encoding="utf-8")
        assert body.startswith(DEEPWORKER_OVERRIDE_MARKER), (
            "first line must carry our marker so we can recognise our "
            "file on the next spawn / cleanup"
        )
        assert long_prompt.rstrip("\n") in body, (
            "the full base_instructions must be in the file — no "
            "truncation, no escaping issues"
        )

        await handle.aclose()
        assert not target.exists(), "aclose must remove our override file"

    async def test_spawn_does_not_touch_user_override(
        self, tmp_path: Path
    ) -> None:
        """If the user has their own AGENTS.override.md (no deepworker
        marker), we must leave it as-is and NOT clobber it."""
        from deep_worker.providers.codex_provider.exec_runner import DEEPWORKER_OVERRIDE_FILENAME

        target = tmp_path / DEEPWORKER_OVERRIDE_FILENAME
        user_content = "# User's own Codex override\n\nThis is the user's file."
        target.write_text(user_content, encoding="utf-8")

        lines = [
            (json.dumps({"type": "turn.completed", "usage": {}}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(
                prompt="hi",
                working_directory=str(tmp_path),
                base_instructions="OUR PROMPT THAT MUST NOT OVERWRITE THE USER FILE",
            ))

        assert target.read_text(encoding="utf-8") == user_content, (
            "user-authored AGENTS.override.md must survive untouched"
        )

        await handle.aclose()
        # aclose must also NOT delete the user's file
        assert target.exists()
        assert target.read_text(encoding="utf-8") == user_content

    async def test_spawn_no_override_when_no_base_instructions(
        self, tmp_path: Path
    ) -> None:
        """No base_instructions → no AGENTS.override.md written.

        Avoids polluting the working directory when there is nothing to
        say (e.g. unit tests that don't pass a system prompt).
        """
        from deep_worker.providers.codex_provider.exec_runner import DEEPWORKER_OVERRIDE_FILENAME

        lines = [
            (json.dumps({"type": "turn.completed", "usage": {}}) + "\n").encode(),
        ]
        mock_proc = _make_mock_process(lines)

        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            handle = await runner.spawn(ExecArgs(
                prompt="hi",
                working_directory=str(tmp_path),
                base_instructions=None,
            ))

        assert not (tmp_path / DEEPWORKER_OVERRIDE_FILENAME).exists()
        await handle.aclose()

    async def test_argv_no_longer_passes_base_instructions_config(self) -> None:
        """The old ``--config base_instructions=...`` path is removed.

        Codex CLI 0.130.0 ignored it anyway. Keeping it would just add
        a large no-op argument to every codex exec.
        """
        runner = CodexExecRunner(executable_path="/usr/bin/codex")
        argv = runner.build_argv_for_test(ExecArgs(
            prompt="hi",
            base_instructions="some long system prompt with \"quotes\" and\nnewlines",
        ))
        joined = " ".join(argv)
        assert "base_instructions" not in joined, (
            "argv must not carry the dead --config base_instructions=... key"
        )
