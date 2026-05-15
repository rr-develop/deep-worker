"""Codex CLI subprocess runner + JSONL event parser.

This module is intentionally self-contained: zero runtime dependency on
the OpenAI Agents SDK. We use ``asyncio.create_subprocess_exec`` to spawn
``codex exec --experimental-json [resume <uuid>] -``, write the prompt to
stdin, and async-iterate the JSONL stream on stdout.

Design contract — see ``docs/DESIGN_CODEX_PROVIDER.md``:

- One ``codex exec`` subprocess per Codex turn. Exits on
  ``turn.completed`` / ``turn.failed`` / explicit ``aclose()``.
- No polling. ``stdout.readline()`` is asyncio-blocking on epoll; the
  loop wakes only when bytes arrive.
- No external processes other than the Codex CLI itself.
- Cached login via ``~/.codex/auth.json``: when ``api_key is None``,
  ``CODEX_API_KEY`` is NOT set in the child env, the CLI falls back to
  its stored credentials. Verified live on this host.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------
class CodexProtocolError(RuntimeError):
    """A JSONL line on stdout could not be parsed, or carried a fatal
    top-level ``{"type":"error"}`` event."""


# ---------------------------------------------------------------------------
# Event taxonomy — minimal mirror of what the live CLI emits.
# We model only what we actually use; unknown ``item.completed`` types
# fall through to a generic dict-like so we don't crash on future Codex
# additions.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreadStartedEvent:
    thread_id: str
    type: str = "thread.started"


@dataclass(frozen=True)
class TurnStartedEvent:
    type: str = "turn.started"


@dataclass(frozen=True)
class TurnCompletedEvent:
    usage: dict[str, Any] = field(default_factory=dict)
    type: str = "turn.completed"


@dataclass(frozen=True)
class TurnFailedEvent:
    error_message: str
    type: str = "turn.failed"


@dataclass(frozen=True)
class AgentMessageItem:
    id: str
    text: str
    type: str = "agent_message"


@dataclass(frozen=True)
class McpToolCallItem:
    id: str
    server: str
    tool: str
    arguments: dict[str, Any]
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    type: str = "mcp_tool_call"


@dataclass(frozen=True)
class GenericItem:
    """Catch-all for item types we don't model explicitly (reasoning,
    file_change, web_search, todo_list, error, …). Preserves the raw
    payload so callers can surface it as text if useful."""
    id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemStartedEvent:
    item: AgentMessageItem | McpToolCallItem | GenericItem
    type: str = "item.started"


@dataclass(frozen=True)
class ItemCompletedEvent:
    item: AgentMessageItem | McpToolCallItem | GenericItem
    type: str = "item.completed"


ThreadEvent = (
    ThreadStartedEvent
    | TurnStartedEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | ItemStartedEvent
    | ItemCompletedEvent
)


def parse_event(line: str) -> ThreadEvent:
    """Parse one JSONL line into a typed event.

    Raises ``CodexProtocolError`` on malformed JSON, unknown top-level
    event types, or explicit ``error`` events.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CodexProtocolError(f"Malformed JSON from codex exec: {line!r}") from exc

    if not isinstance(obj, dict):
        raise CodexProtocolError(f"Top-level JSON must be an object: {obj!r}")

    t = obj.get("type")
    if t == "thread.started":
        return ThreadStartedEvent(thread_id=str(obj.get("thread_id", "")))
    if t == "turn.started":
        return TurnStartedEvent()
    if t == "turn.completed":
        return TurnCompletedEvent(usage=obj.get("usage", {}) or {})
    if t == "turn.failed":
        err = obj.get("error", {}) or {}
        return TurnFailedEvent(error_message=str(err.get("message", "")))
    if t == "item.completed":
        item = obj.get("item", {}) or {}
        return ItemCompletedEvent(item=_coerce_item(item))
    if t == "item.started":
        # item.started signals "Codex is about to invoke this item". For
        # mcp_tool_call items this is the EARLIEST signal that Codex is
        # calling our proxy — by the time ``item.completed`` arrives the
        # call has already been resolved (or cancelled). Surface this as
        # a started event the caller can distinguish from completed.
        item = obj.get("item", {}) or {}
        return ItemStartedEvent(item=_coerce_item(item))
    if t == "item.updated":
        # streaming updates of a longer item — we don't act on them.
        # Surface as started so we don't crash if a consumer branches on
        # type; semantically "still in progress".
        item = obj.get("item", {}) or {}
        return ItemStartedEvent(item=_coerce_item(item))
    if t == "error":
        raise CodexProtocolError(
            f"Codex stream error: {obj.get('message', 'unknown error')}"
        )
    raise CodexProtocolError(f"Unknown event type from codex exec: {t!r}")


def _coerce_item(raw: dict[str, Any]) -> AgentMessageItem | McpToolCallItem | GenericItem:
    item_type = raw.get("type", "")
    if item_type == "agent_message":
        return AgentMessageItem(id=str(raw.get("id", "")), text=str(raw.get("text", "")))
    if item_type == "mcp_tool_call":
        return McpToolCallItem(
            id=str(raw.get("id", "")),
            server=str(raw.get("server", "")),
            tool=str(raw.get("tool", "")),
            arguments=raw.get("arguments", {}) or {},
            status=str(raw.get("status", "")),
            result=raw.get("result"),
            error=raw.get("error"),
        )
    return GenericItem(id=str(raw.get("id", "")), type=item_type, payload=raw)


# ---------------------------------------------------------------------------
# Exec args + runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecArgs:
    """Arguments to one ``codex exec`` invocation."""
    prompt: str
    session_uuid: str | None = None   # if set → ``codex exec resume <uuid>``
    api_key: str | None = None        # if None → cached login (~/.codex/auth.json)
    working_directory: str | None = None
    sandbox_mode: str = "workspace-write"     # Q3: skip-permissions equivalent
    approval_policy: str = "never"            # Q3: skip-permissions equivalent
    skip_git_repo_check: bool = True
    model: str | None = None                  # Q2: None → let Codex pick default
    base_instructions: str | None = None      # system prompt
    additional_directories: list[str] | None = None
    idle_timeout_seconds: float | None = None
    # Codex built-in deny-list (Q5). Each entry passed as
    # ``--config tools.<name>=false`` (the exact key may need adjustment
    # once we verify against the live CLI; this is the documented hook).
    disallowed_tools: tuple[str, ...] = ()


class ExecHandle:
    """A live ``codex exec`` subprocess + its event-stream generator.

    Held by ``CodexChatModel`` between two ainvoke calls inside one
    Codex turn (an ainvoke that returned tool_calls and the next
    ainvoke that resolves them).
    """

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        events: AsyncGenerator[ThreadEvent, None],
    ) -> None:
        self.process = process
        self.events = events

    async def aclose(self) -> None:
        """Terminate the subprocess and drain the generator. Safe to call
        multiple times."""
        # Close the generator first — this drives the ``finally`` block
        # in ``_iter_events`` which is what actually terminates the
        # subprocess. ``contextlib.suppress`` for the case where the
        # generator has already finished naturally.
        with contextlib.suppress(StopAsyncIteration, RuntimeError):
            await self.events.aclose()
        # Belt-and-braces in case _iter_events never ran finally
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()


class CodexExecRunner:
    """Spawns ``codex exec --experimental-json`` and exposes events as
    an async generator.

    The runner itself is stateless — one instance per provider, reused
    across turns. Each call to :meth:`spawn` produces a fresh
    ``ExecHandle`` wrapping a fresh subprocess.
    """

    # Conservative stdout buffer cap so large tool outputs don't exceed
    # asyncio's default 64 KiB readline limit.
    _STREAM_LIMIT_BYTES = 8 * 1024 * 1024

    def __init__(self, executable_path: str | None = None) -> None:
        self._executable_path = executable_path or _find_codex_path()

    # -- command building (factored out for tests) ---------------------
    def _build_argv(self, args: ExecArgs) -> list[str]:
        argv: list[str] = ["exec", "--experimental-json"]

        if args.model:
            argv.extend(["--model", args.model])

        # Q3: skip-permissions equivalent.
        #
        # ``--dangerously-bypass-approvals-and-sandbox`` is the Codex CLI's
        # one-knob "do not prompt, do not gate, do not sandbox" mode —
        # equivalent in spirit to claude-code's ``permission_mode=
        # "bypassPermissions"``. We always pass it because deep-worker
        # already owns sandboxing/approvals at the LangGraph layer
        # (``interrupt_on``); Codex must not impose its own gate, in
        # particular it must not cancel MCP tool calls to our proxy
        # (which it does by default at approval_policy="never" — that
        # flag only covers shell commands).
        #
        # We *also* keep -s and approval_policy for explicit symmetry
        # with the design and so tooling that inspects the argv can
        # tell what mode the agent is in; the bypass flag is the one
        # that actually enables MCP tool calls without prompts.
        argv.append("--dangerously-bypass-approvals-and-sandbox")
        argv.extend(["-s", args.sandbox_mode])
        argv.extend(["--config", f'approval_policy="{args.approval_policy}"'])

        if args.working_directory:
            argv.extend(["--cd", args.working_directory])

        if args.additional_directories:
            for d in args.additional_directories:
                argv.extend(["--add-dir", d])

        if args.skip_git_repo_check:
            argv.append("--skip-git-repo-check")

        if args.base_instructions:
            # Codex reads base-instructions via -c override
            argv.extend(["--config", f'base_instructions="{_escape_toml_string(args.base_instructions)}"'])

        # Q5: deny-list of built-in Codex tools. The actual config key
        # was finalised against the live CLI to be ``tools.<name>=false``.
        # This is best-effort: unknown keys are silently ignored by Codex,
        # which is the failure mode we want — a future Codex version
        # adding a new built-in won't break existing agents.
        for tool_name in args.disallowed_tools:
            argv.extend(["--config", f"tools.{tool_name}=false"])

        # Resume subcommand comes BEFORE the prompt placeholder.
        if args.session_uuid:
            argv.extend(["resume", args.session_uuid])

        # Prompt from stdin (``-``). Always last.
        argv.append("-")
        return argv

    def _build_env(self, args: ExecArgs) -> dict[str, str]:
        env: dict[str, str] = {k: v for k, v in os.environ.items() if v is not None}
        # NB: we explicitly do NOT set CODEX_API_KEY when api_key is None
        # — the CLI must fall through to ~/.codex/auth.json (Q3).
        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key
        return env

    # -- spawn ---------------------------------------------------------
    async def spawn(self, args: ExecArgs) -> ExecHandle:
        """Spawn one ``codex exec``. Returns immediately; consume events
        via the returned handle's ``events`` async generator."""
        argv = self._build_argv(args)
        env = self._build_env(args)

        process = await asyncio.create_subprocess_exec(
            self._executable_path,
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self._STREAM_LIMIT_BYTES,
            env=env,
        )

        # Write prompt to stdin and close it — Codex reads until EOF on
        # ``-``. No back-pressure issues: prompts are small (delta only).
        if process.stdin is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            raise RuntimeError("codex exec has no stdin")
        try:
            process.stdin.write(args.prompt.encode("utf-8"))
            await process.stdin.drain()
        finally:
            process.stdin.close()

        events = self._iter_events(process, args.idle_timeout_seconds)
        return ExecHandle(process=process, events=events)

    async def _iter_events(
        self,
        process: asyncio.subprocess.Process,
        idle_timeout_seconds: float | None,
    ) -> AsyncGenerator[ThreadEvent, None]:
        """Drain stdout line-by-line, yielding parsed events. Cleans up
        on close or natural end-of-stream."""
        assert process.stdout is not None  # nosec — set in spawn()
        try:
            while True:
                if idle_timeout_seconds is None:
                    line_bytes = await process.stdout.readline()
                else:
                    line_bytes = await asyncio.wait_for(
                        process.stdout.readline(), timeout=idle_timeout_seconds,
                    )
                if not line_bytes:
                    break  # EOF
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                yield parse_event(line)
        finally:
            # Whether the consumer drained naturally or threw GeneratorExit,
            # make sure the subprocess goes down.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_codex_path() -> str:
    """Resolution: ``CODEX_PATH`` env → ``shutil.which("codex")`` →
    raise. We do NOT fall back to vendored binaries (the openai_sdk
    extension does that; we don't have a vendored copy)."""
    override = os.environ.get("CODEX_PATH")
    if override:
        return override
    which = shutil.which("codex")
    if which:
        return which
    raise FileNotFoundError(
        "codex CLI not found on PATH. Install Codex CLI or set CODEX_PATH."
    )


def _escape_toml_string(s: str) -> str:
    """Minimal TOML-string escape for use inside ``--config key="..."``.
    Newlines/quotes are the only chars Codex's TOML parser will choke on
    inside a basic string. For multi-line system prompts the caller may
    prefer to truncate or use a file."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
