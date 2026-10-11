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
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Marker written as the first line of the AGENTS.override.md file we drop
# into the Codex working directory. Lets us recognise our own file on the
# next spawn (so we never overwrite a user-authored override) and refuse
# to delete files that were not created by us.
DEEPWORKER_OVERRIDE_MARKER = "<!-- deepworker:codex_override v1 -->"

# Filename Codex CLI 0.130.0 recognises as the per-directory override of
# the standard AGENTS.md instructions (see ``core/src/agents_md.rs``
# strings ``AGENTS.override.md`` / ``AGENTS.md`` in the binary). Both
# files are surfaced to the model as the very first ``user`` message of
# the developer turn, so anything we put in the override is read before
# the user's actual prompt and overrides any project-level AGENTS.md.
DEEPWORKER_OVERRIDE_FILENAME = "AGENTS.override.md"


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------
class CodexProtocolError(RuntimeError):
    """A JSONL line on stdout could not be parsed, or carried a fatal
    top-level ``{"type":"error"}`` event."""


class CodexUsageLimitError(CodexProtocolError):
    """Raised when the Codex CLI reports that the account-level usage cap
    has been hit. Subclass of ``CodexProtocolError`` so existing handlers
    that catch the parent still work.

    The marker in the error message:
        "Codex stream error: You've hit your usage limit. ... try again at <TIME>."
    """


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
    event types, or explicit ``error`` events.  Raises
    ``CodexUsageLimitError`` (a subclass of ``CodexProtocolError``) when
    the error event carries the ``You've hit your usage limit`` marker.
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
        # Codex sends errors in two shapes:
        #   flat:  {"type":"error","message":"..."}
        #   nested: {"type":"error","status":<int>,"error":{"message":"...","type":"..."}}
        nested_err = obj.get("error") or {}
        msg = (
            (obj.get("message") or nested_err.get("message"))
            or "unknown error"
        )
        status = obj.get("status")
        parts = [f"Codex stream error: {msg}"]
        if status is not None:
            parts.append(f"(status={status})")
        error_text = " ".join(parts)
        if "You've hit your usage limit" in msg:
            raise CodexUsageLimitError(error_text)
        raise CodexProtocolError(error_text)
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
    # Mirrors deep-worker's ``-sp``. Fail-closed default: a caller that
    # forgets it gets the SAFE mode. Only with it set does codex run with
    # ``--dangerously-bypass-approvals-and-sandbox`` (see _build_argv).
    auto_approve: bool = False
    skip_git_repo_check: bool = True
    model: str | None = None                  # Q2: None → let Codex pick default
    base_instructions: str | None = None      # system prompt
    additional_directories: list[str] | None = None
    idle_timeout_seconds: float | None = None
    # Q5: native-tool deny-list, expressed as Codex CLI 0.130+
    # feature flags. Each entry becomes one ``--disable <feature>``
    # argument. The previous ``--config disabled_tools=[...]`` channel
    # is gone — Codex 0.130 ignored it (verified live via
    # ``RUST_LOG=codex_api=trace``: the ``response.create`` payload's
    # ``tools=[...]`` list was identical with or without it).
    disabled_features: tuple[str, ...] = ()
    # Q5: top-level Codex config overrides for ``-c key=value`` flags
    # whose values are NOT feature-flag booleans. The canonical case is
    # ``("web_search", '"disabled"')`` — Codex moved web_search out of
    # the feature-flag namespace into a top-level string config in
    # 0.130. Each pair becomes one ``--config <key>=<value>`` argument
    # pair. The ``value`` half is interpolated verbatim — callers
    # therefore quote string values themselves (``'"disabled"'``,
    # ``'"live"'``) so the resulting argv parses as TOML.
    config_overrides: tuple[tuple[str, str], ...] = ()
    # Per-agent MCP server isolation (learned from a related internal incident).
    #
    # The agent's OWN proxy MCP server, injected per-invocation via
    # ``--config mcp_servers.<name>.url=...``. When set, ``_build_argv``
    # also passes ``--ignore-user-config`` so codex does NOT load the
    # shared ``~/.codex/config.toml`` and inherit stale
    # ``deepworker_<dead_pid>`` servers left there by other agents.
    #
    # The provider MUST NOT write to the global config (no
    # ``codex mcp add``) — there is no sanctioned global MCP-server
    # list; agents must be fully independent. See
    # ``docs/DESIGN_CODEX_PROVIDER.md`` §"MCP server isolation".
    #
    # Both fields default to None so a tool-less turn (no proxy) emits
    # no ``mcp_servers.*`` override at all.
    mcp_server_name: str | None = None
    mcp_server_port: int | None = None


def _write_override_file(args: "ExecArgs") -> Path | None:
    """Write the system prompt into ``<cwd>/AGENTS.override.md``.

    Returns the path of the file we created, or ``None`` if we did not
    create one (no working directory, no base_instructions, or a non-ours
    file is already there).

    Refuses to overwrite a file whose first line does NOT contain
    :data:`DEEPWORKER_OVERRIDE_MARKER` — that file is the user's own
    Codex override and we must leave it alone. In that case we log a
    warning to stderr and the Codex turn proceeds with whatever
    instructions the user's file carries (no harm: their override is at
    least as important to them as ours).
    """
    if not args.base_instructions or not args.working_directory:
        return None
    target = Path(args.working_directory) / DEEPWORKER_OVERRIDE_FILENAME

    if target.exists():
        try:
            first_line = target.open("r", encoding="utf-8").readline().rstrip("\n")
        except OSError:
            first_line = ""
        if DEEPWORKER_OVERRIDE_MARKER not in first_line:
            # Someone else's override — never touch it.
            print(
                f"[codex_provider] WARNING: {target} already exists and is not "
                f"owned by deep-worker. Leaving it as-is; deep-worker's system "
                f"prompt will NOT be delivered to this Codex turn.",
                file=sys.stderr,
            )
            return None

    body = (
        f"{DEEPWORKER_OVERRIDE_MARKER}\n"
        f"<!-- Auto-generated by codex_provider for this codex exec subprocess. -->\n"
        f"<!-- Removed automatically when the subprocess exits. -->\n"
        f"\n"
        f"{args.base_instructions}\n"
    )

    try:
        # Atomic-ish write: write to temp file then rename, so a concurrent
        # codex process never sees a half-written override.
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        print(
            f"[codex_provider] WARNING: failed to write {target}: {exc}. "
            f"deep-worker's system prompt will NOT be delivered to this "
            f"Codex turn.",
            file=sys.stderr,
        )
        return None
    return target


def _safe_remove_override_file(path: Path) -> None:
    """Remove ``path`` iff it still carries :data:`DEEPWORKER_OVERRIDE_MARKER`.

    Defensive: a concurrent codex process for a sibling deep-worker agent
    could have rewritten the file in between our spawn and aclose. In
    that case the file is no longer "ours" and we must leave it for that
    other process to clean up.
    """
    try:
        if not path.exists():
            return
        first_line = path.open("r", encoding="utf-8").readline().rstrip("\n")
        if DEEPWORKER_OVERRIDE_MARKER not in first_line:
            return  # someone else's content now — do not touch
        path.unlink()
    except OSError:
        # Best-effort cleanup. A stale file with our marker is annoying
        # but not catastrophic — the next spawn will overwrite it.
        pass


class ExecHandle:
    """A live ``codex exec`` subprocess plus the background event pump.

    Held by ``CodexChatModel`` between two ainvoke calls inside one
    Codex turn. The fields beyond the raw subprocess are the
    plumbing the event pump uses to route Codex events to two
    destinations:

    - ``ainvoke_queue`` — events ``CodexChatModel.ainvoke`` cares
      about (thread.started, turn.started/completed/failed,
      item.started for both proxy MCP and native, item.completed for
      AgentMessageItem).
    - ``native_futures`` — per-``item.id`` ``asyncio.Future`` parked
      at ``item.started`` for native tools and resolved by the pump
      at ``item.completed``. The matching async tool body in
      ``codex_provider.native_tools`` awaits the future. No polling.

    ``pump_task`` is the singleton asyncio Task that drains
    ``self._events`` (the raw JSONL generator). It is created by
    ``CodexExecRunner.spawn`` and cancelled by ``aclose``.

    See ``docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md`` for the full design.

    ``override_file_to_cleanup`` is the path of an ``AGENTS.override.md``
    file that ``spawn`` dropped for this subprocess.  ``aclose``
    removes it iff the file still carries our marker.
    """

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        events: AsyncGenerator[ThreadEvent, None],
        override_file_to_cleanup: Path | None = None,
    ) -> None:
        self.process = process
        # Raw event generator — the pump is the only reader. ainvoke
        # reads ``ainvoke_queue`` instead.
        self.events = events
        self.override_file_to_cleanup = override_file_to_cleanup

        # Filled in by ``CodexExecRunner.spawn`` after the pump is
        # ready. Bounded queue means natural back-pressure on the
        # rare path where Codex emits faster than ainvoke consumes.
        self.ainvoke_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        # Created on native ``item.started``, resolved on the matching
        # ``item.completed``. Tool bodies in
        # ``codex_provider.native_tools`` await these.
        self.native_futures: dict[str, asyncio.Future[str]] = {}
        # The single reader of ``events``. Created by ``spawn``.
        self.pump_task: asyncio.Task[None] | None = None
        # Set by ``event_pump`` to the fatal stream exception (e.g.
        # CodexProtocolError from an API rejection) so that
        # ``CodexChatModel.ainvoke`` — which only ever observes the
        # END_OF_STREAM sentinel on ``ainvoke_queue`` — can re-raise the
        # real cause instead of returning an empty response and stalling
        # the agent (observed live).
        self.pump_error: BaseException | None = None

    async def aclose(self) -> None:
        """Terminate the subprocess and the event pump. Safe to call
        multiple times."""
        # 1) Stop the pump first so it can no longer touch
        #    ``native_futures``. The pump's ``finally`` block releases
        #    every still-parked future with a meaningful exception so
        #    any awaiting tool body wakes up immediately.
        if self.pump_task is not None and not self.pump_task.done():
            self.pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self.pump_task
            self.pump_task = None

        # 2) Belt-and-braces: cancel any future the pump did not get to.
        for fut in list(self.native_futures.values()):
            if not fut.done():
                fut.cancel()
        self.native_futures.clear()

        # 3) Drain the events generator. Pump cancellation already
        #    drove the ``finally`` in ``_iter_events``, but we double-
        #    close in case the pump never started.
        with contextlib.suppress(StopAsyncIteration, RuntimeError):
            await self.events.aclose()

        # 4) Belt-and-braces in case ``_iter_events`` never ran its
        #    finally block — make sure the subprocess goes down.
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()

        # 5) Remove our AGENTS.override.md, if any.
        if self.override_file_to_cleanup is not None:
            _safe_remove_override_file(self.override_file_to_cleanup)
            self.override_file_to_cleanup = None


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

    # ------------------------------------------------------------------
    # Public test-helpers
    #
    # The two methods below are the only ``_build_*`` internals that
    # we deliberately expose under a public name so tests can verify
    # argv/env construction without importing protected attributes
    # (pyright strict treats ``_name`` as private and would block
    # ``runner._build_argv(...)`` in test code with reportPrivateUsage).
    # ------------------------------------------------------------------
    def build_argv_for_test(self, args: ExecArgs) -> list[str]:
        """Public alias for ``_build_argv`` — for tests only.

        Mirrors what ``spawn()`` passes to ``asyncio.create_subprocess_exec``.
        Production code does not call this; ``spawn()`` calls
        ``_build_argv`` directly.
        """
        return self._build_argv(args)

    def build_env_for_test(self, args: ExecArgs) -> dict[str, str]:
        """Public alias for ``_build_env`` — for tests only.

        Mirrors what ``spawn()`` passes as the subprocess ``env``.
        """
        return self._build_env(args)

    # -- command building (factored out for tests) ---------------------
    def _build_argv(self, args: ExecArgs) -> list[str]:
        argv: list[str] = ["exec", "--experimental-json"]

        if args.model:
            argv.extend(["--model", args.model])

        # ``-sp``, with the SAME meaning as on every other provider
        # (owner directive 2026-08-25: "the policy really must be
        # unified"). This used to be unconditional, which made the operator
        # flag decoration on codex: an agent launched WITHOUT ``-sp`` still
        # ran with approvals and sandbox switched off, while grok and
        # claude-code denied every permission request.
        #
        # ``--dangerously-bypass-approvals-and-sandbox`` is the Codex CLI's
        # one-knob "do not prompt, do not gate, do not sandbox" mode —
        # equivalent in spirit to claude-code-sdk's ``permission_mode=
        # "bypassPermissions"``. It is also what lets codex call MCP tools
        # on our proxy without prompting (approval_policy alone covers only
        # shell commands), so WITHOUT ``-sp`` a codex agent is restricted in
        # exactly the way a grok agent without ``-sp`` is.
        #
        # WITHOUT ``-sp``: ``--approve-for-me`` — codex reviews its own
        # approval requests automatically inside its workspace-write
        # sandbox. This is the closest codex equivalent of "restricted but
        # usable", and it is REQUIRED rather than cosmetic: with neither
        # flag, codex announces an MCP tool call and never sends the
        # control request, so every deep-worker tool call hangs for the
        # full 120 s proxy bound and comes back "not_started" (measured
        # live 2026-08-25 — 129.5 s for one ``execute``). The CLI rejects
        # ``--approve-for-me`` together with ``-s`` ("cannot be used with
        # --sandbox"), so the sandbox/approval-policy pair is passed only
        # on the bypass path, where it is accepted.
        if args.auto_approve:
            argv.append("--dangerously-bypass-approvals-and-sandbox")
            argv.extend(["-s", args.sandbox_mode])
            argv.extend([
                "--config", f'approval_policy="{args.approval_policy}"'
            ])
        else:
            argv.append("--approve-for-me")

        # ── Per-agent MCP isolation (observed live) ──
        # Do NOT load the shared ``~/.codex/config.toml``. That file is
        # global to every codex process on the host and used to
        # accumulate one ``[mcp_servers.deepworker_<pid>]`` entry per
        # agent ever launched (the provider wrote them via
        # ``codex mcp add`` — that path is now removed). Loading it made
        # every ``codex exec`` try to connect to dozens of dead ports,
        # each hanging for the 120 s proxy timeout before failover.
        # ``--ignore-user-config`` skips the file entirely; auth still
        # comes from CODEX_HOME and ``resume <uuid>`` still works
        # (sessions live in ``CODEX_HOME/sessions``, not in config.toml).
        #
        # DO NOT remove this without re-introducing global pollution.
        argv.append("--ignore-user-config")
        # Inject ONLY this agent's own proxy MCP server, per-invocation.
        if args.mcp_server_name and args.mcp_server_port:
            url = f"http://127.0.0.1:{args.mcp_server_port}/mcp"
            argv.extend([
                "--config",
                f'mcp_servers.{args.mcp_server_name}.url="{url}"',
            ])

        if args.working_directory:
            argv.extend(["--cd", args.working_directory])

        if args.additional_directories:
            for d in args.additional_directories:
                argv.extend(["--add-dir", d])

        if args.skip_git_repo_check:
            argv.append("--skip-git-repo-check")

        # NOTE: we intentionally do NOT pass ``--config base_instructions=...``
        # here. Codex CLI 0.130.0 silently ignores that key (verified by
        # comparing the live ``session_meta.payload.base_instructions.text``
        # against the value we tried to set — Codex always kept its
        # built-in default GPT-5 base prompt). The system prompt is instead
        # injected via an ``AGENTS.override.md`` file dropped into the
        # working directory in :meth:`spawn`; Codex reads that file and
        # prepends its contents as a ``user`` message at the start of every
        # turn (see ``core/src/agents_md.rs`` in the Codex binary).
        # See a related internal investigation analysis (not included here)
        # for the full investigation.

        # Q5: native-tool deny-list via per-feature ``--disable`` flags.
        #
        # Codex CLI 0.130 silently ignores the older
        # ``--config disabled_tools=[...]`` channel — we verified live
        # via ``RUST_LOG=codex_api=trace`` that the
        # ``response.create`` websocket payload's ``tools=[...]`` list
        # is identical with or without it (see Implementation note 6 in
        # docs/DESIGN_CODEX_PROVIDER.md). The working contract is one
        # ``--disable <feature>`` per feature whose native tools we
        # want stripped from the API request.
        #
        # Mapping (verified live, ``codex features list``):
        #   shell_tool, unified_exec  → exec_command, write_stdin
        #   multi_agent               → spawn_agent, send_input,
        #                               resume_agent, wait_agent,
        #                               close_agent
        #   image_generation          → image_generation
        #   tool_search               → tool_search
        #   apps, plugins             → mcp__codex_apps__*
        for feat in args.disabled_features:
            argv.extend(["--disable", feat])

        # Top-level config overrides for keys that are NOT feature flags
        # (e.g. ``web_search``: Codex moved it out of ``[features]``
        # into a top-level string config in 0.130). Each pair becomes
        # one ``--config <key>=<value>`` arg. The caller already quoted
        # string values so the result parses as TOML.
        for key, value in args.config_overrides:
            argv.extend(["--config", f"{key}={value}"])

        # Resume subcommand comes BEFORE the prompt placeholder.
        if args.session_uuid:
            argv.extend(["resume", args.session_uuid])

        # Prompt from stdin (``-``). Always last.
        argv.append("-")
        return argv

    def _build_env(self, args: ExecArgs) -> dict[str, str]:
        # ``os.environ`` keys + values are always strings — the ``v is
        # not None`` filter is belt-and-braces for monkey-patched
        # environments in tests. pyright sees ``v: str`` here so the
        # check looks redundant; keep it for the runtime safety.
        env: dict[str, str] = {
            k: v for k, v in os.environ.items()
            if v is not None  # pyright: ignore[reportUnnecessaryComparison]
        }
        # NB: we explicitly do NOT set CODEX_API_KEY when api_key is None
        # — the CLI must fall through to ~/.codex/auth.json (Q3).
        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key
        return env

    # -- spawn ---------------------------------------------------------
    async def spawn(
        self,
        args: ExecArgs,
        *,
        proxy_mcp_server_name: str | None = None,
    ) -> ExecHandle:
        """Spawn one ``codex exec``. Returns immediately; consume events
        via the returned handle's ``ainvoke_queue`` (the background
        event pump fills it).

        ``proxy_mcp_server_name`` is the name of the deep-worker proxy
        MCP server registered with this codex subprocess (typically
        ``deepworker_<pid>``). The pump uses it to discriminate
        between OUR proxy mcp_tool_call events (which need
        ainvoke-level handling) and OTHER MCP server events (passed
        through to ainvoke as-is). When ``None`` (e.g. in tests that
        do not exercise the proxy path) the pump treats every MCP
        event as "not ours" — safe default.
        """
        # Local import — event_pump depends on exec_runner so we
        # import lazily to avoid a circular module at load time.
        from .event_pump import event_pump

        argv = self._build_argv(args)
        env = self._build_env(args)

        # Drop AGENTS.override.md with our system prompt into the working
        # directory BEFORE spawning codex, so the very first turn sees it.
        override_file = _write_override_file(args)

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
            if override_file is not None:
                _safe_remove_override_file(override_file)
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            raise RuntimeError("codex exec has no stdin")
        try:
            process.stdin.write(args.prompt.encode("utf-8"))
            await process.stdin.drain()
        finally:
            process.stdin.close()

        events = self._iter_events(process, args.idle_timeout_seconds)
        handle = ExecHandle(
            process=process,
            events=events,
            override_file_to_cleanup=override_file,
        )
        # Spawn the background event pump — the single reader of
        # ``handle.events``. It fills ``handle.ainvoke_queue`` and
        # ``handle.native_futures`` event-driven (no polling).
        handle.pump_task = asyncio.create_task(
            event_pump(
                handle,
                proxy_mcp_server_name=proxy_mcp_server_name or "",
            ),
            name=f"codex-event-pump-{process.pid}",
        )
        return handle

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

            # ── Non‑zero exit check (SIGKILL / crash / error) ──────────
            # If the subprocess exited while we were still reading,
            # ``readline`` returned EOF before the process was ready to
            # close — the JSONL stream ended prematurely.  Wait for the
            # actual exit code now and raise if it indicates failure.
            # This catches ``kill -9`` (exit code -9), segfaults, etc.
            # and surfaces them as a ``CodexProtocolError`` through the
            # event pump's error path instead of silently returning a
            # partial / empty assistant message.
            #
            # If ``returncode`` is still ``None`` the process is still
            # alive — EOF happened because aclose/cancel closed the pipe,
            # not because the process crashed.
            ret = await process.wait()
            if ret != 0:
                sig_hint = " (SIGKILL)" if ret == -9 else ""
                raise CodexProtocolError(
                    f"codex exec exited with abnormal code {ret}{sig_hint}"
                )
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
    raise. We do NOT fall back to vendored binaries (the OpenAI Agents SDK
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


