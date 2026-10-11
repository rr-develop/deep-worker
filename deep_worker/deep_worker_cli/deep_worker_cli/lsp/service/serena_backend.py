"""Serena backend attached to the shared per-project LSP service.

See ``docs/DESIGN_LSP_SERENA_BACKEND_ATTACH.md``.

This is the ONLY place in the tree that spawns Serena — the module
lives under ``deep_worker_cli.lsp.service`` which is the allowlisted
package per POLICY §8 (pinned by ``policy_guard``). Bridge, agent and
every other module reach the backend exclusively through the gRPC
surface of ``LSPServiceServer``.

Design invariants realised here:

* **Single-flight on the whole backend surface.** One
  ``asyncio.Lock`` guards ``call_tool`` and ``refresh``. Serena stdio
  is a single pair of pipes; MCP JSON-RPC needs request/response id
  match; the underlying language server is often single-threaded.
* **Fail-loud.** Every failure path returns a ``pb.ToolResult`` with
  ``is_error=True`` and a non-empty ``error_message``; there is no
  path that returns an empty content on error.
* **No stdio fallback.** If Serena did not start, the service stays
  up but reports ``Health.alive=False`` and the tool RPCs continue to
  answer ``UNIMPLEMENTED``. A silent in-process backend would be the
  exact regression policy §8 exists to prevent.
* **Snake-case tool names on the MCP side.** RPC names on gRPC are
  camel-case (``FindSymbol``); Serena expects ``find_symbol``. The
  mapping table is authoritative here and mirrors the bridge_loop.
* **JSON nulls dropped from ``args_json``** — Serena tool schemas
  reject strict optionals set to None (same fix as ``external_mcp``
  ``_UNSET``).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncContextManager, Awaitable, Callable, Final

from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

_log = logging.getLogger(__name__)


def _blocking_waitpid(pid: int) -> None:
    """Block in a worker thread until ``pid`` exits.

    Used as the no-pidfd fallback for child-death detection on CPython
    builds where ``os.pidfd_open`` is unavailable (it needs
    ``<sys/pidfd.h>`` at compile time — uv-installed CPython 3.11.14
    ships without it).  Runs to completion once the kernel reaps the
    child; ``WNOHANG`` is NOT used, so we never busy-loop.  Idempotent
    against ECHILD: by the time this is called, the child has either
    been reaped by its real parent (and the call returns ``(0, 0)``
    harmlessly) or still belongs to us and we block until it dies.
    """
    while True:
        try:
            waited_pid, _status = os.waitpid(pid, 0)
        except ChildProcessError:
            # Already reaped (e.g. by a previous test teardown).
            return
        if waited_pid == pid:
            return


#: gRPC RPC name → MCP tool name Serena advertises. Mirrors the 11
#: entries in ``bridge_loop._TOOLS`` exactly.
_RPC_TO_TOOL: Final[dict[str, str]] = {
    "FindSymbol": "find_symbol",
    "FindReferencingSymbols": "find_referencing_symbols",
    "FindDeclaration": "find_declaration",
    "FindImplementations": "find_implementations",
    "GetSymbolsOverview": "get_symbols_overview",
    "GetDiagnosticsForFile": "get_diagnostics_for_file",
    "RenameSymbol": "rename_symbol",
    "ReplaceSymbolBody": "replace_symbol_body",
    "InsertBeforeSymbol": "insert_before_symbol",
    "InsertAfterSymbol": "insert_after_symbol",
    "SafeDeleteSymbol": "safe_delete_symbol",
}

#: Names by which Serena has, at various versions, exposed the
#: "reload the project / restart the language server" operation.
#: Order = preference. A7: we PICK by matching against ``list_tools``
#: output rather than hardcoding one name that drifts across versions.
#:
#: Serena 1.6.x exposes ``restart_language_server`` (opted in via our
#: shipped ``--add-mode`` YAML — see
#: ``deep_worker_cli.lsp.resources.SERENA_MODE_YML``); older versions
#: shipped ``activate_project`` / ``refresh_project`` /
#: ``serena__refresh``. The first hit wins.
_REFRESH_TOOL_CANDIDATES: Final[tuple[str, ...]] = (
    "restart_language_server",
    "activate_project",
    "refresh_project",
    "serena__refresh",
)

#: LSP error codes a language server uses to cancel its OWN in-flight
#: requests while it mutates the workspace (background reindex, file
#: open, edit applied): ``-32800`` RequestCancelled, ``-32801``
#: ContentModified. Both are transient — the server is telling us to
#: come back, not that the request was wrong.
#:
#: They reach us as TEXT, not as an exception: Serena catches the
#: solidlsp error and answers with a normal MCP result carrying
#: ``isError=True`` and ``LSPError.__str__`` output ("<msg> (<code>)"),
#: possibly nested inside "... (caused by ...)". There is no
#: structured error code on the MCP surface, so matching the rendered
#: code is the only signal available.
#:
#: The pattern is deliberately strict (design A1): a bare ``-32800``
#: substring would also match ``-328001`` and turn an unrelated,
#: unrecoverable failure into a retry loop.
_TRANSIENT_CODE_RE: Final = re.compile(r"\(-3280[01]\)|(?<![\d-])-3280[01](?![\d])")

#: One original attempt plus two retries. Kept small on purpose: the
#: point is to ride out a short mutation window, not to wait out a
#: long reindex — the latter is reported to the agent instead.
_TRANSIENT_MAX_ATTEMPTS: Final = 3

#: Exponential backoff between retries; total <= 0.75s, i.e. ~0.6% of
#: the 120s call budget.
_TRANSIENT_BACKOFF_S: Final[tuple[float, ...]] = (0.25, 0.5)

#: Below this many seconds left in the series budget an attempt cannot
#: plausibly succeed, so we stop and explain instead.
_MIN_ATTEMPT_S: Final = 0.05

#: What the agent sees when the retries did not clear the mutation
#: window. Three things it MUST convey (design §4.3): the cause, that
#: it is temporary, and — explicitly — that this is not a restriction
#: on issuing several tool calls at once. The last sentence exists
#: because an agent previously read a raw ``cancelled (-32800)`` as
#: "concurrent tool calls are forbidden" and serialised itself.
#: ``{attempts}`` is the number of attempts that actually reached the
#: server, not the configured cap: the series also ends when the call
#: budget runs out, and quoting the cap would be a lie in exactly the
#: message that exists to stop the agent from guessing. Phrased
#: neutrally ("attempts made: N") so it reads correctly for N = 1.
_TRANSIENT_EXHAUSTED_MSG: Final = (
    "serena {tool}: the language server is busy reindexing the project "
    "and cancelled the request (LSP -32800/-32801); attempts made: "
    "{attempts}. This is temporary — retry the same call in a few "
    "seconds. This is NOT a limit on concurrent tool calls: calls from "
    "one or many agents are queued and executed one after another."
)


#: Text markers that mean "the language server / the Serena transport
#: is GONE", as opposed to "the server is busy" (``_TRANSIENT_CODE_RE``)
#: or "your request was wrong" (everything else).
#:
#: Matched case-insensitively as substrings, because — exactly like the
#: transient codes — this reaches us as rendered TEXT and not as a
#: structured code: Serena catches the solidlsp exception and answers
#: with an MCP result carrying ``isError=True``.
#:
#: Provenance of every entry (verified against live logs in
#: ``/home/<user>/.serena/logs/`` and upstream sources):
#:
#: * the first three come from ``solidlsp/ls_process.py`` lines
#:   595 / 634 / 642 — the reader thread noticing ``poll() is not None``;
#: * ``transport error`` / ``broken pipe`` are our own wrapper
#:   (``call_tool`` below) and ``ls_process._send_payload``;
#: * the last two are states of OUR backend object between the death
#:   and the supervisor's restart (``call_tool`` guards below).
#:
#: The list is deliberately CLOSED (design §4.1): matching "any error"
#: would turn an unrecoverable, correctly-reported failure into a retry
#: loop — pinned by ``test_non_terminated_error_is_not_retried``.
_TERMINATED_MARKERS: Final[tuple[str, ...]] = (
    "languageserverterminatedexception",
    "language server stdout read process terminated",
    "language server process terminated",
    "transport error",
    "broken pipe",
    "serena backend not running",
    "serena backend torn down mid-call",
)

#: What the agent sees when a read-only call could not be recovered even
#: after the retry. Three jobs (design §4.5): name the cause, place the
#: blame on our infrastructure, and — explicitly — kill the two WRONG
#: conclusions a real agent drew from the raw exception, namely "I must
#: not issue concurrent tool calls" and "I must not use absolute paths
#: or broad searches". Both were disproven (design §1.3): 2 of 3
#: observed crashes happened on a narrow ``relative_path``.
_LS_TERMINATED_MSG: Final = (
    "serena {tool}: the language server process died and did not come "
    "back in time. This is an infrastructure fault on our side, not a "
    "problem with your request: it was NOT caused by concurrent tool "
    "calls, and NOT by using an absolute path or a broad search. The "
    "known upstream trigger is the Kotlin language server crashing on "
    "an unhandled 'workspace/diagnostic/refresh'. Retry the same call "
    "in a few seconds; if it keeps failing, check "
    "`deep-worker lsp status`."
)

#: Mutating tools get their own text because "just retry" would be
#: actively harmful advice: the edit is written to disk BEFORE Serena
#: asks the language server for diagnostics
#: (``serena/tools/symbol_tools.py`` + ``tools_base.DiagnosticsContext``),
#: so a death at that point leaves an AMBIGUOUS COMMIT — the change may
#: already be on disk. Replaying ``insert_*`` would duplicate the body.
_LS_TERMINATED_MUTATING_MSG: Final = (
    "serena {tool}: the language server process died while this edit "
    "was in flight, so the outcome is UNKNOWN — the edit may or may "
    "not have been written to the file. It was NOT retried "
    "automatically, on purpose, because repeating it could apply the "
    "same change twice. This is an infrastructure fault on our side: "
    "it was NOT caused by concurrent tool calls, and NOT by using an "
    "absolute path or a broad search. Read the affected file/symbol "
    "first to see whether the change is already there, then decide "
    "whether to repeat the call."
)

#: Distinct from both of the above: the call ran out of the service-side
#: budget WITHOUT any sign of the server dying. Reporting that as a
#: termination would be a false diagnostic (design §4.4), so it gets its
#: own text pointing at the likely real cause — a big project still
#: being indexed — and an actionable narrowing hint.
_DISPATCH_TIMEOUT_MSG: Final = (
    "serena {tool}: the call exceeded the service-side budget of "
    "{budget:.0f}s and was cancelled. The language server did not "
    "report a crash — it is most likely still indexing a large "
    "project. This is NOT a limit on concurrent tool calls and NOT "
    "caused by an absolute path. Retry in a few seconds, or narrow the "
    "request (pass relative_path to scope it to one file)."
)


class _BackendDied(RuntimeError):
    """Raised when the MCP child died while a call awaited its reply.

    Carries the word "transport error" indirectly: ``call_tool`` wraps
    it into that prefix, so ``_is_backend_terminated`` classifies it as
    a termination and the dispatcher retries on the fresh backend.
    """

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"the serena process died while {tool_name} was in flight"
        )


def _is_transient_cancel(message: str) -> bool:
    """True if ``message`` carries a transient LSP cancellation code."""
    if not message:
        return False
    return _TRANSIENT_CODE_RE.search(message) is not None


def _is_backend_terminated(message: str) -> bool:
    """True if ``message`` says the language server / transport is gone.

    See ``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md`` §4.1. Deliberately
    NOT a catch-all: a wrong path, a missing symbol and a transient
    cancellation must all answer False so they reach the agent
    unchanged instead of being retried.
    """
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _TERMINATED_MARKERS)


def build_serena_spawn_argv(project_root: Path | str) -> list[str]:
    """argv Serena for ``asyncio.create_subprocess_exec``.

    A4: identical structural shape to
    ``deep_worker_cli.lsp.serena_config.build_serena_mcp_entry`` —
    same command, same args, same project resolution — so we don't
    invent a second CLI dialect. The mcp.json path uses this via the
    bridge for legacy configs; the service uses it directly.
    """
    from deep_worker_cli.lsp.serena_config import build_serena_mcp_entry

    entry = build_serena_mcp_entry(workspace_root=project_root)
    return [entry["command"], *entry["args"]]


def pick_refresh_tool(advertised: list[str]) -> str | None:
    """Pick the first name from ``_REFRESH_TOOL_CANDIDATES`` present
    in ``advertised``. Returns ``None`` if none match — see A7.
    """
    for name in _REFRESH_TOOL_CANDIDATES:
        if name in advertised:
            return name
    return None


def _strip_json_nulls(args: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` — A8."""
    return {k: v for k, v in args.items() if v is not None}


# ── MCP session factory (patched in unit tests) ──────────────────────


class _OwnedStdioSession:
    """Minimal ``mcp.ClientSession`` transport with a captured child pid.

    We deliberately do NOT use ``mcp.client.stdio.stdio_client``:

    * It hides the ``anyio.Process`` from callers, so we cannot open a
      ``pidfd`` for the child (needed for :class:`SerenaSupervisor`).
    * It closes the process via an ``anyio`` cancel scope pinned to
      the entering task. In this service the backend is started from
      one asyncio task (attach_backend) and torn down from another
      (signal-handler-triggered ``server.stop`` or the supervisor's
      restart task). Cross-task ``aclose`` raises
      ``RuntimeError: Attempted to exit cancel scope in a different
      task than it was entered in`` and takes the whole service down.

    Instead we spawn Serena with :func:`asyncio.create_subprocess_exec`
    (task-agnostic), bridge its stdio to a pair of anyio in-memory
    streams (JSON-RPC per line, exactly the MCP stdio spec) and hand
    those streams to a real ``mcp.ClientSession``.

    Public shape mirrors the previous ``_SessionCM``: an async context
    manager whose ``__aenter__`` returns an initialised session and
    whose ``__aexit__`` shuts everything down. The captured pid is
    exposed via :meth:`pid`.
    """

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv
        self._stack = AsyncExitStack()
        self._proc: asyncio.subprocess.Process | None = None
        self._session: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._exit_task: asyncio.Task[None] | None = None
        self._read_stream_writer: Any = None
        self._write_stream_reader: Any = None
        #: Invoked once when the child process is observed to exit.
        #: :class:`SerenaBackend` installs ``_mark_lost`` here so a call
        #: parked on a session nobody will answer stops waiting.
        self._on_death: "Callable[[], None] | None" = None
        #: ``serverInfo.version`` from the single ``initialize()``
        #: handshake below (A1: the handshake is never repeated just
        #: to read a version). Empty string when the peer omits it.
        self.server_version: str = ""

    # ── public ───────────────────────────────────────────────────────

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def set_on_death(self, hook: "Callable[[], None] | None") -> None:
        """Install the child-death notification (see ``_on_death``)."""
        self._on_death = hook

    async def __aenter__(self) -> Any:
        import anyio  # noqa: PLC0415 — local import keeps cold-import light
        from mcp import ClientSession  # noqa: PLC0415
        from mcp.shared.message import SessionMessage  # noqa: PLC0415
        from mcp import types as mcp_types  # noqa: PLC0415

        # Spawn the child ourselves; ``start_new_session=True`` mirrors
        # what the mcp SDK does so signals don't escape to our group.
        self._proc = await asyncio.create_subprocess_exec(
            self._argv[0],
            *self._argv[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Anyio in-memory streams; ClientSession consumes read_stream
        # and produces write_stream.
        read_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_reader = anyio.create_memory_object_stream(0)
        self._read_stream_writer = read_writer
        self._write_stream_reader = write_reader

        proc = self._proc
        stdout = proc.stdout
        stdin = proc.stdin
        assert stdout is not None and stdin is not None

        async def _reader() -> None:
            try:
                while True:
                    line = await stdout.readline()
                    if not line:  # EOF -> child gone
                        return
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not text:
                        continue
                    try:
                        message = mcp_types.JSONRPCMessage.model_validate_json(
                            text
                        )
                    except Exception as exc:  # noqa: BLE001 — surface parse errors
                        await read_writer.send(exc)
                        continue
                    await read_writer.send(SessionMessage(message))
            except (asyncio.CancelledError, anyio.ClosedResourceError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    await read_writer.aclose()

        async def _writer() -> None:
            try:
                async for session_message in write_reader:
                    payload = session_message.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    stdin.write((payload + "\n").encode("utf-8"))
                    with contextlib.suppress(Exception):
                        await stdin.drain()
            except (asyncio.CancelledError, anyio.ClosedResourceError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    stdin.close()

        async def _watch_exit() -> None:
            """Turn child death into session death — EOF is not enough.

            Serena spawns the language servers as its own children, and
            they INHERIT this stdout pipe. So when Serena dies the write
            end stays open in the grandchildren, ``readline()`` never
            returns EOF, ``ClientSession`` never learns about the break
            and a call that is already in flight hangs until the call
            budget expires. The dispatcher then sees a timeout instead
            of "backend terminated" and never retries — exactly the
            failure this whole feature exists to remove.

            Death is observed through a ``pidfd`` — the same primitive
            :class:`SerenaSupervisor` already uses — and NOT through
            ``proc.wait()``: asyncio only resolves ``wait()`` once the
            child has exited AND every stdio pipe has disconnected, so
            the leaked grandchild pipe would block that too. Closing
            the read stream makes the MCP session fail every pending
            request with "Connection closed", which
            ``_is_backend_terminated`` recognises.
            """
            death = asyncio.Event()
            loop = asyncio.get_running_loop()
            pidfd_supported = True
            try:
                fd = os.pidfd_open(proc.pid, 0)
            except (OSError, AttributeError) as e:
                # ``os.pidfd_open`` is unavailable on some CPython builds
                # (it needs <sys/pidfd.h> at compile time — uv-installed
                # CPython 3.11.14 ships without it).  Without a kernel
                # descriptor we cannot make ``loop.add_reader`` work
                # against the child, and ``os.kill(pid, 0)`` is unsafe
                # (the slot can be reused within microseconds).  Fall
                # back to a blocking ``waitpid`` running in the default
                # executor — that is event-driven from the asyncio
                # point of view (one Future resolves exactly once when
                # the child dies) and complies with the no-polling
                # policy enforced by ``test_no_polling_policy``.
                _log.warning(
                    "pidfd_open(%d) failed: %s; falling back to "
                    "executor waitpid so in-flight calls still fail "
                    "fast when the MCP child dies",
                    proc.pid, e,
                )
                pidfd_supported = False
                fd = -1  # marker so the post-block cleanup skips close()
                waitpid_future: asyncio.Future[None] = (
                    loop.run_in_executor(None, _blocking_waitpid, proc.pid)
                )

                def _on_waitpid_done(_fut: asyncio.Future[None]) -> None:
                    if not death.is_set():
                        death.set()

                waitpid_future.add_done_callback(_on_waitpid_done)

            cancelled = False
            try:
                if pidfd_supported:
                    loop.add_reader(fd, death.set)
                try:
                    await death.wait()
                finally:
                    if pidfd_supported:
                        loop.remove_reader(fd)
            except asyncio.CancelledError:
                cancelled = True
            finally:
                if pidfd_supported and fd >= 0:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            if cancelled:
                return

            _log.warning(
                "mcp child %d exited; failing the session so in-flight "
                "calls report a dead backend instead of hanging",
                proc.pid,
            )
            with contextlib.suppress(Exception):
                await read_writer.aclose()
            hook = self._on_death
            if hook is not None:
                try:
                    hook()
                except Exception:  # noqa: BLE001 — never take the loop down
                    _log.exception("mcp child-death hook raised")

        self._reader_task = asyncio.create_task(_reader())
        self._writer_task = asyncio.create_task(_writer())
        self._exit_task = asyncio.create_task(_watch_exit())

        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        try:
            init_result = await session.initialize()
        except BaseException:
            # Best-effort cleanup before re-raising.
            await self._teardown()
            raise
        # Version reporting is cosmetic: a peer that omits serverInfo
        # must not break the attach, so read it fail-quiet.
        with contextlib.suppress(Exception):
            self.server_version = (
                getattr(getattr(init_result, "serverInfo", None), "version", "")
                or ""
            )
        self._session = session
        return session

    async def __aexit__(self, *_exc: object) -> None:
        await self._teardown()

    # ── internals ────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        # Close the client session first (in whatever task calls us —
        # ClientSession itself does NOT use anyio cancel scopes across
        # tasks, so this is safe).
        with contextlib.suppress(Exception):
            await self._stack.aclose()

        # Kill the child if still alive.
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(proc.wait(), timeout=2.0)
            # The child can already be dead while its stdout/stderr
            # pipes stay open in GRANDchildren — Serena's language
            # servers inherit them. asyncio resolves ``proc.wait()``
            # only once the process has exited AND every pipe has
            # disconnected, so the waits above can burn their whole
            # timeout on a corpse and still leave the subprocess
            # transport unfinished (its ``__del__`` then fires against
            # a closed loop). Closing our end disconnects the pipes on
            # our side, which lets asyncio finish and reap.
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.close()
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(proc.wait(), timeout=2.0)

        # Cancel reader/writer/exit-watcher tasks.
        for t in (self._reader_task, self._writer_task, self._exit_task):
            if t is not None:
                t.cancel()
                with contextlib.suppress(
                    asyncio.CancelledError, Exception
                ):
                    await t

        # Close anyio streams.
        for s in (self._read_stream_writer, self._write_stream_reader):
            if s is not None:
                with contextlib.suppress(Exception):
                    await s.aclose()


async def _open_mcp_session(argv: list[str]) -> "AsyncContextManager[Any]":
    """Return an async context manager yielding a live ``ClientSession``.

    Wrapped in its own function so tests replace this single seam with
    a fake session, exercising the whole backend without spawning
    Serena or the language server. Production uses
    :class:`_OwnedStdioSession` which captures the child pid for
    pidfd-driven auto-restart (see :class:`SerenaSupervisor`).
    """
    return _OwnedStdioSession(argv)


# ── SerenaBackend ────────────────────────────────────────────────────


class SerenaBackend:
    """Facade over one Serena MCP subprocess for one project.

    Lifecycle:

        backend = SerenaBackend(project_root)
        await backend.start()          # spawn + initialize + list_tools
        server.set_tool_handler(backend.call_tool)
        server.set_backend_refresh_handler(backend.refresh)
        server.mark_backend_ready()
        ...
        await backend.stop()           # EOF + SIGTERM + SIGKILL as needed

    Every failure surfaces as ``pb.ToolResult(is_error=True,
    error_message=<non-empty>)`` — never an empty content path.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        call_timeout_s: float = 120.0,
        on_lost: "Callable[[], None] | None" = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._call_timeout_s = call_timeout_s
        self._argv = build_serena_spawn_argv(self._project_root)
        self._lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._refresh_tool: str | None = None
        self._advertised_tools: list[str] = []
        #: Called ONCE, from ``_mark_lost``, the first time we notice the
        #: backend is unusable — either the underlying MCP transport
        #: raised during a call, or ``stop`` ran. Server side installs
        #: this at ``attach_backend`` so ``Health.alive`` flips back to
        #: False. Kept intentionally narrow (no arguments, no async) so
        #: it can never itself deadlock the tool path.
        self._on_lost = on_lost
        self._lost = False
        #: Set by ``_mark_lost``. A call already parked on the MCP
        #: session waits on this alongside its response: nobody else
        #: will ever wake it. Failing pending requests is the job of
        #: ``_receive_loop``'s ``finally`` (``mcp/shared/session.py``),
        #: but a request registered AFTER that loop has exited — the
        #: exact race when the child dies between "call scheduled" and
        #: "request written" — is invisible to it and would otherwise
        #: hang for the whole call budget, hiding the death behind a
        #: timeout and suppressing the dispatcher's retry.
        self._lost_event = asyncio.Event()
        #: Called once per RETRY (never for the first attempt) when a
        #: transient cancellation was observed. Purely observational —
        #: the server wires a counter here; see design §4.5.
        self._on_transient_retry: "Callable[[], None] | None" = None
        #: Real OS pid of the spawned Serena. Populated by ``start()``
        #: from the owned transport (:class:`_OwnedStdioSession`);
        #: ``None`` before start, after stop, or when the test seam
        #: bypasses the real transport. Used by
        #: :class:`SerenaSupervisor` for ``os.pidfd_open`` — see
        #: ``docs/DESIGN_LSP_SERENA_PIDFD_RESTART.md``.
        self._pid: int | None = None
        #: Session-lifetime task: owns the ``session_cm``/``ClientSession``
        #: context managers and only tears them down when
        #: ``_stop_event`` is set. Keeping enter/exit in a single task
        #: is what lets ``SerenaBackend.stop()`` be safely called
        #: cross-task (anyio cancel scopes pin exit to the entering
        #: task; we hide that with an event-driven proxy instead).
        self._session_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready_event: asyncio.Event | None = None
        self._start_error: BaseException | None = None
        #: ``serverInfo.version`` copied from the transport's single
        #: ``initialize()`` handshake (A1). Empty until the session is
        #: up, and cleared again when it goes away so the service never
        #: reports the version of a backend that is no longer running.
        self._serena_version: str = ""

    # ── introspection ────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._session is not None

    @property
    def serena_version(self) -> str:
        return self._serena_version

    @property
    def refresh_tool(self) -> str | None:
        return self._refresh_tool

    @property
    def pid(self) -> int | None:
        return self._pid

    def set_on_lost(self, hook: "Callable[[], None] | None") -> None:
        """Replace / install the not-ready hook (A3).

        Callable from the server-side ``attach_backend`` after
        construction so tests can construct a plain backend without
        wiring a server. Idempotent w.r.t. ``_lost``: if the backend
        has already been lost, the new hook is called immediately so
        late installers still see the correct state.
        """
        self._on_lost = hook
        if self._lost and hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001 — defensive
                _log.exception("SerenaBackend on_lost hook raised")

    def set_on_transient_retry(
        self, hook: "Callable[[], None] | None"
    ) -> None:
        """Install the retry-observed hook (design A5).

        Invoked once per retry, never for the initial attempt. Kept
        synchronous and argument-free for the same reason as
        ``set_on_lost``: it runs on the tool path and must not be able
        to block or deadlock it.
        """
        self._on_transient_retry = hook

    def _note_transient_retry(self) -> None:
        hook = self._on_transient_retry
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001 — metrics must never break a call
            _log.exception("SerenaBackend on_transient_retry hook raised")

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bring the MCP session up in a **dedicated** task.

        The session context managers (transport + ``ClientSession``)
        both use anyio cancel scopes that must exit in the same task
        they entered. ``SerenaBackend`` is torn down from other tasks
        (signal handler → server.stop → owner.stop; supervisor's
        _on_death task on restart). To satisfy anyio without exposing
        those constraints to callers, we do all enter/exit in a single
        long-lived task (``_session_task``) driven by ``_stop_event``.
        The public API waits on ``_ready_event`` for handshake success
        and re-raises whatever the session task recorded on failure.
        """
        if self._session_task is not None:
            return
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._start_error = None
        # A restarted backend must not inherit the previous life's
        # death: ``_lost_event`` gates every call, so a stale set flag
        # would fail the very first call on a healthy process.
        self._lost = False
        self._lost_event = asyncio.Event()
        self._session_task = asyncio.create_task(self._run_session())
        await self._ready_event.wait()
        if self._start_error is not None:
            err = self._start_error
            self._start_error = None
            raise err

    async def _run_session(self) -> None:
        assert self._ready_event is not None and self._stop_event is not None
        try:
            session_cm = await _open_mcp_session(self._argv)
            # The owned transport knows first when the child is gone.
            # Test seams that fake the session simply do not expose the
            # hook, and the backend keeps its previous behaviour.
            set_on_death = getattr(session_cm, "set_on_death", None)
            if callable(set_on_death):
                set_on_death(self._mark_lost)
            async with AsyncExitStack() as stack:
                session = await stack.enter_async_context(session_cm)
                listed = await session.list_tools()
                names = [t.name for t in getattr(listed, "tools", [])]
                self._advertised_tools = names
                self._refresh_tool = pick_refresh_tool(names)
                self._session = session
                # ``_OwnedStdioSession`` exposes the real child pid;
                # fake sessions used by unit tests may not, in which
                # case the supervisor treats ``backend.pid is None`` as
                # "no pidfd monitor" and the fail-loud A3 path still
                # applies.
                self._pid = getattr(session_cm, "pid", None)
                # A1: no second handshake — the transport already
                # captured the version during its own ``initialize()``.
                self._serena_version = (
                    getattr(session_cm, "server_version", "") or ""
                )
                self._stack = stack  # kept for backwards-compat only
                self._ready_event.set()
                await self._stop_event.wait()
        except BaseException as e:  # noqa: BLE001 — surface via ready_event
            self._start_error = e
            if not self._ready_event.is_set():
                self._ready_event.set()
        finally:
            self._session = None
            self._stack = None
            self._pid = None
            self._serena_version = ""
            self._mark_lost()

    async def stop(self) -> None:
        """Cross-task-safe planned shutdown.

        Sets ``_stop_event`` and awaits the session task. All enter/
        exit for the anyio-owned context managers happens inside that
        one task, so this method may be called from any asyncio task
        without triggering "Attempted to exit cancel scope in a
        different task than it was entered in".
        """
        task = self._session_task
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._session_task = None
        self._stop_event = None
        self._ready_event = None
        self._session = None
        self._stack = None
        self._pid = None
        self._serena_version = ""
        self._mark_lost()

    def _mark_lost(self) -> None:
        """Notify the owner once that the backend is unusable.

        Idempotent: repeated transport errors on the same session must
        not re-fire the hook. Called both from ``stop`` (planned
        shutdown) and from the transport-error branch of ``call_tool``
        / ``refresh`` (unplanned death).
        """
        if self._lost:
            return
        self._lost = True
        self._lost_event.set()
        self._session = None
        hook = self._on_lost
        if hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001 — never let the hook take us down
                _log.exception("SerenaBackend on_lost hook raised")

    # ── tool dispatch ────────────────────────────────────────────────

    async def call_tool(self, rpc_name: str, args_json: str) -> pb.ToolResult:
        tool_name = _RPC_TO_TOOL.get(rpc_name)
        if tool_name is None:
            return pb.ToolResult(
                is_error=True,
                error_message=f"unknown RPC {rpc_name!r}",
            )
        if self._session is None:
            return pb.ToolResult(
                is_error=True,
                error_message=f"serena backend not running (call to {rpc_name})",
            )
        try:
            raw = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return pb.ToolResult(
                is_error=True,
                error_message=f"{rpc_name}: args_json is not valid JSON: {e}",
            )
        if not isinstance(raw, dict):
            return pb.ToolResult(
                is_error=True,
                error_message=f"{rpc_name}: args_json must decode to an object",
            )
        args = _strip_json_nulls(raw)

        async with self._lock:
            session = self._session
            if session is None:
                return pb.ToolResult(
                    is_error=True,
                    error_message=f"serena backend torn down mid-call ({rpc_name})",
                )
            # Transient-cancel retry (design §4.2). The whole loop —
            # call, convert, classify, retry — stays INSIDE the lock
            # (A4): releasing it between attempts would send this call
            # to the back of the queue while the server is still busy
            # with the very mutation we are waiting out.
            #
            # One shared budget for the series (A2): per-attempt
            # timeouts alone could stack to N x call_timeout_s and blow
            # past the client deadline, so the agent would get
            # DEADLINE_EXCEEDED instead of the explanation below.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._call_timeout_s
            last_raw = ""
            #: Attempts that actually reached Serena. NOT the loop
            #: counter: the series can also end because the budget ran
            #: out, and reporting the cap instead of the real number
            #: would hand the agent a false diagnostic — the exact
            #: failure mode this whole change exists to remove.
            attempts_made = 0
            for attempt in range(1, _TRANSIENT_MAX_ATTEMPTS + 1):
                remaining = deadline - loop.time()
                if remaining < _MIN_ATTEMPT_S:
                    break
                try:
                    result = await self._call_or_death(
                        session, tool_name, args, remaining
                    )
                except asyncio.TimeoutError:
                    # Not retried: the budget is already spent, and a
                    # repeat would only double the wait.
                    return pb.ToolResult(
                        is_error=True,
                        error_message=(
                            f"serena {tool_name} timed out after "
                            f"{self._call_timeout_s:.0f}s"
                        ),
                    )
                except Exception as e:  # noqa: BLE001 — every failure fail-loud
                    # Transport error on a session pipe means Serena is
                    # gone (broken pipe / connection closed). Flip the
                    # owner's readiness flag off so subsequent Health
                    # answers False — otherwise A3 would be dead code
                    # and clients would keep seeing a stale alive=True.
                    # Never retried: the backend is dead, not busy.
                    self._mark_lost()
                    return pb.ToolResult(
                        is_error=True,
                        error_message=f"serena {tool_name} transport error: {e}",
                    )

                attempts_made = attempt
                tool_result = _to_tool_result(tool_name, result)
                if not tool_result.is_error or not _is_transient_cancel(
                    tool_result.error_message
                ):
                    return tool_result

                last_raw = tool_result.error_message
                if attempt >= _TRANSIENT_MAX_ATTEMPTS:
                    break
                backoff = _TRANSIENT_BACKOFF_S[
                    min(attempt - 1, len(_TRANSIENT_BACKOFF_S) - 1)
                ]
                if deadline - loop.time() - backoff < _MIN_ATTEMPT_S:
                    break  # no room left for another attempt after sleeping
                # Delegated so the sleep does not sit directly in a loop
                # body: this is a bounded backoff between two attempts,
                # not a poll, and the no-polling AST guard rightly
                # refuses to distinguish the two by eye.
                await self._backoff_before_retry(
                    tool_name, last_raw, attempt, backoff
                )

            # Budget or attempts exhausted while the server kept
            # cancelling. The raw protocol text goes to the log; the
            # agent gets something it can act on (§4.3).
            _log.warning(
                "serena %s: giving up after %d transient cancellation(s) "
                "(last: %s)",
                tool_name, attempts_made, last_raw or "<none>",
            )
            return pb.ToolResult(
                is_error=True,
                error_message=_TRANSIENT_EXHAUSTED_MSG.format(
                    tool=tool_name, attempts=attempts_made
                ),
            )

    async def _call_or_death(
        self, session: Any, tool_name: str, args: dict[str, Any], remaining: float
    ) -> Any:
        """One MCP call, but never outliving the backend it was sent to.

        Racing the call against ``_lost_event`` is not belt-and-braces:
        it is the ONLY thing that ends a request which was written to
        the session after ``_receive_loop`` had already exited. That
        loop fails the requests it knows about; one registered a
        moment later is nobody's responsibility, and the child is dead,
        so no response is ever coming. Waiting it out would spend the
        whole call budget and report a timeout — telling the dispatcher
        "still indexing" when the truth is "the process died", which is
        precisely the diagnosis this feature exists to prevent.

        Raises ``asyncio.TimeoutError`` when ``remaining`` runs out, and
        ``_BackendDied`` when the backend was lost first; both are
        classified by the caller.

        NOTHING may outlive this method — not on the normal paths, and
        not when we are cancelled from OUTSIDE (the dispatcher's budget
        ``wait_for``, or a gRPC client dropping the RPC). An orphaned
        ``call`` task would keep using the SHARED MCP session after
        ``_lock`` is released, i.e. concurrently with the next call:
        exactly the interleaving the lock exists to prevent. So both
        children are cancelled AND drained in ``finally``, before the
        outer exception propagates.
        """
        call = asyncio.ensure_future(session.call_tool(tool_name, args))
        death = asyncio.ensure_future(self._lost_event.wait())
        done: set[asyncio.Future] = set()
        try:
            done, _pending = await asyncio.wait(
                (call, death),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if call in done:
                # Asymmetric on purpose: a reply that already arrived
                # wins even if the death landed in the same tick.
                # Discarding it would lose a finished result and force
                # a pointless — for mutating tools, dangerous — retry.
                return call.result()
            if not done:
                raise asyncio.TimeoutError
            raise _BackendDied(tool_name)
        finally:
            await self._settle(call, death)

    @staticmethod
    async def _settle(*tasks: asyncio.Future) -> None:
        """Cancel and DRAIN every child task; never raise.

        Draining (awaiting after cancel) is the point: a cancelled task
        is not finished until the loop has run it once more, and a
        merely-cancelled-but-unawaited task is still free to touch the
        MCP session. Exceptions are swallowed deliberately — this runs
        in a ``finally`` and must not replace the outcome the caller is
        already carrying.
        """
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task

    async def _backoff_before_retry(
        self, tool_name: str, last_raw: str, attempt: int, backoff: float
    ) -> None:
        """Wait out a transient cancellation before the next attempt.

        A single bounded sleep between two attempts — NOT a poll: there
        is no condition being re-checked on a timer, the retry count is
        capped at ``_TRANSIENT_MAX_ATTEMPTS`` and the whole series is
        capped by the call budget.
        """
        _log.warning(
            "serena %s: transient LSP cancellation (%s); retry %d/%d in %.2fs",
            tool_name, last_raw, attempt, _TRANSIENT_MAX_ATTEMPTS - 1, backoff,
        )
        self._note_transient_retry()
        await asyncio.sleep(backoff)

    async def refresh(self, reason: str) -> None:
        """Ask Serena to re-scan the project.

        Called from ``LSPServiceServer._run_refresh`` under the
        server's ``_refresh_lock``; here we additionally take our own
        ``_lock`` so a refresh never races a tool call on the shared
        MCP session. Lock order is always outer → backend, per the
        design; no cycle.

        A7: if no refresh tool was advertised at ``start``, raise so
        the server records the refresh cycle as an error rather than
        silently doing nothing.
        """
        if self._refresh_tool is None:
            raise RuntimeError(
                f"no serena refresh tool advertised "
                f"(tried {list(_REFRESH_TOOL_CANDIDATES)}; got "
                f"{self._advertised_tools})"
            )
        if self._session is None:
            raise RuntimeError("serena backend not running")
        async with self._lock:
            session = self._session
            if session is None:  # pragma: no cover — torn down mid-wait
                raise RuntimeError("serena backend torn down mid-refresh")
            try:
                await session.call_tool(self._refresh_tool, {})
            except Exception:
                # Same reason as call_tool: pipe death → not-ready.
                self._mark_lost()
                raise


def _to_tool_result(tool_name: str, result: Any) -> pb.ToolResult:
    """Convert an ``mcp.types.CallToolResult`` to ``pb.ToolResult``.

    Real shape: ``CallToolResult(content=[TextContent(...)|...],
    isError: bool)`` with ``content[i].text`` the JSON payload for a
    text response. We flatten to a JSON string; if Serena reported
    ``isError=True``, we surface it as fail-loud on our side too, with
    the flattened content as the error message.
    """
    is_error = bool(getattr(result, "isError", False))
    content_json = _flatten_content(result)
    if is_error:
        return pb.ToolResult(
            is_error=True,
            error_message=content_json or f"serena {tool_name} reported isError",
        )
    return pb.ToolResult(content_json=content_json)


def _flatten_content(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


__all__ = [
    "SerenaBackend",
    "build_serena_spawn_argv",
    "pick_refresh_tool",
    "_is_backend_terminated",
    "_DISPATCH_TIMEOUT_MSG",
    "_LS_TERMINATED_MSG",
    "_LS_TERMINATED_MUTATING_MSG",
    "_RPC_TO_TOOL",
]
