"""``AcpClient`` — the long-lived ACP CLI JSON-RPC channel.

One instance per agent process (process-global). All ``agent_thread_id``s
share a single CLI subprocess but each owns a distinct ACP ``session_id``
(separate ``session/new`` calls); ACP-compliant CLIs run multiple
concurrent sessions over one stdio channel.

The client owns exactly ONE reader of ``proc.stdout`` (``_read_loop``). It
dispatches each line to one of three handlers:
  * JSON-RPC reply (``id`` + ``result``/``error``)  -> resolve pending future
  * Agent->client request (``id`` + ``method``)     -> answer (fs/terminal
    declared disabled, so -32601; permission requests go through the
    profile's permission policy)
  * Notification (``method`` only)                  -> route ``session/update``
    to the per-session sink queue; drop vendor notifications.

Non-blocking end-to-end: ``await readline()`` (epoll), ``await drain()``,
``await wait_for(future)``. No polling, no extra threads.

``limit=8*1024*1024`` on the subprocess is mandatory — ACP CLIs can emit
very large tool-result lines (full shell output) and the default 64 KiB
StreamReader limit would raise ``LimitOverrunError`` and wedge the reader.

Lifecycle (§5.3): the subprocess is started with ``start_new_session=True``
so we own the whole process group and can ``killpg`` on teardown; terminate
order is session_cancel → SIGTERM → grace → SIGKILL.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from typing import Any

from . import acp_types
from .profile import SessionNewValidator
from .custom_types import (
    MAX_PENDING_REQUESTS,
    MAX_TOOL_NAME_ENTRIES,
    NOTIF_QUEUE_MAX_ITEMS,
)

logger = logging.getLogger(__name__)

_READ_LIMIT = 8 * 1024 * 1024  # 8 MiB — see module docstring.
_JSONRPC_METHOD_NOT_FOUND = -32601
_DEFAULT_CALL_TIMEOUT = 120.0
# ``session/close`` is a teardown on an abandonment path: the caller has
# already given the session up, so waiting the full call timeout on a wedged
# adapter would stall a live turn to reap a process. Short and bounded — a
# missed close leaks one subprocess, a hung close leaks the whole turn.
_SESSION_CLOSE_TIMEOUT = 15.0
# Notification sink capacity (DESIGN §4.5): bounded, fail-loud backpressure.
_GRACE_AFTER_TERM = 5.0


class AcpClientError(RuntimeError):
    """Raised on a JSON-RPC error reply or a protocol-level failure."""


class AcpConcurrentPromptError(AcpClientError):
    """Raised before writing a second prompt for one active ACP session."""


class AcpSinkOverflowError(AcpClientError):
    """Raised (fail-loud) when a per-turn accumulator overflows its hard cap.

    The remaining raiser is the tool_call accumulator
    (``_append_tool_call_bounded``, cap ``QUEUED_TOOL_CALLS_MAX``): a silently
    dropped tool_call would hang LangGraph's ToolNode waiting for a
    ToolMessage that can never arrive, so the active turn is failed instead.

    The notification sink no longer raises this. It applies backpressure
    (``await sink.put`` in ``_put_notification``), so a burst is absorbed
    rather than fatal; the name is kept because the exception is still the
    turn-level overflow signal the consumer unwinds on.
    """


def raise_for_error_reply(
    method: str, reply: dict[str, Any], *, extra_limit_markers: tuple = ()
) -> None:
    """Raise if ``reply`` is a JSON-RPC error reply; return silently otherwise.

    Single source of truth for turning a CLI error reply into the right
    exception. Two callers rely on this:

      * :meth:`AcpClient._call` — the request/response path (handshake,
        session/new, session/load …).
      * ``chat_model._finalize_response`` — the ``session/prompt`` terminal
        path, which resolves its future with the *raw* reply and bypasses
        ``_call`` entirely. Before this helper existed, an error reply there
        silently produced an empty ``AIMessage`` (no ``result``, no failover).

    Account-level exhaustion (usage cap OR billing/payment) is raised as
    :class:`AcpUsageLimitError` so ``agent_process`` fails over; any other
    error reply becomes :class:`AcpClientError`.

    Classification reads the WHOLE error object, not just the outermost
    ``message``: the real account-terminal text may nest one level down
    (``error.data.message`` / ``error.data.http_status``) under a generic
    top-level ``message``. Matching only the top-level field misclassifies a
    billing cap as a generic error — no failover, empty AIMessages.
    """
    if "error" not in reply:
        return
    from .limit_errors import usage_limit_error_from_text  # noqa: PLC0415

    err_obj = reply["error"]
    err_text = _flatten_error_text(err_obj)
    limit_err = usage_limit_error_from_text(err_text, extra_limit_markers)
    if limit_err is not None:
        raise limit_err
    raise AcpClientError(f"{method} failed: {reply['error']}")


def _flatten_error_text(err_obj: Any) -> str:
    """Collect all human-readable text from a JSON-RPC ``error`` payload so
    limit/billing detection sees every level, not just the outermost message.

    Providers nest the account-terminal detail in ``error.data.message`` /
    ``error.data.http_status`` while the top-level ``message`` is generic.
    We concatenate the outer ``message``, any nested ``data.message``, and —
    because an ``http_status`` of 402 is itself account-terminal regardless
    of wording — an explicit ``"HTTP <status>"`` token. Non-dict payloads
    (bare strings) are returned as-is.
    """
    if not isinstance(err_obj, dict):
        return str(err_obj)
    parts: list[str] = []
    msg = err_obj.get("message")
    if msg is not None:
        parts.append(str(msg))
    data = err_obj.get("data")
    if isinstance(data, dict):
        data_msg = data.get("message")
        if data_msg is not None:
            parts.append(str(data_msg))
        http_status = data.get("http_status")
        if http_status is not None:
            parts.append(f"HTTP {http_status}")
    elif data is not None:
        parts.append(str(data))
    if not parts:
        return str(err_obj)
    return " | ".join(parts)


class AcpClient:
    def __init__(
        self,
        client_state: Any | None = None,
        *,
        cmd: tuple[str, ...],
        vendor_prefix: str = "",
        permission_policy: Any | None = None,
        client_name: str = "deep-worker",
        client_version: str = "0.1.0",
        extra_limit_markers: tuple = (),
        env: dict[str, str] | None = None,
        requires_authenticate: bool = False,
        session_new_validator: SessionNewValidator | None = None,
    ) -> None:
        # client_state is the AcpClientState singleton (or None in unit
        # tests that exercise the channel in isolation). It is only used by
        # the higher-level chat_model; the reader routes by the explicit
        # per-session sink registered via ``register_sink``.
        self._client_state = client_state
        self._cmd = cmd
        self._vendor_prefix = vendor_prefix
        self._permission_policy = permission_policy
        self._client_name = client_name
        self._client_version = client_version
        self._extra_limit_markers = tuple(extra_limit_markers)
        self._env = env
        self._requires_authenticate = requires_authenticate
        self._session_new_validator = session_new_validator
        self._proc: Any | None = None
        self._reader_task: asyncio.Task | None = None
        # "The reader is parked waiting for new bytes and holds no frame it has
        # not finished routing." SET while blocked in ``readline()``, CLEAR for
        # the whole time a frame is in hand (including while parked in a
        # backpressured ``await sink.put()``).
        #
        # This is the end-of-stream signal. ``queue.empty()`` alone is NOT one:
        # the moment the reader became interruptible (it must, or a burst can
        # never drain), an empty queue stopped meaning "the stream is
        # exhausted" — it can equally mean "the reader is holding the next
        # frame". Draining until the queue is empty AND this event is set is
        # what keeps the post-reply tail, with no timeout and no counter from
        # the protocol (ACP's prompt reply carries only ``stopReason``).
        #
        # Starts SET: before the reader exists there is no frame in flight, so a
        # consumer must not be made to wait on a reader that will never run.
        self._reader_idle: asyncio.Event = asyncio.Event()
        self._reader_idle.set()
        self._next_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        # session_id -> notification sink queue for the active turn. Bounded
        # (one sink per session, capacity-capped queue).
        self._sinks: dict[str, asyncio.Queue] = {}
        # ACP permits only one in-flight prompt per session. This executable
        # invariant prevents a later caller from replacing the session sink
        # and corrupting both turns.
        self._active_prompts: dict[str, asyncio.Future] = {}
        # session_id -> one-shot "session is ready" future, resolved by the
        # reader on the FIRST session-scoped notification the CLI emits for
        # that session (many CLIs announce readiness right after session/new
        # completes its async init). ainvoke awaits this before the first
        # ``session/prompt`` so it does not race a not-yet-ready session —
        # which some CLIs silently drop, yielding an empty turn. Event-driven,
        # no polling.
        self._session_ready: dict[str, asyncio.Future] = {}
        # toolCallId -> the CLI's own tool name, harvested from the
        # ``tool_call`` notification the adapter emits BEFORE it asks for
        # permission. claude-agent-acp attaches the name to a permission
        # request's ``toolCall._meta`` only for sub-agent calls, and its
        # ``toolCall.title`` is the tool's *content* (the shell command, the
        # file path) — never the tool's name. The notification's
        # ``_meta.claudeCode.toolName`` is the one shape that always carries
        # it, so the permission handler resolves the name through this map.
        # Bounded: entries are dropped when the call completes and the whole
        # map is cleared on session teardown.
        self._tool_name_by_call_id: dict[str, str] = {}
        self._closed = False
        # Handshake idempotence: two AcpChatModel instances (main + TaskManager)
        # share ONE client/process (review item 6); initialize/authenticate
        # must not be re-run for the second model. Reset on EOF/crash so a
        # re-spawned CLI gets a fresh handshake.
        self._initialized = False
        self._authenticated = False
        # Set from the ``initialize`` reply's advertised
        # ``sessionCapabilities.close``. Until the handshake runs we assume
        # NO support, so a close is never sent to an unknown CLI.
        self._session_close_supported = False
        # Single-flight guard for the startup section (spawn + initialize +
        # optional authenticate). Concurrent first callers (main agent model
        # and TaskManager model entering startup together, hard constraint
        # #5) must produce ONE subprocess and ONE handshake. Created lazily
        # on the first ensure_started() so the lock binds to the caller's
        # running loop. There is no cached result Future: on failure the
        # lock is simply released and the flags stay cleared, so the next
        # call genuinely retries instead of inheriting a poisoned future.
        self._start_lock: asyncio.Lock | None = None

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_READ_LIMIT,
            start_new_session=True,  # own process group → killpg teardown
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def ensure_started(self) -> None:
        """Idempotent start + handshake, driven by the CLIENT's liveness.

        SINGLE-FLIGHT (review round 3, hard constraint #5): concurrent first
        callers (main agent model + TaskManager model entering startup
        together) must produce ONE spawn and ONE handshake. A bare
        check-then-act on ``_proc``/``_initialized`` races — both callers
        observe ``_proc is None`` and both await ``start()``, which only
        re-checks ``_proc`` before the (awaiting) ``create_subprocess_exec``.
        The lock serializes the whole spawn + initialize + optional
        authenticate section; the second caller re-checks under the lock and
        finds the state already up. There is no cached result Future: a
        FAILED startup releases the lock with the flags still cleared, so the
        next call genuinely retries instead of inheriting a poisoned future.
        ``_on_eof`` resets the flags in place, so this same path transparently
        re-spawns a recovered client on the next invocation of the SAME
        AcpChatModel — no stale per-model boolean can wedge a recovery.
        """
        if (
            self._proc is not None
            and self._initialized
            and (not self._requires_authenticate or self._authenticated)
        ):
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            # Re-derive state under the lock: the first caller may have
            # completed the whole startup while we were waiting.
            if self._proc is None:
                await self.start()
            if not self._initialized:
                await self.initialize()
            if self._requires_authenticate and not self._authenticated:
                await self.authenticate()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        # Fail any in-flight request waiters.
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()
        # Fail in-flight PROMPT futures before dropping them. Cancelling the
        # reader above means ``_read_loop`` re-raises CancelledError and
        # ``_on_eof`` is NEVER reached, so this is the only place a live turn
        # learns the channel is gone. Clearing the map without failing them
        # left the turn's consumer waiting on a future nobody would ever
        # complete — and ``_sinks.clear()`` below puts nothing in the queue
        # either, so even a two-source ``wait({queue.get(), response_fut})``
        # hung. EOF and reader-crash both fail their waiters loudly; close must
        # not be the one terminal path that goes quiet.
        closed_err = AcpClientError("ACP client closed while a turn was in flight")
        for fut in list(self._active_prompts.values()):
            if not fut.done():
                fut.set_exception(closed_err)
        self._active_prompts.clear()
        self._sinks.clear()
        self._tool_name_by_call_id.clear()
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            await self._terminate_process_group(proc)

    async def _terminate_process_group(self, proc: Any) -> None:
        """Orderly teardown: SIGTERM to the whole process group, grace, then
        SIGKILL (§5.3). Never leaves an orphan process group."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_GRACE_AFTER_TERM)
            return
        except asyncio.TimeoutError:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5)

    # -- sink registration --------------------------------------------------
    def register_sink(self, session_id: str, queue: asyncio.Queue) -> None:
        """Route subsequent ``session/update`` notifications for
        ``session_id`` into ``queue`` (the active turn's notification sink)."""
        self._sinks[session_id] = queue

    def unregister_sink(self, session_id: str) -> None:
        self._sinks.pop(session_id, None)

    @property
    def _idle_event(self) -> asyncio.Event:
        """The reader-idle event, created on demand.

        Some call sites build an ``AcpClient`` with ``object.__new__`` to
        exercise one method in isolation, so ``__init__`` has not run and the
        attribute may be missing. The reader must not die on that — it is the
        one task whose death takes every waiter with it.
        """
        event = getattr(self, "_reader_idle", None)
        if event is None:
            event = asyncio.Event()
            event.set()
            self._reader_idle = event
        return event

    def reader_is_idle(self) -> bool:
        """True iff the reader is parked on the pipe holding no frame.

        A consumer may conclude "the stream is exhausted" only when this is True
        AND its sink is empty. See ``_reader_idle``.
        """
        return self._idle_event.is_set()

    async def wait_reader_idle(self) -> None:
        """Wait until the reader holds no unrouted frame. Event-driven."""
        await self._idle_event.wait()

    # -- JSON-RPC primitives ------------------------------------------------
    def _write(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def _call(
        self, method: str, params: dict | None = None, *, timeout: float | None = None
    ) -> dict:
        """Send a JSON-RPC request and await the full reply dict.

        Returns the whole reply object (``{"id", "result"}`` or
        ``{"id", "error"}``). Raises :class:`AcpClientError` on an error
        reply.
        """
        self._next_id += 1
        rid = self._next_id
        if len(self._pending_requests) >= MAX_PENDING_REQUESTS:
            raise AcpClientError(
                f"too many in-flight ACP requests ({MAX_PENDING_REQUESTS}) — "
                "rejecting new request (reject-new, DESIGN §4.5)"
            )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_requests[rid] = fut
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self._write(req)
        if self._proc is not None and self._proc.stdin is not None:
            await self._proc.stdin.drain()
        try:
            reply = await asyncio.wait_for(
                fut, timeout=timeout or _DEFAULT_CALL_TIMEOUT
            )
        finally:
            self._pending_requests.pop(rid, None)
        raise_for_error_reply(method, reply, extra_limit_markers=self._extra_limit_markers)
        return reply

    # -- high-level handshake ----------------------------------------------
    async def initialize(self) -> dict:
        if self._initialized:
            return {}
        reply = await self._call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
            },
        )
        self._initialized = True
        # Whether this CLI implements ``session/close``. Read from the LIVE
        # handshake, never hardcoded per provider: an adapter that drops the
        # capability must degrade to "no close" rather than have us keep
        # sending a method it will reject. Absent/garbage → False.
        caps = reply.get("result", {}).get("agentCapabilities", {})
        session_caps = caps.get("sessionCapabilities")
        self._session_close_supported = (
            isinstance(session_caps, dict) and "close" in session_caps
        )
        return reply

    async def authenticate(self) -> dict:
        if self._authenticated:
            return {}
        # Most ACP CLIs use a cached local credential (subscription /
        # logged-in CLI). The method id is provider-specific; the profile may
        # override the whole call via ``authenticate_override``.
        reply = await self._call("authenticate", {"methodId": "cached_token"})
        self._authenticated = True
        return reply

    async def session_new(self, *, cwd: str, mcp_servers: list[dict]) -> str:
        reply = await self._call(
            "session/new", {"cwd": cwd, "mcpServers": mcp_servers}
        )
        # Provider-specific post-session/new gate (e.g. claude-code
        # validates the pinned model against the live picker returned in
        # ``configOptions``). No hardcoded whitelists live in the client:
        # the profile hands us an opaque callable and we let it raise.
        validator = self._session_new_validator
        if validator is not None:
            validator(reply, self._env)
        session_id = reply["result"]["sessionId"]
        logger.info(
            "acp session/new -> %s cwd=%s mcp=%s",
            session_id,
            cwd,
            bool(mcp_servers),
        )
        print(
            f"acp session/new -> {session_id} cwd={cwd} mcp={bool(mcp_servers)}",
            file=sys.stderr,
            flush=True,
        )
        # Arm the readiness future BEFORE returning so the reader can resolve
        # it the instant the CLI emits the session's first notification.
        loop = asyncio.get_running_loop()
        if session_id not in self._session_ready:
            self._session_ready[session_id] = loop.create_future()
        return session_id

    async def wait_session_ready(
        self, session_id: str, *, timeout: float = 10.0
    ) -> None:
        """Block until the CLI has emitted its first notification for
        ``session_id`` (its session-ready signal) or ``timeout`` elapses.

        A freshly created session is NOT immediately able to accept a
        ``session/prompt`` — some CLIs finish async session init and announce
        it via a notification. Prompting before that is silently dropped
        (empty turn). This is a one-shot, event-driven wait (resolved by the
        reader); on timeout we proceed anyway so a missing signal degrades to
        the old behaviour rather than hanging."""
        fut = self._session_ready.get(session_id)
        if fut is None:
            return
        if fut.done():
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)

    async def session_load(
        self, session_id: str, *, cwd: str, mcp_servers: list[dict] | None = None
    ) -> dict:
        """Re-attach a session that was created in a PRIOR subprocess.

        Requires the profile to support session/load (``session_load: bool``).
        After a successful load the session is live in THIS subprocess and
        ``session/prompt`` against it works again (memory preserved). Arms
        the same readiness future ``session/new`` does so callers can wait
        for the reloaded session to be promptable.

        ROUND 8 DEFECT #5: the post-session gate (model-pin validation) used
        to run on ``session/new`` ONLY. A resume after a restart therefore
        skipped it entirely, so a pin the adapter had since DROPPED silently
        fell back to the CLI's default — the agent kept running, on a
        different model than the operator pinned, with nothing logged. The
        picker rotates between processes, which is exactly when a resume
        happens, so this is the likeliest moment for the pin to be stale.
        Same validator, same fail-loud contract, both entry points."""
        logger.info(
            "acp session/load session=%s cwd=%s mcp=%s",
            session_id,
            cwd,
            bool(mcp_servers),
        )
        print(
            f"acp session/load session={session_id} cwd={cwd} mcp={bool(mcp_servers)}",
            file=sys.stderr,
            flush=True,
        )
        reply = await self._call(
            "session/load",
            {"sessionId": session_id, "cwd": cwd, "mcpServers": mcp_servers or []},
        )
        validator = self._session_new_validator
        if validator is not None:
            validator(reply, self._env)
        loop = asyncio.get_running_loop()
        if session_id not in self._session_ready:
            self._session_ready[session_id] = loop.create_future()
        return reply

    async def session_close(self, session_id: str | None) -> bool:
        """Ask the CLI to tear down ``session_id``, then drop local state.

        This is the ONLY message that reaps the per-session SDK subprocess.
        ``session/cancel`` interrupts the in-flight turn but deliberately
        leaves the process alive (the adapter's ``closeQueryStream`` doc is
        explicit: only ``query.close()`` terminates the subprocess, and only
        ``session/close``/``session/delete`` reach it). Abandoning a session
        id without this call therefore leaks the whole subprocess — plus the
        stdio MCP servers it spawned — for the lifetime of the adapter.
        That leak is what stacked ~4.3 GB of orphaned ``claude`` processes
        under a single bridge over one day.

        Best-effort and non-fatal by contract: every caller is on an
        abandonment path where the session is already logically gone, so a
        CLI that refuses or never answers must not break the turn. Returns
        True iff the CLI acknowledged the close. Local bookkeeping is
        released either way — a session we cannot close is still one we must
        stop routing updates to.
        """
        if session_id is None:
            return False
        if not self._session_close_supported:
            # Adapter never advertised the capability — releasing local
            # state is all we can honestly do.
            self.release_session(session_id)
            return False
        acked = False
        try:
            # Bounded: teardown cancels an in-flight prompt CLI-side, and a
            # wedged adapter must not stall the caller's turn forever.
            await asyncio.wait_for(
                self._call("session/close", {"sessionId": session_id}),
                timeout=_SESSION_CLOSE_TIMEOUT,
            )
            acked = True
            logger.info("acp session/close -> %s", session_id)
        except Exception as exc:  # noqa: BLE001 — abandonment path, never fatal
            # Loud enough to diagnose a recurring leak, quiet enough not to
            # scare on a benign "Session not found" after a CLI restart.
            logger.warning(
                "acp session/close failed for %s (%s: %s); the CLI subprocess "
                "may leak until the adapter exits",
                session_id, type(exc).__name__, exc,
            )
        finally:
            self.release_session(session_id)
        return acked

    def release_session(self, session_id: str | None) -> None:
        """Drop local ACP bookkeeping for ``session_id`` (best-effort)."""
        if session_id is None:
            return
        active = self._active_prompts.pop(session_id, None)
        if active is not None and not active.done():
            active.cancel()
        self._sinks.pop(session_id, None)
        self._session_ready.pop(session_id, None)

    async def session_cancel(self, session_id: str | None) -> None:
        """Abandon the outstanding ``session/prompt`` for ``session_id``.

        Writes a ``session/cancel`` message to the CLI's stdin AND releases
        the local sticky guard state for ``session_id`` (``_active_prompts`` /
        ``_sinks``). Both steps are required — the guard is only auto-released
        via the pending future's ``done_callback``, which fires only when the
        CLI writes back a JSON-RPC reply for the outstanding prompt. If the
        CLI never replies (crash, wedge, session/load discarding the request,
        mid-turn ``CancelledError``), the entry would stay forever and every
        later ``session_prompt`` on the same session id would raise
        ``AcpConcurrentPromptError`` synchronously, permanently poisoning the
        agent thread.
        """
        logger.info("acp session/cancel session=%s", session_id)
        print(f"acp session/cancel session={session_id}", file=sys.stderr, flush=True)
        if session_id is None:
            return
        with contextlib.suppress(Exception):
            self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": session_id},
                }
            )
            if self._proc is not None and self._proc.stdin is not None:
                await self._proc.stdin.drain()
        active = self._active_prompts.get(session_id)
        if active is not None and not active.done():
            active.cancel()
        self._sinks.pop(session_id, None)

    def session_prompt(
        self,
        *,
        session_id: str,
        prompt: list[dict],
        notification_sink: asyncio.Queue,
    ) -> asyncio.Future:
        """Fire a ``session/prompt`` and return the response Future.

        This is intentionally NOT a coroutine: the caller (``ainvoke``)
        stores the returned future on the per-thread state and drives the
        turn by racing it against the notification sink. The sink is
        registered here so the reader routes ``session/update`` events for
        ``session_id`` into it for the duration of the turn.
        """
        active = self._active_prompts.get(session_id)
        if active is not None and not active.done():
            raise AcpConcurrentPromptError(
                f"session {session_id!r} already has an active prompt"
            )

        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_requests[rid] = fut
        self._active_prompts[session_id] = fut
        self.register_sink(session_id, notification_sink)

        def release_prompt_owner(done: asyncio.Future) -> None:
            if self._active_prompts.get(session_id) is done:
                self._active_prompts.pop(session_id, None)
            if self._sinks.get(session_id) is notification_sink:
                self._sinks.pop(session_id, None)
            if self._pending_requests.get(rid) is done:
                self._pending_requests.pop(rid, None)

        fut.add_done_callback(release_prompt_owner)
        try:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "method": "session/prompt",
                    "params": {"sessionId": session_id, "prompt": prompt},
                }
            )
        except BaseException:
            self._pending_requests.pop(rid, None)
            if not fut.done():
                fut.cancel()
            raise
        # Best-effort drain without blocking the caller's control flow.
        if self._proc is not None and self._proc.stdin is not None:
            loop.create_task(self._proc.stdin.drain())
        return fut

    # -- reader -------------------------------------------------------------
    async def _read_loop(self) -> None:
        """Own the single reader of the CLI's stdout.

        ROUND 8 DEFECT: this loop used to run with NO exception guard, so any
        error killed the task silently and left the client looking healthy —
        ``_proc`` still set, so ``ensure_started`` short-circuits, every
        subsequent prompt is written to a CLI whose replies nobody reads, and
        (with no liveness bound on ``session/prompt``) the agent hangs
        forever with nothing logged until GC.

        Two real paths reach it:

        * ``readline()`` raises ``ValueError`` (from ``LimitOverrunError``)
          when one line exceeds the stream limit — and this module's own
          docstring notes the CLI emits very large lines, so an 8 MiB line is
          a question of when, not if.
        * ``_dispatch`` raises on a malformed frame.

        Any escape is now treated as a crash: same terminal transition as
        EOF, so every waiter fails loudly and the next call re-spawns.
        A dispatch error for ONE frame must not kill the channel, so it is
        contained per-frame and the loop continues.
        """
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            while True:
                # IDLE: parked on the pipe with nothing in hand. A consumer that
                # has drained the queue may now conclude the stream is
                # exhausted — see ``_reader_idle``.
                self._idle_event.set()
                line = await stdout.readline()
                self._idle_event.clear()
                if not line:  # EOF — subprocess exited.
                    self._on_eof()
                    return
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "acp_provider: non-JSON line dropped: %r", line[:200]
                    )
                    continue
                try:
                    # Stays CLEAR across this await: a backpressured
                    # ``session/update`` parks here holding a frame, which is
                    # exactly when a consumer must NOT finalize.
                    await self._dispatch_async(obj)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One bad frame must not take down the whole channel.
                    logger.exception(
                        "acp_provider: dispatch failed for a frame — "
                        "dropping it and continuing to read"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # readline() over-limit, or anything else unforeseen. Fail every
            # waiter exactly as EOF does — a dead reader with live waiters is
            # the silent-hang shape we must never leave behind.
            logger.exception(
                "acp_provider: reader task died (%s) — treating as CLI crash",
                type(exc).__name__,
            )
            self._on_eof(reason=f"ACP reader task died: {type(exc).__name__}: {exc}")
            return
        finally:
            # The reader is gone (EOF, crash, or cancellation by ``aclose``).
            # It holds no frame and never will again, so release anyone waiting
            # on "the reader is idle". Liveness on these paths comes from the
            # two-source wait failing loudly on the response future; leaving
            # this CLEAR would add a second, silent way to hang.
            self._idle_event.set()

    async def _dispatch_async(self, obj: dict) -> None:
        """Route one frame, applying backpressure to ``session/update`` ONLY.

        A full sink makes the reader WAIT for space instead of failing the turn.
        The pressure travels where it belongs — into the pipe buffer and then
        into the CLI's own ``write`` — so nothing is dropped and nothing needs a
        timeout: ``Queue.put`` wakes on the consumer's ``get``.

        Control frames (JSON-RPC replies, agent->client requests) must NOT go
        through the blocking path. Measured: with the turn-ending reply behind a
        backpressured notification, 1500/50 completed but 5000/200 and
        20000/200 DEADLOCKED — the reader waits for space while the consumer
        waits for the turn to end. Keeping replies synchronous breaks that cycle.
        """
        method = obj.get("method")
        has_id = "id" in obj and obj["id"] is not None
        if method == "session/update" and not has_id:
            await self._put_notification(obj)
            return
        self._dispatch(obj)

    async def _put_notification(self, obj: dict) -> None:
        """Backpressured delivery of one ``session/update`` into its sink."""
        params = obj.get("params") or {}
        session_id = params.get("sessionId")
        if session_id:
            ready = self._session_ready.get(session_id)
            if ready is not None and not ready.done():
                ready.set_result(None)
        # Harvest toolCallId -> toolName BEFORE the sink check: see
        # ``_handle_notification`` for why this must happen even with no sink.
        self._remember_tool_name(params)
        sink = self._sinks.get(session_id) if session_id else None
        if sink is None:
            # No active turn for this session — drop (a late chunk after the
            # turn resolved, or a replay during session/load).
            return
        if sink.full():
            logger.debug(
                "acp_provider: sink for session %r at capacity (%d) — reader "
                "waiting for space (backpressure)",
                session_id,
                NOTIF_QUEUE_MAX_ITEMS,
            )
        await sink.put(obj)

    def _dispatch(self, obj: dict) -> None:
        has_id = "id" in obj and obj["id"] is not None
        is_reply = has_id and ("result" in obj or "error" in obj)
        if is_reply:
            # POP, not get: the ``_call`` path also pops in its finally, but
            # ``session_prompt`` registers a pending request and bypasses
            # ``_call`` — so without popping here every turn would leak one
            # stale entry (O(turns) growth).
            fut = self._pending_requests.pop(obj["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(obj)
            return
        if has_id and "method" in obj:
            # Agent -> client request.
            self._handle_agent_request(obj)
            return
        if "method" in obj:
            self._handle_notification(obj)
            return
        logger.warning("acp_provider: unroutable line: %r", obj)

    def _handle_agent_request(self, obj: dict) -> None:
        method = obj.get("method", "")
        # We declared fs:false / terminal:false. Any such request is answered
        # with method-not-found; the CLI must gracefully fail the built-in.
        if method.startswith(("fs/", "terminal/")):
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": obj["id"],
                    "error": {
                        "code": _JSONRPC_METHOD_NOT_FOUND,
                        "message": f"{method} not supported (capability disabled)",
                    },
                }
            )
            return
        if method == "session/request_permission":
            self._handle_permission_request(obj)
            return
        logger.warning("acp_provider: unhandled agent request %r", method)
        self._write(
            {
                "jsonrpc": "2.0",
                "id": obj["id"],
                "error": {"code": _JSONRPC_METHOD_NOT_FOUND,
                          "message": f"{method} not supported"},
            }
        )

    def _handle_permission_request(self, obj: dict) -> None:
        """Fail-closed by default, BLANKET approval under ``-sp``
        (DESIGN §4.6).

        Without the operator flag NOTHING is approved. With it, everything
        the CLI asks about is approved — there is no per-tool list to fall
        out of date (see :class:`PermissionPolicy` and the 2026-08-24
        incident it documents).

        A DENIAL IS NEVER SILENT. The branch that used to return here
        without a word is what made a four-day production outage invisible:
        grok terminates a turn whose permission was rejected, so the
        operator saw only ``AcpTurnCancelledError`` with no cause anywhere
        in the logs.

        The selected option is always one the CLI actually offered (option
        ids are opaque and vary by version) — inventing an id makes the CLI
        reject the response and terminate the turn."""
        policy = self._permission_policy
        params = obj.get("params") or {}
        options = params.get("options") or []
        offered = [option for option in options if isinstance(option, dict)]
        deny_kind_prefix = (
            policy.deny_kind_prefix if policy is not None else "deny"
        )
        deny_option = next(
            (
                option
                for option in offered
                if str(option.get("kind", "")).startswith(deny_kind_prefix)
                and option.get("optionId")
            ),
            None,
        )
        kind = self._permission_request_kind_for(params)
        if policy is None or not policy.is_allowed_kind(kind):
            logger.info(
                "acp_provider: denying permission for %r — the agent was "
                "started WITHOUT -sp (the CLI will end this turn; pass -sp "
                "to approve tool permissions)",
                kind,
            )
            self._reply_permission(obj, deny_option)
            return
        allow_kind_prefix = policy.allow_kind_prefix
        allow_option = next(
            (
                option
                for option in offered
                if str(option.get("kind", "")).startswith(allow_kind_prefix)
                and option.get("optionId")
            ),
            None,
        )
        if allow_option is None:
            logger.error(
                "acp_provider: permission request %r has no selectable allow option",
                obj.get("id"),
            )
            self._reply_permission(obj, deny_option)
            return
        outcome = {"outcome": "selected", "optionId": allow_option["optionId"]}
        self._write(
            {
                "jsonrpc": "2.0",
                "id": obj["id"],
                "result": {"outcome": outcome},
            }
        )

    def _remember_tool_name(self, params: dict) -> None:
        """Record ``toolCallId -> toolName`` from a ``tool_call`` /
        ``tool_call_update`` notification, and forget it once the call
        reaches a terminal status so the map cannot grow without bound.

        claude-agent-acp builds every such notification through
        ``claudeCodeMetaFromToolUse``, which ALWAYS sets
        ``_meta.claudeCode.toolName``. Verified against the installed
        adapter 0.68.0.
        """
        update = params.get("update")
        if not isinstance(update, dict):
            return
        kind = update.get("sessionUpdate")
        if kind not in ("tool_call", "tool_call_update"):
            return
        call_id = update.get("toolCallId")
        if not isinstance(call_id, str) or not call_id:
            return
        status = update.get("status")
        if status in ("completed", "failed"):
            self._tool_name_by_call_id.pop(call_id, None)
            return
        meta = update.get("_meta")
        if not isinstance(meta, dict):
            return
        claude_meta = meta.get("claudeCode")
        if not isinstance(claude_meta, dict):
            return
        name = claude_meta.get("toolName")
        if isinstance(name, str) and name:
            self._tool_name_by_call_id[call_id] = name
            # Backstop for a CLI that never reports the call terminal (see
            # MAX_TOOL_NAME_ENTRIES). dicts preserve insertion order, so
            # popping the first key drops the OLDEST id — never the one a
            # just-arrived permission ask is about.
            while len(self._tool_name_by_call_id) > MAX_TOOL_NAME_ENTRIES:
                self._tool_name_by_call_id.pop(
                    next(iter(self._tool_name_by_call_id))
                )

    def _permission_request_kind_for(self, params: dict) -> str:
        """Tool name for a permission ask, preferring the notification-derived
        map over the request's own (lossy) shapes.

        ``_permission_request_kind`` stays a pure static fallback chain; this
        instance method adds the ``toolCallId`` hop that makes a TOP-LEVEL
        ask resolvable at all.
        """
        tool_call = params.get("toolCall")
        if isinstance(tool_call, dict):
            call_id = tool_call.get("toolCallId")
            if isinstance(call_id, str):
                remembered = self._tool_name_by_call_id.get(call_id)
                if remembered:
                    return remembered
        return self._permission_request_kind(params)

    @staticmethod
    def _permission_request_kind(params: dict) -> str:
        """Best-effort name of the tool the permission request is about.

        ACP spec has ``params.kind``, but claude-agent-acp does not fill it
        for tool permission asks — the tool name lives in the embedded
        ``params.toolCall._meta.claudeCode.toolName``, and ONLY for sub-agent
        calls (the adapter spreads ``_meta`` in behind
        ``parentToolUseId ? … : {}``). Falls back to empty string, which the
        fail-closed policy denies.

        ``toolCall.title`` is deliberately NOT consulted: it is the tool's
        CONTENT, not its name — ``toolInfoFromToolUse`` sets it to the shell
        command for ``Bash`` and to the file path for ``Read``. Treating it
        as a name made every top-level ask miss the allowlist, so ``-sp``
        approved nothing (live incident 2026-08-16). The recoverable path for
        a top-level ask is ``_permission_request_kind_for``, which resolves
        the name through the preceding ``tool_call`` notification.
        """
        kind = params.get("kind")
        if isinstance(kind, str) and kind:
            return kind
        tool_call = params.get("toolCall")
        if isinstance(tool_call, dict):
            meta = tool_call.get("_meta")
            if isinstance(meta, dict):
                cc = meta.get("claudeCode")
                if isinstance(cc, dict):
                    name = cc.get("toolName")
                    if isinstance(name, str) and name:
                        return name
        return ""

    def _reply_permission(self, obj: dict, deny_option: dict | None) -> None:
        if deny_option is not None:
            outcome = {"outcome": "selected", "optionId": deny_option["optionId"]}
        else:
            outcome = {"outcome": "cancelled"}
        self._write(
            {
                "jsonrpc": "2.0",
                "id": obj["id"],
                "result": {"outcome": outcome},
            }
        )

    def _handle_notification(self, obj: dict) -> None:
        method = obj.get("method")
        if acp_types.is_ignored_vendor_notification(method, self._vendor_prefix):
            return
        if method != "session/update":
            # Other standard notifications we don't act on.
            return
        params = obj.get("params") or {}
        session_id = params.get("sessionId")
        # Resolve the one-shot readiness future for this session on the FIRST
        # notification we see for it — regardless of whether a turn sink is
        # registered yet.
        if session_id:
            ready = self._session_ready.get(session_id)
            if ready is not None and not ready.done():
                ready.set_result(None)
        # Harvest toolCallId -> toolName BEFORE the sink check: the adapter
        # emits the ``tool_call`` notification and the permission request
        # back-to-back, and a permission ask can arrive while no sink is
        # registered (the notification would then be dropped and the name
        # lost). Cheap and bounded — see ``_tool_name_by_call_id``.
        self._remember_tool_name(params)
        sink = self._sinks.get(session_id) if session_id else None
        if sink is None:
            # No active turn for this session — drop (e.g. a late chunk after
            # the turn already resolved, or a replay during session/load).
            return
        sink.put_nowait(obj)

    def _on_eof(self, reason: str | None = None) -> None:
        """Deterministic recoverable-crash transition (DESIGN §5.3).

        The CLI subprocess exited (EOF on stdout). EVERY waiter is failed so
        nothing hangs: pending JSON-RPC request futures, in-flight prompt
        futures, and per-session readiness futures all get the same terminal
        error. All session bookkeeping (sinks, ready maps, active prompts) is
        cleared, and the dead process/reader references are dropped so a
        subsequent ``start()`` re-spawns a fresh CLI. The next ainvoke either
        restarts cleanly or fails visibly — never silently hangs.

        The reader task that called us exits right after, so there is no
        orphan task. ``proc.wait()`` is scheduled best-effort to reap the
        zombie.

        ``reason`` overrides the error text: the reader also calls this when
        it dies from an exception rather than EOF (round 8), and the operator
        needs to see WHICH of the two happened.
        """
        err = AcpClientError(reason or "ACP CLI exited (EOF on stdout)")
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.set_exception(err)
        self._pending_requests.clear()
        for fut in list(self._active_prompts.values()):
            if not fut.done():
                fut.set_exception(err)
        self._active_prompts.clear()
        for fut in list(self._session_ready.values()):
            if not fut.done():
                fut.set_exception(err)
        self._session_ready.clear()
        self._sinks.clear()
        # Fail per-thread waiters (builtin passthrough Futures the ToolNode is
        # blocked on) and reset session liveness across all threads sharing
        # this client. Best-effort: client_state is None in unit tests that
        # exercise the channel in isolation.
        if self._client_state is not None:
            self._client_state.on_cli_crash(err)
        # The old CLI is gone: a re-spawned process needs a fresh handshake.
        self._initialized = False
        self._authenticated = False
        # Re-derived from the next handshake; the replacement CLI may be a
        # different adapter version with different capabilities.
        self._session_close_supported = False
        # Reset process/reader so the next start() re-spawns. Reap the old
        # process (fire-and-forget — it already exited; this just reaps).
        proc, self._proc = self._proc, None
        self._reader_task = None
        if proc is not None:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().create_task(proc.wait())


# ---------------------------------------------------------------------------
# Process-global shared clients (review item 6).
# ---------------------------------------------------------------------------
# The main agent model AND the TaskManager model each build an AcpChatModel.
# Without sharing, each would spawn its OWN CLI subprocess — two adapter
# processes for one agent. We key one AcpClient (one CLI process, one reader)
# per profile identity and hand it to every AcpChatModel; ACP sessions are
# still per-thread (distinct ``session/new`` ids over the same stdio channel).
_GLOBAL_ACP_CLIENTS: dict[tuple[Any, ...], "AcpClient"] = {}


def get_global_acp_client(profile: Any, client_state: Any) -> "AcpClient":
    """Return the process-wide shared :class:`AcpClient` for ``profile``.

    Keyed by the profile's CLI identity (provider id + command) so two
    ``AcpChatModel`` instances in one agent process resolve to the SAME
    client/subprocess. Safe to call repeatedly — the client is created once
    and cached for the life of the process (until EOF resets it in place).
    """
    # The profile env is part of the CLI IDENTITY: a client pinned to one
    # model must never be handed to a model expecting another, so two
    # profiles differing only in env get two subprocesses.
    profile_env = tuple(getattr(profile, "env", ()) or ())
    key = (profile.provider, tuple(profile.cmd), profile_env)
    client = _GLOBAL_ACP_CLIENTS.get(key)
    if client is None:
        env: dict[str, str] | None = None
        if profile_env:
            # MERGE over the inherited environment — the CLI needs PATH/HOME
            # and the user's credential env; a bare dict would strip them.
            env = dict(os.environ)
            env.update({str(k): str(v) for k, v in profile_env})
        client = AcpClient(
            client_state=client_state,
            cmd=profile.cmd,
            vendor_prefix=profile.vendor_prefix,
            permission_policy=profile.permission_policy,
            extra_limit_markers=profile.extra_limit_markers,
            requires_authenticate=profile.requires_authenticate,
            env=env,
            session_new_validator=getattr(profile, "session_new_validator", None),
        )
        _GLOBAL_ACP_CLIENTS[key] = client
    return client


def reset_global_acp_clients() -> None:
    """Test hook: drop all shared clients. Callers must ``aclose()`` first."""
    _GLOBAL_ACP_CLIENTS.clear()
