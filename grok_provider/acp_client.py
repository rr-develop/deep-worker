"""``GrokAcpClient`` — the long-lived ``grok agent stdio`` JSON-RPC channel.

One instance per agent process (process-global, like the codex proxy MCP).
All ``agent_thread_id``s share a single ``grok agent stdio`` subprocess but
each owns a distinct ``grok_session_id`` (separate ``session/new`` calls);
grok happily runs multiple concurrent sessions over one stdio channel.

The client owns exactly ONE reader of ``proc.stdout`` (``_read_loop``). It
dispatches each line to one of three handlers:
  * JSON-RPC reply (``id`` + ``result``/``error``)  -> resolve pending future
  * Agent->client request (``id`` + ``method``)     -> answer (fs/terminal
    declared disabled, so -32601)
  * Notification (``method`` only)                  -> route ``session/update``
    to the per-session sink queue; drop vendor ``_x.ai/*`` noise.

Non-blocking end-to-end: ``await readline()`` (epoll), ``await drain()``,
``await wait_for(future)``. No polling, no extra threads. See DESIGN §9.

``limit=8*1024*1024`` on the subprocess is mandatory — grok can emit very
large tool-result lines (full shell output) and the default 64 KiB
StreamReader limit would raise ``LimitOverrunError`` and wedge the reader.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from grok_provider import acp_types

logger = logging.getLogger(__name__)

_READ_LIMIT = 8 * 1024 * 1024  # 8 MiB — see module docstring.
_JSONRPC_METHOD_NOT_FOUND = -32601
_DEFAULT_CALL_TIMEOUT = 120.0


class GrokAcpClientError(RuntimeError):
    """Raised on a JSON-RPC error reply or a protocol-level failure."""


class GrokConcurrentPromptError(GrokAcpClientError):
    """Raised before writing a second prompt for one active ACP session."""


class GrokAcpClient:
    def __init__(
        self,
        client_state: Any | None = None,
        *,
        cmd: tuple[str, ...] = ("grok", "agent", "stdio"),
    ) -> None:
        # client_state is the GrokClientState singleton (or None in unit
        # tests that exercise the channel in isolation). It is only used by
        # the higher-level chat_model; the reader routes by the explicit
        # per-session sink registered via ``register_sink``.
        self._client_state = client_state
        self._cmd = cmd
        self._proc: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._next_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        # grok_session_id -> notification sink queue for the active turn.
        self._sinks: dict[str, asyncio.Queue] = {}
        # ACP permits only one in-flight prompt per session. This executable
        # invariant prevents a later caller from replacing the session sink and
        # corrupting both turns.
        self._active_prompts: dict[str, asyncio.Future] = {}
        # grok_session_id -> one-shot "session is ready" future, resolved by
        # the reader on the FIRST session-scoped notification grok emits for
        # that session (it sends ``available_commands_update`` right after
        # ``session/new`` completes its async init). ainvoke awaits this
        # before the first ``session/prompt`` so it does not race a
        # not-yet-ready session — which grok silently drops, yielding an
        # empty turn (live integration 2026-06-18). Event-driven, no polling.
        self._session_ready: dict[str, asyncio.Future] = {}
        self._closed = False

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
        )
        self._reader_task = asyncio.create_task(self._read_loop())

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
        self._active_prompts.clear()
        self._sinks.clear()
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)

    # -- sink registration --------------------------------------------------
    def register_sink(self, session_id: str, queue: asyncio.Queue) -> None:
        """Route subsequent ``session/update`` notifications for
        ``session_id`` into ``queue`` (the active turn's notification sink)."""
        self._sinks[session_id] = queue

    def unregister_sink(self, session_id: str) -> None:
        self._sinks.pop(session_id, None)

    # -- JSON-RPC primitives ------------------------------------------------
    def _write(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def _call(
        self, method: str, params: dict | None = None, *, timeout: float | None = None
    ) -> dict:
        """Send a JSON-RPC request and await the full reply dict.

        Returns the whole reply object (``{"id", "result"}`` or
        ``{"id", "error"}``). Raises :class:`GrokAcpClientError` on an error
        reply.
        """
        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_event_loop()
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
        if "error" in reply:
            from grok_provider.limit_errors import (  # noqa: PLC0415
                grok_usage_limit_error_from_text,
            )

            err_obj = reply["error"]
            err_text = (
                str(err_obj.get("message", err_obj))
                if isinstance(err_obj, dict)
                else str(err_obj)
            )
            limit_err = grok_usage_limit_error_from_text(err_text)
            if limit_err is not None:
                raise limit_err
            raise GrokAcpClientError(f"{method} failed: {reply['error']}")
        return reply

    # -- high-level handshake ----------------------------------------------
    async def initialize(self) -> dict:
        return await self._call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "deep-worker", "version": "0.1.0"},
            },
        )

    async def authenticate(self) -> dict:
        # supergrok subscription — cached token in ~/.grok/auth.json.
        return await self._call("authenticate", {"methodId": "cached_token"})

    async def session_new(self, *, cwd: str, mcp_servers: list[dict]) -> str:
        reply = await self._call(
            "session/new", {"cwd": cwd, "mcpServers": mcp_servers}
        )
        session_id = reply["result"]["sessionId"]
        # Arm the readiness future BEFORE returning so the reader can resolve
        # it the instant grok emits the session's first notification.
        loop = asyncio.get_event_loop()
        if session_id not in self._session_ready:
            self._session_ready[session_id] = loop.create_future()
        return session_id

    async def wait_session_ready(
        self, session_id: str, *, timeout: float = 10.0
    ) -> None:
        """Block until grok has emitted its first notification for
        ``session_id`` (its session-ready signal) or ``timeout`` elapses.

        A freshly created session is NOT immediately able to accept a
        ``session/prompt`` — grok finishes async session init and announces
        it via ``available_commands_update``. Prompting before that is
        silently dropped (empty turn). This is a one-shot, event-driven wait
        (resolved by the reader); on timeout we proceed anyway so a missing
        signal degrades to the old behaviour rather than hanging."""
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

        grok requires ``mcpServers`` here exactly like ``session/new`` (it
        rejects the call with -32602 "missing field mcpServers" otherwise).
        After a successful load the session is live in THIS subprocess and
        ``session/prompt`` against it works again (memory preserved). Arms
        the same readiness future ``session/new`` does so callers can wait
        for the reloaded session to be promptable."""
        reply = await self._call(
            "session/load",
            {"sessionId": session_id, "cwd": cwd, "mcpServers": mcp_servers or []},
        )
        loop = asyncio.get_event_loop()
        if session_id not in self._session_ready:
            self._session_ready[session_id] = loop.create_future()
        return reply

    async def session_cancel(self, session_id: str | None) -> None:
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
            raise GrokConcurrentPromptError(
                f"session {session_id!r} already has an active prompt"
            )

        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_requests[rid] = fut
        self._active_prompts[session_id] = fut
        self.register_sink(session_id, notification_sink)

        def release_prompt_owner(done: asyncio.Future) -> None:
            if self._active_prompts.get(session_id) is done:
                self._active_prompts.pop(session_id, None)
            if self._sinks.get(session_id) is notification_sink:
                self._sinks.pop(session_id, None)

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
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            line = await stdout.readline()
            if not line:  # EOF — subprocess exited.
                self._on_eof()
                return
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                logger.warning("grok_provider: non-JSON line dropped: %r", line[:200])
                continue
            self._dispatch(obj)

    def _dispatch(self, obj: dict) -> None:
        has_id = "id" in obj and obj["id"] is not None
        is_reply = has_id and ("result" in obj or "error" in obj)
        if is_reply:
            # POP, not get: the ``_call`` path also pops in its finally, but
            # ``session_prompt`` registers a pending request and bypasses
            # ``_call`` — so without popping here every turn would leak one
            # stale entry (O(turns) growth). Popping on reply is safe for
            # both paths (a double-pop just returns None). Review fix
            # observed live, 2026-06-17.
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
        logger.warning("grok_provider: unroutable line: %r", obj)

    def _handle_agent_request(self, obj: dict) -> None:
        method = obj.get("method", "")
        # We declared fs:false / terminal:false. Any such request is answered
        # with method-not-found; grok must gracefully fail the built-in.
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
            # Auto-approve by selecting an option the agent ACTUALLY offered.
            # Option ids are opaque and vary by Grok version/tool (for example
            # ``approved-for-session``); inventing ``allow`` makes Grok reject
            # the response and terminate the whole turn as permission_rejected.
            params = obj.get("params") or {}
            options = params.get("options") or []
            offered = [option for option in options if isinstance(option, dict)]
            selected = next(
                (
                    option
                    for option in offered
                    if str(option.get("kind", "")).startswith("allow")
                    and option.get("optionId")
                ),
                offered[0] if offered else None,
            )
            option_id = selected.get("optionId") if selected else None
            if option_id:
                outcome = {"outcome": "selected", "optionId": option_id}
            else:
                logger.error(
                    "grok_provider: permission request %r has no selectable options",
                    obj.get("id"),
                )
                outcome = {"outcome": "cancelled"}
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": obj["id"],
                    "result": {"outcome": outcome},
                }
            )
            return
        logger.warning("grok_provider: unhandled agent request %r", method)
        self._write(
            {
                "jsonrpc": "2.0",
                "id": obj["id"],
                "error": {"code": _JSONRPC_METHOD_NOT_FOUND,
                          "message": f"{method} not supported"},
            }
        )

    def _handle_notification(self, obj: dict) -> None:
        method = obj.get("method")
        if acp_types.is_ignored_vendor_notification(method):
            return
        if method != "session/update":
            # Other standard notifications we don't act on.
            return
        params = obj.get("params") or {}
        session_id = params.get("sessionId")
        # Resolve the one-shot readiness future for this session on the FIRST
        # notification we see for it — regardless of whether a turn sink is
        # registered yet (grok announces readiness via
        # available_commands_update before any prompt is sent).
        if session_id:
            ready = self._session_ready.get(session_id)
            if ready is not None and not ready.done():
                ready.set_result(None)
        sink = self._sinks.get(session_id) if session_id else None
        if sink is None:
            # No active turn for this session — drop (e.g. a late chunk after
            # the turn already resolved, or a replay during session/load).
            return
        sink.put_nowait(obj)

    def _on_eof(self) -> None:
        err = GrokAcpClientError("grok agent stdio exited (EOF on stdout)")
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.set_exception(err)
        self._pending_requests.clear()
