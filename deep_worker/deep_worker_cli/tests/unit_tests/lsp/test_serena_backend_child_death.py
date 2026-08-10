"""Death of the MCP process must fail the call, not hang it.

Why this file exists (found by the retry integration test, §6.3 of the
``docs/DESIGN_LSP_LS_TERMINATED_RETRY.md`` design): the retry on
``LanguageServerTerminatedException`` did not fire in the most important
scenario — Serena dies while a call is ALREADY in flight.

The cause is not the retry logic but the transport. ``_OwnedStdioSession``
treats only EOF on the child's ``stdout`` as death. But Serena launches
language servers as its own subprocesses, and they INHERIT the same
stdout pipe. So after Serena dies the write end of the pipe stays open
in the grandchildren, EOF never arrives, ``ClientSession`` does not learn
about the break, and the call waiting for a reply hangs until the end of
the budget. A timeout bubbles up instead of "backend terminated", the
retry never starts — exactly the error the agent originally saw.

This is reproduced here without Serena: a fake MCP server spawns a
grandchild that keeps stdout open and never answers ``tools/call``.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

import pytest

from deep_worker_cli.lsp.service.serena_backend import (
    _OwnedStdioSession,
    _is_backend_terminated,
)


#: Fake MCP server: answers the handshake, spawns a grandchild with the
#: same stdout (as Serena spawns a language server) and stays silent on
#: tool calls, imitating long-running work.
_LEAKY_MCP = r"""
import json, subprocess, sys, time

# The grandchild inherits stdout: the write end of the pipe stays open
# even after this process dies — like Serena's language server.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        msg = json.loads(raw)
    except Exception:
        continue
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "leaky", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"],
              "result": {"tools": []}})
    elif method == "tools/call":
        time.sleep(300)  # no reply will come: the call is "in flight"
"""

#: How long we wait for the call to fail after the process dies. Child
#: death is an event (``proc.wait()``), so the reaction must be nearly
#: instant; the threshold has a large margin but is finite, so the test
#: fails rather than hangs.
_FAIL_FAST_S = 10.0


@pytest.mark.asyncio
async def test_in_flight_call_fails_when_child_dies_with_leaked_pipe() -> None:
    """The child is killed, a grandchild holds stdout — the call must fail fast."""
    argv = [sys.executable, "-u", "-c", _LEAKY_MCP]
    sess = _OwnedStdioSession(argv)
    async with sess:
        session = sess._session
        assert session is not None
        pid = sess.pid
        assert isinstance(pid, int)

        call = asyncio.create_task(session.call_tool("anything", {}))
        # Let the call reach the server: otherwise we would kill the child
        # BEFORE the request is sent and would be testing the wrong scenario.
        await asyncio.sleep(0.5)
        assert not call.done(), "the fake server must not have replied"

        os.kill(pid, signal.SIGKILL)

        try:
            result = await asyncio.wait_for(call, timeout=_FAIL_FAST_S)
        except asyncio.TimeoutError:
            call.cancel()
            pytest.fail(
                "the call did not finish after the MCP process died: EOF will "
                "not arrive while the grandchild holds stdout open"
            )
        except Exception as exc:  # noqa: BLE001 — an exception is also acceptable
            assert _is_backend_terminated(f"transport error: {exc}"), (
                f"the dispatcher does not recognize this as backend death: {exc!r}"
            )
        else:
            text = str(getattr(result, "content", result))
            assert _is_backend_terminated(text) or getattr(
                result, "isError", False
            ), f"the call returned success after the process died: {text!r}"


@pytest.mark.asyncio
async def test_transport_notifies_owner_about_child_death() -> None:
    """The transport must NOTIFY the owner about the child's death.

    Closing the read stream is not enough: only already-registered
    requests would fail that way. The owner (``SerenaBackend``) learns
    about the death precisely through this hook and only thanks to it can
    finish a call that raced with the death. Without the notification the
    other checks would stay green while prod would stay broken
    (mutants T25/T29).
    """
    argv = [sys.executable, "-u", "-c", _LEAKY_MCP]
    fired = asyncio.Event()
    sess = _OwnedStdioSession(argv)
    sess.set_on_death(fired.set)
    async with sess:
        pid = sess.pid
        assert isinstance(pid, int)
        os.kill(pid, signal.SIGKILL)
        try:
            await asyncio.wait_for(fired.wait(), timeout=_FAIL_FAST_S)
        except asyncio.TimeoutError:
            pytest.fail("the transport did not notify the owner about the child's death")


@pytest.mark.asyncio
async def test_backend_installs_death_hook_into_the_transport() -> None:
    """``SerenaBackend`` must subscribe to death at the transport.

    The hook exists only for this subscription: if the backend does not
    install it, it will not learn about the death and the old hang will
    come back (mutant T29).
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    installed: list[object] = []

    class _CM:
        def set_on_death(self, hook) -> None:
            installed.append(hook)

        async def __aenter__(self):
            class _S:
                async def initialize(self) -> None: ...

                async def list_tools(self):
                    class _L:
                        tools: list = []

                    return _L()

            return _S()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend("/tmp")
        await backend.start()
        try:
            assert installed, (
                "the backend did not subscribe to process death at the transport"
            )
            assert installed[0] == backend._mark_lost
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_restarted_backend_does_not_inherit_previous_death() -> None:
    """A restart must clear the death flag of the previous life.

    ``_lost_event`` gates EVERY call. If ``start()`` does not reset it,
    the resurrected backend will fail the very first healthy call and the
    dispatcher will retry against a live process (mutant T28).
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _OkSession:
        async def initialize(self) -> None: ...

        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

        async def call_tool(self, _name, _args):
            # A real call does not reply instantly. Without this pause the
            # race with death would resolve in a single quantum in favor of
            # the reply and would not notice the inherited flag.
            await asyncio.sleep(0.05)

            class _R:
                content: list = []
                isError = False

            return _R()

    class _CM:
        async def __aenter__(self):
            return _OkSession()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend("/tmp")
        await backend.start()
        backend._mark_lost()  # the "previous life" died
        await backend.stop()

        await backend.start()  # resurrection
        try:
            result = await asyncio.wait_for(
                backend.call_tool("FindSymbol", '{"name_path":"X"}'),
                timeout=5.0,
            )
            assert not result.is_error, (
                f"a healthy backend failed the call because of the previous death: "
                f"{result.error_message!r}"
            )
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_reply_wins_over_a_simultaneous_death() -> None:
    """An arriving reply outweighs a death that happened at the same moment.

    The race must be asymmetric: the result is already in hand, so the
    call succeeded, and declaring it a failure means discarding a ready
    reply and making an extra retry (dangerous for mutating tools).
    Mutant T30 gave the victory to death when both fired at once.
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _DyingSession:
        def __init__(self, backend) -> None:
            self._backend = backend

        async def initialize(self) -> None: ...

        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

        async def call_tool(self, _name, _args):
            # The reply and the death become known in the same
            # quantum of event-loop work.
            self._backend._mark_lost()

            class _R:
                content: list = []
                isError = False

            return _R()

    holder: dict[str, object] = {}

    class _CM:
        async def __aenter__(self):
            return _DyingSession(holder["backend"])

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend("/tmp")
        holder["backend"] = backend
        await backend.start()
        try:
            result = await asyncio.wait_for(
                backend.call_tool("FindSymbol", '{"name_path":"X"}'),
                timeout=5.0,
            )
            assert not result.is_error, (
                "a ready reply was discarded because of a simultaneous death: "
                f"{result.error_message!r}"
            )
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_outer_cancellation_does_not_orphan_the_mcp_call() -> None:
    """An outer cancellation must also cancel the inner MCP call.

    Review note (BLOCKER, round 5). ``_call_or_death`` is cancelled from
    outside in two regular cases: the dispatcher's ``wait_for`` at the
    budget boundary and client cancellation of the gRPC request. If the
    inner ``session.call_tool`` task stays alive, it will keep using the
    SHARED MCP session after ``_lock`` has been released — that is, in
    parallel with the next call. This is exactly the class of
    desynchronization that the lock serializes calls to prevent.

    We check the contract itself, not its consequence: the inner call
    must receive the cancellation, and afterwards the backend must remain
    usable.
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    started = asyncio.Event()
    inner_cancelled = asyncio.Event()
    inner_tasks: list[asyncio.Task] = []

    class _HangingSession:
        async def initialize(self) -> None: ...

        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

        async def call_tool(self, _name, _args):
            task = asyncio.current_task()
            assert task is not None
            inner_tasks.append(task)
            started.set()
            try:
                await asyncio.Event().wait()  # no reply will come
            except asyncio.CancelledError:
                inner_cancelled.set()
                raise

    class _CM:
        async def __aenter__(self):
            return _HangingSession()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend("/tmp")
        await backend.start()
        try:
            outer = asyncio.create_task(
                backend.call_tool("FindSymbol", '{"name_path":"X"}')
            )
            await asyncio.wait_for(started.wait(), timeout=5.0)

            outer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await outer

            try:
                await asyncio.wait_for(inner_cancelled.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pytest.fail(
                    "outer cancellation left the MCP call alive: it will keep "
                    "using the shared session in parallel with the next one"
                )
            # Cancelling is not enough — a cancelled task that has not been
            # awaited to completion still counts as alive and can run code
            # after the cancellation point. We require actual COMPLETION,
            # immediately after control returns, without extra
            # "spins" of the event loop.
            assert all(t.done() for t in inner_tasks), (
                "an orphaned call task was left in the pending state"
            )
            # And no stray live tasks of this call: neither the death-wait
            # task nor the MCP task itself.
            leftovers = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
                and ("_call_or_death" in repr(t) or "call_tool" in repr(t)
                     or "_lost_event" in repr(t))
            ]
            assert not leftovers, (
                f"live call tasks remained after cancellation: {leftovers}"
            )
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_timed_out_call_leaves_no_live_children() -> None:
    """When the budget expires, both child tasks must be COMPLETED.

    Cancelling is not enough: a cancelled task that has not been awaited
    to completion is formally alive and can run code after the
    cancellation point — after ``call_tool`` has already returned control
    and released ``_lock``, i.e. in parallel with the next call on the
    SHARED MCP session. This must be checked exactly here, on the timeout
    path: it returns control without any intermediate ``await`` at the
    caller, so a "spin" of the loop cannot hide the difference between
    "cancelled" and "cancelled and awaited".
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    inner: dict[str, asyncio.Task] = {}
    started = asyncio.Event()

    class _HangingSession:
        async def initialize(self) -> None: ...

        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

        async def call_tool(self, _name, _args):
            task = asyncio.current_task()
            assert task is not None
            inner["task"] = task
            started.set()
            await asyncio.Event().wait()

    class _CM:
        async def __aenter__(self):
            return _HangingSession()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        # A tiny budget: the boundary is what matters, not the waiting.
        backend = sb.SerenaBackend("/tmp", call_timeout_s=0.2)
        await backend.start()
        try:
            before = {t for t in asyncio.all_tasks() if not t.done()}
            result = await asyncio.wait_for(
                backend.call_tool("FindSymbol", '{"name_path":"X"}'),
                timeout=5.0,
            )
            assert result.is_error

            # No additional await before these checks.
            assert inner["task"].done(), (
                "the MCP task was cancelled but not awaited to completion: it "
                "can still touch the shared session after the lock is released"
            )
            leaked = {
                t
                for t in asyncio.all_tasks()
                if not t.done() and t not in before
                and t is not asyncio.current_task()
            }
            assert not leaked, (
                f"live tasks remained after the budget expired: {leaked}"
            )
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_call_started_after_child_death_fails_fast() -> None:
    """A call started AFTER the death must also fail fast.

    A separate sub-case, and it is trickier than the previous one:
    failing pending requests is done by the MCP session's ``_receive_loop``
    in its ``finally`` (``mcp/shared/session.py``). A request registered
    after that loop has already exited will not be failed by anyone — it
    will hang until the end of the budget. That is exactly what the race
    looked like on the live stack: the death landed between task creation
    and request dispatch, so the retry never started and the call burned
    the whole budget.
    """
    argv = [sys.executable, "-u", "-c", _LEAKY_MCP]
    sess = _OwnedStdioSession(argv)
    async with sess:
        session = sess._session
        assert session is not None
        pid = sess.pid
        assert isinstance(pid, int)

        os.kill(pid, signal.SIGKILL)
        # Let the watcher process the death (pidfd event), but do NOT
        # send any request before that.
        await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(
                session.call_tool("anything", {}), timeout=_FAIL_FAST_S
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "a call after process death hung: the session does not know that "
                "there is no one left to write to"
            )
        except Exception as exc:  # noqa: BLE001 — expected path
            assert _is_backend_terminated(f"transport error: {exc}")


@pytest.mark.asyncio
async def test_call_racing_the_death_does_not_wait_out_the_budget() -> None:
    """A call that "missed" the death must fail immediately.

    The narrowest and most harmful case, measured on the live stack
    (a diagnostic log): the death happens BETWEEN creation of the call
    task and dispatch of the request. Failing pending requests is done by
    the MCP session's ``_receive_loop`` in its ``finally``, but by then
    it has already finished, and our request registers AFTER it — no one
    is left to fail it. The call hung for the whole budget (110 s), the
    dispatcher saw a timeout instead of a death and did not retry:
    exactly the failure this feature exists to eliminate.

    Here the session is faked to pin precisely the contract: ``call_tool``
    finishes when the backend dies, without waiting for a reply that will
    never come.
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _HangingSession:
        async def initialize(self) -> None: ...

        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

        async def call_tool(self, _name, _args):
            await asyncio.Event().wait()  # no reply will ever come

    class _CM:
        async def __aenter__(self):
            return _HangingSession()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    orig = sb._open_mcp_session
    sb._open_mcp_session = _fake_open  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend("/tmp")
        await backend.start()
        try:
            call = asyncio.create_task(
                backend.call_tool("FindSymbol", '{"name_path":"X"}')
            )
            await asyncio.sleep(0.1)
            assert not call.done()

            # The death is noticed by the transport/supervisor — the call must
            # get an outcome immediately rather than wait out the budget.
            backend._mark_lost()

            try:
                result = await asyncio.wait_for(call, timeout=5.0)
            except asyncio.TimeoutError:
                call.cancel()
                pytest.fail(
                    "call_tool kept waiting for a reply from a dead backend"
                )
            assert result.is_error
            assert _is_backend_terminated(result.error_message), (
                f"the dispatcher does not recognize this as death: "
                f"{result.error_message!r}"
            )
        finally:
            await backend.stop()
    finally:
        sb._open_mcp_session = orig


@pytest.mark.asyncio
async def test_backend_reports_terminated_marker_on_child_death(
    tmp_path,
) -> None:
    """Through ``SerenaBackend``: the error text must be "terminated".

    The dispatcher decides about retry based on the error TEXT
    (``_is_backend_terminated``), so failing fast is not enough — the
    failure must be recognizable.
    """
    from deep_worker_cli.lsp.service import serena_backend as sb

    argv = [sys.executable, "-u", "-c", _LEAKY_MCP]
    orig = sb.build_serena_spawn_argv
    sb.build_serena_spawn_argv = lambda _root: argv  # ty: ignore[invalid-assignment]
    try:
        backend = sb.SerenaBackend(tmp_path)
        await backend.start()
        try:
            pid = backend.pid
            assert isinstance(pid, int)
            call = asyncio.create_task(
                backend.call_tool("FindSymbol", '{"name_path":"X"}')
            )
            await asyncio.sleep(0.5)
            assert not call.done()
            os.kill(pid, signal.SIGKILL)
            try:
                result = await asyncio.wait_for(call, timeout=_FAIL_FAST_S)
            except asyncio.TimeoutError:
                call.cancel()
                pytest.fail("call_tool hung after the MCP process died")
            assert result.is_error
            assert _is_backend_terminated(result.error_message), (
                f"the text is not recognized as backend death: "
                f"{result.error_message!r}"
            )
        finally:
            await backend.stop()
    finally:
        sb.build_serena_spawn_argv = orig
