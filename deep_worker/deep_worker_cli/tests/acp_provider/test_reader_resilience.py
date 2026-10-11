"""The stdout reader must never die silently (review round 8, item 3).

``AcpClient`` owns exactly ONE reader of the CLI's stdout. It used to run
with no exception guard, so any error killed the task while leaving the
client looking perfectly healthy:

* ``_proc`` stays set, so ``ensure_started()`` short-circuits;
* every later prompt is written to a CLI whose replies nobody reads;
* ``session/prompt`` has no liveness bound, so the agent hangs FOREVER;
* nothing is logged until GC eventually reports a dead task.

Two paths reach it in production. ``StreamReader.readline()`` raises
``ValueError`` (from ``LimitOverrunError``) when a single line exceeds the
stream limit — and the module docstring itself notes the CLI emits very
large lines — and ``_dispatch`` can raise on a malformed frame.

Contract:

* reader death from ANY exception == the EOF transition: every waiter fails
  loudly with a reason naming the cause, and the client resets so the next
  call re-spawns;
* a failure dispatching ONE frame must NOT kill the channel — it is
  contained and the loop keeps reading.
"""
from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.acp.acp_client import AcpClient, AcpClientError


class _FakeProc:
    """Stands in for the CLI subprocess; only stdout/stdin are touched."""

    def __init__(self, stdout: asyncio.StreamReader) -> None:
        self.stdout = stdout
        self.stdin = None
        self.returncode = None

    def wait(self):  # pragma: no cover - reaped best-effort
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(0)
        return fut


def _client(stdout: asyncio.StreamReader) -> AcpClient:
    client = object.__new__(AcpClient)
    client._proc = _FakeProc(stdout)
    client._pending_requests = {}
    client._active_prompts = {}
    client._session_ready = {}
    client._sinks = {}
    client._client_state = None
    client._initialized = True
    client._authenticated = True
    client._reader_task = None
    return client


class TestReaderSurvivesBadFrames:
    def test_dispatch_failure_does_not_kill_the_reader(self) -> None:
        """One malformed frame must not take down the whole channel."""

        async def scenario() -> tuple[list[dict], bool]:
            stdout = asyncio.StreamReader()
            client = _client(stdout)
            seen: list[dict] = []

            def _dispatch(obj: dict) -> None:
                if obj.get("boom"):
                    raise RuntimeError("bad frame")
                seen.append(obj)

            client._dispatch = _dispatch
            task = asyncio.create_task(client._read_loop())
            stdout.feed_data(b'{"boom": true}\n')
            stdout.feed_data(b'{"ok": 1}\n')
            await asyncio.sleep(0.05)
            alive = not task.done()
            stdout.feed_eof()
            await asyncio.wait_for(task, timeout=2)
            return seen, alive

        seen, alive_after_bad_frame = asyncio.run(scenario())
        assert alive_after_bad_frame, "reader died on a single bad frame"
        assert seen == [{"ok": 1}], (
            f"frames after the bad one were not processed: {seen!r}"
        )


class TestReaderDeathIsTreatedAsCrash:
    def test_overlong_line_fails_waiters_instead_of_hanging(self) -> None:
        """``readline()`` over the limit raises ValueError. Waiters must FAIL,
        not wait forever on a channel with no reader."""

        async def scenario() -> tuple[BaseException | None, bool]:
            stdout = asyncio.StreamReader(limit=64)
            client = _client(stdout)
            waiter = asyncio.get_running_loop().create_future()
            client._pending_requests = {1: waiter}

            task = asyncio.create_task(client._read_loop())
            stdout.feed_data(b"x" * 5000)  # no newline -> over the limit
            await asyncio.wait_for(task, timeout=2)

            exc: BaseException | None = None
            if waiter.done():
                exc = waiter.exception()
            return exc, client._proc is None

        exc, proc_reset = asyncio.run(scenario())
        assert exc is not None, (
            "reader died and the pending request was left hanging forever — "
            "this is the silent-hang shape the guard must prevent"
        )
        assert isinstance(exc, AcpClientError)
        assert "reader" in str(exc).lower(), (
            f"error must name the cause, got: {exc!r}"
        )
        assert proc_reset, "client did not reset, so it will never re-spawn"

    def test_reason_distinguishes_reader_death_from_plain_eof(self) -> None:
        """An operator must be able to tell the two apart in the log."""

        async def eof_case() -> str:
            stdout = asyncio.StreamReader()
            client = _client(stdout)
            waiter = asyncio.get_running_loop().create_future()
            client._pending_requests = {1: waiter}
            task = asyncio.create_task(client._read_loop())
            stdout.feed_eof()
            await asyncio.wait_for(task, timeout=2)
            return str(waiter.exception())

        msg = asyncio.run(eof_case())
        assert "EOF" in msg, f"plain EOF lost its wording: {msg!r}"

    def test_cancellation_is_not_swallowed(self) -> None:
        """Shutdown cancels the reader; that must stay a CancelledError and
        must NOT be reported as a CLI crash."""

        async def scenario() -> bool:
            stdout = asyncio.StreamReader()
            client = _client(stdout)
            task = asyncio.create_task(client._read_loop())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return client._proc is not None

        still_set = asyncio.run(scenario())
        assert still_set, (
            "cancellation was treated as a crash and reset the client"
        )
