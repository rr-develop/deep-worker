"""``aclose()`` during a live turn must fail in-flight prompt futures.

DEFECT: ``aclose`` cancels ``_reader_task``, so ``_read_loop``'s
``except asyncio.CancelledError: raise`` runs and ``_on_eof`` is NEVER called.
``aclose`` then does ``_active_prompts.clear()`` *without* failing the futures,
so they are simply dropped. A consumer waiting on its turn's response future
waits on a future nobody will ever complete — and because ``_sinks.clear()``
puts nothing in the queue either, a two-source
``asyncio.wait({queue.get(), response_fut})`` consumer hangs just as badly as a
naive ``await queue.get()`` one.

Measured before the fix: EOF and reader-crash both release a guarded consumer
with ``AcpClientError``; ``aclose()`` hangs it. This is the only terminal path
with that asymmetry, which is what makes it a defect rather than a trade.

No timeout is involved in the fix: the future is failed eagerly, in the same
synchronous section that clears the map.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from deep_worker.providers.acp.acp_client import AcpClient, AcpClientError


def _client() -> AcpClient:
    client = AcpClient(cmd=("true",))
    client._write = MagicMock()
    return client


class _Stdin:
    def write(self, data: bytes) -> None:  # pragma: no cover - not asserted
        pass

    async def drain(self) -> None:  # pragma: no cover - not asserted
        pass

    def close(self) -> None:
        pass


class _Stdout:
    """Blocks forever without EOF: models a CLI that is alive and quiet, which
    is the state a turn is in when the client is closed underneath it."""

    def __init__(self) -> None:
        self._never = asyncio.Event()

    async def readline(self) -> bytes:
        await self._never.wait()
        return b""


class _Proc:
    def __init__(self) -> None:
        self.stdout = _Stdout()
        self.stdin = _Stdin()
        self.pid = 4242

    async def wait(self) -> int:
        return 0


async def _guarded_wait(queue: asyncio.Queue, response_fut: asyncio.Future) -> str:
    """The two-source drain production uses: whichever source completes first.

    Returns a classification string, never a bare bool — a negative result has
    to name its cause.
    """
    get_task = asyncio.ensure_future(queue.get())
    done, _ = await asyncio.wait(
        {get_task, response_fut}, return_when=asyncio.FIRST_COMPLETED
    )
    if get_task in done:
        get_task.result()
        return "queue-item"
    get_task.cancel()
    try:
        response_fut.result()
    except BaseException as exc:  # noqa: BLE001 - classification, not handling
        return f"released-by-future:{type(exc).__name__}"
    return "released-by-future:result"


@pytest.mark.asyncio
async def test_aclose_releases_a_live_turn_loudly() -> None:
    """FAILS before the fix: the guarded consumer hangs instead of being
    released, because ``_active_prompts.clear()`` drops the future silently."""
    client = _client()
    client._proc = _Proc()  # type: ignore[assignment]
    client._initialized = True
    session_id = "sess-live"
    queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    client.register_sink(session_id, queue)
    response_fut: asyncio.Future = asyncio.get_running_loop().create_future()
    client._active_prompts[session_id] = response_fut
    client._reader_task = asyncio.create_task(client._read_loop())
    await asyncio.sleep(0)

    waiter = asyncio.create_task(_guarded_wait(queue, response_fut))
    await asyncio.sleep(0)

    await client.aclose()

    # Event-driven: the fix completes the future, so the waiter is already
    # releasable. wait_for only bounds the FAILURE mode so a regression reports
    # instead of wedging the suite; it is not part of the mechanism.
    outcome = await asyncio.wait_for(waiter, timeout=5)
    assert outcome == "released-by-future:AcpClientError", outcome


@pytest.mark.asyncio
async def test_aclose_error_names_the_cause() -> None:
    """The error must say the client was closed — not masquerade as an EOF or a
    CLI crash, which are different operator-facing situations."""
    client = _client()
    client._proc = _Proc()  # type: ignore[assignment]
    client._initialized = True
    response_fut: asyncio.Future = asyncio.get_running_loop().create_future()
    client._active_prompts["s"] = response_fut
    client._reader_task = asyncio.create_task(client._read_loop())
    await asyncio.sleep(0)

    await client.aclose()

    assert response_fut.done()
    with pytest.raises(AcpClientError, match="closed"):
        response_fut.result()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_leaves_no_pending_futures() -> None:
    """A second aclose must not raise, and must not resurrect state. Guards the
    fix against being written as an unconditional set_exception on a map that a
    prior close already emptied."""
    client = _client()
    client._proc = _Proc()  # type: ignore[assignment]
    client._initialized = True
    response_fut: asyncio.Future = asyncio.get_running_loop().create_future()
    client._active_prompts["s"] = response_fut
    client._reader_task = asyncio.create_task(client._read_loop())
    await asyncio.sleep(0)

    await client.aclose()
    await client.aclose()

    assert client._active_prompts == {}
    assert client._sinks == {}
    assert response_fut.done()


@pytest.mark.asyncio
async def test_aclose_does_not_override_an_already_resolved_turn() -> None:
    """A turn that already finished successfully must keep its result: closing
    the client afterwards is routine teardown, not a failure of that turn."""
    client = _client()
    client._proc = _Proc()  # type: ignore[assignment]
    client._initialized = True
    response_fut: asyncio.Future = asyncio.get_running_loop().create_future()
    response_fut.set_result({"id": 1, "result": {"stopReason": "end_turn"}})
    client._active_prompts["s"] = response_fut
    client._reader_task = asyncio.create_task(client._read_loop())
    await asyncio.sleep(0)

    await client.aclose()

    assert response_fut.result()["result"]["stopReason"] == "end_turn"
