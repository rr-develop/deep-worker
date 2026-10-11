"""A burst of notifications must never overflow the sink, and never lose a chunk.

Two mechanisms, both required, both measured:

1. BACKPRESSURE. ``session/update`` is delivered with ``await sink.put()``, so a
   full sink makes the reader wait for space instead of failing the turn. The
   pressure goes into the pipe buffer and then into the CLI's own ``write``.
   Control frames (replies, agent->client requests) must stay OFF the blocking
   path: with the turn-ending reply behind a backpressured notification, 1500/50
   completed but 5000/200 and 20000/200 DEADLOCKED (reader waits for space,
   consumer waits for the turn to end).

2. READER-IDLE as the end-of-stream signal. ``queue.empty()`` is not one. The
   reader must be interruptible or a buffered burst can never drain (measured:
   zero yields, consumer never scheduled, sink overflows). The instant it became
   interruptible, an empty queue stopped meaning "stream exhausted" — it can
   equally mean "the reader is holding the next frame". Because the two-source
   wait resolves a tie in favour of the response future, the consumer finalized
   while the tail was still unread in the StreamReader: measured 4802 of 5001
   frames read at the break, LOST = tail (50 of 50, 200 of 200).

Neither a timeout nor a protocol counter is involved: ACP's prompt reply carries
only ``stopReason``. Both conditions are events.

Measured with both mechanisms in place (cap 1000, permission ask mid-stream,
long post-reply tail): LOST=0 and order preserved at 20/2, 900/200, 1500/50,
5000/200, 20000/200 and 50000/500, with qmax pinned at 1000 and the reader
blocking 48501 times at 50000 frames.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker.providers.acp.custom_types import NOTIF_QUEUE_MAX_ITEMS

SESSION = "sess-burst"
PROMPT_ID = 42


def _notif(index: int, filler: int = 200) -> bytes:
    return (
        json.dumps(
            {
                "method": "session/update",
                "params": {
                    "sessionId": SESSION,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"c{index}|" + "x" * filler},
                    },
                },
            }
        )
        + "\n"
    ).encode()


def _reply() -> bytes:
    return (
        json.dumps(
            {"jsonrpc": "2.0", "id": PROMPT_ID, "result": {"stopReason": "end_turn"}}
        )
        + "\n"
    ).encode()


def _permission_ask() -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "session/request_permission",
                "params": {
                    "sessionId": SESSION,
                    "toolCall": {"toolCallId": "tc-1"},
                    "options": [{"optionId": "no", "kind": "deny"}],
                },
            }
        )
        + "\n"
    ).encode()


class _Stdin:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Proc:
    def __init__(self, stdout: asyncio.StreamReader) -> None:
        self.stdout = stdout
        self.stdin = _Stdin()
        self.pid = 4242

    async def wait(self) -> int:
        return 0


def _client(payload: bytes, *, eof: bool = True) -> AcpClient:
    """A real AcpClient reading ``payload`` from a real StreamReader.

    A real StreamReader matters: ``readline()`` does not yield when the bytes are
    already buffered, which is the condition the whole defect lives in. A list of
    pre-split frames would hide it.
    """
    client = AcpClient(cmd=("node", "/nonexistent"))
    stream = asyncio.StreamReader(limit=8 * 1024 * 1024)
    stream.feed_data(payload)
    if eof:
        stream.feed_eof()
    client._proc = _Proc(stream)  # type: ignore[assignment]
    client._write = lambda obj: None  # type: ignore[assignment]
    return client


def _model(client: AcpClient):
    """An AcpChatModel without constructor side effects, wired to ``client``.

    Only the fields ``_apply_update`` / ``_drain_until_stream_exhausted`` touch
    are set — the point is to exercise the REAL helpers, not to build a session.
    """
    from deep_worker.providers.acp.chat_model import AcpChatModel
    from deep_worker.providers.acp.client_state import AcpClientState
    from deep_worker.providers.acp.profile import AcpProfile

    model = object.__new__(AcpChatModel)
    model._acp_client = client
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
    )
    model._client_state = AcpClientState()
    return model


def _state(queue: asyncio.Queue, response_fut: asyncio.Future):
    from deep_worker.providers.acp.custom_types import ThreadState

    state = ThreadState()
    state.session_id = SESSION
    state.current_notif_queue = queue
    state.pending_session_prompt_future = response_fut
    return state


async def _run_turn(
    total: int,
    tail: int,
    *,
    with_permission: bool = False,
    capacity: int = NOTIF_QUEUE_MAX_ITEMS,
    count_at: str = "text",
) -> dict:
    """Drive one turn the way production does and report DELIVERY, not drainage.

    The consumer mirrors ``_consume_turn``: a two-source wait in which the
    response future WINS a tie, then a drain that ends only when the queue is
    empty AND the reader is idle.
    """
    head = total - tail
    payload = b"".join(_notif(i) for i in range(head))
    if with_permission:
        payload += _permission_ask()
    payload += _reply()
    payload += b"".join(_notif(head + i) for i in range(tail))

    client = _client(payload)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
    client.register_sink(SESSION, queue)
    response_fut: asyncio.Future = loop.create_future()
    client._pending_requests[PROMPT_ID] = response_fut
    client._active_prompts[SESSION] = response_fut

    # The REAL consumer helper, on a real AcpChatModel. Driving an inline copy
    # of the drain loop here would prove nothing about production: a mutation
    # that removed the reader-idle check from the helper left such a test green.
    model = _model(client)

    stats = {"blocked": 0, "qmax": 0}
    real_put = client._put_notification

    arrived: list[str] = []

    async def counting_put(obj: dict) -> None:
        sink = client._sinks.get(SESSION)
        if sink is not None and sink.full():
            stats["blocked"] += 1
        text = ((obj.get("params") or {}).get("update") or {}).get("content", {}).get(
            "text"
        )
        if text:
            arrived.append(text.split("|")[0])
        await real_put(obj)
        if sink is not None:
            stats["qmax"] = max(stats["qmax"], sink.qsize())

    client._put_notification = counting_put  # type: ignore[assignment]
    client._reader_task = asyncio.create_task(client._read_loop())

    text_acc: list[str] = []
    tool_calls: list[dict] = []
    state = _state(queue, response_fut)

    try:
        while True:
            get_task = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {get_task, response_fut}, return_when=asyncio.FIRST_COMPLETED
            )
            if response_fut in done:
                # Production's tie-break: the future wins, checked first.
                if get_task in done and not get_task.cancelled():
                    model._apply_update(
                        state, "thread-1", get_task.result(), text_acc, tool_calls
                    )
                else:
                    get_task.cancel()
                await model._drain_until_stream_exhausted(
                    state, "thread-1", queue, text_acc, tool_calls
                )
                break
            model._apply_update(
                state, "thread-1", get_task.result(), text_acc, tool_calls
            )
    finally:
        if client._reader_task is not None:
            client._reader_task.cancel()

    delivered = (
        arrived if count_at == "transport" else [p.split("|")[0] for p in text_acc]
    )
    return {
        "delivered": delivered,
        "lost": total - len(delivered),
        "ordered": delivered == [f"c{i}" for i in range(total)],
        **stats,
    }


class TestBurstDoesNotOverflowAndDoesNotLose:
    @pytest.mark.asyncio
    async def test_burst_far_beyond_capacity_delivers_every_notification(self) -> None:
        """50 000 notifications against a 1000-slot sink. FAILS pre-fix with
        AcpSinkOverflowError / LOST=49000.

        Delivery is counted at the TRANSPORT, because 50 000 chunks exceed
        ``QUEUED_TEXT_MAX_CHUNKS`` (10 000) and the accumulator then drops the
        excess *by design*, loudly (DESIGN §4.5). Those are two different
        guarantees and conflating them would make this test assert that a
        documented cap does not exist.
        """
        result = await asyncio.wait_for(
            _run_turn(50_000, 500, with_permission=True, count_at="transport"),
            timeout=180,
        )
        assert result["lost"] == 0, f"transport lost {result['lost']} of 50000"
        assert result["ordered"], "notifications arrived out of order"
        assert result["qmax"] <= NOTIF_QUEUE_MAX_ITEMS, result["qmax"]
        assert result["blocked"] > 10_000, (
            f"limiter engaged only {result['blocked']} times — the burst did not "
            "exercise backpressure"
        )

    @pytest.mark.asyncio
    async def test_a_turn_within_the_text_cap_loses_no_text(self) -> None:
        """Below ``QUEUED_TEXT_MAX_CHUNKS`` the answer must be complete, so the
        two guarantees above are both pinned: transport delivery AND text."""
        result = await asyncio.wait_for(_run_turn(9000, 300), timeout=120)
        assert result["lost"] == 0, f"lost {result['lost']} chunks of text"
        assert result["ordered"]
        assert result["blocked"] > 0, "limiter never engaged at 9000 chunks"

    @pytest.mark.asyncio
    async def test_queue_never_exceeds_capacity(self) -> None:
        for total, tail in ((1500, 50), (5000, 200)):
            result = await asyncio.wait_for(_run_turn(total, tail), timeout=90)
            assert result["qmax"] <= NOTIF_QUEUE_MAX_ITEMS, (
                f"{total}/{tail}: queue reached {result['qmax']}"
            )
            assert result["lost"] == 0, f"{total}/{tail}: lost {result['lost']}"

    @pytest.mark.asyncio
    async def test_no_deadlock_when_the_limiter_engages_hard(self) -> None:
        """The turn must still COMPLETE while the reader is parked thousands of
        times. Backpressure with the reply on the blocking path deadlocked here.

        Counted at the transport: 20 000 exceeds the text cap (see the 50 000
        case), and what this test is about is completion, not accumulation."""
        result = await asyncio.wait_for(
            _run_turn(20_000, 200, count_at="transport"), timeout=120
        )
        assert result["blocked"] > 1000, (
            f"limiter barely engaged ({result['blocked']}) — this configuration "
            "no longer exercises the deadlock risk"
        )
        assert result["lost"] == 0


class TestPostReplyTailSurvives:
    @pytest.mark.asyncio
    async def test_tail_in_the_same_read_batch_is_delivered(self) -> None:
        """Chunks that arrive AFTER the prompt reply, in the same buffered batch,
        must still reach the answer — the premise the finalization drain exists
        for. FAILS with a queue-only drain once the reader can be interrupted."""
        result = await asyncio.wait_for(_run_turn(900, 200), timeout=60)
        assert result["lost"] == 0, f"lost {result['lost']} tail chunks"
        assert result["delivered"][-1] == "c899"

    @pytest.mark.asyncio
    async def test_tail_survives_WITH_the_limiter_engaged(self) -> None:
        """The sixth property. A tail test that only runs below capacity proves
        nothing about the mechanism, because the limiter never engages — that is
        the exact hole an earlier measurement of mine fell into."""
        result = await asyncio.wait_for(_run_turn(5000, 200), timeout=90)
        assert result["blocked"] > 0, (
            "the limiter never engaged, so this run says nothing about the "
            "interaction between backpressure and the tail"
        )
        assert result["lost"] == 0, f"lost {result['lost']} with backpressure active"
        assert result["delivered"][-1] == "c4999"


class TestReaderIdleSignal:
    @pytest.mark.asyncio
    async def test_idle_is_clear_while_the_reader_holds_a_frame(self) -> None:
        """The signal must be FALSE exactly while a frame is in flight, which is
        what makes it a usable end-of-stream test."""
        client = _client(b"".join(_notif(i) for i in range(2000)), eof=False)
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        client.register_sink(SESSION, queue)
        client._reader_task = asyncio.create_task(client._read_loop())
        try:
            # Let the reader fill the sink and park inside await put().
            for _ in range(50):
                await asyncio.sleep(0)
            assert queue.full()
            assert not client.reader_is_idle(), (
                "reader reported idle while parked on a full sink holding a frame"
            )
        finally:
            if client._reader_task is not None:
                client._reader_task.cancel()

    @pytest.mark.asyncio
    async def test_idle_is_set_once_the_stream_is_exhausted(self) -> None:
        client = _client(b"".join(_notif(i) for i in range(5)))
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        client.register_sink(SESSION, queue)
        client._reader_task = asyncio.create_task(client._read_loop())
        try:
            await asyncio.wait_for(client.wait_reader_idle(), timeout=5)
            while not queue.empty():
                queue.get_nowait()
            assert client.reader_is_idle()
        finally:
            if client._reader_task is not None:
                client._reader_task.cancel()

    @pytest.mark.asyncio
    async def test_idle_is_released_when_the_reader_dies(self) -> None:
        """A dead reader must not leave a consumer waiting on "idle" forever —
        that would be a second, silent way to hang."""
        client = _client(b"", eof=False)

        async def boom() -> bytes:
            raise ValueError("simulated over-limit line")

        client._proc.stdout.readline = boom  # type: ignore[assignment]
        client._reader_task = asyncio.create_task(client._read_loop())
        await asyncio.wait_for(client.wait_reader_idle(), timeout=5)
        assert client.reader_is_idle()


class TestTerminalPathsStayLoud:
    """EOF / crash / aclose during the drain: a visible error, never a hang."""

    async def _terminal(self, kind: str) -> str:
        client = _client(b"".join(_notif(i) for i in range(5)), eof=(kind == "eof"))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        client.register_sink(SESSION, queue)
        response_fut: asyncio.Future = loop.create_future()
        client._active_prompts[SESSION] = response_fut
        if kind == "crash":
            async def boom() -> bytes:
                raise ValueError("simulated over-limit line")

            client._proc.stdout.readline = boom  # type: ignore[assignment]
        client._reader_task = asyncio.create_task(client._read_loop())
        await asyncio.sleep(0)
        if kind == "aclose":
            await client.aclose()
        try:
            get_task = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {get_task, response_fut}, return_when=asyncio.FIRST_COMPLETED
            )
            if response_fut in done:
                get_task.cancel()
                try:
                    response_fut.result()
                    return "released:result"
                except BaseException as exc:  # noqa: BLE001 - classification
                    return f"released:{type(exc).__name__}"
            get_task.cancel()
            while True:
                while not queue.empty():
                    queue.get_nowait()
                if client.reader_is_idle():
                    break
                await client.wait_reader_idle()
            try:
                await asyncio.wait_for(asyncio.shield(response_fut), timeout=2)
                return "released:result"
            except asyncio.TimeoutError:
                return "no-error"
            except BaseException as exc:  # noqa: BLE001 - classification
                return f"released:{type(exc).__name__}"
        finally:
            if client._reader_task is not None:
                client._reader_task.cancel()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["eof", "crash", "aclose"])
    async def test_terminal_path_releases_with_a_named_error(self, kind: str) -> None:
        outcome = await asyncio.wait_for(self._terminal(kind), timeout=20)
        assert outcome == "released:AcpClientError", f"{kind} -> {outcome}"
