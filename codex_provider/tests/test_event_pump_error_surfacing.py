"""Regression: a fatal error inside the codex event stream must be
SURFACED to ainvoke, not silently swallowed.

Incident observed 2026-05-20: the agent ran with ``--model unknown``;
the Codex API rejected it with HTTP 400, which the JSONL parser turned
into a ``CodexProtocolError`` raised from ``handle.events``. That
exception propagated out of ``event_pump`` and killed the pump task —
landing as a bare ``Task exception was never retrieved`` in the logs.
The pump's ``finally`` still pushed ``END_OF_STREAM`` onto the
ainvoke queue, so ``CodexChatModel.ainvoke`` saw a clean end-of-stream,
returned an EMPTY assistant message, and the upper layers treated the
turn as "no response" rather than "the model is wrong". The agent then
got stuck (status RUNNING, no error surfaced to the user).

Contract under test:
  * ``event_pump`` records the fatal exception on
    ``handle.pump_error`` before it finishes (so a consumer that only
    sees ``END_OF_STREAM`` can still discover the cause), AND still
    re-raises it (so ``Task.exception()`` is meaningful too).
  * After the pump finishes via error, ``END_OF_STREAM`` is still
    enqueued (ainvoke must never block forever).
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider.event_pump import END_OF_STREAM, event_pump  # noqa: E402
from codex_provider.exec_runner import (  # noqa: E402
    CodexProtocolError,
    ThreadEvent,
    ThreadStartedEvent,
)


async def _events_then_raise(
    pre: list[ThreadEvent], exc: BaseException
) -> AsyncGenerator[ThreadEvent, None]:
    """Yield ``pre`` events, then raise ``exc`` from the stream —
    mimicking ``_iter_events`` hitting a Codex error mid-stream."""
    for ev in pre:
        yield ev
    raise exc


class _StubHandle:
    """Stand-in exposing what the pump touches + the new pump_error."""

    def __init__(self, gen: AsyncGenerator[ThreadEvent, None]) -> None:
        self.events = gen
        self.ainvoke_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, asyncio.Future[str]] = {}
        # New field the fix introduces. Defaulting it here keeps the
        # stub forward-compatible regardless of attribute order.
        self.pump_error: BaseException | None = None


PROXY = "deepworker_test"


@pytest.mark.asyncio
async def test_pump_records_error_on_handle() -> None:
    """The fatal stream error must be recorded on ``handle.pump_error``."""
    boom = CodexProtocolError(
        'Codex stream error: {"status":400,"error":{"message":'
        '"The \'unknown\' model is not supported"}}'
    )
    handle = _StubHandle(_events_then_raise([], boom))

    # The pump still raises (so Task.exception() is meaningful). We
    # don't care about the raised instance here — we care that it was
    # ALSO recorded on the handle for the END_OF_STREAM-only consumer.
    with pytest.raises(CodexProtocolError):
        await event_pump(handle, PROXY)  # type: ignore[arg-type]

    assert handle.pump_error is boom, (
        "event_pump must record the fatal stream error on "
        "handle.pump_error so ainvoke (which only sees END_OF_STREAM) "
        "can re-raise it instead of returning an empty response"
    )


@pytest.mark.asyncio
async def test_pump_still_enqueues_end_of_stream_on_error() -> None:
    """Even on a fatal error, END_OF_STREAM must be enqueued so the
    ainvoke consumer never blocks forever."""
    boom = CodexProtocolError("Codex stream error: boom")
    handle = _StubHandle(_events_then_raise([ThreadStartedEvent(thread_id="t1")], boom))

    with pytest.raises(CodexProtocolError):
        await event_pump(handle, PROXY)  # type: ignore[arg-type]

    # Drain the queue; END_OF_STREAM must be present.
    saw_eos = False
    while not handle.ainvoke_queue.empty():
        item = handle.ainvoke_queue.get_nowait()
        if item is END_OF_STREAM:
            saw_eos = True
    assert saw_eos, "END_OF_STREAM must be enqueued even on fatal error"


@pytest.mark.asyncio
async def test_pump_clean_end_leaves_pump_error_none() -> None:
    """A normal end-of-stream (generator exhausts cleanly) must leave
    ``pump_error`` as None — no false positives."""
    handle = _StubHandle(_events_then_clean([ThreadStartedEvent(thread_id="t1")]))

    await event_pump(handle, PROXY)  # type: ignore[arg-type]

    assert handle.pump_error is None


async def _events_then_clean(
    pre: list[ThreadEvent],
) -> AsyncGenerator[ThreadEvent, None]:
    for ev in pre:
        yield ev
