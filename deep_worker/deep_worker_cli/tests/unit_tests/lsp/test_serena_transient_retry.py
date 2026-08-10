"""Retry of transient language-server cancellations.

See ``docs/DESIGN_LSP_TRANSIENT_CANCEL_RETRY.md``.

Why this is needed at all: the language server cancels its own in-flight
requests with code ``-32800`` (RequestCancelled) / ``-32801``
(ContentModified) while background reindexing is running. Serena
returns this NOT as an exception, but as a successful MCP response with
``isError=True`` and text like ``cancelled (-32800)``. Previously we
handed this text to the agent as an ordinary tool error, and the agent drew
the wrong conclusion from it: "you cannot call two tools at once".

Pinned here:

* a transient cancellation is retried and the call eventually succeeds;
* exhausting attempts/budget yields an EXPLANATORY text, not a raw code;
* non-transient errors, timeouts and transport errors are NOT retried;
* the retry does not release the shared lock (peak_concurrent == 1);
* backoff is an ``await``, not a thread-blocking sleep;
* the series is bounded by the shared ``call_timeout_s`` budget (A2).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deep_worker_cli.lsp.service.serena_backend import (
    SerenaBackend,
    _TRANSIENT_MAX_ATTEMPTS,
    _is_transient_cancel,
)

#: Only async tests are marked; ``test_code_pattern_is_strict`` is
#: synchronous (a pure predicate function), and a global asyncio marker
#: on it would produce a PytestWarning.
pytestmark: list = []


#: Real text forms arriving from solidlsp. The first is the direct
#: ``LSPError.__str__``; the second is the same inside the
#: ``SolidLSPException.__str__`` wrapper (``... (caused by ...)``).
_RAW_CANCELLED = "cancelled (-32800)"
_RAW_WRAPPED = (
    "Error processing request textDocument/documentSymbol with params:\n{} "
    "(caused by cancelled (-32800))"
)
_RAW_CONTENT_MODIFIED = "content modified (-32801)"


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResult:
    def __init__(self, *, text: str = "", is_error: bool = False) -> None:
        self.content = [_FakeContent(text)]
        self.isError = is_error


class _RetrySession:
    """MCP session with a programmable response queue.

    ``responses`` is a FIFO; when the queue is empty, the last element is
    served again (so "server still busy" is modelled without an infinite
    list).
    """

    def __init__(
        self,
        *,
        responses: list[_FakeResult] | None = None,
        tools_list: list[str] | None = None,
        sleep_before_reply: float = 0.0,
        raise_on_call: BaseException | None = None,
    ) -> None:
        self.responses = list(responses or [_FakeResult(text="{}")])
        self.tools_list = tools_list or ["find_symbol"]
        self.sleep_before_reply = sleep_before_reply
        self.raise_on_call = raise_on_call
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.concurrent_calls = 0
        self.peak_concurrent = 0
        #: Order of "who called" — by the args["caller"] value, if the
        #: test passed it. Needed to distinguish "attempts of one call run
        #: consecutively" from "calls interleave" (a released lock).
        self.caller_order: list[str] = []

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[SimpleNamespace(name=n) for n in self.tools_list]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeResult:
        self.concurrent_calls += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent_calls)
        try:
            self.call_log.append((name, dict(arguments)))
            caller = arguments.get("caller")
            if caller is not None:
                self.caller_order.append(str(caller))
            if self.raise_on_call is not None:
                raise self.raise_on_call
            if self.sleep_before_reply:
                await asyncio.sleep(self.sleep_before_reply)
            if len(self.responses) > 1:
                return self.responses.pop(0)
            return self.responses[0]
        finally:
            self.concurrent_calls -= 1


def _install(monkeypatch: pytest.MonkeyPatch, session: _RetrySession) -> None:
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _CM:
        async def __aenter__(self) -> _RetrySession:
            return session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    async def _fake_open(_argv: list[str]) -> Any:
        return _CM()

    monkeypatch.setattr(sb, "_open_mcp_session", _fake_open)


# ── §4.1: code recognition (A1) ──────────────────────────────────────


def test_code_pattern_is_strict() -> None:
    """A1: we match the code, not a bare substring.

    ``-328001`` contains ``-32800`` as a prefix; a naive
    ``"-32800" in message`` check would take a foreign error for a
    cancellation and loop the retry on an unrecoverable failure.
    """
    assert _is_transient_cancel(_RAW_CANCELLED)
    assert _is_transient_cancel(_RAW_WRAPPED)
    assert _is_transient_cancel(_RAW_CONTENT_MODIFIED)
    assert not _is_transient_cancel("boom (-328001)")
    assert not _is_transient_cancel("no such symbol")
    assert not _is_transient_cancel("")


# ── §4.2: retry ──────────────────────────────────────────────────────


async def test_transient_cancel_is_retried_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _RetrySession(
        responses=[
            _FakeResult(text=_RAW_CANCELLED, is_error=True),
            _FakeResult(text='{"ok":true}'),
        ]
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is False, result.error_message
        assert json.loads(result.content_json) == {"ok": True}
        assert len(session.call_log) == 2, "there must be exactly one retry"
    finally:
        await backend.stop()


async def test_content_modified_is_also_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _RetrySession(
        responses=[
            _FakeResult(text=_RAW_CONTENT_MODIFIED, is_error=True),
            _FakeResult(text='{"ok":true}'),
        ]
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is False
        assert len(session.call_log) == 2
    finally:
        await backend.stop()


async def test_non_transient_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _RetrySession(
        responses=[_FakeResult(text="no such symbol", is_error=True)]
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert len(session.call_log) == 1, "an ordinary error is not retried"
        assert result.error_message == "no such symbol", (
            "the text of an ordinary error must not be substituted"
        )
    finally:
        await backend.stop()


# ── §4.3: explanatory error (A6) ─────────────────────────────────────


async def test_transient_retry_exhausted_returns_explanatory_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A6: the agent receives an explanation, not a protocol code.

    It is precisely from the raw ``cancelled (-32800)`` that the agent in
    the original report inferred a non-existent limit on concurrent calls.
    """
    session = _RetrySession(
        responses=[_FakeResult(text=_RAW_CANCELLED, is_error=True)]
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        msg = result.error_message
        low = msg.lower()
        assert "reindex" in low or "busy" in low, msg
        assert "retry" in low, msg
        assert "NOT a limit on concurrent tool calls" in msg, msg
        assert msg.strip() != _RAW_CANCELLED, (
            "the raw protocol text must not be the only explanation"
        )
        assert len(session.call_log) == 3, "three attempts: 1 original + 2 retries"
        assert "attempts made: 3" in msg, msg
    finally:
        await backend.stop()


# ── §4.2: lock and backoff ───────────────────────────────────────────


async def test_transient_retry_does_not_release_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A4: conversion and retry — under the same lock.

    If the lock is released between attempts, a neighbouring call will
    interleave, and ours will go to the end of the queue, losing its budget.
    """
    # A response delay is mandatory: without it, calls manage to run "in
    # queue order" even with a released lock, and a "release/acquire
    # around backoff" mutation would stay alive.
    session = _RetrySession(
        responses=[_FakeResult(text=_RAW_CANCELLED, is_error=True)],
        sleep_before_reply=0.1,
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        await asyncio.gather(
            backend.call_tool("FindSymbol", json.dumps({"caller": "A"})),
            backend.call_tool("FindSymbol", json.dumps({"caller": "B"})),
        )
        assert session.peak_concurrent == 1, (
            f"retry released the lock (peak={session.peak_concurrent})"
        )
        # The main check: all attempts of one call run CONSECUTIVELY.
        # If backoff releases the lock, the neighbour interleaves between
        # them and the order becomes alternating (A,B,A,B...).
        order = session.caller_order
        assert len(order) == 6, order
        assert order == sorted(order, key=order.index), order
        first = order[0]
        assert order[:3] == [first] * 3, (
            f"attempts of one call were broken up by a neighbour: {order}"
        )
    finally:
        await backend.stop()


async def test_backoff_is_awaited_not_slept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Policy: no event-loop blocking and no new threads."""
    session = _RetrySession(
        responses=[
            _FakeResult(text=_RAW_CANCELLED, is_error=True),
            _FakeResult(text="{}"),
        ]
    )
    _install(monkeypatch, session)

    from deep_worker_cli.lsp.service import serena_backend as sb

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def _spy(delay: float, *a: Any, **k: Any) -> Any:
        slept.append(delay)
        return await real_sleep(0)

    monkeypatch.setattr(sb.asyncio, "sleep", _spy)

    backend = SerenaBackend(tmp_path)
    await backend.start()
    try:
        await backend.call_tool("FindSymbol", "{}")
        assert slept, "backoff must go through await asyncio.sleep"
        assert all(d > 0 for d in slept)
    finally:
        await backend.stop()


# ── A2: shared series budget ─────────────────────────────────────────


async def test_series_budget_is_bounded_by_call_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: the whole series fits within one ``call_timeout_s``.

    Without a shared ceiling, 3 attempts × per-attempt timeout would
    outlive the client deadline, and the agent would get DEADLINE_EXCEEDED
    instead of an explanatory text — exactly what this whole thing is for.
    """
    # Each attempt on its own fits within call_timeout_s, but THREE
    # attempts + backoff no longer do. Without a shared ceiling the series
    # would survive ~3x the budget and outlive the client deadline.
    call_timeout_s = 0.6
    session = _RetrySession(
        responses=[_FakeResult(text=_RAW_CANCELLED, is_error=True)],
        sleep_before_reply=0.4,
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path, call_timeout_s=call_timeout_s)
    await backend.start()
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await backend.call_tool("FindSymbol", "{}")
        elapsed = loop.time() - started
        # The series ceiling is the budget itself; a tolerance for the scheduler.
        assert elapsed <= call_timeout_s + 0.25, (
            f"series exceeded the shared budget {call_timeout_s}s: {elapsed:.2f}s "
            f"(without a shared ceiling it would have been ~"
            f"{3 * 0.4 + 0.75:.2f}s)"
        )
        assert result.is_error is True
        assert "NOT a limit on concurrent tool calls" in result.error_message
        # The series was cut off by the budget, not by the number of
        # attempts: the message must name the REAL number of calls to the
        # server. Otherwise the agent gets a false diagnostic ("after 3
        # attempts") in exactly the text that was written to be truthful.
        made = len(session.call_log)
        assert made < _TRANSIENT_MAX_ATTEMPTS, (
            f"the test must cut the series by budget; attempts: {made}"
        )
        assert f"attempts made: {made}" in result.error_message, (
            result.error_message
        )
    finally:
        await backend.stop()


async def test_timeout_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: a timeout is not retried — a retry would only double the wait."""
    session = _RetrySession(
        responses=[_FakeResult(text="{}")], sleep_before_reply=5.0
    )
    _install(monkeypatch, session)

    backend = SerenaBackend(tmp_path, call_timeout_s=0.2)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert "timed out" in result.error_message
        assert len(session.call_log) == 1, "a timeout must not be retried"
    finally:
        await backend.stop()


async def test_transport_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: a transport error = backend death → _mark_lost, no retry."""
    session = _RetrySession(raise_on_call=BrokenPipeError("pipe is gone"))
    _install(monkeypatch, session)

    lost: list[bool] = []
    backend = SerenaBackend(tmp_path, on_lost=lambda: lost.append(True))
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert "transport error" in result.error_message
        assert len(session.call_log) == 1
        assert lost == [True], "a transport error must mark the backend lost"
    finally:
        await backend.stop()


# ── A5: observability hook ───────────────────────────────────────────


async def test_transient_retry_notifies_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A5: the hook is called on each RETRY (the original attempt is not a retry)."""
    session = _RetrySession(
        responses=[_FakeResult(text=_RAW_CANCELLED, is_error=True)]
    )
    _install(monkeypatch, session)

    hits: list[int] = []
    backend = SerenaBackend(tmp_path)
    backend.set_on_transient_retry(lambda: hits.append(1))
    await backend.start()
    try:
        await backend.call_tool("FindSymbol", "{}")
        assert len(hits) == 2, "3 attempts → exactly 2 retries"
    finally:
        await backend.stop()


async def test_hook_exception_does_not_break_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A5: observability must not be allowed to break the working path."""
    session = _RetrySession(
        responses=[
            _FakeResult(text=_RAW_CANCELLED, is_error=True),
            _FakeResult(text='{"ok":true}'),
        ]
    )
    _install(monkeypatch, session)

    def _boom() -> None:
        raise RuntimeError("metrics exploded")

    backend = SerenaBackend(tmp_path)
    backend.set_on_transient_retry(_boom)
    await backend.start()
    try:
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is False, result.error_message
    finally:
        await backend.stop()
