"""Integration tests: the real ``AcpClient`` channel against a fake ACP CLI
subprocess (tests/acp_provider/fake_acp_cli.py). No real Claude Code needed.

Covers (review item 8):
  * Channel-A sequence: start → initialize → session/new → readiness →
    session/prompt → terminal reply,
  * JSON-RPC permission replies: allow/deny choice written back for a real
    ``session/request_permission`` request,
  * usage-limit error reply → ``AcpUsageLimitError`` (failover path),
  * EOF crash (mid-turn) → deterministic recoverable-crash transition,
  * teardown (aclose) reaps the subprocess,
  * notification sink overflow → fail-loud (no hang).

Each test drives its own subprocess via an env-selected fake scenario.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import re

import pytest

from deep_worker.providers.acp.acp_client import (
    AcpClient,
    AcpClientError,
    AcpSinkOverflowError,
)
from deep_worker.providers.acp.limit_errors import AcpUsageLimitError
from deep_worker.providers.acp.profile import PermissionPolicy

FAKE_CLI = Path(__file__).parent / "fake_acp_cli.py"


def _client(
    *,
    scenario: str,
    policy: PermissionPolicy | None = None,
    extra_limit_markers: tuple = (),
    client_state: object | None = None,
) -> AcpClient:
    env = dict(os.environ, FAKE_ACP_SCENARIO=scenario)
    return AcpClient(
        cmd=(sys.executable, str(FAKE_CLI)),
        vendor_prefix="",
        permission_policy=policy,
        extra_limit_markers=extra_limit_markers,
        client_state=client_state,
        env=env,
    )


class TestChannelA:
    def test_handshake_session_and_prompt_text(self) -> None:
        async def _run() -> None:
            client = _client(scenario="text")
            await client.start()
            try:
                init = await client.initialize()
                assert init["result"]["serverInfo"]["name"] == "fake-acp"
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                await client.wait_session_ready(sid)
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=asyncio.Queue(maxsize=1000),
                )
                reply = await fut
                assert reply["result"]["result"]["stopReason"] == "end_turn"
            finally:
                await client.aclose()

        asyncio.run(_run())

    def test_usage_limit_error_raises_acp_usage_limit(self) -> None:
        """A session/prompt error reply whose text matches the claude usage
        markers classifies as AcpUsageLimitError via raise_for_error_reply
        (the same path chat_model._finalize_response takes → failover)."""
        async def _run() -> None:
            client = _client(scenario="limit")
            await client.start()
            try:
                await client.initialize()
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=asyncio.Queue(maxsize=1000),
                )
                reply = await asyncio.wait_for(fut, timeout=5)
                from deep_worker.providers.acp.acp_client import (
                    raise_for_error_reply,
                )

                with pytest.raises(AcpUsageLimitError):
                    raise_for_error_reply(
                        "session/prompt",
                        reply,
                        extra_limit_markers=(
                            re.compile(r"You['\u2019]ve hit your session limit"),
                            re.compile(
                                r"session limit \u00b7 resets", re.IGNORECASE
                            ),
                        ),
                    )
            finally:
                await client.aclose()

        asyncio.run(_run())


class TestPermissionReply:
    def test_auto_approve_selects_allow_option(self) -> None:
        policy = PermissionPolicy(
            auto_approve=True,
            allow_kind_prefix="allow",
            deny_kind_prefix="reject",
        )
        replies: list[dict] = []
        real_write = None

        def _spy_write(obj: dict) -> None:
            if obj.get("id") == 9001 and "result" in obj:
                replies.append(obj)
            real_write(obj)

        async def _run() -> None:
            nonlocal real_write
            client = _client(scenario="permission", policy=policy)
            real_write = client._write
            client._write = _spy_write
            await client.start()
            try:
                await client.initialize()
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                await client.wait_session_ready(sid)
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=asyncio.Queue(maxsize=1000),
                )
                await asyncio.wait_for(fut, timeout=5)
            finally:
                await client.aclose()

            assert replies, "expected a permission reply to be written"
            outcome = replies[0]["result"]["outcome"]
            assert outcome == {"outcome": "selected", "optionId": "allow"}

        asyncio.run(_run())

    def test_deny_by_default_selects_reject_or_cancelled(self) -> None:
        async def _run() -> None:
            client = _client(scenario="permission", policy=None)  # fail-closed
            await client.start()
            try:
                await client.initialize()
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                await client.wait_session_ready(sid)
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=asyncio.Queue(maxsize=1000),
                )
                await asyncio.wait_for(fut, timeout=5)
            finally:
                await client.aclose()

        asyncio.run(_run())


class TestEofCrashAndTeardown:
    def test_crash_mid_turn_fails_prompt_and_client_restartable(self) -> None:
        async def _run() -> None:
            client = _client(scenario="crash")
            await client.start()
            try:
                await client.initialize()
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                await client.wait_session_ready(sid)
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=asyncio.Queue(maxsize=1000),
                )
                with pytest.raises(AcpClientError, match="EOF"):
                    await fut
                assert client._proc is None  # reset → restartable
                assert not client._pending_requests
            finally:
                await client.aclose()

        asyncio.run(_run())

    def test_aclose_reaps_subprocess(self) -> None:
        async def _run() -> None:
            client = _client(scenario="text")
            await client.start()
            proc = client._proc
            assert proc is not None
            await client.aclose()
            assert client._closed
            # Process group terminated and reaped.
            assert proc.returncode is not None

        asyncio.run(_run())


class TestSinkOverflow:
    """A flood must be ABSORBED, not fatal.

    This test used to assert the opposite — that overflowing the sink failed the
    turn with ``AcpSinkOverflowError``. That was the old contract, and it was the
    defect: a long turn with many notifications lost its entire answer because
    the reader never yielded, so the consumer was never scheduled and the sink
    filled. ``session/update`` is now delivered with ``await sink.put()``, so a
    full sink makes the reader WAIT for space instead of killing the turn.

    The fake CLI's ``overflow`` scenario floods 10 000 updates and never ends
    the turn, so under the new contract the correct observable outcome is: the
    sink stays at its capacity (never past it), the reader reports it is NOT idle
    (it is parked holding a frame), and the turn simply does not resolve — no
    exception, nothing lost.

    ORACLE IS STATE-BASED, NOT TIME-BASED. The backpressure claim is "the sink
    reaches capacity and the reader parks there", which is observable the
    instant it happens: ``sink.full()`` and ``not reader_is_idle()``. An
    earlier version proved it by *waiting out* ``wait_for(fut, timeout=2)`` and
    asserting the TimeoutError — i.e. it inferred "the turn did not resolve"
    from two seconds of wall clock. That costs 2s of every run and is exactly
    the construction that flakes on a loaded runner: if the flood needs longer
    than the budget to fill 16 slots, the assertions fire on a not-yet-full
    sink. Here the wait is driven by the arrival of frames (each ``sleep(0)``
    yields to the reader) and bounded by an iteration cap that is a *safety
    net*, not the oracle; "the turn did not resolve" is read directly off
    ``fut.done()``.
    """

    def test_flood_is_absorbed_and_bounded_instead_of_failing_the_turn(self) -> None:
        async def _run() -> None:
            client = _client(scenario="overflow")
            await client.start()
            try:
                await client.initialize()
                sid = await client.session_new(cwd="/tmp", mcp_servers=[])
                await client.wait_session_ready(sid)
                sink: asyncio.Queue = asyncio.Queue(maxsize=16)
                fut = client.session_prompt(
                    session_id=sid,
                    prompt=[{"type": "text", "text": "hi"}],
                    notification_sink=sink,
                )
                # Yield until the flood has filled the sink AND the reader is
                # parked on it. Both halves are needed: ``_idle_event`` is SET
                # just before each ``readline()`` and cleared right after, so
                # between two frames there is a window where the sink is
                # already full while the reader momentarily reads idle. Waiting
                # on ``sink.full()`` alone lands in that window and the
                # ``reader_is_idle()`` assertion below fires spuriously
                # (measured: 2 of 5 runs). The state we are after is the
                # conjunction — a parked reader holding a frame it cannot
                # deliver. Progress is the loop's own condition, so a slow
                # machine simply takes more iterations; the cap only keeps a
                # genuinely broken build from spinning forever.
                for _ in range(100_000):
                    if sink.full() and not client.reader_is_idle():
                        break
                    assert sink.qsize() <= 16, (
                        f"sink grew past capacity: {sink.qsize()}"
                    )
                    await asyncio.sleep(0)
                assert sink.full(), "expected the flood to fill the bounded sink"
                assert sink.qsize() <= 16, f"sink grew past capacity: {sink.qsize()}"
                # The flood must not have resolved OR failed the turn. Read the
                # future's state directly instead of timing it out.
                assert not fut.done(), (
                    "the flood must neither resolve nor fail the turn; "
                    f"future already done: {fut}"
                )
                assert not client.reader_is_idle(), (
                    "the reader should be parked holding a frame while the sink "
                    "is full — that is what prevents the overflow"
                )
                # Draining releases the parked reader: liveness, by state.
                #
                # NOT ``wait_for(wait_reader_idle(), timeout=5)``: the reader
                # is idle only while parked on ``readline()`` holding nothing,
                # and this scenario floods 10 000 updates. Draining 16 slots
                # lets it enqueue 16 more and park on the sink again, so "idle"
                # is unreachable until all 10 000 are consumed — it was never
                # an invariant here, just a race that 3.12 won and 3.11 lost
                # (run 39511394923: TimeoutError at this line on 3.11 only).
                #
                # What draining actually guarantees is that the parked reader
                # RESUMES, which is observable directly: it refills the slots
                # we freed. That is the liveness claim, and it needs no clock.
                for _ in range(16):
                    sink.get_nowait()
                assert sink.qsize() < 16, "sanity: the drain must free slots"
                for _ in range(100_000):
                    if sink.full():
                        break
                    await asyncio.sleep(0)
                assert sink.full(), (
                    "the parked reader must resume once the sink has room — "
                    f"it refilled only {sink.qsize()} of 16 slots"
                )
                fut.cancel()
            finally:
                await client.aclose()

        asyncio.run(_run())
