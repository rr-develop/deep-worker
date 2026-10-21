"""TDD for ``AcpClient`` — the long-lived ``grok agent stdio``
JSON-RPC channel.

We never launch the real grok binary. Instead we drive the client with a
fake ``asyncio.subprocess.Process``-like object exposing ``stdin`` (a sink
we can read what the client wrote) and ``stdout`` (a stream we feed lines
into). This lets us assert the wire protocol and the reader_task routing
deterministically.

Covered (DESIGN §9.2-§9.4):
 - initialize / authenticate / session_new / session_prompt request shapes
 - reader resolves the matching pending_requests Future by id
 - notifications routed to the registered per-session sink queue
 - vendor ``_x.ai/*`` notifications dropped
 - agent->client fs/terminal requests answered with JSON-RPC -32601
 - aclose() cancels reader + pending futures and closes stdin
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.acp.acp_client import (  # noqa: E402
    AcpClient,
    AcpClientError,
    AcpConcurrentPromptError,
    raise_for_error_reply,
)
from deep_worker.providers.acp.limit_errors import AcpUsageLimitError  # noqa: E402
from deep_worker.providers.grok_provider.profile import build_profile  # noqa: E402


def _grok_client(*, auto_approve: bool = True) -> AcpClient:
    """Build an ``AcpClient`` carrying the GROK profile's parameters.

    Since 3fe834d1 the ACP client is provider-agnostic: grok's command, its
    ``_x.ai/`` vendor namespace, its ``allow-*``/``deny-*`` permission kinds
    and its ``authenticate`` requirement all arrive as constructor arguments
    from :func:`deep_worker.providers.grok_provider.profile.build_profile`.
    This file asserts GROK behaviour, so the values are taken from the live
    grok profile rather than restated here — a profile change that would alter
    grok's wire behaviour shows up as a failure in these tests instead of
    silently leaving them asserting claude-code defaults.

    ``auto_approve`` mirrors deep-worker's ``-sp`` flag; the permission tests
    assert the approval path, which only exists with the flag set.
    """
    profile = build_profile(auto_approve=auto_approve)
    return AcpClient(
        client_state=None,
        cmd=profile.cmd,
        vendor_prefix=profile.vendor_prefix,
        permission_policy=profile.permission_policy,
        extra_limit_markers=profile.extra_limit_markers,
        requires_authenticate=profile.requires_authenticate,
        session_new_validator=profile.session_new_validator,
    )


class _FakeStdin:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def lines(self) -> list[dict]:
        out = []
        for raw in bytes(self.buffer).splitlines():
            if raw.strip():
                out.append(json.loads(raw))
        return out


class _FakeStdout:
    """An asyncio-readline-able stream we push lines into."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    def feed_line(self, obj: dict) -> None:
        self._q.put_nowait((json.dumps(obj) + "\n").encode())

    def feed_eof(self) -> None:
        self._q.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._q.get()


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.stderr = _FakeStdout()
        self._waited = False
        self.returncode = None
        # ``aclose`` tears the CLI down by process GROUP (§5.3):
        # ``os.killpg(os.getpgid(proc.pid), SIGTERM)``. A fake proc must still
        # own a ``pid``, and it must be one that can never name a real process
        # on this host — ``-1`` raises ProcessLookupError, which the client
        # suppresses before falling back to ``proc.terminate()`` below. So the
        # real teardown path runs end to end and signals nothing live.
        self.pid = -1
        # Filled in by the spawn stub (see ``client_with_proc``).
        self.spawn_argv: tuple = ()
        self.spawn_kwargs: dict = {}

    async def wait(self) -> int:
        self._waited = True
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture()
async def client_with_proc(monkeypatch):
    proc = _FakeProc()

    async def fake_create(*args, **kwargs):
        # Record the argv the client spawned: the command is a constructor
        # argument since 3fe834d1, so this is the only place a test can see
        # that it is GROK's CLI being launched and not another provider's.
        proc.spawn_argv = tuple(args)
        proc.spawn_kwargs = kwargs
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    client = _grok_client()
    await client.start()
    try:
        yield client, proc
    finally:
        await client.aclose()


async def _reply_to_last(proc: _FakeProc, result: dict) -> None:
    """Feed a JSON-RPC reply matching the id of the last request written."""
    last = proc.stdin.lines()[-1]
    proc.stdout.feed_line({"jsonrpc": "2.0", "id": last["id"], "result": result})


class TestHandshake:
    async def test_start_spawns_the_grok_stdio_command(
        self, client_with_proc
    ) -> None:
        """``start()`` must launch GROK's ACP command.

        Before 3fe834d1 the argv was hardcoded in grok's own forked client;
        it now arrives from ``build_profile().cmd``. Asserting it here is what
        keeps this file a test of the grok channel rather than of whichever
        command the shared client happens to be handed.
        """
        _client, proc = client_with_proc
        assert proc.spawn_argv == ("grok", "agent", "stdio")
        # Large-line limit is mandatory — a 64 KiB default wedges the reader
        # on grok's big tool-result lines (module docstring).
        assert proc.spawn_kwargs["limit"] == 8 * 1024 * 1024
        # Own process group, so teardown can killpg the whole tree (§5.3).
        assert proc.spawn_kwargs["start_new_session"] is True

    async def test_initialize_request_shape(self, client_with_proc) -> None:
        client, proc = client_with_proc
        task = asyncio.create_task(client.initialize())
        await asyncio.sleep(0.01)
        sent = proc.stdin.lines()[-1]
        assert sent["method"] == "initialize"
        assert sent["params"]["protocolVersion"] == 1
        # fs / terminal declared disabled
        caps = sent["params"]["clientCapabilities"]
        assert caps["terminal"] is False
        await _reply_to_last(proc, {"protocolVersion": 1})
        await asyncio.wait_for(task, timeout=1)

    async def test_authenticate_uses_cached_token(self, client_with_proc) -> None:
        client, proc = client_with_proc
        task = asyncio.create_task(client.authenticate())
        await asyncio.sleep(0.01)
        sent = proc.stdin.lines()[-1]
        assert sent["method"] == "authenticate"
        assert sent["params"]["methodId"] == "cached_token"
        await _reply_to_last(proc, {})
        await asyncio.wait_for(task, timeout=1)

    async def test_session_new_returns_session_id(self, client_with_proc) -> None:
        client, proc = client_with_proc
        task = asyncio.create_task(
            client.session_new(cwd="/tmp", mcp_servers=[])
        )
        await asyncio.sleep(0.01)
        sent = proc.stdin.lines()[-1]
        assert sent["method"] == "session/new"
        assert sent["params"]["cwd"] == "/tmp"
        await _reply_to_last(proc, {"sessionId": "sess-123"})
        sid = await asyncio.wait_for(task, timeout=1)
        assert sid == "sess-123"


class TestNotificationRouting:
    async def test_update_routed_to_session_sink(self, client_with_proc) -> None:
        client, proc = client_with_proc
        sink: asyncio.Queue = asyncio.Queue()
        client.register_sink("sess-A", sink)
        proc.stdout.feed_line(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-A",
                    "update": {"sessionUpdate": "agent_message_chunk",
                               "content": {"text": "hi"}},
                },
            }
        )
        notif = await asyncio.wait_for(sink.get(), timeout=1)
        assert notif["params"]["update"]["content"]["text"] == "hi"

    async def test_vendor_notification_dropped(self, client_with_proc) -> None:
        client, proc = client_with_proc
        # The client must be carrying GROK's vendor namespace, not another
        # provider's: the prefix is a constructor argument since 3fe834d1, so
        # a profile that stopped supplying ``_x.ai/`` would otherwise leave
        # this test quietly asserting some other CLI's drop rule.
        assert client._vendor_prefix == "_x.ai/"
        sink: asyncio.Queue = asyncio.Queue()
        client.register_sink("sess-A", sink)
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "method": "_x.ai/queue/changed",
             "params": {"sessionId": "sess-A"}}
        )
        # Then a real update, to prove ordering / that the vendor one was skipped.
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "method": "session/update",
             "params": {"sessionId": "sess-A",
                        "update": {"sessionUpdate": "agent_message_chunk",
                                   "content": {"text": "real"}}}}
        )
        notif = await asyncio.wait_for(sink.get(), timeout=1)
        assert notif["params"]["update"]["content"]["text"] == "real"
        assert sink.empty()


class TestAgentRequests:
    async def test_fs_read_answered_with_method_not_found(
        self, client_with_proc
    ) -> None:
        client, proc = client_with_proc
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "id": 9001, "method": "fs/read_text_file",
             "params": {"path": "/etc/passwd"}}
        )
        await asyncio.sleep(0.02)
        replies = [m for m in proc.stdin.lines() if m.get("id") == 9001]
        assert replies, "client must reply to the agent request"
        assert replies[-1]["error"]["code"] == -32601

    async def test_terminal_request_method_not_found(self, client_with_proc) -> None:
        client, proc = client_with_proc
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "id": 9002, "method": "terminal/create",
             "params": {}}
        )
        await asyncio.sleep(0.02)
        replies = [m for m in proc.stdin.lines() if m.get("id") == 9002]
        assert replies[-1]["error"]["code"] == -32601

    async def test_permission_selects_offered_allow_option(
        self, client_with_proc
    ) -> None:
        """Regression: Grok rejects invented option ids and cancels the turn.

        The client must answer with an optionId grok ACTUALLY offered. The
        option kinds here are grok's own ``allow-*``/``deny-*`` family (see
        ``grok_provider.profile.build_profile``), and the selected id is the
        real grok id ``approved-for-session`` — not the literal ``allow``
        that an inventing client would send.
        """
        _client, proc = client_with_proc
        proc.stdout.feed_line(
            {
                "jsonrpc": "2.0",
                "id": 9003,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "sess-A",
                    "toolCall": {"toolCallId": "tc-edit", "title": "StrReplace"},
                    "options": [
                        {"optionId": "reject-once", "name": "Reject", "kind": "deny-once"},
                        {"optionId": "approved-for-session", "name": "Allow", "kind": "allow-always"},
                    ],
                },
            }
        )
        await asyncio.sleep(0.02)
        replies = [m for m in proc.stdin.lines() if m.get("id") == 9003]
        assert replies[-1]["result"] == {
            "outcome": {
                "outcome": "selected",
                "optionId": "approved-for-session",
            }
        }

    async def test_permission_denied_without_sp_selects_grok_deny_option(
        self, monkeypatch
    ) -> None:
        """Without ``-sp`` NOTHING is approved, and the denial picks the id
        grok offered (DESIGN §4.6, incident 2026-08-24).

        This is the other half of "never invent an option id": the deny
        branch is bound to grok's ``deny_kind_prefix`` from the profile, so
        the reply carries grok's ``reject-once`` rather than a literal
        ``deny``. The forked client approved every request unconditionally,
        so this path had no coverage at all on the grok side.
        """
        proc = _FakeProc()

        async def fake_create(*args, **kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        client = _grok_client(auto_approve=False)
        await client.start()
        try:
            proc.stdout.feed_line(
                {
                    "jsonrpc": "2.0",
                    "id": 9005,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": "sess-A",
                        "toolCall": {"toolCallId": "tc-sh", "title": "run_terminal_command"},
                        "options": [
                            {"optionId": "reject-once", "name": "Reject", "kind": "deny-once"},
                            {"optionId": "approved-for-session", "name": "Allow", "kind": "allow-always"},
                        ],
                    },
                }
            )
            await asyncio.sleep(0.02)
            replies = [m for m in proc.stdin.lines() if m.get("id") == 9005]
            assert replies[-1]["result"] == {
                "outcome": {"outcome": "selected", "optionId": "reject-once"}
            }
        finally:
            await client.aclose()


class TestSessionPrompt:
    async def test_session_prompt_returns_response_future(
        self, client_with_proc
    ) -> None:
        client, proc = client_with_proc
        sink: asyncio.Queue = asyncio.Queue()
        fut = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "hello"}],
            notification_sink=sink,
        )
        assert isinstance(fut, asyncio.Future)
        await asyncio.sleep(0.01)
        sent = proc.stdin.lines()[-1]
        assert sent["method"] == "session/prompt"
        assert sent["params"]["sessionId"] == "sess-A"
        # Resolve it.
        await _reply_to_last(proc, {"stopReason": "end_turn"})
        resp = await asyncio.wait_for(fut, timeout=1)
        assert resp["result"]["stopReason"] == "end_turn"

    async def test_session_prompt_does_not_leak_pending_request(
        self, client_with_proc
    ) -> None:
        """Regression (agent_h review 2026-06-17): session_prompt bypasses
        _call's finally-pop, so the reply dispatch MUST pop the pending
        request — otherwise every turn leaks one stale entry."""
        client, proc = client_with_proc
        sink: asyncio.Queue = asyncio.Queue()
        fut = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "hi"}],
            notification_sink=sink,
        )
        await asyncio.sleep(0.01)
        assert len(client._pending_requests) == 1  # registered
        await _reply_to_last(proc, {"stopReason": "end_turn"})
        await asyncio.wait_for(fut, timeout=1)
        # Popped on reply — no leak.
        assert len(client._pending_requests) == 0

    async def test_second_prompt_for_same_session_is_rejected_without_write(
        self, client_with_proc
    ) -> None:
        client, proc = client_with_proc
        first_sink: asyncio.Queue = asyncio.Queue()
        first = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "first"}],
            notification_sink=first_sink,
        )
        writes_after_first = list(proc.stdin.lines())

        second_sink: asyncio.Queue = asyncio.Queue()
        with pytest.raises(AcpConcurrentPromptError):
            client.session_prompt(
                session_id="sess-A",
                prompt=[{"type": "text", "text": "second"}],
                notification_sink=second_sink,
            )

        assert proc.stdin.lines() == writes_after_first
        assert client._sinks["sess-A"] is first_sink
        assert client._active_prompts["sess-A"] is first

    async def test_prompt_slot_reopens_after_owner_finishes(
        self, client_with_proc
    ) -> None:
        client, proc = client_with_proc
        first = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "first"}],
            notification_sink=asyncio.Queue(),
        )
        await _reply_to_last(proc, {"stopReason": "end_turn"})
        await asyncio.wait_for(first, timeout=1)
        await asyncio.sleep(0)

        second = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "second"}],
            notification_sink=asyncio.Queue(),
        )
        assert client._active_prompts["sess-A"] is second
        assert proc.stdin.lines()[-1]["params"]["prompt"][0]["text"] == "second"

    async def test_stale_prompt_completion_preserves_new_owner(
        self, client_with_proc
    ) -> None:
        client, _proc = client_with_proc
        first_sink: asyncio.Queue = asyncio.Queue()
        first = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "first"}],
            notification_sink=first_sink,
        )
        replacement = asyncio.get_running_loop().create_future()
        replacement_sink: asyncio.Queue = asyncio.Queue()
        client._active_prompts["sess-A"] = replacement
        client._sinks["sess-A"] = replacement_sink

        first.set_result({"result": {"stopReason": "end_turn"}})
        await asyncio.sleep(0)

        assert client._active_prompts["sess-A"] is replacement
        assert client._sinks["sess-A"] is replacement_sink


class TestSessionCancelReleasesGuard:
    """RC-A (incident 2026-07-01 §9b/§10 — grok_provider root cause).

    ``session_cancel(session_id)`` used to only write ``session/cancel`` to
    grok's stdin without touching the local ``_active_prompts`` guard. If grok
    never wrote back a JSON-RPC reply for the outstanding ``session/prompt``
    (crash, wedge, mid-turn CancelledError, session/load discarding the
    request), the guard entry stayed forever and every subsequent
    ``session_prompt`` on the same session id raised
    ``AcpConcurrentPromptError`` synchronously — permanently poisoning the
    agent thread. Cancel means "abandon this prompt", so the local guard MUST
    be released too.
    """

    async def test_cancel_releases_active_prompt_guard(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        client, proc = client_with_proc
        sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "first"}],
            notification_sink=sink,
        )
        assert "sess-A" in client._active_prompts

        await client.session_cancel("sess-A")
        # Give the future's done_callback one loop tick to run.
        await asyncio.sleep(0)

        # Sticky guard MUST be gone after cancel — otherwise the next
        # session_prompt raises AcpConcurrentPromptError synchronously.
        assert "sess-A" not in client._active_prompts
        # Sink for the abandoned prompt is also gone (so a new turn's sink
        # can be registered without racing the stale one).
        assert "sess-A" not in client._sinks
        # And session/cancel was actually written to grok's stdin (we didn't
        # skip the wire message by mistake).
        methods = [ln.get("method") for ln in proc.stdin.lines()]
        assert "session/cancel" in methods

    async def test_next_prompt_after_cancel_is_accepted(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        client, proc = client_with_proc
        first_sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "wedged"}],
            notification_sink=first_sink,
        )
        # Grok never replies to the first prompt — cancel abandons it.
        await client.session_cancel("sess-A")
        await asyncio.sleep(0)

        # A brand-new turn on the same session id must NOT hit the guard.
        second_sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        fut = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "second"}],
            notification_sink=second_sink,
        )
        assert client._active_prompts["sess-A"] is fut
        assert client._sinks["sess-A"] is second_sink
        # Last wire message is the second session/prompt (not the cancel).
        prompts = [
            ln for ln in proc.stdin.lines()
            if ln.get("method") == "session/prompt"
        ]
        assert prompts[-1]["params"]["prompt"][0]["text"] == "second"

    async def test_cancel_does_not_touch_other_sessions(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        client, _proc = client_with_proc
        sink_a: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        sink_b: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        fut_a = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "a"}],
            notification_sink=sink_a,
        )
        fut_b = client.session_prompt(
            session_id="sess-B",
            prompt=[{"type": "text", "text": "b"}],
            notification_sink=sink_b,
        )

        await client.session_cancel("sess-A")
        await asyncio.sleep(0)

        # A is gone.
        assert "sess-A" not in client._active_prompts
        assert "sess-A" not in client._sinks
        # B is untouched — releasing A's guard must not race B's turn.
        assert client._active_prompts["sess-B"] is fut_b
        assert client._sinks["sess-B"] is sink_b
        # A's future was cancelled by session_cancel.
        assert fut_a.cancelled()

    async def test_cancel_none_or_unknown_is_safe_noop(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        client, proc = client_with_proc
        # Establish B so we can prove it survives noop cancels for None/unknown.
        sink_b: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        fut_b = client.session_prompt(
            session_id="sess-B",
            prompt=[{"type": "text", "text": "b"}],
            notification_sink=sink_b,
        )
        writes_before = list(proc.stdin.lines())

        await client.session_cancel(None)      # None → return immediately
        await client.session_cancel("nope")    # unknown → still safe

        # Unknown session_id: cancel MAY still emit a session/cancel wire
        # message (grok will just ignore it) but the client's guard state
        # for the other sessions must not change.
        assert client._active_prompts["sess-B"] is fut_b
        assert client._sinks["sess-B"] is sink_b
        # The None call must not have written anything.
        writes_after_none = writes_before + [
            ln for ln in proc.stdin.lines()[len(writes_before):]
            if ln.get("method") == "session/cancel"
            and ln["params"].get("sessionId") == "nope"
        ]
        assert writes_after_none == proc.stdin.lines()

    async def test_cancel_pops_pending_request_entry(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        """RC-1 (agent_h review). Cancelling an outstanding prompt must
        also drop its entry from ``_pending_requests`` — otherwise every
        wedged-then-cancelled prompt leaks an entry there, one per hot-loop
        retry, exactly in the incident scenario. The active future's
        done-callback (``release_prompt_owner``) is the natural place to
        clean it up, since ``fut.cancel()`` alone does not remove the
        dict entry."""
        client, _proc = client_with_proc
        sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "wedged"}],
            notification_sink=sink,
        )
        pending_before = list(client._pending_requests.keys())
        assert len(pending_before) == 1
        rid = pending_before[0]

        await client.session_cancel("sess-A")
        # Give the future's done-callback one loop tick to run.
        await asyncio.sleep(0)

        assert rid not in client._pending_requests
        assert client._pending_requests == {}

    async def test_cancel_pops_only_its_own_pending_entry(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        """Cancel on sess-A must not evict sess-B's pending JSON-RPC
        request — B's turn is still in flight and its future waiter would
        never resolve if the entry vanished."""
        client, _proc = client_with_proc
        sink_a: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        sink_b: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "a"}],
            notification_sink=sink_a,
        )
        fut_b = client.session_prompt(
            session_id="sess-B",
            prompt=[{"type": "text", "text": "b"}],
            notification_sink=sink_b,
        )
        # rids are assigned in call order.
        rids = list(client._pending_requests.keys())
        assert len(rids) == 2
        rid_a, rid_b = rids
        assert client._pending_requests[rid_b] is fut_b

        await client.session_cancel("sess-A")
        await asyncio.sleep(0)

        assert rid_a not in client._pending_requests
        # B's pending entry survives so its awaiter still resolves normally.
        assert client._pending_requests.get(rid_b) is fut_b

    async def test_late_reply_for_cancelled_rid_is_a_safe_noop(
        self, client_with_proc: tuple[AcpClient, "_FakeProc"]
    ) -> None:
        """If grok eventually writes a JSON-RPC reply for the cancelled
        prompt (e.g. its wedge clears seconds later), the reader must not
        crash or corrupt any newly-registered prompt on the same session.
        With the pending entry popped on cancel, the reply becomes an
        unknown-id dispatch and is dropped silently."""
        client, proc = client_with_proc
        sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "wedged"}],
            notification_sink=sink,
        )
        first_rid = list(client._pending_requests.keys())[0]

        await client.session_cancel("sess-A")
        await asyncio.sleep(0)

        # A fresh turn on the same session id (proves guard was released).
        second_sink: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        second_fut = client.session_prompt(
            session_id="sess-A",
            prompt=[{"type": "text", "text": "second"}],
            notification_sink=second_sink,
        )
        second_rid = client._next_id
        assert client._pending_requests[second_rid] is second_fut

        # Grok now dribbles in a late reply for the FIRST rid. Reader must
        # not resolve the new second_fut and must not raise.
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "id": first_rid, "result": {"stopReason": "cancelled"}}
        )
        await asyncio.sleep(0.02)

        assert not second_fut.done()
        # And the new turn's pending entry is still there for its own reply.
        assert client._pending_requests.get(second_rid) is second_fut


class TestRaiseForErrorReply:
    """Shared error extractor — single source of truth for both the
    ``_call`` path and the ``session/prompt`` terminal path (chat_model)."""

    def test_no_error_key_is_noop(self) -> None:
        # A normal result reply must pass through silently.
        raise_for_error_reply("session/prompt", {"id": 1, "result": {}})

    def test_billing_error_raises_usage_limit(self) -> None:
        reply = {"id": 1, "error": {"code": 402, "message": "Payment Required"}}
        with pytest.raises(AcpUsageLimitError):
            raise_for_error_reply("session/prompt", reply)

    def test_insufficient_credits_raises_usage_limit(self) -> None:
        reply = {"id": 1, "error": {"message": "insufficient credits on account"}}
        with pytest.raises(AcpUsageLimitError):
            raise_for_error_reply("session/prompt", reply)

    def test_generic_error_raises_acp_client_error(self) -> None:
        reply = {"id": 1, "error": {"code": -32602, "message": "bad params"}}
        with pytest.raises(AcpClientError) as excinfo:
            raise_for_error_reply("session/new", reply)
        # Not misclassified as a usage-limit error.
        assert not isinstance(excinfo.value, AcpUsageLimitError)

    def test_error_string_payload(self) -> None:
        # error may be a bare string, not a dict.
        with pytest.raises(AcpUsageLimitError):
            raise_for_error_reply("session/prompt", {"id": 1, "error": "402 payment required"})

    def test_nested_data_message_402_raises_usage_limit(self) -> None:
        """REAL production shape (test_grok2, incident 2026-07-04): the grok
        JSON-RPC error wraps the account-terminal 402 inside ``error.data`` and
        puts a generic ``"Internal error"`` in the top-level ``error.message``.

        The extractor MUST read ``error.data.message`` (not only the outer
        ``error.message``) or it misclassifies a billing cap as a generic
        ``AcpClientError`` — no failover, empty AIMessages. This is the
        exact reply copied verbatim from test_grok2.err.log."""
        reply = {
            "id": 5,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": {
                    "message": (
                        "API error (status 402 Payment Required): "
                        "Grok Build usage balance exhausted\n\n"
                        "Request URL: https://cli-chat-proxy.grok.com/v1/responses"
                    ),
                    "http_status": 402,
                },
            },
        }
        with pytest.raises(AcpUsageLimitError):
            raise_for_error_reply("session/prompt", reply)

    def test_nested_data_http_status_402_alone_raises_usage_limit(self) -> None:
        """Even if the nested ``data.message`` were opaque, an
        ``http_status == 402`` is itself account-terminal and must be
        classified as a usage/billing cap → failover."""
        reply = {
            "id": 7,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": {"message": "upstream said no", "http_status": 402},
            },
        }
        with pytest.raises(AcpUsageLimitError):
            raise_for_error_reply("session/prompt", reply)

    def test_nested_generic_error_still_acp_client_error(self) -> None:
        """A genuinely generic nested error (no billing/limit markers, non-402
        status) must NOT be misclassified as a usage limit — it stays a plain
        ``AcpClientError`` so we never burn a false failover step."""
        reply = {
            "id": 9,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": {"message": "invalid session id", "http_status": 400},
            },
        }
        with pytest.raises(AcpClientError) as excinfo:
            raise_for_error_reply("session/prompt", reply)
        assert not isinstance(excinfo.value, AcpUsageLimitError)


class TestCallErrorReplies:
    async def test_call_raises_usage_limit_on_billing(self, client_with_proc) -> None:
        client, proc = client_with_proc
        task = asyncio.create_task(client.initialize())
        await asyncio.sleep(0.01)
        last = proc.stdin.lines()[-1]
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "id": last["id"],
             "error": {"code": 402, "message": "Payment Required"}}
        )
        with pytest.raises(AcpUsageLimitError):
            await asyncio.wait_for(task, timeout=1)

    async def test_call_raises_client_error_on_generic(self, client_with_proc) -> None:
        client, proc = client_with_proc
        task = asyncio.create_task(client.session_new(cwd="/tmp", mcp_servers=[]))
        await asyncio.sleep(0.01)
        last = proc.stdin.lines()[-1]
        proc.stdout.feed_line(
            {"jsonrpc": "2.0", "id": last["id"],
             "error": {"code": -32602, "message": "missing field mcpServers"}}
        )
        with pytest.raises(AcpClientError):
            await asyncio.wait_for(task, timeout=1)


class TestClose:
    async def test_aclose_closes_stdin_and_cancels(self, monkeypatch) -> None:
        proc = _FakeProc()

        async def fake_create(*a, **k):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        client = _grok_client()
        await client.start()
        # A dangling pending request future.
        pend = asyncio.create_task(client.initialize())
        await asyncio.sleep(0.01)
        await client.aclose()
        assert proc.stdin.closed is True
        with pytest.raises((asyncio.CancelledError, Exception)):
            await pend
