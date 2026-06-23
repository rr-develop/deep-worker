"""TDD for ``GrokAcpClient`` — the long-lived ``grok agent stdio``
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

from grok_provider.acp_client import (  # noqa: E402
    GrokAcpClient,
    GrokConcurrentPromptError,
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
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    client = GrokAcpClient(client_state=None)
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
        """Regression: Grok rejects invented option ids and cancels the turn."""
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
                        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
                        {"optionId": "approved-for-session", "name": "Allow", "kind": "allow_always"},
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

    async def test_permission_falls_back_to_first_offered_option(
        self, client_with_proc
    ) -> None:
        """Never invent ``allow`` when an agent uses unfamiliar option kinds."""
        _client, proc = client_with_proc
        proc.stdout.feed_line(
            {
                "jsonrpc": "2.0",
                "id": 9004,
                "method": "session/request_permission",
                "params": {
                    "options": [
                        {"optionId": "proceed", "name": "Proceed", "kind": "custom"}
                    ]
                },
            }
        )
        await asyncio.sleep(0.02)
        replies = [m for m in proc.stdin.lines() if m.get("id") == 9004]
        assert replies[-1]["result"]["outcome"]["optionId"] == "proceed"


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
        """Regression (observed live, 2026-06-17): session_prompt bypasses
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
        with pytest.raises(GrokConcurrentPromptError):
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


class TestClose:
    async def test_aclose_closes_stdin_and_cancels(self, monkeypatch) -> None:
        proc = _FakeProc()

        async def fake_create(*a, **k):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        client = GrokAcpClient(client_state=None)
        await client.start()
        # A dangling pending request future.
        pend = asyncio.create_task(client.initialize())
        await asyncio.sleep(0.01)
        await client.aclose()
        assert proc.stdin.closed is True
        with pytest.raises((asyncio.CancelledError, Exception)):
            await pend
