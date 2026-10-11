"""Progressive ``rawInput`` streaming — a proxy MCP tool must not be
dispatched with a partially-streamed argument set.

REGRESSION for an owner-observed production failure: every deep-worker MCP
tool appeared to "hang" for a claude-code agent, while its local built-ins
worked. The agent log showed, for the SAME call:

    no pending MCP handler for 'write_file' (toolCallId=...) within 10s
    TOOL_ERROR: 1 validation error for write_file
      content: Field required [type=missing, input_value={'file_path': ...}]

The tool never ran and the model was left waiting, so from the operator's seat
the tool channel was simply dead.

ROOT CAUSE, captured live from the pinned adapter (0.68.0) rather than
assumed — the real update sequence for one Write call is:

    tool_call        rawInput={}                                  status=pending
    tool_call_update rawInput={"file_path": "..."}                 status=None
    tool_call_update rawInput={"file_path": "...", "content": ...} status=None
    tool_call_update rawInput=None                                 status=completed

``rawInput`` is streamed PROGRESSIVELY across several refine events. The
implementation assumed a single refine ("rawInput={} then the real args"), so
it released the hold on the FIRST refine and dispatched with only
``file_path`` — the required ``content`` had not streamed yet. The completing
refine then arrived for a call that had already been dispatched.

The guard only rejected an EMPTY rawInput, so a partial-but-non-empty one
passed straight through. The fix must hold a streaming call until its
arguments actually satisfy the tool's declared schema.
"""

from __future__ import annotations

import asyncio

from deep_worker.providers.acp.chat_model import (
    AcpChatModel,
    _required_args_present,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker.providers.acp.profile import AcpProfile

PROXY_SERVER = "deep_worker_acp_proxy_1"

# Verbatim from the live capture (a recorded adapter trace),
# trimmed only in the size of the content payload.
LIVE_CONTENT = "BANANA\n" * 8


class _FakeProxy:
    """Minimal stand-in for the in-process proxy MCP server."""

    def __init__(self, server_name: str = PROXY_SERVER) -> None:
        self.server_name = server_name


def _profile() -> AcpProfile:
    return AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read", "Write"}),
        tool_name_from_update="from_claude_code_meta",
    )


def _model() -> AcpChatModel:
    """Build the model without running its constructor (no subprocess)."""
    model = object.__new__(AcpChatModel)
    model._profile = _profile()
    model._proxy_mcp = _FakeProxy()
    model._client_state = AcpClientState()
    # write_file is a deep-worker MCP tool: content is REQUIRED.
    model._tools_metadata = [
        {
            "name": "write_file",
            "description": "Write a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "list_jobs",
            "description": "List jobs",
            "input_schema": {
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
        {
            "name": "send_message_to_agent",
            "description": "Send a message",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["agent_name", "message"],
            },
        },
    ]
    return model


def _mcp_name(tool: str) -> str:
    return f"mcp__{PROXY_SERVER}__{tool}"


def _pending(tcid: str, tool: str) -> dict:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": "Preparing file…",
        "rawInput": {},
        "status": "pending",
        "_meta": {"claudeCode": {"toolName": _mcp_name(tool)}},
    }


def _refine(tcid: str, tool: str, args: dict) -> dict:
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tcid,
        "title": f"Write {tool}",
        "rawInput": args,
        "_meta": {"claudeCode": {"toolName": _mcp_name(tool)}},
    }


def _terminal(tcid: str, tool: str, status: str = "completed") -> dict:
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tcid,
        "title": None,
        "rawInput": None,
        "status": status,
        "_meta": {"claudeCode": {"toolName": _mcp_name(tool)}},
    }


def _notif(upd: dict) -> dict:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"update": upd}}


def _drive(model: AcpChatModel, state: ThreadState, updates: list[dict]) -> list[dict]:
    """Feed updates through _apply_update exactly as the pump does."""

    async def _runner() -> list[dict]:
        text: list[str] = []
        calls: list[dict] = []
        for upd in updates:
            model._apply_update(state, "thread-1", _notif(upd), text, calls)
        return calls

    return asyncio.run(_runner())


class TestProgressiveRawInput:
    def test_partial_args_are_not_dispatched(self) -> None:
        """THE DEFECT: a refine carrying only some required args must not run.

        Dispatching here is what produced the owner-visible
        "content: Field required" error and the dead tool channel.
        """
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t1", "write_file"),
                _refine("t1", "write_file", {"file_path": "/tmp/x.txt"}),
            ],
        )
        assert calls == [], (
            "tool dispatched with an incomplete argument set: "
            f"{calls!r} — 'content' had not streamed yet"
        )

    def test_call_dispatches_once_args_complete(self) -> None:
        """The completing refine must release the held call, exactly once."""
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t1", "write_file"),
                _refine("t1", "write_file", {"file_path": "/tmp/x.txt"}),
                _refine(
                    "t1",
                    "write_file",
                    {"file_path": "/tmp/x.txt", "content": LIVE_CONTENT},
                ),
            ],
        )
        assert len(calls) == 1, f"expected exactly one dispatch, got {calls!r}"
        assert calls[0]["name"] == "write_file"
        assert calls[0]["args"] == {
            "file_path": "/tmp/x.txt",
            "content": LIVE_CONTENT,
        }

    def test_full_live_sequence_dispatches_complete_args(self) -> None:
        """Replay of the captured live trace, terminal update included."""
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t1", "write_file"),
                _refine("t1", "write_file", {"file_path": "/tmp/x.txt"}),
                _refine(
                    "t1",
                    "write_file",
                    {"file_path": "/tmp/x.txt", "content": LIVE_CONTENT},
                ),
                _terminal("t1", "write_file"),
            ],
        )
        assert len(calls) == 1
        assert calls[0]["args"].get("content") == LIVE_CONTENT

    def test_send_message_partial_not_dispatched(self) -> None:
        """The owner's other symptom: send_message_to_agent lost `message`."""
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t2", "send_message_to_agent"),
                _refine("t2", "send_message_to_agent", {"agent_name": "agent_a"}),
            ],
        )
        assert calls == [], f"dispatched without 'message': {calls!r}"

    def test_send_message_dispatches_when_complete(self) -> None:
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t2", "send_message_to_agent"),
                _refine("t2", "send_message_to_agent", {"agent_name": "agent_a"}),
                _refine(
                    "t2",
                    "send_message_to_agent",
                    {"agent_name": "agent_a", "message": "hi"},
                ),
            ],
        )
        assert len(calls) == 1
        assert calls[0]["args"] == {"agent_name": "agent_a", "message": "hi"}

    def test_single_refine_with_complete_args_still_works(self) -> None:
        """The original (single-refine) shape must keep working unchanged."""
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t3", "write_file"),
                _refine(
                    "t3",
                    "write_file",
                    {"file_path": "/tmp/y.txt", "content": "hello"},
                ),
            ],
        )
        assert len(calls) == 1
        assert calls[0]["args"]["content"] == "hello"

    def test_terminal_without_complete_args_releases_hold(self) -> None:
        """A call whose args never complete must not leak the hold forever."""
        model, state = _model(), ThreadState()
        calls = _drive(
            model,
            state,
            [
                _pending("t4", "write_file"),
                _refine("t4", "write_file", {"file_path": "/tmp/z.txt"}),
                _terminal("t4", "write_file", status="failed"),
            ],
        )
        assert calls == []
        assert "t4" not in state.refine_pending_tool_calls, (
            "hold leaked after a terminal status — the set would grow unbounded"
        )


class TestRequiredArgsPresentFailOpen:
    """Contract of the schema helper (round-5 review point 3).

    Fail-OPEN means: if we don't understand the schema, we do NOT hold a
    call hostage — the empty-rawInput guard still catches pending calls, and
    releasing here just preserves the pre-fix behaviour for shapes the
    helper cannot reason about.
    """

    def test_missing_schema_is_satisfied(self) -> None:
        assert _required_args_present(None, {}) is True

    def test_non_dict_schema_is_satisfied(self) -> None:
        assert _required_args_present("not-a-schema", {}) is True  # type: ignore[arg-type]

    def test_no_required_is_satisfied(self) -> None:
        assert _required_args_present({"type": "object"}, {}) is True

    def test_empty_required_is_satisfied(self) -> None:
        assert _required_args_present({"required": []}, {}) is True

    def test_non_list_required_is_satisfied(self) -> None:
        assert _required_args_present({"required": "content"}, {}) is True

    def test_malformed_required_entries_fail_open(self) -> None:
        """``required=[123]`` is nonsense — we release rather than wedge.

        Documented contract: an unknown-typed ``required`` entry means the
        schema is untrusted end-to-end, so we treat the whole check as
        satisfied. Anything else would silently hold calls indefinitely
        whenever a third-party tool exposed a slightly-off schema.
        """
        assert _required_args_present({"required": [123]}, {}) is True
        assert (
            _required_args_present(
                {"required": ["file_path", 42]}, {"file_path": "/tmp/x"}
            )
            is True
        )

    def test_present_required_args_are_satisfied(self) -> None:
        schema = {"required": ["file_path", "content"]}
        assert _required_args_present(
            schema, {"file_path": "/tmp/x", "content": "hi"}
        ) is True

    def test_missing_required_arg_is_not_satisfied(self) -> None:
        schema = {"required": ["file_path", "content"]}
        assert _required_args_present(
            schema, {"file_path": "/tmp/x"}
        ) is False

    def test_none_value_counts_as_missing(self) -> None:
        """A key present with ``None`` cannot satisfy the schema — otherwise
        a stream that briefly carried ``"content": null`` would dispatch."""
        schema = {"required": ["content"]}
        assert _required_args_present(schema, {"content": None}) is False


class TestOptionalArgsSurviveStreaming:
    """ROUND 8 DEFECT (reported by a tester, reproduced live): the hold is
    released as soon as the REQUIRED args are present, so any OPTIONAL
    argument that streams in a LATER refine is silently dropped.

    The reporter's controlled experiment, on a live agent:

        read_file(file_path=..., offset=100, limit=3)   -> returned lines 1-100
        read_file(offset=100, limit=3, file_path=...)   -> returned lines 101-103

    Key ORDER in the model's JSON decides the outcome — the signature of
    progressive parsing. Blast radius is wide and SILENT: no validation
    error, the tool just runs with fewer arguments than the model asked for
    (``execute`` without ``timeout``, ``grep`` without filters, ``edit_file``
    without ``replace_all``).

    THE FIX is at correlation time, not mapping time: the MCP ``tools/call``
    payload is complete by construction, so it is the authoritative source.
    These tests therefore drive ``_correlate_proxy_tool_calls`` (what the
    real turn does before returning its AIMessage), not ``_map_tool_call``
    in isolation.
    """

    @staticmethod
    def _correlate(model: AcpChatModel, tool_calls: list[dict], parked: dict):
        """Run correlation against a proxy whose handler holds ``parked``."""

        class _Proxy:
            server_name = PROXY_SERVER

            async def wait_for_pending_calls(self, *, tool_name, arguments, timeout):
                if tool_name != parked["name"]:
                    raise asyncio.TimeoutError
                return [parked]

        model._proxy_mcp = _Proxy()
        state = AcpClientState()
        state.proxy_call_id_by_acp_id = {}
        asyncio.run(model._correlate_proxy_tool_calls(state, tool_calls))
        return tool_calls

    def test_optional_args_recovered_from_handler(self) -> None:
        """The mapped call lost offset/limit; the handler has them."""
        model = _model()
        mapped = [{
            "id": "t9", "name": "read_file",
            "args": {"file_path": "/tmp/big.txt"}, "type": "tool_call",
        }]
        out = self._correlate(model, mapped, {
            "id": "h1", "name": "read_file",
            "args": {"file_path": "/tmp/big.txt", "offset": 100, "limit": 3},
        })
        assert out[0]["args"] == {
            "file_path": "/tmp/big.txt", "offset": 100, "limit": 3,
        }, (
            "optional args streamed after the required set were DROPPED — "
            "this is the reporter's lines-1-100 instead of lines-101-103"
        )

    def test_no_required_args_tool_recovers_optionals(self) -> None:
        """Empty ``required`` is the worst case: the hold never engages, so
        the FIRST non-empty refine wins and later args are lost."""
        model = _model()
        mapped = [{
            "id": "t11", "name": "list_jobs",
            "args": {"verbose": True}, "type": "tool_call",
        }]
        out = self._correlate(model, mapped, {
            "id": "h2", "name": "list_jobs",
            "args": {"verbose": True, "limit": 5},
        })
        assert out[0]["args"] == {"verbose": True, "limit": 5}

    def test_conflicting_handler_args_are_not_adopted(self) -> None:
        """Correlation matches by tool NAME only, so a CONCURRENT call to the
        same tool can park first. Splicing two different calls together would
        be worse than the bug: refuse when a shared key disagrees."""
        model = _model()
        mapped = [{
            "id": "t12", "name": "read_file",
            "args": {"file_path": "/tmp/A.txt"}, "type": "tool_call",
        }]
        out = self._correlate(model, mapped, {
            "id": "h3", "name": "read_file",
            "args": {"file_path": "/tmp/B.txt", "offset": 5},
        })
        assert out[0]["args"] == {"file_path": "/tmp/A.txt"}, (
            "adopted a DIFFERENT call's arguments"
        )

    def test_handler_missing_a_mapped_key_is_not_adopted(self) -> None:
        """Handler payload must be a strict SUPERSET to be trusted."""
        model = _model()
        mapped = [{
            "id": "t13", "name": "read_file",
            "args": {"file_path": "/tmp/A.txt", "limit": 9}, "type": "tool_call",
        }]
        out = self._correlate(model, mapped, {
            "id": "h4", "name": "read_file", "args": {"file_path": "/tmp/A.txt"},
        })
        assert out[0]["args"] == {"file_path": "/tmp/A.txt", "limit": 9}

    def test_identical_args_are_left_alone(self) -> None:
        model = _model()
        mapped = [{
            "id": "t14", "name": "read_file",
            "args": {"file_path": "/tmp/A.txt"}, "type": "tool_call",
        }]
        out = self._correlate(model, mapped, {
            "id": "h5", "name": "read_file", "args": {"file_path": "/tmp/A.txt"},
        })
        assert out[0]["args"] == {"file_path": "/tmp/A.txt"}
