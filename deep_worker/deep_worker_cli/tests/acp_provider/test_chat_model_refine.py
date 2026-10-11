"""Unit tests for claude-agent-acp streaming/refine handling in the shared
chat model.

claude-agent-acp surfaces a ``tool_call`` with ``rawInput={}`` (status
``pending``) and streams the REAL arguments in a follow-up
``tool_call_update`` WITHOUT a status (the "refine" event). Until the refine
event (or a terminal status) the call must NOT be handed to LangGraph —
running a tool with empty args would be a silent no-op at best.

Also covers the bounded built-in passthrough map (reject-new on overflow)
and the MCP-name-prefix dispatch (CASE A).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import CALL_ID_MAP_MAX_ENTRIES, ThreadState
from deep_worker.providers.acp.profile import AcpProfile


def _profile(
    style: str = "mcp_name_prefix",
    known_builtins: tuple[str, ...] = ("Bash", "Read"),
    provider: str = "claude-code",
) -> AcpProfile:
    return AcpProfile(
        provider=provider,
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style=style,
        known_builtins=frozenset(known_builtins),
        tool_name_from_update="from_claude_code_meta",
    )


def _model(
    profile: AcpProfile | None = None,
    proxy: object | None = None,
    *,
    known_builtins: tuple[str, ...] = ("Bash", "Read"),
) -> AcpChatModel:
    """Build an AcpChatModel WITHOUT running its constructor (no subprocess,
    no global-singleton mutation). Only the attributes used by the pure
    mapping methods are set."""
    model = object.__new__(AcpChatModel)
    model._profile = profile or _profile(known_builtins=known_builtins)
    model._proxy_mcp = proxy
    model._tools_metadata = []
    model._client_state = AcpClientState()
    return model


def _state() -> ThreadState:
    return ThreadState()


def _pending_tool_call(tcid: str, title: str = "Bash") -> dict:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": title,
        "rawInput": {},
        "status": "pending",
        "_meta": {"claudeCode": {"toolName": title}},
    }


def _refine_update(tcid: str, title: str, args: dict) -> dict:
    # tool_call_update WITHOUT status = refine event.
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tcid,
        "title": title,
        "rawInput": args,
        "_meta": {"claudeCode": {"toolName": title}},
    }


def _terminal_update(tcid: str, title: str, status: str = "completed") -> dict:
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tcid,
        "title": title,
        "rawInput": {},
        "status": status,
        "_meta": {"claudeCode": {"toolName": title}},
    }


def _notif(upd: dict) -> dict:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"update": upd}}


def _run_sync(fn, *args, **kwargs):
    """Run a sync callable inside a fresh running event loop.

    ``_map_tool_call`` / ``_apply_update`` register builtin passthrough
    Futures, which REQUIRE a live event loop (``get_event_loop`` outside a
    loop is deprecated/error). Calls that create no Future work fine without
    one, but wrapping is harmless and keeps the tests loop-safe.
    """

    async def _runner():
        return fn(*args, **kwargs)

    return asyncio.run(_runner())


class TestMapToolCall:
    def test_pending_tool_call_with_empty_args_not_mapped(self) -> None:
        """The pending tool_call (rawInput={}, status=pending) MUST NOT
        surface — running it would invoke the tool with no arguments."""
        model = _model()
        assert model._map_tool_call(_state(), _pending_tool_call("t1")) is None

    def test_refined_tool_call_maps_to_builtin(self) -> None:
        model = _model()
        tc = _run_sync(
            model._map_tool_call, _state(),
            _refine_update("t1", "Bash", {"command": "ls"}),
        )
        assert tc == {
            "id": "t1",
            "name": "Bash",
            "args": {"command": "ls"},
            "type": "tool_call",
        }

    def test_unknown_builtin_routed_via_generic_name(self) -> None:
        model = _model(known_builtins=("Bash",))
        tc = _run_sync(
            model._map_tool_call, _state(),
            _refine_update("t1", "WeirdTool", {"x": 1}),
        )
        assert tc is not None
        # The generic name is DERIVED from the profile's provider id
        # (sanitised: separators stripped), so it is asserted by derivation
        # rather than as a literal. Hardcoding it made this test fail on the
        # claude-code-acp -> claude-code rename even though production was
        # behaving correctly.
        expected = (
            f"{model._profile.provider}Builtin".replace("-", "").replace("_", "")
        )
        assert tc["name"] == expected
        assert tc["args"]["_acp_original_title"] == "WeirdTool"

    def test_mcp_name_prefix_dispatch_case_a(self) -> None:
        proxy = MagicMock()
        proxy.server_name = "proxy_x"
        model = _model(proxy=proxy)
        upd = {
            "sessionUpdate": "tool_call",
            "toolCallId": "m1",
            "title": "read_file",
            "rawInput": {"path": "/etc/hosts"},
            "_meta": {"claudeCode": {"toolName": "mcp__proxy_x__read_file"}},
        }
        tc = model._map_tool_call(_state(), upd)
        assert tc is not None
        assert tc["name"] == "read_file"
        assert tc["args"] == {"path": "/etc/hosts"}


class TestApplyUpdateRefineFlow:
    def test_pending_then_refine_then_terminal(self) -> None:
        async def _run() -> None:
            model = _model()
            state = _state()
            tool_calls: list[dict] = []
            text_acc: list[str] = []

            # 1. pending tool_call → held back in refine_pending_tool_calls.
            model._apply_update(state, "agent-1", _notif(_pending_tool_call("t1")), text_acc, tool_calls)
            assert "t1" in state.refine_pending_tool_calls
            assert tool_calls == []

            # 2. refine (no status, real args) → surfaced, hold released.
            model._apply_update(state, "agent-1", _notif(_refine_update("t1", "Bash", {"command": "ls"})), text_acc, tool_calls)
            assert "t1" not in state.refine_pending_tool_calls
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "Bash"
            assert tool_calls[0]["args"] == {"command": "ls"}

            # 3. terminal completed → no double surface, hold already gone.
            model._apply_update(state, "agent-1", _notif(_terminal_update("t1", "Bash")), text_acc, tool_calls)
            assert len(tool_calls) == 1

        asyncio.run(_run())

    def test_terminal_status_releases_never_refined_hold(self) -> None:
        model = _model()
        state = _state()
        tool_calls: list[dict] = []
        text_acc: list[str] = []

        model._apply_update(state, "agent-1", _notif(_pending_tool_call("t2")), text_acc, tool_calls)
        assert "t2" in state.refine_pending_tool_calls

        # Terminal update with no refine in between → hold released.
        model._apply_update(state, "agent-1", _notif(_terminal_update("t2", "Bash")), text_acc, tool_calls)
        assert "t2" not in state.refine_pending_tool_calls
        assert tool_calls == []  # never surfaced with empty args

    def test_agent_text_chunk_accumulated(self) -> None:
        model = _model()
        text_acc: list[str] = []
        tool_calls: list[dict] = []
        upd = {"sessionUpdate": "agent_message_chunk", "content": {"text": "hello"}}
        model._apply_update(_state(), "agent-1", _notif(upd), text_acc, tool_calls)
        assert text_acc == ["hello"]

    def test_usage_update_records_gauge(self) -> None:
        model = _model()
        state = _state()
        upd = {
            "sessionUpdate": "usage_update",
            "content": {"used": 123, "size": 1000},
        }
        model._apply_update(state, "agent-1", _notif(upd), [], [])
        assert state.usage_snapshot == {"used": 123, "size": 1000}

    def test_plan_update_surfaces_tool_call(self) -> None:
        model = _model()
        tool_calls: list[dict] = []
        upd = {"sessionUpdate": "plan", "toolCallId": "p1", "title": "update_plan"}
        model._apply_update(_state(), "agent-1", _notif(upd), [], tool_calls)
        assert len(tool_calls) == 1


class TestBoundedBuiltinPassthrough:
    def test_overflow_rejects_new_entries(self) -> None:
        """reject-new (DESIGN §4.5): at CALL_ID_MAP_MAX_ENTRIES a new
        built-in call is dropped, not stored."""
        async def _run() -> None:
            model = _model()
            state = _state()
            # Pre-fill the map to capacity with done futures.
            for i in range(CALL_ID_MAP_MAX_ENTRIES):
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"text": "ok"})
                state.builtin_passthrough_futures[f"t{i}"] = fut

            upd = _refine_update("overflow", "Bash", {"command": "ls"})
            tc = model._map_tool_call(state, upd)
            assert tc is None
            assert len(state.builtin_passthrough_futures) == CALL_ID_MAP_MAX_ENTRIES

        asyncio.run(_run())


class TestToolNameFrom:
    def test_uses_profile_extraction_style(self) -> None:
        model = _model()
        upd = {
            "title": "Read /etc/hosts (1-30)",
            "_meta": {"claudeCode": {"toolName": "Read"}},
        }
        assert model._tool_name_from(upd) == "Read"
