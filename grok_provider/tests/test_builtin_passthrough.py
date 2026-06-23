"""TDD for ``builtin_tool_passthrough`` — the MANDATORY main channel that
records grok's locally-executed built-in tools (Shell/Read/Edit/...) into
LangGraph history.

Grok runs built-ins itself (Q1: cannot be disabled / no permission
prompt). We observe them on the ACP ``tool_call`` / ``tool_call_update``
stream and reconstruct the same ``AIMessage(tool_calls=[...])`` +
``ToolMessage`` pair LangGraph produces for proxy-MCP tools. The
passthrough BaseTool body parks an ``asyncio.Future`` on
``ThreadState.builtin_passthrough_futures[tool_call_id]`` and awaits it;
the reader resolves it on ``status=completed``.

DESIGN §11A.4-§11A.6. Hermetic: no grok binary, no subprocess.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.runnables.config import RunnableConfig

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from grok_provider import builtin_tool_passthrough as bp  # noqa: E402
from grok_provider.client_state import GrokClientState  # noqa: E402


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def client_state(fake_home, monkeypatch):
    cs = GrokClientState()
    # The passthrough tools resolve the singleton — point it at our instance.
    monkeypatch.setattr(bp, "get_global_client_state", lambda: cs)
    return cs


def _cfg(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _tool_call(name: str, args: dict, tcid: str) -> dict:
    """The exact shape LangGraph's ToolNode uses to invoke a tool that has
    an InjectedToolCallId field."""
    return {"args": args, "name": name, "type": "tool_call", "id": tcid}


class TestToolSet:
    def test_builds_one_tool_per_known_builtin_plus_generic(self) -> None:
        tools = bp.build_passthrough_tools()
        names = {t.name for t in tools}
        for n in ("Shell", "Read", "Write", "Edit", "Glob", "Grep"):
            assert n in names
        for n in ("GenerateImage", "image_gen", "image_edit"):
            assert n in names
        # CallMcpTool is NOT a passthrough tool (routes to channel A).
        assert "CallMcpTool" not in names

    def test_tools_have_open_schema(self) -> None:
        tools = bp.build_passthrough_tools()
        shell = next(t for t in tools if t.name == "Shell")
        schema = shell.get_input_schema().model_json_schema()
        # OPEN schema — we forward grok's rawInput verbatim.
        assert schema.get("additionalProperties", False) is True


class TestRoundTrip:
    async def test_shell_round_trip(self, client_state) -> None:
        st = client_state.get_or_create("agentA")
        tools = bp.build_passthrough_tools()
        shell = next(t for t in tools if t.name == "Shell")

        tcid = "tc-1"
        # Reader-side: register the Future BEFORE ToolNode runs (ainvoke does
        # this on the tool_call event).
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        st.builtin_passthrough_futures[tcid] = fut

        body = asyncio.create_task(
            shell.ainvoke(
                _tool_call("Shell", {"command": "ls"}, tcid),
                config=_cfg("agentA"),
            )
        )
        await asyncio.sleep(0)
        # Reader resolves the future with rendered result + metadata.
        bp.resolve_builtin_future(
            client_state,
            "agentA",
            tcid,
            {"text": "file1\nfile2", "metadata": {"exit_code": 0, "cwd": "/tmp"}},
        )
        msg = await asyncio.wait_for(body, timeout=1)
        result = msg.content if hasattr(msg, "content") else msg
        assert "file1" in result
        # Future popped after resolution.
        assert tcid not in st.builtin_passthrough_futures

    async def test_shell_round_trip_attaches_grok_meta(self, client_state) -> None:
        st = client_state.get_or_create("agentA")
        tools = bp.build_passthrough_tools()
        shell = next(t for t in tools if t.name == "Shell")
        tcid = "tc-meta"
        st.builtin_passthrough_futures[tcid] = (
            asyncio.get_event_loop().create_future()
        )
        body = asyncio.create_task(
            shell.ainvoke(
                _tool_call("Shell", {"command": "ls"}, tcid),
                config=_cfg("agentA"),
            )
        )
        await asyncio.sleep(0)
        bp.resolve_builtin_future(
            client_state,
            "agentA",
            tcid,
            {
                "text": "ok",
                "metadata": {"exit_code": 0},
                "grok_meta": {"exit_code": 0, "artifact_path": "/tmp/x.png"},
            },
        )
        msg = await asyncio.wait_for(body, timeout=1)
        assert msg.additional_kwargs.get("grok_builtin") is True
        assert msg.additional_kwargs["grok_meta"]["artifact_path"] == "/tmp/x.png"

    async def test_failed_status_returns_error_text(self, client_state) -> None:
        st = client_state.get_or_create("agentA")
        tools = bp.build_passthrough_tools()
        shell = next(t for t in tools if t.name == "Shell")
        tcid = "tc-err"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        st.builtin_passthrough_futures[tcid] = fut

        body = asyncio.create_task(
            shell.ainvoke(_tool_call("Shell", {"command": "false"}, tcid), config=_cfg("agentA"))
        )
        await asyncio.sleep(0)
        bp.resolve_builtin_future(
            client_state, "agentA", tcid,
            {"text": "boom", "is_error": True, "metadata": {}},
        )
        msg = await asyncio.wait_for(body, timeout=1)
        result = msg.content if hasattr(msg, "content") else msg
        assert "boom" in result
        assert msg.status == "error"


class TestResolveHelper:
    def test_resolve_unknown_thread_is_noop(self, client_state) -> None:
        # Must not raise.
        bp.resolve_builtin_future(client_state, "ghost", "x", {"text": "y"})

    def test_resolve_unknown_tcid_is_noop(self, client_state) -> None:
        client_state.get_or_create("agentA")
        bp.resolve_builtin_future(client_state, "agentA", "nope", {"text": "y"})


class TestGenericFallback:
    def test_generic_tool_present_for_unknown(self) -> None:
        tools = bp.build_passthrough_tools()
        # The generic wildcard is exposed so an unknown grok built-in can be
        # surfaced. It must exist and be invocable.
        generic = bp.get_generic_passthrough_tool()
        assert generic is not None
        assert generic.name  # has a name

    async def test_generic_round_trip(self, client_state) -> None:
        st = client_state.get_or_create("agentA")
        generic = bp.get_generic_passthrough_tool()
        tcid = "tc-gen"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        st.builtin_passthrough_futures[tcid] = fut
        body = asyncio.create_task(
            generic.ainvoke(
                _tool_call(generic.name, {"raw": {"x": 1}}, tcid), config=_cfg("agentA")
            )
        )
        await asyncio.sleep(0)
        bp.resolve_builtin_future(client_state, "agentA", tcid, {"text": "done"})
        msg = await asyncio.wait_for(body, timeout=1)
        result = msg.content if hasattr(msg, "content") else msg
        assert "done" in result
