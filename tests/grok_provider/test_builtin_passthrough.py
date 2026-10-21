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

from deep_worker.providers.grok_provider import builtin_tool_passthrough as bp  # noqa: E402
from deep_worker.providers.acp import builtin_tool_passthrough as acp_bp  # noqa: E402
from deep_worker.providers.acp.client_state import AcpClientState  # noqa: E402


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def client_state(fake_home, monkeypatch):
    cs = AcpClientState()
    # The passthrough tools resolve the singleton from the SHARED acp module
    # (the grok module is a thin binding over it since 3fe834d1) — point it
    # at our instance.
    monkeypatch.setattr(acp_bp, "get_global_client_state", lambda: cs)
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
                # The INPUT key is ``meta`` (``metadata`` is the fallback) —
                # it is not the same string as the OUTPUT marker. 3fe834d1
                # renamed the output markers grok_* -> acp_*; the grok module
                # still exports them under its GROK_* aliases, so assert
                # through the constants rather than literals.
                "meta": {"exit_code": 0, "artifact_path": "/tmp/x.png"},
            },
        )
        msg = await asyncio.wait_for(body, timeout=1)
        assert msg.additional_kwargs.get(bp.GROK_BUILTIN_MARKER) is True
        assert msg.additional_kwargs[bp.GROK_META_KEY]["artifact_path"] == "/tmp/x.png"

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
        # ``get_generic_passthrough_tool()`` was dropped by 3fe834d1 (it was
        # consumed only by the fork's own chat model). The BEHAVIOUR it
        # guarded is unchanged: the wildcard tool that surfaces an unknown
        # grok built-in is still built, so reach it through the live path.
        #
        # The name is asserted as a LITERAL, not via
        # ``bp.GENERIC_BUILTIN_NAME``: selecting the tool by that constant
        # and then comparing against the same constant is a tautology that
        # survives renaming it (profile.py:39 is the single source, and
        # ``bp`` only re-exports it). ``GrokBuiltin`` is the wire name grok
        # itself emits, so it is a value, not an implementation detail.
        tools = bp.build_passthrough_tools()
        assert bp.GENERIC_BUILTIN_NAME == "GrokBuiltin"
        names = [t.name for t in tools]
        assert "GrokBuiltin" in names
        # Exactly one wildcard, and it does not shadow a known built-in.
        assert names.count("GrokBuiltin") == 1
        assert "GrokBuiltin" not in bp.KNOWN_GROK_BUILTINS

    async def test_generic_round_trip(self, client_state) -> None:
        st = client_state.get_or_create("agentA")
        generic = next(
            t for t in bp.build_passthrough_tools() if t.name == bp.GENERIC_BUILTIN_NAME
        )
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
