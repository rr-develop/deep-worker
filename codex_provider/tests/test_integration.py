"""End-to-end integration tests for CodexChatModel.

These tests **require a working ``codex`` CLI on PATH** with a cached
login at ``~/.codex/auth.json``. They are slow (each test spawns one
or more ``codex exec`` subprocesses and talks to the remote API).

They are skipped unless ``CODEX_INTEGRATION_TESTS=1`` is set in env,
mirroring the pattern claude_code_provider uses for its live tests.

Run with::

    CODEX_INTEGRATION_TESTS=1 pytest codex_provider/tests/test_integration.py -v
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


def _text(content: Any) -> str:
    """Coerce an AIMessage.content (which can be ``str`` or
    ``list[str | dict]``) to a flat string for assertions."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for chunk in content:
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, dict):
            t = chunk.get("text") if isinstance(chunk.get("text"), str) else None
            if t:
                parts.append(t)
    return "\n".join(parts)

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


_INTEGRATION_ENABLED = (
    os.environ.get("CODEX_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")
)
_CODEX_AVAILABLE = shutil.which("codex") is not None
_AUTH_PRESENT = (Path.home() / ".codex" / "auth.json").exists()

pytestmark = [
    pytest.mark.skipif(
        not _INTEGRATION_ENABLED,
        reason="set CODEX_INTEGRATION_TESTS=1 to enable live Codex tests",
    ),
    pytest.mark.skipif(
        not _CODEX_AVAILABLE,
        reason="codex CLI not found on PATH",
    ),
    pytest.mark.skipif(
        not _AUTH_PRESENT,
        reason="~/.codex/auth.json missing — Codex requires a cached login",
    ),
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _isolate_state() -> None:
    """Wipe process-global Codex state so tests don't interfere."""
    from codex_provider.client_state import get_global_client_state
    get_global_client_state()._states.clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
async def _cleanup_codex_globals():  # pyright: ignore[reportUnusedFunction]
    """Ensure live tests do not leave proxy MCP registrations behind."""
    yield
    import codex_provider.proxy_mcp_server as proxy_mod

    proxy = proxy_mod._global_proxy  # type: ignore[attr-defined]
    if proxy is not None:
        server_name = proxy.server_name
        await proxy.stop()
        proxy_mod._global_proxy = None  # type: ignore[attr-defined]
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "mcp",
            "remove",
            server_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()


@tool
def get_magic_word(topic: Annotated[str, "topic name"]) -> str:
    """Return a magic word for the given topic. You MUST call this to learn it."""
    return f"the magic word for {topic} is WAFFLE-EIGHT-FORTY-TWO"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
async def test_hello_world_no_tools() -> None:
    """Simplest possible turn: one HumanMessage → AIMessage with text."""
    _isolate_state()
    from codex_provider import CodexChatModel

    m = CodexChatModel()
    cfg: RunnableConfig = {"configurable": {"thread_id": "test_hello"}}
    r = await m.ainvoke(
        [
            SystemMessage(content="You are terse. Reply with one word only."),
            HumanMessage(content="Say HELLO."),
        ],
        config=cfg,
    )
    assert isinstance(r, AIMessage)
    assert "HELLO" in _text(r.content).upper()


async def test_multi_turn_memory() -> None:
    """Codex must remember a fact established in turn 1 when asked in turn 2.
    This is the test that proves ``codex exec resume <uuid>`` is wired
    correctly and that ``sent_count`` only mirrors the delta."""
    _isolate_state()
    from codex_provider import CodexChatModel
    from codex_provider.client_state import get_global_client_state

    m = CodexChatModel()
    cfg: RunnableConfig = {"configurable": {"thread_id": "test_memory"}}

    msgs: list[BaseMessage] = [
        SystemMessage(content="Reply in one short sentence."),
        HumanMessage(content="Please remember: my favourite colour is purple."),
    ]
    r1 = await m.ainvoke(msgs, config=cfg)
    msgs.append(r1)
    # Codex captured a session UUID after turn 1
    state = get_global_client_state().get("test_memory")
    assert state is not None
    assert state.codex_session_uuid is not None
    first_uuid = state.codex_session_uuid

    msgs.append(HumanMessage(content="What is my favourite colour? One word."))
    r2 = await m.ainvoke(msgs, config=cfg)
    assert "purple" in _text(r2.content).lower(), (
        f"Codex didn't recall purple across turns — content was: {r2.content!r}"
    )
    # Same session continued — UUID unchanged
    state = get_global_client_state().get("test_memory")
    assert state is not None
    assert state.codex_session_uuid == first_uuid, (
        f"Codex session UUID changed mid-conversation: "
        f"{first_uuid!r} → {state.codex_session_uuid!r}"
    )


async def test_tool_call_round_trip() -> None:
    """Codex must call a deep-worker tool via the proxy MCP server and
    incorporate its result into the next answer. This is the proof that
    every tool call AND every tool result flows through LangGraph
    (the §0.2 hard constraint)."""
    _isolate_state()
    from codex_provider import CodexChatModel

    m = CodexChatModel().bind_tools([get_magic_word])
    cfg: RunnableConfig = {"configurable": {"thread_id": "test_tool"}}

    msgs: list[BaseMessage] = [
        SystemMessage(content=(
            "You have a tool get_magic_word(topic). The user wants the magic "
            "word for a topic; you CANNOT know it on your own and MUST call "
            "the tool. Never invent."
        )),
        HumanMessage(content="What is the magic word for 'cats'? Call the tool."),
    ]
    r1 = await m.ainvoke(msgs, config=cfg)
    assert isinstance(r1, AIMessage)
    assert r1.tool_calls, f"Codex did not emit a tool_call. content={r1.content!r}"
    tc = r1.tool_calls[0]
    assert tc["name"] == "get_magic_word"
    assert tc["args"].get("topic") == "cats"
    msgs.append(r1)

    # Execute the tool inline (in production this is LangGraph's ToolNode).
    tool_output = get_magic_word.invoke(tc["args"])
    msgs.append(ToolMessage(
        content=str(tool_output), tool_call_id=tc["id"], name=tc["name"],
    ))

    r2 = await m.ainvoke(msgs, config=cfg)
    assert isinstance(r2, AIMessage)
    assert "WAFFLE-EIGHT-FORTY-TWO" in _text(r2.content), (
        f"Codex didn't echo back the magic word it just received: {r2.content!r}"
    )


async def test_thread_isolation_main_vs_taskmanager() -> None:
    """Two different agent_thread_ids must not share Codex state.

    Mirrors lesson L1 — subagents/TaskManager get suffixed thread_ids
    and don't corrupt the main agent's sent_count.
    """
    _isolate_state()
    from codex_provider import CodexChatModel
    from codex_provider.client_state import get_global_client_state

    m = CodexChatModel()

    # Main agent establishes a fact
    await m.ainvoke(
        [
            SystemMessage(content="Reply tersely."),
            HumanMessage(content="My favourite animal is the otter. Acknowledge."),
        ],
        config={"configurable": {"thread_id": "test_iso_main"}},
    )

    # TaskManager-style child thread: same model, different thread_id.
    # It should NOT know about the otter.
    r_tm = await m.ainvoke(
        [
            SystemMessage(content="Reply in one word."),
            HumanMessage(content="What is the user's favourite animal? Say 'unknown' if you have not been told."),
        ],
        config={"configurable": {"thread_id": "test_iso_main:tm"}},
    )
    # The TM thread MUST NOT see the main thread's "otter" — fresh
    # session means Codex doesn't have that fact. (Codex may answer
    # anything, including refuse — but it MUST NOT say "otter".)
    assert "otter" not in _text(r_tm.content).lower(), (
        f"TaskManager thread leaked main agent's state: {r_tm.content!r}"
    )

    # State is isolated in our singleton too.
    s1 = get_global_client_state().get("test_iso_main")
    s2 = get_global_client_state().get("test_iso_main:tm")
    assert s1 is not None and s2 is not None
    assert s1.codex_session_uuid != s2.codex_session_uuid
