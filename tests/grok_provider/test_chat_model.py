"""TDD for ``GrokChatModel`` — the LangChain BaseChatModel that drives the
grok CLI over ACP.

Hermetic: no real grok binary. We inject a fake ACP client (``session_new``
returns a canned id; ``session_prompt`` returns a Future we resolve, after
pushing ``session/update`` notifications into the sink) and let the model
use the real proxy MCP server object but stub its ``url_async`` so nothing
binds a socket. Covers (DESIGN §8, §10, §11A):

 - construction is cheap + side-effect-free; _llm_type == "grok"
 - _generate raises (async-only)
 - bind_tools extracts MCP metadata
 - pydantic dump excludes un-serialisable callables (checkpoint pickle)
 - ainvoke requires an explicit thread_id
 - new-turn: session/new + session/prompt, system prompt delivered inline
   as a <system> block in the FIRST prompt only
 - agent_message_chunk accumulation -> AIMessage.content
 - proxy-MCP tool_call surfaced as AIMessage.tool_calls (channel A)
 - built-in tool_call surfaced + future registered (channel B)
 - ToolMessage resolution: proxy-MCP resolves the parked call; a built-in
   (``BUILTIN_MARKER``) ToolMessage does NOT
 - summarization rotation drops the session id, keeps the thread id

Since 3fe834d1 the grok provider is a thin subclass of the shared
``AcpChatModel``: ``grok_provider/chat_model.py`` holds only ``GrokChatModel``
plus its profile wiring, while every module-level helper and every singleton
accessor lives in ``deep_worker/providers/acp/chat_model.py``. Tests therefore
import the grok class from the grok module (that is the production entry
point) but reach the helpers — and monkeypatch the singleton accessors — on
``acp.chat_model``, which is the module the executing code resolves them from.
Grok specifics (``_x.ai/`` prefix, ``CallMcpTool`` dispatch, ``GenerateImage``/
``image_edit``, ``GrokBuiltin``, ``allow-*``/``deny-*`` kinds) are asserted
through the live grok PROFILE, never against claude-code defaults.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# ``acp_cm`` is where the shared implementation lives: AcpChatModel.__init__
# calls ``get_global_client_state()`` and ``_ensure_proxy_started`` calls
# ``get_global_proxy_mcp_server()`` through THIS module's globals, so this is
# the only module where patching them has any effect.
from deep_worker.providers.acp import chat_model as acp_cm  # noqa: E402
from deep_worker.providers.acp.acp_client import AcpClientError  # noqa: E402
from deep_worker.providers.grok_provider.chat_model import GrokChatModel  # noqa: E402
from deep_worker.providers.acp.chat_model import AcpTurnCancelledError  # noqa: E402
from deep_worker.providers.acp.builtin_tool_passthrough import (  # noqa: E402
    BUILTIN_MARKER,
)
from deep_worker.providers.acp.limit_errors import AcpUsageLimitError  # noqa: E402
from deep_worker.providers.acp.client_state import AcpClientState  # noqa: E402
from deep_worker.providers.grok_provider.profile import build_profile  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeAcpClient:
    """Minimal stand-in for AcpClient driven by the test.

    Every signature mirrors the real ``acp.acp_client.AcpClient``, including
    the keyword-only ``meta`` on ``session_new``/``session_load`` and the
    idempotent ``ensure_started`` the shared ``_ensure_acp_started`` calls on
    EVERY turn (the fork had a per-model ``_acp_started`` latch instead). A
    fake that drifts from the real signature turns a green suite into a lie.
    """

    def __init__(self) -> None:
        self.started = False
        self.sinks: dict[str, asyncio.Queue] = {}
        self.prompts: list[dict] = []
        self.session_counter = 0
        self.last_prompt_future: asyncio.Future | None = None
        self.session_new_calls: list[dict] = []
        self.session_load_calls: list[dict] = []

    async def start(self) -> None:
        self.started = True

    async def ensure_started(self) -> None:
        self.started = True

    async def initialize(self) -> dict:
        return {"result": {}}

    async def authenticate(self) -> dict:
        return {"result": {}}

    async def aclose(self) -> None:
        self.started = False

    async def session_new(
        self, *, cwd: str, mcp_servers: list[dict], meta: dict | None = None
    ) -> str:
        self.session_counter += 1
        self.session_new_calls.append(
            {"cwd": cwd, "mcp_servers": mcp_servers, "meta": meta}
        )
        return f"sess-{self.session_counter}"

    async def session_load(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[dict],
        meta: dict | None = None,
    ) -> None:
        self.session_load_calls.append(
            {"session_id": session_id, "cwd": cwd,
             "mcp_servers": mcp_servers, "meta": meta}
        )

    async def session_close(self, session_id: str | None) -> bool:
        return True

    async def wait_session_ready(self, session_id: str) -> None:
        return None

    def register_sink(self, session_id: str, queue: asyncio.Queue) -> None:
        self.sinks[session_id] = queue

    def session_prompt(self, *, session_id, prompt, notification_sink):
        self.register_sink(session_id, notification_sink)
        self.prompts.append({"session_id": session_id, "prompt": prompt})
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.last_prompt_future = fut
        return fut

    async def session_cancel(self, session_id) -> None:
        return None

    def has_active_prompt(self, session_id: str | None) -> bool:
        fut = self.last_prompt_future
        return fut is not None and not fut.done()

    def release_session(self, session_id: str | None) -> None:
        return None

    # Test helpers ---------------------------------------------------------
    def push_update(self, session_id: str, update: dict) -> None:
        self.sinks[session_id].put_nowait(
            {"method": "session/update",
             "params": {"sessionId": session_id, "update": update}}
        )

    def finish(self, stop_reason: str = "end_turn", meta: dict | None = None) -> None:
        assert self.last_prompt_future is not None
        if not self.last_prompt_future.done():
            result: dict[str, Any] = {"stopReason": stop_reason}
            if meta is not None:
                result["_meta"] = meta
            self.last_prompt_future.set_result({"result": result})


class _FakeProxy:
    def __init__(self) -> None:
        self.server_name = "deep_worker_proxy_test"
        self.resolved: list[tuple[str, Any]] = []
        self.resolved_tool_names: list[str | None] = []
        self._tools: list[dict] = []

    def update_tools(self, md: list[dict]) -> None:
        self._tools = md

    async def url_async(self) -> str:
        return "http://127.0.0.1:0/mcp"

    async def start(self) -> None:
        return None

    def resolve_call(
        self,
        call_id: str,
        result: Any,
        *,
        tool_name: str | None = None,
    ) -> None:
        # Mirrors the real ProxyMCPServer.resolve_call signature, including
        # the keyword-only ``tool_name`` production passes. A fake that
        # drifts from the real signature turns a green suite into a lie.
        self.resolved.append((call_id, result))
        self.resolved_tool_names.append(tool_name)


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def install_client_state(monkeypatch) -> AcpClientState:
    """A fresh ``AcpClientState`` installed as the process singleton.

    Patching ``acp_cm.get_global_client_state`` — not the grok module — is what
    makes ``AcpChatModel.__init__`` pick it up, and going through the model's
    own constructor is also what calls ``set_profile(provider, fingerprint())``
    on it. So the container under test is profiled exactly as production
    profiles it (``provider="grok"`` + the grok session fingerprint) rather
    than by hand-fed arguments, and persistence stays scoped to the fake home.
    """
    cs = AcpClientState()
    monkeypatch.setattr(acp_cm, "get_global_client_state", lambda: cs)
    return cs


@pytest.fixture()
def model(fake_home, monkeypatch):
    cs = install_client_state(monkeypatch)
    m = GrokChatModel(model="grok-4")
    m._client_state = cs
    m._acp_client = _FakeAcpClient()
    m._proxy_mcp = _FakeProxy()
    m._proxy_started = True
    m._acp_started = True
    m._working_dir = "/tmp"
    return m


def _cfg(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


# --------------------------------------------------------------------------
# Construction / surface
# --------------------------------------------------------------------------
class TestSurface:
    def test_llm_type(self, fake_home) -> None:
        assert GrokChatModel()._llm_type == "grok"

    def test_generate_raises(self, fake_home) -> None:
        with pytest.raises(NotImplementedError):
            GrokChatModel()._generate()

    def test_construction_is_cheap(self, fake_home) -> None:
        m = GrokChatModel(model="grok-4")
        assert m._model_name == "grok-4"

    def test_model_dump_excludes_callables(self, fake_home) -> None:
        dumped = GrokChatModel().model_dump()
        for k in ("custom_get_token_ids", "callbacks", "rate_limiter"):
            assert k not in dumped

    def test_json_schema_strips_callables(self, fake_home) -> None:
        schema = GrokChatModel.model_json_schema()
        props = schema.get("properties", {})
        for k in ("custom_get_token_ids", "callbacks", "rate_limiter"):
            assert k not in props


class TestBindTools:
    def test_extracts_metadata(self, fake_home) -> None:
        @tool
        def read_file(path: str) -> str:
            """Read a file."""
            return ""

        m = GrokChatModel().bind_tools([read_file])
        names = [t["name"] for t in m._tools_metadata]
        assert "read_file" in names

    def test_passthrough_tools_excluded_from_proxy_metadata(
        self, fake_home
    ) -> None:
        """REGRESSION (test_grok 2026-06-18): channel-B built-in passthrough
        tools must NOT be published as MCP tools by proxy_mcp_server. Otherwise
        grok sees a deepworker MCP tool with the same name as its native
        built-in (e.g. ``Shell``), dispatches the model's call through
        CallMcpTool, and parks a Future on the proxy that the built-in path
        will never resolve → 120s job-handle notice and a stuck turn.
        """
        from deep_worker.providers.grok_provider.builtin_tool_passthrough import (  # noqa: PLC0415
            build_passthrough_tools,
        )

        @tool
        def execute(command: str) -> str:
            """Real deepworker MCP tool."""
            return ""

        m = GrokChatModel().bind_tools([execute, *build_passthrough_tools()])
        names = [t["name"] for t in m._tools_metadata]
        assert "execute" in names, "real MCP tool must still be published"
        # NONE of the passthrough names — grok's built-in inventory — may leak
        # into proxy MCP's tools/list.
        forbidden = {
            "Shell", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep",
            "WebSearch", "WebFetch", "Task", "Plan", "GrokBuiltin",
            "GenerateImage", "image_gen", "image_edit",
        }
        assert forbidden.isdisjoint(set(names)), (
            f"passthrough tool(s) leaked into proxy MCP metadata: "
            f"{sorted(forbidden & set(names))!r}"
        )


# --------------------------------------------------------------------------
# Proxy startup
# --------------------------------------------------------------------------
class TestEnsureProxyStarted:
    async def test_proxy_started_even_with_no_tools(
        self, fake_home, monkeypatch
    ) -> None:
        """REGRESSION (live integration 2026-06-18): a turn with NO bound
        deep-worker MCP tools must STILL start the proxy MCP server, because
        every ``session/new`` references it as grok's MCP endpoint. The old
        code returned early when ``_tools_metadata`` was empty, leaving
        ``_proxy_mcp`` as None — and the very next line of ainvoke
        (``self._proxy_mcp.url_async()``) raised AttributeError on any
        tool-less turn (hello-world / memory / thread-isolation).
        """
        cs = install_client_state(monkeypatch)
        m = GrokChatModel(model="grok-4")
        m._client_state = cs
        # No bind_tools — _tools_metadata stays [].
        assert m._tools_metadata == []

        started = {"count": 0}

        class _Proxy:
            server_name = "deep_worker_proxy_test"

            async def start(self) -> None:
                started["count"] += 1

            async def url_async(self) -> str:
                return "http://127.0.0.1:0/mcp"

        # Patched on the SHARED module: ``_ensure_proxy_started`` is inherited
        # from AcpChatModel and resolves the factory from acp.chat_model.
        monkeypatch.setattr(
            acp_cm, "get_global_proxy_mcp_server", lambda md: _Proxy()
        )
        await m._ensure_proxy_started()
        assert m._proxy_mcp is not None, (
            "proxy must be created even with zero tools bound"
        )
        assert m._proxy_started is True
        assert started["count"] == 1
        # url_async must be callable now (the line that used to crash).
        assert await m._proxy_mcp.url_async() == "http://127.0.0.1:0/mcp"


# --------------------------------------------------------------------------
# ainvoke
# --------------------------------------------------------------------------
class TestAinvokeContract:
    async def test_requires_thread_id(self, model) -> None:
        with pytest.raises(ValueError):
            await model.ainvoke([HumanMessage(content="hi")])


class TestNewTurn:
    async def test_simple_text_turn(self, model) -> None:
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke(
                [SystemMessage(content="be helpful"), HumanMessage(content="hello")],
                config=_cfg("agentA"),
            )
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "hi there"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        msg: AIMessage = await asyncio.wait_for(task, timeout=1)
        assert msg.content == "hi there"
        # System prompt delivered inline as <system> block in the first prompt.
        first_text = acp.prompts[0]["prompt"][0]["text"]
        assert "<system>" in first_text and "be helpful" in first_text
        assert "hello" in first_text

    async def test_first_turn_replays_history_into_the_new_session(
        self, model
    ) -> None:
        """A restarted agent (or one switched over from another provider)
        has a LangGraph history but no live grok session. The brand-new
        session must receive the prior turns, not just the newest line.

        Before the fix the 7b re-slice fired on this first turn — its only
        guard was ``sent_count < len(state_messages)``, trivially true when
        ``sent_count`` seeds to 0 — and collapsed the whole history down to
        the trailing HumanMessage. Mirrors the 2026-08-16 claude-code
        amnesia; grok shares the code shape.
        """
        acp = model._acp_client
        history = [
            SystemMessage(content="sys"),
            HumanMessage(content="remember: the codeword is ALPHA"),
            AIMessage(content="noted"),
            HumanMessage(content="what is the codeword?"),
        ]
        task = asyncio.create_task(
            model.ainvoke(history, config=_cfg("restarted"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "ALPHA"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)

        first_text = acp.prompts[0]["prompt"][0]["text"]
        assert "what is the codeword?" in first_text
        assert "the codeword is ALPHA" in first_text, (
            "the first grok turn dropped the agent's prior history"
        )

    async def test_session_registered_in_index(self, model) -> None:
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="x")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        assert model._client_state.thread_for_session(sid) is not None
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)

    async def test_no_tools_means_no_mcp_server_registered(self, model) -> None:
        """REGRESSION (live integration 2026-06-18): grok 0.2.54 aborts a turn
        (empty output) when handed an MCP server with an empty tools/list.
        With no bound deep-worker tools, session/new must register ZERO mcp
        servers — grok's built-ins still flow via channel-B passthrough."""
        acp = model._acp_client
        assert model._tools_metadata == []  # no bind_tools
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="hi")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)
        assert acp.session_new_calls, "session_new should have been called"
        assert acp.session_new_calls[0]["mcp_servers"] == [], (
            "no proxy MCP server may be registered when zero tools are bound"
        )

    async def test_bound_tools_register_proxy_mcp_server(self, fake_home, monkeypatch) -> None:
        """The mirror: when deep-worker tools ARE bound, session/new must
        register exactly the proxy MCP server so grok can dispatch them."""
        cs = install_client_state(monkeypatch)

        @tool
        def read_file(path: str) -> str:
            """Read a file."""
            return ""

        m = GrokChatModel(model="grok-4").bind_tools([read_file])
        m._client_state = cs
        m._acp_client = _FakeAcpClient()
        m._proxy_mcp = _FakeProxy()
        m._proxy_started = True
        m._acp_started = True
        m._working_dir = "/tmp"

        task = asyncio.create_task(
            m.ainvoke([HumanMessage(content="hi")], config=_cfg("agentB"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{m._acp_client.session_counter}"
        m._acp_client.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                                        "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        m._acp_client.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)
        servers = m._acp_client.session_new_calls[0]["mcp_servers"]
        assert len(servers) == 1
        assert servers[0]["name"] == m._proxy_mcp.server_name


class TestProxyToolCall:
    async def test_proxy_mcp_tool_surfaced_as_tool_call(self, model) -> None:
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="run")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        # grok dispatches our proxy tool via CallMcpTool.
        acp.push_update(sid, {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-1",
            "title": "CallMcpTool",
            "rawInput": {"server": model._proxy_mcp.server_name,
                         "toolName": "execute", "arguments": {"command": "ls"}},
        })
        await asyncio.sleep(0.01)
        msg: AIMessage = await asyncio.wait_for(task, timeout=1)
        assert msg.tool_calls
        call = msg.tool_calls[0]
        assert call["name"] == "execute"
        assert call["id"] == "tc-1"
        assert call["args"] == {"command": "ls"}


class TestMapToolCall:
    async def test_unknown_builtin_routes_to_grok_builtin(self, model) -> None:
        st = model._client_state.get_or_create("agentA")
        tc = model._map_tool_call(
            st,
            {
                "toolCallId": "tc-img",
                "title": "future_builtin_xyz",
                "rawInput": {"prompt": "test"},
            },
        )
        assert tc is not None
        assert tc["name"] == "GrokBuiltin"
        assert tc["args"]["_grok_original_title"] == "future_builtin_xyz"

    def test_builtin_future_uses_the_running_loop_not_get_event_loop(
        self, model
    ) -> None:
        """``_map_tool_call`` must take the RUNNING loop.

        ``asyncio.get_event_loop()`` no longer creates a loop on demand
        (Python 3.12) — it raises "There is no current event loop" whenever
        the thread has none, which is exactly the state an earlier async test
        leaves behind. That made three tests here pass in isolation and fail
        in a full-suite run.

        Asserted at the source level on purpose: from INSIDE a running loop
        both spellings behave identically, so a behavioural test cannot tell
        them apart and would not catch a regression back to the broken form.

        Resolved off ``GrokChatModel`` rather than off a module, so it reads
        whatever implementation grok ACTUALLY executes: since 3fe834d1 that is
        the inherited ``AcpChatModel._map_tool_call``, and it would follow an
        override back to the grok subclass if one were ever added.
        """
        import inspect

        source = inspect.getsource(GrokChatModel._map_tool_call)
        assert "get_running_loop()" in source
        assert "get_event_loop()" not in source, (
            "_map_tool_call must not use asyncio.get_event_loop(): it raises "
            "when the calling thread has no loop set"
        )

    async def test_known_imagine_builtin_surfaces_by_acp_title(self, model) -> None:
        st = model._client_state.get_or_create("agentA")
        tc = model._map_tool_call(
            st,
            {
                "toolCallId": "tc-gen",
                "title": "GenerateImage",
                "rawInput": {"description": "a red circle", "filename": "circle.png"},
            },
        )
        assert tc is not None
        assert tc["name"] == "GenerateImage"
        assert "_grok_original_title" not in tc["args"]


class TestRenderCompletedPayload:
    def test_render_imagine_payload_with_path(self, tmp_path) -> None:
        img = tmp_path / "out.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        upd = {
            "content": [{"text": f"saved to {img}"}],
            "rawOutput": {"path": str(img), "mime": "image/png"},
        }
        # The payload's structured half is keyed ``meta`` since 3fe834d1 (the
        # fork called it ``grok_meta``). It is the INPUT half of the
        # passthrough contract — ``_payload_to_tool_message`` reads
        # ``payload["meta"]`` and only then republishes it on the ToolMessage
        # under the OUTPUT marker ``META_KEY``. The two are not the same key.
        payload = acp_cm._render_completed_payload(upd, "completed")
        assert str(img) in payload["text"]
        assert payload["meta"]["artifact_path"] == str(img)
        assert payload["meta"]["mime"] == "image/png"
        assert payload["meta"]["kind"] == "image"


class TestToolContract:
    """The per-turn ``<system>`` tool contract, always on the GROK profile.

    ``_build_tool_contract`` is provider-parametric now: the vendor facts the
    assertions below turn on (grok's built-in inventory, the bare
    ``<server>__<tool>`` mangling rather than claude-code's ``mcp__`` prefix)
    come from ``grok_provider.profile.build_profile()``, so running it under
    the default claude-code profile would prove nothing about grok.
    """

    @staticmethod
    def _contract(tool_names: list[str], server_name: str = "deep_worker_proxy_1") -> str:
        return acp_cm._build_tool_contract(
            tool_names, build_profile(), server_name=server_name
        )

    def test_live_verified_imagine_guidance_included(self) -> None:
        contract = self._contract(["execute"])
        assert "GenerateImage" in contract
        assert "image_edit" in contract

    def test_grok_builtins_named_not_claude_code_ones(self) -> None:
        """The built-in inventory in the contract is GROK's.

        Guards the profile wiring itself: a contract built on the default
        claude-code profile would still contain "Read"/"Shell" (shared names)
        and would silently pass the assertions above, so pin a name that only
        the grok profile knows.
        """
        contract = self._contract(["execute"])
        assert "run_terminal_command" in contract  # grok 1.0.5 native
        assert "image_to_video" in contract

    def test_bound_tool_named_under_groks_mangled_call_form(self) -> None:
        """Replaces ``test_str_replace_prefers_bound_edit_file``.

        The old test asserted the contract rendered a built-in→MCP preference
        pair (``StrReplace -> `edit_file```) out of grok's ``_BUILTIN_EQUIVALENTS``
        map. That map was deliberately deleted with the fork (3fe834d1): the
        shared contract no longer tells the model which built-in to prefer per
        name, it states the preference once in prose and lists the built-ins.
        What the contract still owes ``edit_file`` is the CALL NAME grok can
        actually resolve — bare ``<server>__<tool>``, NOT claude-code's
        ``mcp__<server>__<tool>`` (incidents 2026-06-28 / 2026-08-24) — plus
        the prose preference. That is asserted here.
        """
        contract = self._contract(["edit_file"], server_name="deep_worker_proxy_7")
        assert "`deep_worker_proxy_7__edit_file`" in contract
        assert "mcp__deep_worker_proxy_7__edit_file" not in contract
        assert "no deep-worker equivalent" in contract

    def test_contract_disambiguates_mcp_vs_native(self) -> None:
        # §8 of DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md: the contract
        # must state these are MCP tools, NOT grok's native built-ins, and
        # must still forbid hunting for the schema on disk.
        contract = self._contract(["send_message_to_agent"])
        assert "MCP tool" in contract
        assert "NOT" in contract and "built-in" in contract
        # Disk-hunt prohibition preserved. The literal ``mcps/`` path glob
        # moved to the system-prompt note (proxy_tool_hints, asserted in
        # test_grok_system_prompt_still_forbids_the_disk_hunt below); the
        # per-turn contract carries the prohibition in its own words.
        assert "do NOT search the filesystem for their schema" in contract
        # No misleading literal mcp__deepworker__ prefix advertised to grok.
        assert "mcp__deepworker" not in contract

    def test_grok_system_prompt_still_forbids_the_disk_hunt(self) -> None:
        """The ``mcps/.../tools/*.json`` prohibition did not disappear — it
        moved from grok's forked contract into the shared provider note that
        the grok system prompt renders. Asserted on provider ``"grok"``: the
        note is per-provider (grok is in ``ACP_PROXY_MANGLED``) and a
        regression that dropped grok from that bucket would show up here.
        """
        from deep_worker_cli.prompts.proxy_tool_hints import (  # noqa: PLC0415
            provider_tool_invocation_note,
        )

        note = provider_tool_invocation_note("grok", pid=4242)
        assert "mcps/" in note
        assert "mcp__deepworker" not in note


class TestBuiltinToolCall:
    async def test_builtin_surfaced_and_future_registered(self, model) -> None:
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="edit")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-builtin",
            "title": "Shell",
            "rawInput": {"command": "ls -la"},
        })
        await asyncio.sleep(0.01)
        msg: AIMessage = await asyncio.wait_for(task, timeout=1)
        assert msg.tool_calls
        call = msg.tool_calls[0]
        assert call["name"] == "Shell"
        assert call["id"] == "tc-builtin"
        # A passthrough future must have been registered for the reader.
        st = model._client_state.get("agentA")
        assert "tc-builtin" in st.builtin_passthrough_futures


class TestBuiltinMarkerWireContract:
    """The ``additional_kwargs`` marker keys are a PERSISTED contract.

    Both the stamper (``_payload_to_tool_message``) and the reader
    (``ainvoke`` step 7b) import the same constant, so a rename is internally
    consistent and no behavioural test can see it — verified by mutation: the
    whole suite stays green when ``BUILTIN_MARKER`` is renamed. But these
    strings are baked into every already-checkpointed grok history: after a
    rename, a resumed agent's historical built-in ToolMessages stop reading as
    built-ins and get re-dispatched into the proxy MCP. The renaming of
    ``grok_builtin``/``grok_meta`` to these shared names at 3fe834d1 was a
    deliberate one-off; the values are pinned here so the next one cannot be
    silent.
    """

    def test_marker_keys_are_the_persisted_literals(self) -> None:
        from deep_worker.providers.acp.builtin_tool_passthrough import (  # noqa: PLC0415
            META_KEY,
        )

        assert BUILTIN_MARKER == "acp_builtin"
        assert META_KEY == "acp_builtin_meta"

    def test_grok_aliases_still_resolve_to_the_shared_constants(self) -> None:
        """Production code imports these by their historical grok names
        (``grok_provider.builtin_tool_passthrough``); they must stay aliases,
        not drift into a second pair of values."""
        from deep_worker.providers.acp.builtin_tool_passthrough import (  # noqa: PLC0415
            META_KEY,
        )
        from deep_worker.providers.grok_provider.builtin_tool_passthrough import (  # noqa: PLC0415
            GROK_BUILTIN_MARKER,
            GROK_META_KEY,
        )

        assert GROK_BUILTIN_MARKER == BUILTIN_MARKER
        assert GROK_META_KEY == META_KEY


class TestToolResolution:
    async def test_proxy_tool_message_resolves_call(self, model) -> None:
        # Simulate mid-turn: an in-flight prompt future + a proxy ToolMessage.
        st = model._client_state.get_or_create("agentA")
        st.session_id = "sess-9"
        st.sent_count = 2
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        st.pending_session_prompt_future = fut
        st.current_notif_queue = asyncio.Queue()
        model._acp_client.register_sink("sess-9", st.current_notif_queue)
        model._acp_client.last_prompt_future = fut

        messages = [
            HumanMessage(content="run"),
            AIMessage(content="", tool_calls=[
                {"id": "tc-1", "name": "execute", "args": {"command": "ls"},
                 "type": "tool_call"}]),
            ToolMessage(content="output", tool_call_id="tc-1"),
        ]
        task = asyncio.create_task(model.ainvoke(messages, config=_cfg("agentA")))
        await asyncio.sleep(0.01)
        # The proxy MCP call should have been resolved.
        assert any(cid == "tc-1" for cid, _ in model._proxy_mcp.resolved)
        # Finish the in-flight turn.
        fut.set_result({"result": {"stopReason": "end_turn"}})
        await asyncio.wait_for(task, timeout=1)

    async def test_builtin_tool_message_does_not_resolve_proxy(self, model) -> None:
        st = model._client_state.get_or_create("agentA")
        st.session_id = "sess-9"
        st.sent_count = 2
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        st.pending_session_prompt_future = fut
        st.current_notif_queue = asyncio.Queue()
        model._acp_client.register_sink("sess-9", st.current_notif_queue)
        model._acp_client.last_prompt_future = fut

        messages = [
            HumanMessage(content="run"),
            AIMessage(content="", tool_calls=[
                {"id": "tc-b", "name": "Shell", "args": {"command": "ls"},
                 "type": "tool_call"}]),
            # The marker key is BUILTIN_MARKER ("acp_builtin") since 3fe834d1;
            # the fork spelled it "grok_builtin". Taken from the constant the
            # passthrough tool itself stamps, so a rename cannot make this
            # test silently assert nothing.
            ToolMessage(content="output", tool_call_id="tc-b",
                        additional_kwargs={BUILTIN_MARKER: True}),
        ]
        task = asyncio.create_task(model.ainvoke(messages, config=_cfg("agentA")))
        await asyncio.sleep(0.01)
        # A built-in ToolMessage must NOT hit the proxy MCP.
        assert not any(cid == "tc-b" for cid, _ in model._proxy_mcp.resolved)
        fut.set_result({"result": {"stopReason": "end_turn"}})
        await asyncio.wait_for(task, timeout=1)


class TestResponseRaceDrainsQueue:
    async def test_text_chunks_queued_before_response_are_not_lost(
        self, model
    ) -> None:
        """REGRESSION (live integration 2026-06-18): grok streams every
        agent_message_chunk and THEN resolves the session/prompt response in
        the same event-loop tick. If _consume_turn finalized on the response
        future without first draining the queue, it returned an EMPTY
        AIMessage (intermittent content='' failures). Here we pre-load the
        queue with the answer chunks AND resolve the response, then assert
        the returned content includes the queued text."""
        st = model._client_state.get_or_create("agentA")
        st.session_id = "sess-race"
        st.sent_count = 0
        q: asyncio.Queue = asyncio.Queue()
        # Answer chunks already sitting in the queue when the turn resolves.
        for piece in ("Hello", " there", "!"):
            q.put_nowait({
                "method": "session/update",
                "params": {"sessionId": "sess-race",
                           "update": {"sessionUpdate": "agent_message_chunk",
                                      "content": {"text": piece}}},
            })
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        st.current_notif_queue = q
        st.pending_session_prompt_future = fut

        msg = await asyncio.wait_for(
            model._consume_turn(st, "agentA"), timeout=1
        )
        assert msg.content == "Hello there!", (
            f"queued chunks lost on response-race: {msg.content!r}"
        )
        # Queue fully drained; future cleared.
        assert q.empty()
        assert st.pending_session_prompt_future is None


class TestSummarization:
    """Summarization rotation: drop the CLI session, KEEP the thread id.

    The ROUTE changed, not the rotation. The fork detected summarization
    implicitly inside ``ainvoke`` (a message list shorter than
    ``sent_count``) and silently opened a fresh session right there. Since
    the SummarizerPort landed, that detection is a SAFETY NET only: reaching
    it means ``areset_after_summary`` was skipped, and continuing would feed
    the whole visible history into the new session as its first prompt — the
    exact full-history dump of incident 2026-08-19. So the net now resets and
    raises instead. Both halves are pinned below: the port does the rotation,
    the net refuses to re-prompt.
    """

    async def test_summarization_drops_session(self, model) -> None:
        st = model._client_state.get_or_create("agentA")
        st.sent_count = 10
        st.session_id = "sess-old"
        model._client_state.register_session("sess-old", "agentA")

        # The production route: LangGraph's summarizer calls the port.
        await model.areset_after_summary(thread_id="agentA")

        # The old session is evicted from the reverse index and dropped from
        # the slot, while the slot (the deep-worker thread) survives — L3.
        assert model._client_state.thread_for_session("sess-old") is None
        assert model._client_state.get("agentA") is st
        assert st.session_id is None
        assert st.sent_count == 0

        # And the NEXT turn opens a brand-new session, keeping the thread id.
        acp = model._acp_client
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="fresh")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        sid = f"sess-{acp.session_counter}"
        assert sid != "sess-old"
        assert model._client_state.thread_for_session(sid) is st
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)

    async def test_shorter_list_without_the_port_resets_and_refuses(
        self, model
    ) -> None:
        """The safety net. A shorter message list with no ``areset_after_summary``
        must reset the slot AND raise rather than re-prompt a fresh session
        with the whole visible history (incident 2026-08-19)."""
        st = model._client_state.get_or_create("agentA")
        st.sent_count = 10
        st.session_id = "sess-old"
        model._client_state.register_session("sess-old", "agentA")
        acp = model._acp_client

        # Bounded: the FAILURE MODE this pins is a fall-through into
        # session/new + session/prompt, and nothing in this test ever resolves
        # that prompt future. Without the bound a regression hangs the suite
        # instead of failing it (verified: an unbounded version of this test
        # hangs on a mutation that drops the raise).
        with pytest.raises(acp_cm.AcpSummarizationResetRequired):
            await asyncio.wait_for(
                model.ainvoke(
                    [HumanMessage(content="fresh")], config=_cfg("agentA")
                ),
                timeout=2,
            )
        # Reset applied and persisted...
        assert model._client_state.thread_for_session("sess-old") is None
        assert st.session_id is None
        assert st.sent_count == 0
        # ...and NO prompt was sent into a fresh session.
        assert acp.session_new_calls == []
        assert acp.prompts == []


class TestCancellationClassification:
    @pytest.mark.asyncio
    async def test_acp_cancelled_stop_reason_is_not_asyncio_task_cancellation(
        self, model
    ) -> None:
        with pytest.raises(AcpTurnCancelledError):
            await model._finalize_response(
                {"result": {"stopReason": "cancelled"}}, []
            )


class TestErrorReplyFinalization:
    """session/prompt resolves its future with the RAW reply and bypasses
    ``_call``'s error check. An error reply must raise here, NOT collapse
    into an empty AIMessage (incident 2026-07-03 — grok HTTP 402)."""

    @pytest.mark.asyncio
    async def test_billing_error_reply_raises_usage_limit(self, model) -> None:
        # This is the exact shape session/prompt delivers on an xAI 402.
        reply = {"id": 7, "error": {"code": 402, "message": "Payment Required"}}
        with pytest.raises(AcpUsageLimitError):
            await model._finalize_response(reply, ["partial"])

    @pytest.mark.asyncio
    async def test_generic_error_reply_raises_client_error(self, model) -> None:
        reply = {"id": 7, "error": {"code": -32000, "message": "internal boom"}}
        with pytest.raises(AcpClientError):
            await model._finalize_response(reply, [])

    @pytest.mark.asyncio
    async def test_error_reply_does_not_return_empty_aimessage(self, model) -> None:
        # The regression: no exception, empty content, silent no-failover.
        reply = {"id": 7, "error": {"message": "insufficient credits"}}
        with pytest.raises(AcpUsageLimitError):
            await model._finalize_response(reply, [])

    @pytest.mark.asyncio
    async def test_nested_402_reply_raises_usage_limit_real_path(self, model) -> None:
        """REAL production shape (test_grok2, incident 2026-07-04) driven
        through the REAL terminal path ``_finalize_response`` → real
        ``raise_for_error_reply`` → real markers. The 402 is nested inside
        ``error.data.message`` with a generic ``"Internal error"`` on top.

        This is the exact reply session/prompt delivered on the live agent.
        The earlier (2026-07-03) tests fed only a FLAT ``error.message`` the
        real API never sends, so this nested shape still broke in production
        until the extractor learned to read ``error.data`` (POLICIES.md §10)."""
        reply = {
            "id": 7,
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
            await model._finalize_response(reply, ["partial"])

    @pytest.mark.asyncio
    async def test_normal_result_reply_still_returns_content(self, model) -> None:
        reply = {"id": 7, "result": {"stopReason": "end_turn"}}
        msg = await model._finalize_response(reply, ["hello ", "world"])
        assert msg.content == "hello world"


# ---------------------------------------------------------------------------
# Grok arg-compat: _map_tool_call MUST normalize rawInput on the CASE A
# (proxy tool) path per docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md
# §3.2.
#
# On grok there is now exactly ONE channel-A sub-branch: the dispatch wrapper
# (``CallMcpTool{server,toolName,arguments}`` on ≤1.0.4, ``use_tool`` on
# 1.0.5+). The former "grok calls the bound tool by its plain name" sub-branch
# was deliberately REMOVED for every ``mcp_dispatch_style="dispatch_builtin"``
# profile (incident 2026-08-24, D2): grok 1.0.5
# named its own natives after our tools (read_file, grep, execute, …), so a
# bare-title ``read_file`` is grok's LOCAL built-in, and routing it into our
# bound tool double-executed it with the CLI's arguments. Those names now live
# in ``GROK_COLLIDING_BUILTINS`` and are recorded through the generic
# ``GrokBuiltin`` recorder — asserted in
# ``test_bare_title_colliding_native_is_not_channel_a`` below.
# ---------------------------------------------------------------------------
class TestMapToolCallProxyNormalization:
    def test_dispatched_read_file_target_file_renamed(
        self, fake_home, monkeypatch
    ) -> None:
        """``target_file`` → ``file_path``, on grok's real channel-A route.

        Same oracle as before, re-routed: the rename is applied by
        ``normalize_tool_args`` inside the CASE A branch, which grok reaches
        only through the dispatch wrapper.
        """
        cs = install_client_state(monkeypatch)

        @tool
        def read_file(file_path: str) -> str:
            """Read a file."""
            return ""

        m = GrokChatModel(model="grok-4").bind_tools([read_file])
        m._client_state = cs
        m._proxy_mcp = _FakeProxy()

        st = cs.get_or_create("agentA")
        tc = m._map_tool_call(
            st,
            {
                "toolCallId": "tc-42",
                "title": "CallMcpTool",
                "rawInput": {
                    "server": m._proxy_mcp.server_name,
                    "toolName": "read_file",
                    "arguments": {"target_file": "/x"},
                },
            },
        )
        assert tc is not None
        assert tc["name"] == "read_file"
        # target_file MUST have been renamed to file_path.
        assert tc["args"] == {"file_path": "/x"}

    def test_call_mcp_tool_execute_timeout_string_coerced(
        self, fake_home, monkeypatch
    ) -> None:
        cs = install_client_state(monkeypatch)

        @tool
        def execute(command: str, timeout: int = 30) -> str:
            """Run a shell command."""
            return ""

        m = GrokChatModel(model="grok-4").bind_tools([execute])
        m._client_state = cs
        m._proxy_mcp = _FakeProxy()

        st = cs.get_or_create("agentA")
        tc = m._map_tool_call(
            st,
            {
                "toolCallId": "tc-99",
                "title": "CallMcpTool",
                "rawInput": {
                    "server": m._proxy_mcp.server_name,
                    "toolName": "execute",
                    "arguments": {"command": "sleep 1", "timeout": "20s"},
                },
            },
        )
        assert tc is not None
        assert tc["name"] == "execute"
        assert tc["args"] == {"command": "sleep 1", "timeout": 20}

    def test_dispatched_read_file_no_op_when_correct_key(
        self, fake_home, monkeypatch
    ) -> None:
        cs = install_client_state(monkeypatch)

        @tool
        def read_file(file_path: str) -> str:
            """Read a file."""
            return ""

        m = GrokChatModel(model="grok-4").bind_tools([read_file])
        m._client_state = cs
        m._proxy_mcp = _FakeProxy()

        st = cs.get_or_create("agentA")
        tc = m._map_tool_call(
            st,
            {
                "toolCallId": "tc-43",
                "title": "CallMcpTool",
                "rawInput": {
                    "server": m._proxy_mcp.server_name,
                    "toolName": "read_file",
                    "arguments": {"file_path": "/y", "offset": 100},
                },
            },
        )
        assert tc is not None
        assert tc["name"] == "read_file"
        assert tc["args"] == {"file_path": "/y", "offset": 100}

    async def test_bare_title_colliding_native_is_not_channel_a(
        self, fake_home, monkeypatch
    ) -> None:
        """REGRESSION (incident 2026-08-24, D2), grok-specific.

        A bare-title ``read_file`` is grok's OWN native, not a dispatch of our
        same-named bound tool. It must NOT become channel A: it is recorded
        through the generic ``GrokBuiltin`` recorder with its real title
        preserved under grok's ``_grok_original_title`` key, and its rawInput
        must reach us verbatim (CASE B, so NO arg-compat rename — the keys are
        grok's, not ours). Routing it into our tool double-executed it with the
        CLI's own arguments and burned a 10s MCP-handler correlation per call.

        Async on purpose: CASE B registers the passthrough Future, which needs
        a running loop (``asyncio.get_running_loop``).
        """
        cs = install_client_state(monkeypatch)

        @tool
        def read_file(file_path: str) -> str:
            """Read a file."""
            return ""

        m = GrokChatModel(model="grok-4").bind_tools([read_file])
        m._client_state = cs
        m._proxy_mcp = _FakeProxy()

        prof = build_profile()
        assert "read_file" in prof.colliding_builtins  # grok-profile fact

        st = cs.get_or_create("agentA")
        tc = m._map_tool_call(
            st,
            {
                "toolCallId": "tc-collide",
                "title": "read_file",
                "rawInput": {"target_file": "/x"},
            },
        )
        assert tc is not None
        assert tc["name"] == "GrokBuiltin"
        assert tc["args"] == {
            "target_file": "/x",
            "_grok_original_title": "read_file",
        }
        # A passthrough Future was parked — this is channel B, so the reader
        # (not a ToolMessage → proxy resolve_call) will complete it.
        assert "tc-collide" in st.builtin_passthrough_futures

    async def test_builtin_passthrough_rawInput_is_not_normalized(
        self, fake_home, monkeypatch
    ) -> None:
        """CASE B (built-in passthrough) MUST forward rawInput verbatim.

        The compat layer is deliberately scoped to CASE A only —
        built-in tools live inside grok and get their own args from grok's
        own schema (see builtin_tool_passthrough.py). Renaming keys there
        would corrupt the round-trip.
        """
        cs = install_client_state(monkeypatch)
        m = GrokChatModel(model="grok-4")
        m._client_state = cs
        m._proxy_mcp = _FakeProxy()

        st = cs.get_or_create("agentA")
        # "Read" is a KNOWN_GROK_BUILTIN — its rawInput.file_path is
        # grok's own key, unrelated to our tool aliases. It MUST NOT be
        # touched even though "file_path" happens to also be canonical for
        # our own read_file tool.
        tc = m._map_tool_call(
            st,
            {
                "toolCallId": "tc-builtin-1",
                "title": "Read",
                "rawInput": {"file_path": "/x", "offset": 0, "limit": 100},
            },
        )
        assert tc is not None
        assert tc["name"] == "Read"
        assert tc["args"] == {"file_path": "/x", "offset": 0, "limit": 100}

