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
 - ToolMessage resolution: proxy-MCP resolves the parked call; grok_builtin
   ToolMessage does NOT
 - summarization rotation drops the session id, keeps the thread id
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

from grok_provider import chat_model as cm  # noqa: E402
from grok_provider.acp_client import GrokAcpClientError  # noqa: E402
from grok_provider.chat_model import (  # noqa: E402
    GrokChatModel,
    GrokTurnCancelledError,
)
from grok_provider.limit_errors import GrokUsageLimitError  # noqa: E402
from grok_provider.client_state import GrokClientState  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeAcpClient:
    """Minimal stand-in for GrokAcpClient driven by the test."""

    def __init__(self) -> None:
        self.started = False
        self.sinks: dict[str, asyncio.Queue] = {}
        self.prompts: list[dict] = []
        self.session_counter = 0
        self.last_prompt_future: asyncio.Future | None = None
        self.session_new_calls: list[dict] = []

    async def start(self) -> None:
        self.started = True

    async def initialize(self) -> dict:
        return {"result": {}}

    async def authenticate(self) -> dict:
        return {"result": {}}

    async def aclose(self) -> None:
        self.started = False

    async def session_new(self, *, cwd: str, mcp_servers: list[dict]) -> str:
        self.session_counter += 1
        self.session_new_calls.append({"cwd": cwd, "mcp_servers": mcp_servers})
        return f"sess-{self.session_counter}"

    def register_sink(self, session_id: str, queue: asyncio.Queue) -> None:
        self.sinks[session_id] = queue

    def session_prompt(self, *, session_id, prompt, notification_sink):
        self.register_sink(session_id, notification_sink)
        self.prompts.append({"session_id": session_id, "prompt": prompt})
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.last_prompt_future = fut
        return fut

    async def session_cancel(self, session_id) -> None:
        return None

    # Test helpers ---------------------------------------------------------
    def push_update(self, session_id: str, update: dict) -> None:
        self.sinks[session_id].put_nowait(
            {"method": "session/update",
             "params": {"sessionId": session_id, "update": update}}
        )

    def finish(self, stop_reason: str = "end_turn") -> None:
        assert self.last_prompt_future is not None
        if not self.last_prompt_future.done():
            self.last_prompt_future.set_result(
                {"result": {"stopReason": stop_reason}}
            )


class _FakeProxy:
    def __init__(self) -> None:
        self.server_name = "deep_worker_proxy_test"
        self.resolved: list[tuple[str, Any]] = []
        self._tools: list[dict] = []

    def update_tools(self, md: list[dict]) -> None:
        self._tools = md

    async def url_async(self) -> str:
        return "http://127.0.0.1:0/mcp"

    async def start(self) -> None:
        return None

    def resolve_call(self, call_id: str, result: Any) -> None:
        self.resolved.append((call_id, result))


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def model(fake_home, monkeypatch):
    cs = GrokClientState()
    monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)
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
        from grok_provider.builtin_tool_passthrough import (  # noqa: PLC0415
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
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)
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

        monkeypatch.setattr(
            cm, "get_global_proxy_mcp_server", lambda md: _Proxy()
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
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)

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
    def test_unknown_builtin_routes_to_grok_builtin(self, model) -> None:
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

    def test_known_imagine_builtin_surfaces_by_acp_title(self, model) -> None:
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
        payload = cm._render_completed_payload(upd, "completed")
        assert str(img) in payload["text"]
        assert payload["grok_meta"]["artifact_path"] == str(img)
        assert payload["grok_meta"]["mime"] == "image/png"
        assert payload["grok_meta"]["kind"] == "image"


class TestToolContract:
    def test_live_verified_imagine_guidance_included(self) -> None:
        contract = cm._build_tool_contract(["execute"])
        assert "GenerateImage" in contract
        assert "image_edit" in contract

    def test_str_replace_prefers_bound_edit_file(self) -> None:
        contract = cm._build_tool_contract(["edit_file"])
        assert "StrReplace -> `edit_file`" in contract

    def test_contract_disambiguates_mcp_vs_native(self) -> None:
        # §8 of DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md: the contract
        # must state these are MCP tools, NOT grok's native built-ins, and
        # must still forbid hunting for the schema on disk.
        contract = cm._build_tool_contract(["send_message_to_agent"])
        assert "MCP tool" in contract
        assert "NOT" in contract and "built-in" in contract
        assert "mcps/" in contract  # disk-hunt prohibition preserved
        # No misleading literal mcp__deepworker__ prefix advertised to grok.
        assert "mcp__deepworker" not in contract


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


class TestToolResolution:
    async def test_proxy_tool_message_resolves_call(self, model) -> None:
        # Simulate mid-turn: an in-flight prompt future + a proxy ToolMessage.
        st = model._client_state.get_or_create("agentA")
        st.grok_session_id = "sess-9"
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
        st.grok_session_id = "sess-9"
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
            ToolMessage(content="output", tool_call_id="tc-b",
                        additional_kwargs={"grok_builtin": True}),
        ]
        task = asyncio.create_task(model.ainvoke(messages, config=_cfg("agentA")))
        await asyncio.sleep(0.01)
        # grok_builtin ToolMessage must NOT hit the proxy MCP.
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
        st.grok_session_id = "sess-race"
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
    async def test_summarization_drops_session(self, model) -> None:
        st = model._client_state.get_or_create("agentA")
        st.sent_count = 10
        st.grok_session_id = "sess-old"
        model._client_state.register_session("sess-old", "agentA")

        acp = model._acp_client
        # New, SHORTER message list -> summarization detected.
        task = asyncio.create_task(
            model.ainvoke([HumanMessage(content="fresh")], config=_cfg("agentA"))
        )
        await asyncio.sleep(0.01)
        # The old session must have been evicted from the index.
        assert model._client_state.thread_for_session("sess-old") is None
        sid = f"sess-{acp.session_counter}"
        acp.push_update(sid, {"sessionUpdate": "agent_message_chunk",
                              "content": {"text": "ok"}})
        await asyncio.sleep(0.01)
        acp.finish("end_turn")
        await asyncio.wait_for(task, timeout=1)
        # A brand-new session was created.
        assert sid != "sess-old"


class TestCancellationClassification:
    @pytest.mark.asyncio
    async def test_acp_cancelled_stop_reason_is_not_asyncio_task_cancellation(
        self, model
    ) -> None:
        with pytest.raises(GrokTurnCancelledError):
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
        with pytest.raises(GrokUsageLimitError):
            await model._finalize_response(reply, ["partial"])

    @pytest.mark.asyncio
    async def test_generic_error_reply_raises_client_error(self, model) -> None:
        reply = {"id": 7, "error": {"code": -32000, "message": "internal boom"}}
        with pytest.raises(GrokAcpClientError):
            await model._finalize_response(reply, [])

    @pytest.mark.asyncio
    async def test_error_reply_does_not_return_empty_aimessage(self, model) -> None:
        # The regression: no exception, empty content, silent no-failover.
        reply = {"id": 7, "error": {"message": "insufficient credits"}}
        with pytest.raises(GrokUsageLimitError):
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
        with pytest.raises(GrokUsageLimitError):
            await model._finalize_response(reply, ["partial"])

    @pytest.mark.asyncio
    async def test_normal_result_reply_still_returns_content(self, model) -> None:
        reply = {"id": 7, "result": {"stopReason": "end_turn"}}
        msg = await model._finalize_response(reply, ["hello ", "world"])
        assert msg.content == "hello world"


# ---------------------------------------------------------------------------
# Grok arg-compat: _map_tool_call MUST normalize rawInput on the CASE A
# (proxy tool) path per docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md
# §3.2. Two proxy sub-branches:
#   1) grok calls the bound tool by its plain name (title == tool name).
#   2) grok wraps the call in CallMcpTool with rawInput.toolName / arguments.
# ---------------------------------------------------------------------------
class TestMapToolCallProxyNormalization:
    def test_bound_tool_read_file_target_file_renamed(self, fake_home, monkeypatch) -> None:
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)

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
                "title": "read_file",
                "rawInput": {"target_file": "/x"},
            },
        )
        assert tc is not None
        assert tc["name"] == "read_file"
        # target_file MUST have been renamed to file_path.
        assert tc["args"] == {"file_path": "/x"}

    def test_call_mcp_tool_execute_timeout_string_coerced(
        self, fake_home, monkeypatch
    ) -> None:
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)

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

    def test_bound_tool_read_file_no_op_when_correct_key(
        self, fake_home, monkeypatch
    ) -> None:
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)

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
                "title": "read_file",
                "rawInput": {"file_path": "/y", "offset": 100},
            },
        )
        assert tc is not None
        assert tc["args"] == {"file_path": "/y", "offset": 100}

    def test_builtin_passthrough_rawInput_is_not_normalized(
        self, fake_home, monkeypatch
    ) -> None:
        """CASE B (built-in passthrough) MUST forward rawInput verbatim.

        The compat layer is deliberately scoped to CASE A only —
        built-in tools live inside grok and get their own args from grok's
        own schema (see builtin_tool_passthrough.py). Renaming keys there
        would corrupt the round-trip.
        """
        cs = GrokClientState()
        monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)
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

