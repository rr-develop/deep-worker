"""A turn-ending tool must not leave the ACP turn wedged (round 8 #2).

``send_message_to_agent`` is turn-ending: ``ToolCompletionMiddleware`` jumps
to the graph end as soon as it succeeds, so the NEXT ``ainvoke`` — the only
place that normally resolves a proxy-MCP tool Future — never runs in that
graph run. The parked MCP handler is then handed a FALSE "still running…
job_N" notice 120s later, the model chases the phantom job (parking more
calls), and the turn only unwedges on the next external impulse.

Observed live on a claude-code agent: delivery at 15:41:51, the turn's
own text surfacing at 18:28:56.

``claude-code-sdk`` already solved this with
``InterAgentDeliveryResolveMiddleware``, which is DUCK-TYPED on
``resolve_pending_tool``. It was never registered for ACP models purely
because ``AcpChatModel`` did not provide that method — so the fix is to
provide it and register by CAPABILITY, not by concrete class.

Second half of the same defect: ``record_summarization`` explicitly leaves
``pending_session_prompt_future`` "to the caller", and the caller never
cleared it. With ``sent_count`` reset to 0 every historical ToolMessage sets
``tool_resolved=True``, so step 9 takes the mid-turn-resume branch and
SWALLOWS the first post-summarization prompt.
"""
from __future__ import annotations

import asyncio

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState


class _RecordingProxy:
    server_name = "deep_worker_acp_proxy_1"

    def __init__(self) -> None:
        self.resolved: list[tuple[str, object]] = []

    def resolve_call(self, call_id, content, tool_name=None):
        self.resolved.append((call_id, content))


def _model(proxy=None, client_state=None) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._proxy_mcp = proxy
    model._client_state = client_state or AcpClientState()
    return model


class TestResolvePendingTool:
    def test_resolves_through_the_acp_id_mapping(self) -> None:
        """The CLI knows the ACP id; the handler parked under a synthetic
        one. Without the hop resolve_call is a no-op and we are back to the
        120s false timeout."""
        proxy = _RecordingProxy()
        cs = AcpClientState()
        state = cs.get_or_create("thread-1")
        state.proxy_call_id_by_acp_id["acp-77"] = "handler-9"
        model = _model(proxy, cs)

        model.resolve_pending_tool("thread-1", "acp-77", {"content": []})

        assert proxy.resolved == [("handler-9", {"content": []})], (
            f"did not resolve via the mapped handler id: {proxy.resolved!r}"
        )

    def test_consumes_the_mapping_entry(self) -> None:
        """Resolving must not leave the correlation entry behind to be
        resolved a second time by the normal ToolMessage path."""
        proxy = _RecordingProxy()
        cs = AcpClientState()
        state = cs.get_or_create("thread-1")
        state.proxy_call_id_by_acp_id["acp-77"] = "handler-9"
        model = _model(proxy, cs)

        model.resolve_pending_tool("thread-1", "acp-77", {"content": []})
        assert "acp-77" not in state.proxy_call_id_by_acp_id

    def test_falls_back_to_the_raw_id_when_unmapped(self) -> None:
        proxy = _RecordingProxy()
        model = _model(proxy)
        model.resolve_pending_tool("no-such-thread", "acp-5", {"content": []})
        assert proxy.resolved == [("acp-5", {"content": []})]

    def test_never_raises(self) -> None:
        """Middleware calls this inside a tool turn; it must be inert on
        every failure mode rather than breaking the turn."""
        class _Boom:
            server_name = "x"

            def resolve_call(self, *a, **k):
                raise RuntimeError("boom")

        _model(_Boom()).resolve_pending_tool("t", "id", {"content": []})
        _model(None).resolve_pending_tool("t", "id", {"content": []})


class TestMiddlewareRegistration:
    """The seam is duck-typed, so ACP must qualify by capability."""

    def test_acp_model_exposes_the_duck_typed_method(self) -> None:
        assert callable(getattr(AcpChatModel, "resolve_pending_tool", None)), (
            "AcpChatModel lacks resolve_pending_tool, so the delivery-resolve "
            "middleware silently skips ACP — this IS the defect"
        )

    def test_registration_is_by_capability_not_class(self) -> None:
        from deep_worker_cli.agent import _maybe_add_delivery_resolve_middleware

        class _Capable:
            def resolve_pending_tool(self, *a, **k):
                pass

        stack: list = []
        _maybe_add_delivery_resolve_middleware(stack, _Capable())
        assert len(stack) == 1, (
            "a model exposing resolve_pending_tool was not wired up"
        )

    def test_models_without_the_capability_are_skipped(self) -> None:
        from deep_worker_cli.agent import _maybe_add_delivery_resolve_middleware

        class _Plain:
            pass

        stack: list = []
        _maybe_add_delivery_resolve_middleware(stack, _Plain())
        _maybe_add_delivery_resolve_middleware(stack, "deepseek-chat")
        assert stack == [], "wired a model that cannot be resolved early"

    def test_claude_code_still_registered(self) -> None:
        """Regression: the original beneficiary must keep working."""
        from deep_worker_cli.agent import _maybe_add_delivery_resolve_middleware

        class _ClaudeLike:
            def resolve_pending_tool(self, *a, **k):
                pass

            def drain_after_inter_agent_send(self, *a, **k):
                pass

        stack: list = []
        _maybe_add_delivery_resolve_middleware(stack, _ClaudeLike())
        assert len(stack) == 1


class TestSummarizationDropsStalePromptFuture:
    def test_pending_future_is_cancelled(self) -> None:
        """A future belonging to a dropped session must not survive it."""

        async def scenario():
            state = AcpClientState().get_or_create("t")
            fut = asyncio.get_running_loop().create_future()
            state.pending_session_prompt_future = fut
            AcpChatModel._discard_pending_prompt(state, "summarization")
            return fut, state.pending_session_prompt_future

        fut, still_set = asyncio.run(scenario())
        assert still_set is None, (
            "stale prompt future survived summarization — step 9 will treat "
            "the next fresh turn as a mid-turn resume and swallow the "
            "user's message"
        )
        assert fut.cancelled()

    def test_idempotent_and_safe_when_absent(self) -> None:
        state = AcpClientState().get_or_create("t")
        AcpChatModel._discard_pending_prompt(state, "x")
        AcpChatModel._discard_pending_prompt(state, "x")
        assert state.pending_session_prompt_future is None

    def test_done_future_is_not_cancelled_again(self) -> None:
        async def scenario():
            state = AcpClientState().get_or_create("t")
            fut = asyncio.get_running_loop().create_future()
            fut.set_result({"ok": True})
            state.pending_session_prompt_future = fut
            AcpChatModel._discard_pending_prompt(state, "x")
            return fut

        fut = asyncio.run(scenario())
        assert fut.done() and not fut.cancelled()
