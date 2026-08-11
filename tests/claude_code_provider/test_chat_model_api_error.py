"""Wiring tests: ``ClaudeCodeChatModel.ainvoke`` raises ``ClaudeApiResponseError``
on turns that carry an upstream API error instead of an answer.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §3.3 (wiring)
and §5.1 A1..A8.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from claude_agent_sdk import (
    AssistantMessage as ClaudeAssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker.providers.claude_code_provider.chat_model import (
    ClaudeApiResponseError,
    ClaudeCodeChatModel,
    ClaudeUsageLimitError,
)


REAL_400 = (
    "API Error: 400 messages.18.content.0.tool_result.content.1: "
    "Input tag 'image_url' found using 'type' does not match any of the "
    "expected tags: 'document', 'image', 'search_result', 'text', "
    "'tool_reference'"
)

ALL_ZERO = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

NONZERO = {
    "input_tokens": 12,
    "output_tokens": 34,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def _fresh_model(thread_id: str = "test") -> ClaudeCodeChatModel:
    model = ClaudeCodeChatModel()
    # Discard any per-thread counter left over from other tests.
    model.client_manager._sent_count.pop(thread_id, None)
    return model


def _install_stream(
    model: ClaudeCodeChatModel, stream_factory: Any
) -> tuple[Mock, Mock]:
    """Wire ``get_or_create_client`` to yield the given async stream.

    Returns the (mock_client, mock_proxy) pair.
    """
    mock_client = Mock()
    mock_client.query = AsyncMock()
    mock_client.receive_response = stream_factory
    mock_proxy = Mock()
    mock_proxy.has_pending_calls = Mock(return_value=False)
    mock_proxy.get_pending_calls = Mock(return_value=[])
    return mock_client, mock_proxy


# ===========================================================================
# A1 — the production case
# ===========================================================================


@pytest.mark.asyncio
async def test_A1_raises_on_api_error_text_with_zero_usage() -> None:
    """Verbatim §1.2 payload: single TextBlock('API Error: 400 …') +
    ResultMessage(is_error=False), no tool calls, no bill.
    """
    model = _fresh_model()

    async def stream():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=REAL_400)],
            model="claude-sonnet-4-5-20250929",
            usage=ALL_ZERO,
            stop_reason="end_turn",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=200,
            duration_api_ms=100,
            is_error=False,
            num_turns=1,
            session_id="s",
            usage=ALL_ZERO,
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeApiResponseError) as excinfo:
            await model.ainvoke(
                [HumanMessage("hello")],
                config={"configurable": {"thread_id": "test"}},
            )

    exc = excinfo.value
    # Message = precedence 1 (marker-carrying response_text).
    assert str(exc) == REAL_400
    # No structured signal on this path — enrichment fields are neutral.
    assert exc.assistant_error is None
    assert exc.result_errors == ()


# ===========================================================================
# A2 — no regression: usage-limit path still fatal
# ===========================================================================


@pytest.mark.asyncio
async def test_A2_usage_limit_marker_still_raises_ClaudeUsageLimitError() -> None:
    model = _fresh_model()

    async def stream():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="You've hit your session limit · resets 12:00")],
            model="claude-sonnet-4-5-20250929",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeUsageLimitError):
            await model.ainvoke(
                [HumanMessage("2+2")],
                config={"configurable": {"thread_id": "test"}},
            )


# ===========================================================================
# A3 — normal answer must round-trip unchanged
# ===========================================================================


@pytest.mark.asyncio
async def test_A3_normal_answer_returns_aimessage_unchanged() -> None:
    model = _fresh_model()

    async def stream():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="4")],
            model="claude-sonnet-4-5-20250929",
            usage=NONZERO,
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            usage=NONZERO,
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        result = await model.ainvoke(
            [HumanMessage("2+2")],
            config={"configurable": {"thread_id": "test"}},
        )
    assert isinstance(result, AIMessage)
    assert result.content == "4"


# ===========================================================================
# A4 — returning_tool_calls guard: pending proxy call wins over would-be error
# ===========================================================================


@pytest.mark.asyncio
async def test_A4_returning_tool_calls_bypasses_detector() -> None:
    """When there IS a pending proxy tool call, the detector must NOT be
    consulted — even if the terminal text would otherwise trigger rule 3.

    We simulate the caller returning tool_calls by having the proxy report
    a pending call and by feeding a stream whose text would otherwise fire
    the detector. The guard is: return tool_calls, don't raise.
    """
    model = _fresh_model()

    async def stream():
        # The stream emits an "API-Error-looking" text with zero usage; the
        # guard should still short-circuit into tool-calls, not error.
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=REAL_400)],
            model="claude-sonnet-4-5-20250929",
            usage=ALL_ZERO,
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

    pending_call = {"id": "toolu_abc", "name": "mcp__deepworker__ls", "args": {"path": "/tmp"}}
    client, proxy = _install_stream(model, stream)
    proxy.has_pending_calls = Mock(return_value=True)
    proxy.get_pending_calls = Mock(return_value=[pending_call])

    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        result = await model.ainvoke(
            [HumanMessage("list /tmp")],
            config={"configurable": {"thread_id": "test"}},
        )
    assert isinstance(result, AIMessage)
    # LangChain normalises tool_calls (adds ``type='tool_call'``). Assert on
    # the fields we care about, not on dict-shape equality.
    assert len(result.tool_calls) == 1
    got = result.tool_calls[0]
    assert got["id"] == pending_call["id"]
    assert got["name"] == pending_call["name"]
    assert got["args"] == pending_call["args"]


# ===========================================================================
# A5 — prior built-in ToolUseBlock does NOT suppress terminal ResultMessage.is_error
# ===========================================================================


@pytest.mark.asyncio
async def test_A5_prior_builtin_tool_does_not_suppress_terminal_api_error() -> None:
    """Built-in tools (WebSearch, ToolSearch, ...) run inside the CLI and
    produce a ToolUseBlock in the same turn. That must NOT immunise a
    terminal ResultMessage(is_error=True) — the guard is the authoritative
    pending-tool state, not history of tool blocks.
    """
    model = _fresh_model()

    async def stream():
        # A built-in ToolUseBlock (name does NOT start with 'mcp__deepworker__')
        yield ClaudeAssistantMessage(
            content=[ToolUseBlock(id="tu_builtin", name="WebSearch", input={})],
            model="claude-sonnet-4-5-20250929",
        )
        # Terminal ResultMessage with the SDK's single error bool set.
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="s",
            result="upstream failed",
            errors=["retry later"],
        )

    client, proxy = _install_stream(model, stream)
    # No proxy call is pending — this is a real terminal error turn.
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeApiResponseError) as excinfo:
            await model.ainvoke(
                [HumanMessage("search cats")],
                config={"configurable": {"thread_id": "test"}},
            )
    # Message came from result_text (precedence 2) because response_text is empty.
    assert str(excinfo.value) == "upstream failed"
    assert excinfo.value.result_errors == ("retry later",)


# ===========================================================================
# A6 — resolved proxy tool earlier + terminal marker+known-zero text turn
# ===========================================================================
# Two-phase realistic reproduction:
#   Turn 1: HumanMessage → stream emits proxy ToolUseBlock → ainvoke returns
#           AIMessage(tool_calls=[...]). No detector consultation (guard).
#   Turn 2: [HumanMessage, AIMessage(tool_calls=...), ToolMessage(result)] →
#           tool_resolved=True path re-enters receive_response, which now
#           yields the terminal "API Error: 400 ..." text + KNOWN_ZERO usage
#           ResultMessage with NO pending proxy calls remaining →
#           returning_tool_calls is False → detector fires → raise
#           ClaudeApiResponseError.
# This is the exact "resolved-proxy-then-terminal-API-error" post-resolve
# control flow that A5 (built-in tool + ResultMessage.is_error) does NOT
# cover: A5 pins the built-in + structured signal path, A6 pins the
# resolved-proxy + textual-fallback path. Both must fire.


@pytest.mark.asyncio
async def test_A6_resolved_proxy_then_terminal_marker_raises() -> None:
    from claude_agent_sdk import ToolUseBlock  # noqa: PLC0415
    from langchain_core.messages import ToolMessage  # noqa: PLC0415

    model = _fresh_model()
    proxy_tool_id = "toolu_resolved_ok"
    proxy_tool_name = "mcp__deepworker__ls"

    # ── Phase 1: proxy ToolUseBlock in the first stream ────────────────
    async def stream_phase1():
        yield ClaudeAssistantMessage(
            content=[
                ToolUseBlock(
                    id=proxy_tool_id,
                    name=proxy_tool_name,
                    input={"path": "/tmp"},
                )
            ],
            model="claude-sonnet-4-5-20250929",
        )
        # No ResultMessage — ainvoke breaks on the proxy block and waits
        # for the id-keyed pending registration.

    client, proxy = _install_stream(model, stream_phase1)
    # Prime the proxy pending registry BEFORE the ToolUseBlock arrives so
    # the id-keyed wait resolves synchronously.
    proxy._pending_call_by_id = {
        proxy_tool_id: {
            "id": proxy_tool_id,
            "name": proxy_tool_name,
            "args": {"path": "/tmp"},
        },
    }
    proxy.has_pending_calls = lambda: bool(proxy._pending_call_by_id)
    proxy.get_pending_calls = lambda: list(proxy._pending_call_by_id.values())

    # Register the id-keyed future so wait_for_pending_call_by_id resolves.
    async def _wait_id(_call_id: str, timeout: float | None = None) -> dict[str, Any]:
        return proxy._pending_call_by_id[_call_id]
    proxy.wait_for_pending_call_by_id = _wait_id

    # Track resolve_call so Phase 2 knows the tool was resolved.
    resolved_ids: list[str] = []

    def _resolve(call_id: str, _result: Any) -> None:
        resolved_ids.append(call_id)
        proxy._pending_call_by_id.pop(call_id, None)
    proxy.resolve_call = _resolve

    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        result1 = await model.ainvoke(
            [HumanMessage("list /tmp")],
            config={"configurable": {"thread_id": "test-a6"}},
        )
    assert isinstance(result1, AIMessage)
    assert len(result1.tool_calls) == 1
    assert result1.tool_calls[0]["id"] == proxy_tool_id

    # ── Phase 2: ToolMessage resolves + terminal marker+KNOWN_ZERO ─────
    async def stream_phase2():
        # After resolve_call, Claude finalises the turn — the CLI sends the
        # upstream API error as a plain TextBlock (observed production
        # shape) and closes with a KNOWN_ZERO-usage ResultMessage.
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="API Error: 400 upstream failed after tool run")],
            model="claude-sonnet-4-5-20250929",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        yield ResultMessage(
            subtype="success",         # NOT is_error — structured flag off
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,            # NOT is_error — verbatim production shape
            num_turns=2,
            session_id="s-a6",
            stop_reason="end_turn",
        )

    # Rebind receive_response for the second turn. resolve_call in Phase 2
    # pops the id from _pending_call_by_id, so has_pending_calls() returns
    # False at the end of the receive loop — the guard evaluates
    # returning_tool_calls=False and the detector runs.
    async def _receive2():
        async for ev in stream_phase2():
            yield ev
    client.receive_response = _receive2

    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeApiResponseError) as excinfo:
            await model.ainvoke(
                [
                    HumanMessage("list /tmp"),
                    result1,                          # AIMessage with tool_calls
                    ToolMessage(
                        content="ok",
                        tool_call_id=proxy_tool_id,
                    ),
                ],
                config={"configurable": {"thread_id": "test-a6"}},
            )

    # The detector fired on the resolved-proxy → terminal-textual path.
    assert "API Error: 400" in str(excinfo.value)
    # Sanity: the tool WAS actually resolved on Phase 2 (not bypassed).
    assert resolved_ids == [proxy_tool_id], (
        f"Phase 2 must exercise the tool-resolve control flow; "
        f"resolved_ids={resolved_ids!r}"
    )


# ===========================================================================
# A7 — AssistantMessage.error alone triggers, even with empty text
# ===========================================================================


@pytest.mark.asyncio
async def test_A7_assistant_error_with_empty_text_raises_non_empty() -> None:
    model = _fresh_model()

    async def stream():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="")],
            model="claude-sonnet-4-5-20250929",
            error="invalid_request",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeApiResponseError) as excinfo:
            await model.ainvoke(
                [HumanMessage("hi")],
                config={"configurable": {"thread_id": "test"}},
            )
    exc = excinfo.value
    assert str(exc)  # non-empty
    assert "invalid_request" in str(exc)
    assert exc.assistant_error == "invalid_request"


# ===========================================================================
# A8 — ResultMessage(is_error=True) with no assistant text
# ===========================================================================


@pytest.mark.asyncio
async def test_A8_result_is_error_without_assistant_text_raises() -> None:
    model = _fresh_model()

    async def stream():
        # No AssistantMessage at all — just a terminal ResultMessage.is_error.
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="s",
            errors=["upstream 503"],
            result="Retryable",
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(model.client_manager, "get_or_create_client", return_value=(client, proxy)):
        with pytest.raises(ClaudeApiResponseError) as excinfo:
            await model.ainvoke(
                [HumanMessage("hi")],
                config={"configurable": {"thread_id": "test"}},
            )
    exc = excinfo.value
    # Precedence 2: result_text wins over joined errors when response_text empty.
    assert str(exc) == "Retryable"
    assert exc.result_errors == ("upstream 503",)


@pytest.mark.asyncio
async def test_A9_api_error_rotates_poisoned_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deep_worker.providers.claude_code_provider.session_state import (  # noqa: PLC0415
        load_session_state,
        save_session_state,
    )

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
    model = _fresh_model("test-a9")
    model.client_manager._session_epoch.pop("test-a9", None)
    save_session_state(
        "test-a9", sent_count=0, session_uuid="poisoned-session-id"
    )
    save_session_state("test-a9:tm", sent_count=2, session_uuid="tm-session-id")

    async def stream():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=REAL_400)],
            model="claude-sonnet-4-5-20250929",
            usage=ALL_ZERO,
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="poisoned-session-id",
            stop_reason="end_turn",
        )

    client, proxy = _install_stream(model, stream)
    with patch.object(
        model.client_manager,
        "get_or_create_client",
        return_value=(client, proxy),
    ):
        with pytest.raises(ClaudeApiResponseError):
            await model.ainvoke(
                [HumanMessage("trigger invalid image_url")],
                config={"configurable": {"thread_id": "test-a9"}},
            )

    assert model.client_manager.get_session_epoch("test-a9") == 1
    assert load_session_state("test-a9") is None
    assert load_session_state("test-a9:tm") == {
        "sent_count": 2,
        "session_uuid": "tm-session-id",
    }
