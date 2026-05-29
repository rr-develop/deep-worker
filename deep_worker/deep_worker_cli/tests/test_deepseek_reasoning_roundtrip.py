"""Regression: the deepseek model must PASS reasoning_content BACK to the API.

Incident (observed 2026-05-22): deepseek-v4-flash is a thinking model. DeepSeek
requires that an assistant turn's ``reasoning_content`` be echoed back on
subsequent calls, or it rejects the request:

    400 - The `reasoning_content` in the thinking mode must be passed back to
          the API.

``langchain_deepseek.ChatDeepSeek`` (1.0.1) READS reasoning_content inbound
(into ``additional_kwargs``) but its ``_get_request_payload`` does NOT re-send
it outbound — so the 400 still fired on multi-step turns (verified live:
TURN1 failed even after switching to ChatDeepSeek).

Fix: ``ReasoningChatDeepSeek`` overrides ``_get_request_payload`` to re-inject
``additional_kwargs['reasoning_content']`` onto the matching outgoing
assistant message (input↔payload correlation is positional and 1:1).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_worker_cli.deepseek_model import ReasoningChatDeepSeek


def _model() -> ReasoningChatDeepSeek:
    return ReasoningChatDeepSeek(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="sk-test",
    )


def test_reasoning_content_reinjected_into_outbound_assistant() -> None:
    m = _model()
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="answer", additional_kwargs={"reasoning_content": "STEP1: ponder"}),
        HumanMessage(content="continue"),
    ]
    payload = m._get_request_payload(msgs)

    assistant = [mm for mm in payload["messages"] if mm["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].get("reasoning_content") == "STEP1: ponder", (
        "reasoning_content must be re-sent on the outgoing assistant message "
        "(observed 2026-05-22) — ChatDeepSeek 1.0.1 alone drops it"
    )


def test_positional_correlation_with_system_and_tool() -> None:
    """Re-injection must match the RIGHT assistant message even when system /
    tool messages are interleaved (positional 1:1 mapping)."""
    m = _model()
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="a1", additional_kwargs={"reasoning_content": "R1"}),
        ToolMessage(content="tool out", tool_call_id="t1"),
        AIMessage(content="a2", additional_kwargs={"reasoning_content": "R2"}),
    ]
    payload = m._get_request_payload(msgs)

    assistants = [mm for mm in payload["messages"] if mm["role"] == "assistant"]
    assert [a.get("reasoning_content") for a in assistants] == ["R1", "R2"]


def test_assistant_without_reasoning_is_untouched() -> None:
    """An assistant message with no reasoning_content must not gain a stray
    key (e.g. the very first turn, or a non-thinking response)."""
    m = _model()
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="plain answer"),  # no reasoning_content
    ]
    payload = m._get_request_payload(msgs)
    assistant = [mm for mm in payload["messages"] if mm["role"] == "assistant"][0]
    assert "reasoning_content" not in assistant


def test_inherits_chatdeepseek_content_normalization() -> None:
    """The override must still run ChatDeepSeek's content normalization (it
    calls super()), e.g. list assistant content → string."""
    m = _model()
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content=[{"type": "text", "text": "block-text"}],
            additional_kwargs={"reasoning_content": "R"},
        ),
    ]
    payload = m._get_request_payload(msgs)
    assistant = [mm for mm in payload["messages"] if mm["role"] == "assistant"][0]
    assert assistant["content"] == "block-text"  # normalized by super()
    assert assistant.get("reasoning_content") == "R"  # re-injected by us


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
