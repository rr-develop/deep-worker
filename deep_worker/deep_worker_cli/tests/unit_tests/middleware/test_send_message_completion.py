"""Unit tests for SendMessageCompletionMiddleware.

Design doc:
    deep_worker_cli/docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md

Invariants under test:

* The middleware kicks in ONLY when the last state message is a
  ``ToolMessage`` whose ``name`` is ``send_message_to_agent`` and whose
  parsed payload says ``success=True`` with a non-empty
  ``delivered_content``.
* On success it returns
  ``{"messages": [AIMessage(content=<delivered>)], "jump_to": "end"}``
  — i.e. injects an echo text AIMessage and terminates the LangGraph
  turn. LangGraph's ``jump_to`` mechanism requires the hook to be
  decorated with ``@hook_config(can_jump_to=["end"])`` — also verified.
* On any other shape (other tool, failure, unparseable content, empty
  state, non-ToolMessage last message) it returns ``None`` so the
  normal agent flow runs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# NOTE: These imports fail until the implementation lands — expected
# during the TDD red phase.
from deep_worker_cli.middleware.send_message_completion import (
    TOOL_NAME,
    SendMessageCompletionMiddleware,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_state(messages: list[Any] | None) -> dict[str, Any]:
    """Build a minimal state dict the middleware expects.

    We use a plain dict (not the TypedDict) so tests don't depend on
    langgraph internals. The middleware accesses ``state["messages"]``
    via ``.get("messages", [])``, so a plain dict works.
    """
    state: dict[str, Any] = {}
    if messages is not None:
        state["messages"] = messages
    return state


def _invoke(
    mw: SendMessageCompletionMiddleware,
    messages: list[Any] | None,
) -> dict[str, Any] | None:
    """Run before_model with a fake runtime (None is fine: the middleware
    never touches it)."""
    state = _make_state(messages)
    # runtime arg is unused by our middleware; pass None for simplicity.
    return mw.before_model(state, None)  # type: ignore[arg-type]


def _tool_msg(
    content: Any,
    *,
    name: str = TOOL_NAME,
    tool_call_id: str = "call_1",
) -> ToolMessage:
    # ToolMessage requires str/list content — if caller passes a dict we
    # dump it as str() to mimic how LangGraph sometimes serializes.
    if isinstance(content, dict):
        content_str = str(content)
    else:
        content_str = content
    return ToolMessage(
        content=content_str,
        tool_call_id=tool_call_id,
        name=name,
    )


# ===========================================================================
# before_model hook config
# ===========================================================================


class TestHookConfig:
    """LangGraph only honours ``jump_to="end"`` if the hook method carries
    ``__can_jump_to__`` metadata set by ``@hook_config(can_jump_to=["end"])``.
    Without this decorator, the return value is ignored.
    """

    def test_before_model_is_marked_can_jump_to_end(self) -> None:
        method = SendMessageCompletionMiddleware.before_model
        # The decorator puts the list directly on the function object.
        can_jump_to = getattr(method, "__can_jump_to__", None)
        assert can_jump_to is not None, (
            "SendMessageCompletionMiddleware.before_model must be decorated "
            "with @hook_config(can_jump_to=[...]) for jump_to to take effect"
        )
        assert "end" in can_jump_to, (
            f"before_model must declare 'end' as a valid jump target; "
            f"got {can_jump_to!r}"
        )


# ===========================================================================
# Pass-through cases
# ===========================================================================


class TestPassThrough:
    """All cases where the middleware must do nothing (return None)."""

    def setup_method(self) -> None:
        self.mw = SendMessageCompletionMiddleware()

    def test_empty_messages_list(self) -> None:
        assert _invoke(self.mw, []) is None

    def test_missing_messages_key(self) -> None:
        assert _invoke(self.mw, None) is None

    def test_last_is_human_message(self) -> None:
        msgs = [HumanMessage(content="hello")]
        assert _invoke(self.mw, msgs) is None

    def test_last_is_system_message(self) -> None:
        msgs = [SystemMessage(content="sys")]
        assert _invoke(self.mw, msgs) is None

    def test_last_is_ai_message_with_tool_calls(self) -> None:
        # Before the tool has actually run.
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": TOOL_NAME,
                        "args": {"agent_name": "beta", "message": "hi"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
        assert _invoke(self.mw, msgs) is None

    def test_last_is_tool_message_from_other_tool(self) -> None:
        msgs = [_tool_msg("ok", name="execute")]
        assert _invoke(self.mw, msgs) is None

    def test_last_is_tool_message_unknown_name(self) -> None:
        msgs = [_tool_msg("whatever", name="mystery")]
        assert _invoke(self.mw, msgs) is None

    def test_failed_delivery_passes_through(self) -> None:
        msgs = [
            _tool_msg(
                json.dumps(
                    {
                        "success": False,
                        "error": "agent_not_found",
                        "message": "no such agent",
                    }
                )
            )
        ]
        assert _invoke(self.mw, msgs) is None

    def test_empty_delivered_content(self) -> None:
        msgs = [_tool_msg(json.dumps({"success": True, "delivered_content": ""}))]
        assert _invoke(self.mw, msgs) is None

    def test_whitespace_only_delivered_content(self) -> None:
        msgs = [
            _tool_msg(json.dumps({"success": True, "delivered_content": "   \n\t"}))
        ]
        assert _invoke(self.mw, msgs) is None

    def test_missing_delivered_content(self) -> None:
        msgs = [_tool_msg(json.dumps({"success": True}))]
        assert _invoke(self.mw, msgs) is None

    def test_unparseable_content_string(self) -> None:
        msgs = [_tool_msg("this is not json or a dict repr")]
        assert _invoke(self.mw, msgs) is None

    def test_non_dict_parseable_content(self) -> None:
        # Parses successfully but not as a dict.
        msgs = [_tool_msg(json.dumps(["not", "a", "dict"]))]
        assert _invoke(self.mw, msgs) is None


# ===========================================================================
# Termination cases — the core behaviour
# ===========================================================================


class TestTurnTermination:
    """Successful send_message_to_agent results terminate the turn."""

    def setup_method(self) -> None:
        self.mw = SendMessageCompletionMiddleware()

    def _expect_echo(
        self,
        result: dict[str, Any] | None,
        expected_content: str,
    ) -> None:
        assert result is not None, "Middleware should have returned an update"
        assert result.get("jump_to") == "end", (
            f"jump_to must be 'end', got {result.get('jump_to')!r}"
        )
        new_messages = result.get("messages")
        assert isinstance(new_messages, list) and len(new_messages) == 1
        echo = new_messages[0]
        assert isinstance(echo, AIMessage), (
            f"Echo message must be AIMessage, got {type(echo).__name__}"
        )
        assert echo.content == expected_content
        # tool_calls must be empty — this is a text response, not another tool use.
        assert not getattr(echo, "tool_calls", []), (
            "Echo AIMessage must NOT carry tool_calls"
        )

    def test_json_string_content_success_triggers_echo(self) -> None:
        delivered = "alpha -> beta :> hello beta"
        msgs = [
            _tool_msg(
                json.dumps({"success": True, "delivered_content": delivered})
            )
        ]
        self._expect_echo(_invoke(self.mw, msgs), delivered)

    def test_python_repr_string_content_success_triggers_echo(self) -> None:
        """Some LangChain code paths serialize dict tool results via str()
        which yields a Python-repr string with single quotes. The
        middleware must fall back to ast.literal_eval for that case."""
        delivered = "alpha -> beta :> hello"
        payload = {"success": True, "delivered_content": delivered}
        msgs = [_tool_msg(str(payload))]  # str() gives single-quoted repr
        self._expect_echo(_invoke(self.mw, msgs), delivered)

    def test_multiline_delivered_content_preserved(self) -> None:
        delivered = "alpha -> beta :> line1\n\nline2 with spaces  "
        msgs = [
            _tool_msg(
                json.dumps({"success": True, "delivered_content": delivered})
            )
        ]
        self._expect_echo(_invoke(self.mw, msgs), delivered)

    def test_tool_message_name_matches_constant(self) -> None:
        """Sanity: TOOL_NAME constant matches the actual tool's .name."""
        from deep_worker_cli.tools import make_send_message_to_agent_tool

        tool = make_send_message_to_agent_tool("alpha")
        assert tool.name == TOOL_NAME

    def test_full_flow_ai_tool_call_then_tool_message(self) -> None:
        """Realistic state: AIMessage(tool_calls=...) followed by
        ToolMessage. Only the tail (ToolMessage) matters for the
        middleware; verify the whole message list doesn't confuse it."""
        delivered = "alpha -> beta :> the deed is done"
        msgs = [
            HumanMessage(content="send a message to beta"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": TOOL_NAME,
                        "args": {"agent_name": "beta", "message": "the deed is done"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            _tool_msg(
                json.dumps({"success": True, "delivered_content": delivered}),
                tool_call_id="call_1",
            ),
        ]
        self._expect_echo(_invoke(self.mw, msgs), delivered)


# ===========================================================================
# AgentMiddleware interface conformance
# ===========================================================================


class TestAgentMiddlewareContract:
    """The middleware must be a proper AgentMiddleware subclass the
    deepagents / langchain agent framework can plug in."""

    def test_is_agent_middleware(self) -> None:
        from langchain.agents.middleware.types import AgentMiddleware

        mw = SendMessageCompletionMiddleware()
        assert isinstance(mw, AgentMiddleware)

    def test_exposes_no_extra_tools(self) -> None:
        mw = SendMessageCompletionMiddleware()
        # The class sets `tools = []` — must remain iterable and empty
        # to not accidentally register tools.
        assert list(mw.tools) == []

    def test_no_constructor_arguments_required(self) -> None:
        # The middleware reads everything from state; no sender_name
        # plumbing needed.
        mw = SendMessageCompletionMiddleware()
        assert mw is not None
