"""Unit tests for the generalized turn-ending tool mechanism.

See design document:
    deep_worker_cli/docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_worker_cli.middleware.tool_completion import (
    DuplicateTurnEndingToolError,
    ToolCompletionMiddleware,
    TurnEndingTool,
    _trailing_tool_message_run_length,
)


def _tool_msg(name: str, payload: dict | str, call_id: str = "call-1") -> ToolMessage:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ToolMessage(content=content, name=name, tool_call_id=call_id)


def _ai_with_calls(n: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": f"t{i}", "args": {}, "id": f"call-{i}"} for i in range(n)
        ],
    )


def _state(messages: list) -> dict:
    return {"messages": messages}


# --------------------------------------------------------------------------
# TurnEndingTool: contract and validation
# --------------------------------------------------------------------------


class TestTurnEndingToolValidation:
    def test_defaults(self):
        spec = TurnEndingTool(tool_name="foo")
        assert spec.tool_name == "foo"
        assert spec.echo_field == "delivered_content"
        assert spec.is_success({"success": True}) is True
        assert spec.is_success({"success": False}) is False
        assert spec.is_success({}) is False

    def test_is_frozen(self):
        spec = TurnEndingTool(tool_name="foo")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.tool_name = "bar"  # type: ignore[misc]

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n", None, 123])
    def test_rejects_bad_tool_name(self, bad):
        with pytest.raises((ValueError, TypeError)):
            TurnEndingTool(tool_name=bad)

    def test_rejects_non_callable_is_success(self):
        with pytest.raises(TypeError):
            TurnEndingTool(tool_name="foo", is_success=None)
        with pytest.raises(TypeError):
            TurnEndingTool(tool_name="foo", is_success="not callable")

    @pytest.mark.parametrize("bad", ["", "   ", 123])
    def test_rejects_bad_echo_field(self, bad):
        with pytest.raises((ValueError, TypeError)):
            TurnEndingTool(tool_name="foo", echo_field=bad)

    def test_echo_field_none_is_allowed(self):
        assert TurnEndingTool(tool_name="foo", echo_field=None).echo_field is None


# --------------------------------------------------------------------------
# Registry construction: duplicates are configuration errors
# --------------------------------------------------------------------------


class TestRegistryConstruction:
    def test_empty_registry_is_allowed(self):
        assert ToolCompletionMiddleware()._by_name == {}

    def test_duplicate_user_specs_raise(self):
        with pytest.raises(DuplicateTurnEndingToolError) as exc:
            ToolCompletionMiddleware(
                [TurnEndingTool("foo"), TurnEndingTool("foo", echo_field=None)]
            )
        assert "foo" in str(exc.value)

    def test_duplicate_error_is_a_valueerror(self):
        assert issubclass(DuplicateTurnEndingToolError, ValueError)

    def test_rejects_non_spec_entries(self):
        with pytest.raises(TypeError):
            ToolCompletionMiddleware(["send_message_to_agent"])  # type: ignore[list-item]

    def test_registers_no_tools_of_its_own(self):
        assert ToolCompletionMiddleware([TurnEndingTool("foo")]).tools == []


# --------------------------------------------------------------------------
# _trailing_tool_message_run_length
# --------------------------------------------------------------------------


class TestTrailingToolMessageRunLength:
    def test_empty(self):
        assert _trailing_tool_message_run_length([]) == 0

    def test_tail_not_toolmessage(self):
        msgs = [HumanMessage(content="hi"), _ai_with_calls(1), _tool_msg("a", {}), AIMessage(content="x")]
        assert _trailing_tool_message_run_length(msgs) == 0

    def test_single_trailing(self):
        msgs = [HumanMessage(content="hi"), _ai_with_calls(1), _tool_msg("a", {})]
        assert _trailing_tool_message_run_length(msgs) == 1

    def test_two_trailing(self):
        msgs = [HumanMessage(content="hi"), _ai_with_calls(2), _tool_msg("a", {}), _tool_msg("b", {})]
        assert _trailing_tool_message_run_length(msgs) == 2

    def test_boundary_is_first_non_toolmessage(self):
        """An EARLIER ToolMessage run, separated by an AIMessage, must not count."""
        msgs = [
            HumanMessage(content="hi"),
            _ai_with_calls(2),
            _tool_msg("old1", {}),
            _tool_msg("old2", {}),
            AIMessage(content="thinking"),
            _ai_with_calls(1),
            _tool_msg("new", {}),
        ]
        assert _trailing_tool_message_run_length(msgs) == 1


# --------------------------------------------------------------------------
# before_model gates
# --------------------------------------------------------------------------


class TestBeforeModelGates:
    def _mw(self, *specs: TurnEndingTool) -> ToolCompletionMiddleware:
        return ToolCompletionMiddleware(list(specs))

    def test_empty_registry_never_ends(self):
        mw = self._mw()
        state = _state([_ai_with_calls(1), _tool_msg("anything", {"success": True, "delivered_content": "x"})])
        assert mw.before_model(state) is None

    def test_no_messages(self):
        assert self._mw(TurnEndingTool("foo")).before_model(_state([])) is None

    def test_ignores_non_toolmessage_tail(self):
        mw = self._mw(TurnEndingTool("foo"))
        for tail in (HumanMessage(content="hi"), AIMessage(content="done")):
            assert mw.before_model(_state([_ai_with_calls(1), tail])) is None

    def test_matches_by_exact_name_only(self):
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg("bar", {"success": True, "delivered_content": "x"})])
        assert mw.before_model(state) is None

    @pytest.mark.parametrize(
        "name",
        ["submit_fragments", "mcp__x__submit", "send_message_to_agent_v2", "foo_", "_foo"],
    )
    def test_convention_looking_name_never_ends(self, name):
        """No name-based magic: only explicit registration makes a tool turn-ending."""
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg(name, {"success": True, "delivered_content": "x"})])
        assert mw.before_model(state) is None

    def test_unparseable_payload_no_end(self):
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg("foo", "not json and not a dict repr")])
        assert mw.before_model(state) is None

    def test_is_success_false_no_end(self):
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": False, "delivered_content": "x"})])
        assert mw.before_model(state) is None

    def test_custom_predicate(self):
        mw = self._mw(
            TurnEndingTool("foo", is_success=lambda d: d.get("document_closed") is True, echo_field=None)
        )
        no = _state([_ai_with_calls(1), _tool_msg("foo", {"document_closed": False})])
        assert mw.before_model(no) is None
        yes = _state([_ai_with_calls(1), _tool_msg("foo", {"document_closed": True})])
        assert mw.before_model(yes) == {"jump_to": "end"}

    def test_raising_predicate_no_end_and_warns(self, caplog):
        def boom(_data):
            raise RuntimeError("predicate exploded")

        mw = self._mw(TurnEndingTool("foo", is_success=boom))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True})])
        with caplog.at_level("WARNING"):
            assert mw.before_model(state) is None
        assert any("foo" in r.message or "foo" in r.getMessage() for r in caplog.records)

    def test_echo_field_missing_no_end(self):
        mw = self._mw(TurnEndingTool("foo", echo_field="delivered_content"))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True})])
        assert mw.before_model(state) is None

    @pytest.mark.parametrize("value", ["", "   ", 123, None, {"a": 1}])
    def test_echo_field_empty_or_non_string_no_end(self, value):
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True, "delivered_content": value})])
        assert mw.before_model(state) is None

    def test_echo_field_set_injects_message_and_jumps(self):
        mw = self._mw(TurnEndingTool("foo"))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True, "delivered_content": "hello there"})])
        result = mw.before_model(state)
        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        echo = result["messages"][0]
        assert isinstance(echo, AIMessage)
        assert echo.content == "hello there"

    def test_echo_field_none_jumps_without_message(self):
        """echo_field=None must NOT manufacture an empty AIMessage."""
        mw = self._mw(TurnEndingTool("foo", echo_field=None))
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True})])
        result = mw.before_model(state)
        assert result == {"jump_to": "end"}
        assert "messages" not in result

    def test_parses_json_and_repr_forms(self):
        mw = self._mw(TurnEndingTool("foo"))
        for raw in (
            '{"success": true, "delivered_content": "echoed"}',
            "{'success': True, 'delivered_content': 'echoed'}",
        ):
            state = _state([_ai_with_calls(1), _tool_msg("foo", raw)])
            result = mw.before_model(state)
            assert result is not None, raw
            assert result["messages"][0].content == "echoed"

    def test_accepts_dict_content(self):
        mw = self._mw(TurnEndingTool("foo"))
        msg = ToolMessage(content="", name="foo", tool_call_id="c1")
        object.__setattr__(msg, "content", {"success": True, "delivered_content": "d"})
        result = mw.before_model(_state([_ai_with_calls(1), msg]))
        assert result is not None and result["jump_to"] == "end"

    def test_hook_config_can_jump_end(self):
        hook = ToolCompletionMiddleware.before_model
        assert getattr(hook, "__can_jump_to__", None) == ["end"]


# --------------------------------------------------------------------------
# Parallel tool calls: only a SOLE trailing tool result may end the turn
# --------------------------------------------------------------------------


class TestParallelToolCalls:
    def _mw(self) -> ToolCompletionMiddleware:
        return ToolCompletionMiddleware([TurnEndingTool("foo")])

    def test_single_trailing_tool_message_ends(self):
        state = _state([_ai_with_calls(1), _tool_msg("foo", {"success": True, "delivered_content": "x"})])
        assert self._mw().before_model(state) is not None

    def test_parallel_registered_result_last_does_not_end(self):
        """The case a naive messages[-1] implementation gets wrong."""
        state = _state([
            _ai_with_calls(2),
            _tool_msg("sibling", {"success": False, "error": "boom"}, "call-0"),
            _tool_msg("foo", {"success": True, "delivered_content": "x"}, "call-1"),
        ])
        assert self._mw().before_model(state) is None

    def test_parallel_registered_result_first_does_not_end(self):
        state = _state([
            _ai_with_calls(2),
            _tool_msg("foo", {"success": True, "delivered_content": "x"}, "call-0"),
            _tool_msg("sibling", {"success": False, "error": "boom"}, "call-1"),
        ])
        assert self._mw().before_model(state) is None

    def test_parallel_all_successful_still_does_not_end(self):
        """The rule is run-length based, not failure-detection based."""
        state = _state([
            _ai_with_calls(2),
            _tool_msg("sibling", {"success": True}, "call-0"),
            _tool_msg("foo", {"success": True, "delivered_content": "x"}, "call-1"),
        ])
        assert self._mw().before_model(state) is None
