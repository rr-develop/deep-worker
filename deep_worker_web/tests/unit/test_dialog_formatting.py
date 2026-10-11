"""Unit tests for the rewritten dialog formatter.

Covers all 50 cases from
DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md sections 4.1-4.6 and 4.8.

(The 5 no-limit RPyC-server tests live in the agent-side test suite
(test_get_history_no_limit.py) because they target the agent-side
``_handle_get_history`` and need the CLI import path.)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.dialog_formatter import (
    NO_RESPONSE_TEXT,
    classify_human,
    extract_text,
    format_tool_call,
    render_dialog,
    split_into_turns,
)


# ============================================================================
# Helpers — message factories. These produce the exact dict shape that
# ``AgentProcess._handle_get_history`` emits over RPyC. Tests work directly
# on this representation so they never need a real RPyC connection.
# ============================================================================


def human(content: str, msg_id: str = "h") -> dict[str, Any]:
    return {"type": "HumanMessage", "id": msg_id, "content": content}


def ai(
    content: str | list[Any] = "",
    tool_calls: list[dict[str, Any]] | None = None,
    msg_id: str = "a",
) -> dict[str, Any]:
    msg: dict[str, Any] = {"type": "AIMessage", "id": msg_id, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool(
    content: str = "",
    tool_call_id: str = "tc1",
    name: str = "execute",
    msg_id: str = "t",
) -> dict[str, Any]:
    return {
        "type": "ToolMessage",
        "id": msg_id,
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
        "status": "success",
    }


def tc(name: str = "execute", args: dict[str, Any] | None = None, tc_id: str = "tc1") -> dict[str, Any]:
    return {"id": tc_id, "name": name, "args": args or {}}


def system(content: str, msg_id: str = "s") -> dict[str, Any]:
    return {"type": "SystemMessage", "id": msg_id, "content": content}


def system(content: str, msg_id: str = "s") -> dict[str, Any]:
    return {"type": "SystemMessage", "id": msg_id, "content": content}


def system(content: str, msg_id: str = "s") -> dict[str, Any]:
    return {"type": "SystemMessage", "id": msg_id, "content": content}


def fmt(history: list[dict[str, Any]], verbosity: str = "final") -> str:
    """Shortcut: split + render in one go."""
    turns = split_into_turns(history)
    return render_dialog(turns, verbosity)


NO_HISTORY = "No conversation history yet."
NO_RESPONSE = "(no response yet)"


# ============================================================================
# 4.1  Default mode (verbosity=final)
# ============================================================================


class TestDefaultMode:
    def test_empty_history(self) -> None:
        assert fmt([]) == NO_HISTORY

    def test_single_user_no_response(self) -> None:
        out = fmt([human("Hello")])
        assert out == "--- User ---\nHello\n\n--- Assistant ---\n(no response yet)"

    def test_single_user_with_simple_answer(self) -> None:
        out = fmt([human("Hello"), ai("ok")])
        assert "--- User ---\nHello" in out
        assert "--- Assistant ---\nok" in out

    def test_takes_last_text_ai_not_first(self) -> None:
        out = fmt([
            human("q"),
            ai("preamble"),
            ai("middle"),
            ai("FINAL"),
        ])
        assert "FINAL" in out
        assert "preamble" not in out
        assert "middle" not in out

    def test_skips_tool_only_ai_messages(self) -> None:
        out = fmt([
            human("q"),
            ai("", tool_calls=[tc("execute", {"command": "ls"})]),
            ai("done"),
        ])
        assert "done" in out
        # tool name must NOT appear in final mode
        assert "execute" not in out

    def test_handles_content_blocks(self) -> None:
        out = fmt([
            human("q"),
            ai(content=[{"type": "text", "text": "hi"}]),
        ])
        assert "hi" in out

    def test_handles_mixed_content_blocks(self) -> None:
        out = fmt([
            human("q"),
            ai(content=[
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "name": "x", "input": {}},
            ]),
        ])
        assert "answer" in out

    def test_intermediate_tool_calls_skipped_in_final(self) -> None:
        out = fmt([
            human("q"),
            ai("p1"),
            tool("r1"),
            ai("p2"),
            tool("r2"),
            ai("FINAL"),
        ])
        assert "FINAL" in out
        assert "p1" not in out
        assert "p2" not in out

    def test_multiple_turns_each_takes_last_text(self) -> None:
        out = fmt([
            human("q1"), ai("p1"), ai("F1"),
            human("q2"), ai("p2"), ai("F2"),
            human("q3"), ai("p3"), ai("F3"),
        ])
        for f in ("F1", "F2", "F3"):
            assert f in out
        for p in ("p1", "p2", "p3"):
            assert p not in out

    def test_multiple_turns_chronological_order(self) -> None:
        out = fmt([
            human("first-q"), ai("first-A"),
            human("second-q"), ai("second-A"),
            human("third-q"), ai("third-A"),
        ])
        i1 = out.index("first-A")
        i2 = out.index("second-A")
        i3 = out.index("third-A")
        assert i1 < i2 < i3

    def test_trailing_user_no_ai_renders_no_response(self) -> None:
        out = fmt([
            human("q1"), ai("a1"),
            human("q2"), ai("a2"),
            human("dangling"),
        ])
        # last block must include placeholder and the dangling query
        assert "dangling" in out
        assert NO_RESPONSE in out
        # earlier turns still rendered with their answers
        assert "a1" in out
        assert "a2" in out


# ============================================================================
# 4.2  Non-direct user sources (visible at all verbosity levels)
# ============================================================================


class TestNonDirectUserSources:
    def test_taskmanager_turn_visible_default(self) -> None:
        out = fmt([human("TaskManager:> do X"), ai("done")])
        assert "--- TaskManager ---" in out
        # prefix stripped
        assert "TaskManager:> " not in out.replace("--- TaskManager ---", "")
        assert "do X" in out
        assert "done" in out

    def test_taskmanager_turn_visible_text_mode(self) -> None:
        out = fmt([human("TaskManager:> do X"), ai("done")], verbosity="text")
        assert "--- TaskManager ---" in out
        assert "done" in out

    def test_taskmanager_turn_visible_full_mode(self) -> None:
        out = fmt([human("TaskManager:> do X"), ai("done")], verbosity="full")
        assert "--- TaskManager ---" in out
        assert "done" in out

    def test_taskmanager_no_response_renders_placeholder(self) -> None:
        out = fmt([human("TaskManager:> do X")])
        assert "--- TaskManager ---" in out
        assert NO_RESPONSE in out

    def test_agent_echo_turn_visible_default(self) -> None:
        out = fmt([human("agent:eva:> hi"), ai("ok")])
        assert "--- From agent:eva ---" in out
        # prefix stripped from body
        assert "agent:eva:> " not in out.replace("--- From agent:eva ---", "")
        assert "hi" in out
        assert "ok" in out

    def test_agent_echo_turn_visible_text_mode(self) -> None:
        out = fmt([human("agent:eva:> hi"), ai("ok")], verbosity="text")
        assert "--- From agent:eva ---" in out
        assert "ok" in out

    def test_agent_echo_turn_visible_full_mode(self) -> None:
        out = fmt([human("agent:eva:> hi"), ai("ok")], verbosity="full")
        assert "--- From agent:eva ---" in out
        assert "ok" in out

    def test_agent_echo_no_response_renders_placeholder(self) -> None:
        out = fmt([human("agent:eva:> hi")])
        assert "--- From agent:eva ---" in out
        assert NO_RESPONSE in out

    def test_mixed_sources_all_visible(self) -> None:
        out = fmt([
            human("real user q"),
            ai("user-A"),
            human("TaskManager:> tm prompt"),
            ai("tm-A"),
            human("agent:eva:> peer note"),
            ai("peer-A"),
        ])
        assert "--- User ---" in out
        assert "--- TaskManager ---" in out
        assert "--- From agent:eva ---" in out
        # chronological
        i_user = out.index("user-A")
        i_tm = out.index("tm-A")
        i_peer = out.index("peer-A")
        assert i_user < i_tm < i_peer

    def test_taskmanager_and_user_never_filtered(self) -> None:
        # 4 turns, mixing real user and TaskManager — every one MUST appear.
        history = [
            human("u1"), ai("a1"),
            human("TaskManager:> tm1"), ai("at1"),
            human("u2"), ai("a2"),
            human("TaskManager:> tm2"), ai("at2"),
        ]
        out = fmt(history)
        # 4 user-side headers expected (2 User + 2 TaskManager)
        assert out.count("--- User ---") == 2
        assert out.count("--- TaskManager ---") == 2
        # 4 assistant blocks
        assert out.count("--- Assistant ---") == 4


# ============================================================================
# 4.3  Summarization — MUST be visible, labelled with its own header
#
# Rationale (DESIGN_SUMMARIZATION_VISIBLE.md): hiding the summarization
# node breaks post-mortem analysis because the reader cannot tell why the
# agent's behaviour pivots after a context compaction. The node is the
# explanation and MUST render — but under an unambiguous header so it is
# clearly distinct from real user input.
# ============================================================================


SUMMARIZATION_HEADER = "--- Summarization ---"


class TestSummarizationVisibility:
    def test_summarization_human_is_rendered_with_summarization_header(self) -> None:
        out = fmt([
            human("Context Extraction Assistant\nblah blah"),
            ai("synthetic-summary"),
            human("real query"),
            ai("real-A"),
        ])
        # summarization turn is visible and labelled
        assert SUMMARIZATION_HEADER in out
        # its synthetic AI body (the distilled recap) is visible
        assert "synthetic-summary" in out
        # the marker substring appears only under --- Summarization ---,
        # NOT under a --- User --- block preceding it
        pre_sum = out.split(SUMMARIZATION_HEADER, 1)[0]
        assert "Context Extraction Assistant" not in pre_sum
        # the following real turn is intact
        assert "real query" in out
        assert "real-A" in out

    def test_summarization_legacy_marker_is_rendered(self) -> None:
        out = fmt([
            human("You are in the middle of a conversation that has been summarized\n\nblah"),
            ai("recap-of-prior-context"),
            human("real"),
            ai("real-A"),
        ])
        assert SUMMARIZATION_HEADER in out
        assert "recap-of-prior-context" in out
        assert "real-A" in out

    def test_summarization_only_history_still_renders(self) -> None:
        # Prior contract returned NO_HISTORY; new contract: summarization
        # alone IS content and MUST render.
        out = fmt([
            human("You are in the middle of a conversation that has been summarized: foo"),
        ])
        assert out != NO_HISTORY
        assert SUMMARIZATION_HEADER in out

    def test_summarization_and_following_turn_both_render(self) -> None:
        out = fmt([
            human("You are in the middle of a conversation that has been summarized: ctx"),
            human("real"),
            ai("real-A"),
        ])
        assert out.count(SUMMARIZATION_HEADER) == 1
        assert out.count("--- User ---") == 1
        assert "real" in out
        assert "real-A" in out

    def test_summarization_body_preserved_verbatim(self) -> None:
        """Body is NOT silently truncated — post-mortem readers need the
        full distilled summary."""
        out = fmt([
            human("Context Extraction Assistant\n---\nDistilled: agent did X, found Y."),
            ai("continuation"),
        ])
        assert out.count(SUMMARIZATION_HEADER) == 1
        assert "Distilled: agent did X, found Y." in out
        assert "continuation" in out

    def test_summarization_visible_in_text_mode(self) -> None:
        out = fmt([
            human("Context Extraction Assistant\nsummary-body"),
            ai("synthetic-1"),
            ai("synthetic-2"),
        ], verbosity="text")
        assert SUMMARIZATION_HEADER in out
        # text mode renders all text-bearing AIMessages in the turn
        assert "synthetic-1" in out
        assert "synthetic-2" in out

    def test_summarization_visible_in_full_mode_with_tool_calls(self) -> None:
        out = fmt([
            human("Context Extraction Assistant\nsummary-body"),
            ai("synthetic", tool_calls=[tc("execute", {"command": "ls"})]),
        ], verbosity="full")
        assert SUMMARIZATION_HEADER in out
        assert "synthetic" in out
        # full mode surfaces tool-calls attached to the synthetic AI
        assert "execute" in out


# ============================================================================
# 4.4  verbosity=text
# ============================================================================


class TestTextMode:
    def test_text_mode_includes_all_text_ai_messages(self) -> None:
        out = fmt([
            human("q"),
            ai("step1"),
            ai("step2"),
            ai("FINAL"),
        ], verbosity="text")
        assert "step1" in out
        assert "step2" in out
        assert "FINAL" in out
        # FINAL is last
        assert out.index("step1") < out.index("step2") < out.index("FINAL")

    def test_text_mode_still_skips_tool_only_ai(self) -> None:
        out = fmt([
            human("q"),
            ai("text1"),
            ai("", tool_calls=[tc("execute", {})]),
            ai("text2"),
        ], verbosity="text")
        assert "text1" in out
        assert "text2" in out
        # tool name absent
        assert "execute" not in out

    def test_text_mode_no_response_renders_placeholder(self) -> None:
        out = fmt([human("q")], verbosity="text")
        assert NO_RESPONSE in out

    def test_text_mode_does_not_emit_tool_calls(self) -> None:
        out = fmt([
            human("q"),
            ai("", tool_calls=[tc("execute", {"command": "ls"})]),
            ai("done"),
        ], verbosity="text")
        assert "execute" not in out
        assert "command" not in out
        assert "done" in out


# ============================================================================
# 4.5  verbosity=full
# ============================================================================


class TestFullMode:
    def test_full_mode_renders_tool_call_lines(self) -> None:
        out = fmt([
            human("q"),
            ai("", tool_calls=[tc("execute", {"command": "ls"})]),
            ai("done"),
        ], verbosity="full")
        assert "--- Assistant (tool: execute) ---" in out
        assert "→ execute(" in out
        assert "done" in out

    def test_full_mode_truncates_long_args(self) -> None:
        big = {"command": "x" * 500}
        out = fmt([
            human("q"),
            ai("", tool_calls=[tc("execute", big)]),
        ], verbosity="full")
        # the truncation marker must appear and total args length capped
        assert "..." in out
        # no unbroken 400-char string of x's must survive
        assert "x" * 400 not in out

    def test_full_mode_preserves_chronological_order(self) -> None:
        out = fmt([
            human("q"),
            ai("first-text"),
            ai("", tool_calls=[tc("execute", {"command": "first"})]),
            ai("middle-text"),
            ai("", tool_calls=[tc("read_file", {"path": "/x"})]),
            ai("FINAL"),
        ], verbosity="full")
        i_ft = out.index("first-text")
        i_exec = out.index("→ execute(")
        i_mt = out.index("middle-text")
        i_read = out.index("→ read_file(")
        i_final = out.index("FINAL")
        assert i_ft < i_exec < i_mt < i_read < i_final

    def test_full_mode_handles_unicode_args(self) -> None:
        out = fmt([
            human("q"),
            ai("", tool_calls=[tc("execute", {"command": "search"})]),
        ], verbosity="full")
        assert "search" in out  # not escaped to \u…

    def test_full_mode_no_response_still_renders_placeholder(self) -> None:
        out = fmt([human("q")], verbosity="full")
        assert NO_RESPONSE in out


# ============================================================================
# 4.8  Helper purity (extracted to its own module)
# ============================================================================


class TestClassifyHuman:
    def test_classify_human_user(self) -> None:
        assert classify_human("just a user message") == "user"

    def test_classify_human_taskmanager(self) -> None:
        assert classify_human("TaskManager:> do X") == "taskmanager"

    def test_classify_human_agent_echo(self) -> None:
        assert classify_human("agent:eva:> peer note") == "agent_echo"

    def test_classify_human_agent_echo_malformed_falls_to_user(self) -> None:
        # starts with 'agent:' but no ':> '
        assert classify_human("agent:bad-format-no-arrow") == "user"

    def test_classify_human_summarization_explicit(self) -> None:
        assert classify_human(
            "You are in the middle of a conversation that has been summarized\n\netc"
        ) == "summarization"

    def test_classify_human_summarization_context_extraction(self) -> None:
        assert classify_human(
            "Some preamble Context Extraction Assistant rest of body..."
        ) == "summarization"


class TestExtractText:
    def test_extract_text_from_string_content(self) -> None:
        assert extract_text({"content": "hello"}) == "hello"

    def test_extract_text_from_block_list(self) -> None:
        msg = {
            "content": [
                {"type": "text", "text": "hi "},
                {"type": "tool_use", "name": "x"},
                {"type": "text", "text": "there"},
            ]
        }
        assert extract_text(msg) == "hi there"

    def test_extract_text_returns_empty_for_tool_only(self) -> None:
        msg = {
            "content": [
                {"type": "tool_use", "name": "x", "input": {}},
            ]
        }
        assert extract_text(msg) == ""


# ============================================================================
# Bonus helper test: format_tool_call truncation contract
# ============================================================================


class TestFormatToolCall:
    def test_short_args_kept_intact(self) -> None:
        s = format_tool_call(tc("execute", {"command": "ls"}))
        assert s == '→ execute({"command": "ls"})'

    def test_long_args_truncated_with_ellipsis(self) -> None:
        s = format_tool_call(tc("execute", {"command": "x" * 500}))
        assert s.endswith("...)")
        # full string under reasonable cap
        assert len(s) <= 220


# ============================================================================
# System messages are first-class in history rendering.
#
# POLICY: Rendering full conversation history WITHOUT system prompts is
# misleading and is forbidden. System messages (the base agent prompt AND
# runtime middleware hints such as the inter-agent reply hint) MUST be
# rendered at every verbosity at or above "text" ("All assistant text").
# They are intentionally omitted only from the most compact "final" view.
# See DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §"System messages".
# ============================================================================


class TestSystemMessagesVisible:
    def test_leading_system_prompt_rendered_in_text_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_leading_system_prompt_rendered_in_full_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_midturn_system_hint_rendered_in_text_mode(self) -> None:
        # The inter-agent reply hint is injected AFTER a HumanMessage.
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("You just received a message from agent 'alpha'."),
                ai("Sure, reviewing now."),
            ],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You just received a message from agent 'alpha'." in out
        # And the assistant text is still present, in order.
        assert "Sure, reviewing now." in out

    def test_midturn_system_hint_rendered_in_full_mode_in_order(self) -> None:
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("HINT: reply via send_message_to_agent."),
                ai("On it.", tool_calls=[tc("execute", {"command": "ls"})]),
            ],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "HINT: reply via send_message_to_agent." in out
        # System hint must appear before the assistant text that follows it.
        assert out.index("HINT: reply") < out.index("On it.")

    def test_system_prompt_omitted_in_final_mode(self) -> None:
        # "final" is the most compact view: one assistant block per turn.
        # System messages are intentionally not surfaced there.
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="final",
        )
        assert "--- System ---" not in out
        assert "You are a helpful assistant." not in out

    def test_system_only_history_renders_in_text_mode(self) -> None:
        # A history that is only a system prompt (no human turn yet) must
        # still surface the system prompt at text/full verbosity.
        out = fmt([system("You are a helpful assistant.")], verbosity="text")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_system_only_history_renders_in_full_mode(self) -> None:
        out = fmt([system("You are a helpful assistant.")], verbosity="full")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out


# ============================================================================
# System messages are first-class in history rendering.
#
# POLICY: Rendering full conversation history WITHOUT system prompts is
# misleading and is forbidden. System messages (the base agent prompt AND
# runtime middleware hints such as the inter-agent reply hint) MUST be
# rendered at every verbosity at or above "text" ("All assistant text").
# They are intentionally omitted only from the most compact "final" view.
# See DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §"System messages".
# ============================================================================


class TestSystemMessagesVisible:
    def test_leading_system_prompt_rendered_in_text_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_leading_system_prompt_rendered_in_full_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_midturn_system_hint_rendered_in_text_mode(self) -> None:
        # The inter-agent reply hint is injected AFTER a HumanMessage.
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("You just received a message from agent 'alpha'."),
                ai("Sure, reviewing now."),
            ],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You just received a message from agent 'alpha'." in out
        # And the assistant text is still present, in order.
        assert "Sure, reviewing now." in out

    def test_midturn_system_hint_rendered_in_full_mode_in_order(self) -> None:
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("HINT: reply via send_message_to_agent."),
                ai("On it.", tool_calls=[tc("execute", {"command": "ls"})]),
            ],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "HINT: reply via send_message_to_agent." in out
        # System hint must appear before the assistant text that follows it.
        assert out.index("HINT: reply") < out.index("On it.")

    def test_system_prompt_omitted_in_final_mode(self) -> None:
        # "final" is the most compact view: one assistant block per turn.
        # System messages are intentionally not surfaced there.
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="final",
        )
        assert "--- System ---" not in out
        assert "You are a helpful assistant." not in out

    def test_system_only_history_renders_in_text_mode(self) -> None:
        # A history that is only a system prompt (no human turn yet) must
        # still surface the system prompt at text/full verbosity.
        out = fmt([system("You are a helpful assistant.")], verbosity="text")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_system_only_history_renders_in_full_mode(self) -> None:
        out = fmt([system("You are a helpful assistant.")], verbosity="full")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out


# ============================================================================
# System messages are first-class in history rendering.
#
# POLICY: Rendering full conversation history WITHOUT system prompts is
# misleading and is forbidden. System messages (the base agent prompt AND
# runtime middleware hints such as the inter-agent reply hint) MUST be
# rendered at every verbosity at or above "text" ("All assistant text").
# They are intentionally omitted only from the most compact "final" view.
# See DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §"System messages".
# ============================================================================


class TestSystemMessagesVisible:
    def test_leading_system_prompt_rendered_in_text_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_leading_system_prompt_rendered_in_full_mode(self) -> None:
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_midturn_system_hint_rendered_in_text_mode(self) -> None:
        # The inter-agent reply hint is injected AFTER a HumanMessage.
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("You just received a message from agent 'alpha'."),
                ai("Sure, reviewing now."),
            ],
            verbosity="text",
        )
        assert "--- System ---" in out
        assert "You just received a message from agent 'alpha'." in out
        # And the assistant text is still present, in order.
        assert "Sure, reviewing now." in out

    def test_midturn_system_hint_rendered_in_full_mode_in_order(self) -> None:
        out = fmt(
            [
                human("alpha -> beta :> please review"),
                system("HINT: reply via send_message_to_agent."),
                ai("On it.", tool_calls=[tc("execute", {"command": "ls"})]),
            ],
            verbosity="full",
        )
        assert "--- System ---" in out
        assert "HINT: reply via send_message_to_agent." in out
        # System hint must appear before the assistant text that follows it.
        assert out.index("HINT: reply") < out.index("On it.")

    def test_system_prompt_omitted_in_final_mode(self) -> None:
        # "final" is the most compact view: one assistant block per turn.
        # System messages are intentionally not surfaced there.
        out = fmt(
            [system("You are a helpful assistant."), human("q"), ai("a")],
            verbosity="final",
        )
        assert "--- System ---" not in out
        assert "You are a helpful assistant." not in out

    def test_system_only_history_renders_in_text_mode(self) -> None:
        # A history that is only a system prompt (no human turn yet) must
        # still surface the system prompt at text/full verbosity.
        out = fmt([system("You are a helpful assistant.")], verbosity="text")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out

    def test_system_only_history_renders_in_full_mode(self) -> None:
        out = fmt([system("You are a helpful assistant.")], verbosity="full")
        assert "--- System ---" in out
        assert "You are a helpful assistant." in out


# ---------------------------------------------------------------------------
# Recoverable failures are visible at ``final`` too
#
# ``final`` hides system messages by design — but hiding the runtime's
# failure message made a failed turn read as a finished one: the model's
# last musing was shown as the answer and the error vanished. Reported by
# the owner 2026-09-04: "it just looked like the model's final answer".
# See docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §5.
# ---------------------------------------------------------------------------


_RUNTIME_FAILURE = (
    "[runtime] The previous turn failed with an internal error "
    "(MaxIterationsExceeded: The agent hit its per-turn iteration cap "
    "without producing a final response.)."
)


def _turn_with_runtime_failure() -> list[dict[str, Any]]:
    """The real shape from agent ``test_lsp``: a musing, then the failure."""
    return [
        {"type": "HumanMessage", "content": "Determine what the project does"},
        {"type": "AIMessage", "content": "<think> Let me plan this out. </think>"},
        {
            "type": "SystemMessage",
            "content": _RUNTIME_FAILURE,
            "recoverable_error": True,
        },
    ]


def test_recoverable_failure_is_visible_at_final():
    turns = split_into_turns(_turn_with_runtime_failure())

    out = render_dialog(turns, verbosity="final")

    assert "MaxIterationsExceeded" in out, (
        "the default rendering dropped the failure — a failed turn reads "
        "as a finished one"
    )


def test_recoverable_failure_does_not_replace_the_assistant_text():
    """Both are shown: what the model said, and that the turn failed."""
    turns = split_into_turns(_turn_with_runtime_failure())

    out = render_dialog(turns, verbosity="final")

    assert "Let me plan this out" in out
    assert out.index("Let me plan this out") < out.index("MaxIterationsExceeded")


def test_an_unflagged_system_message_stays_hidden_at_final():
    """Otherwise the fix would just be 'show everything at final'.

    The base prompt and routine middleware hints are precisely what
    ``final`` exists to omit.
    """
    history = [
        {"type": "HumanMessage", "content": "hi"},
        {"type": "AIMessage", "content": "hello"},
        {"type": "SystemMessage", "content": "You are a helpful agent."},
    ]
    turns = split_into_turns(history)

    assert "helpful agent" not in render_dialog(turns, verbosity="final")
    assert "helpful agent" in render_dialog(turns, verbosity="text")


def test_a_failed_turn_with_no_assistant_text_reports_the_failure():
    """``NO_RESPONSE_TEXT`` here would be disinformation.

    "The model said nothing" and "the turn crashed" are different facts;
    the second one is the one the reader needs.
    """
    history = [
        {"type": "HumanMessage", "content": "go"},
        {
            "type": "SystemMessage",
            "content": _RUNTIME_FAILURE,
            "recoverable_error": True,
        },
    ]
    turns = split_into_turns(history)

    out = render_dialog(turns, verbosity="final")

    assert "MaxIterationsExceeded" in out
    assert NO_RESPONSE_TEXT not in out


def test_text_and_full_still_show_every_system_message():
    """The flag must not become a filter at the verbose levels."""
    history = [
        {"type": "HumanMessage", "content": "hi"},
        {"type": "AIMessage", "content": "hello"},
        {"type": "SystemMessage", "content": "You are a helpful agent."},
        {
            "type": "SystemMessage",
            "content": _RUNTIME_FAILURE,
            "recoverable_error": True,
        },
    ]
    turns = split_into_turns(history)

    for verbosity in ("text", "full"):
        out = render_dialog(turns, verbosity=verbosity)
        assert "helpful agent" in out, verbosity
        assert "MaxIterationsExceeded" in out, verbosity
