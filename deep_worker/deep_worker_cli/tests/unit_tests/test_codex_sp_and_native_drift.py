"""codex: one permission semantic with the other providers, and no native
tool that vanishes from history without a word.

Two owner directives, 2026-08-24/25:

1. "The policy really should be unified, as far as possible" —
   ``-sp`` must mean the same thing everywhere. Until now codex passed
   ``--dangerously-bypass-approvals-and-sandbox`` UNCONDITIONALLY, so an
   agent launched WITHOUT ``-sp`` still ran with approvals and sandbox
   switched off, while grok/claude-code denied every permission. The flag
   was decoration for codex.

2. "It makes sense to log new native tools ... I would not ban them
   automatically, since experience shows they do more good than harm
   more often than not." Codex item types not in ``NATIVE_TOOL_ITEM_TYPES`` were dropped
   by a bare ``# else: ... — ignore``: the tool ran inside codex, and its
   call and result never reached the agent's history, with nothing logged.
   The list is a snapshot of codex-cli 0.130.0; the installed CLI is
   0.149.1. So: record the unknown tool through a generic recorder, log it
   once so it is visible, and keep an EXPLICIT ignore list for the item
   types that are genuinely informational (reasoning/plan) or that turn out
   to disrupt the agent loop — the way claude-code's "ready while waiting"
   built-ins (CronCreate/Monitor/PushNotification/RemoteTrigger) are
   blocked because they park the session in idle.
"""

from __future__ import annotations

import logging

import pytest

from deep_worker.providers.codex_provider.chat_model import CodexChatModel
from deep_worker.providers.codex_provider.event_pump import (
    INFORMATIONAL_ITEM_TYPES,
    NATIVE_TOOL_ITEM_TYPES,
    is_recordable_tool_item,
)
from deep_worker.providers.codex_provider.exec_runner import (
    CodexExecRunner,
    ExecArgs,
)
from deep_worker.providers.codex_provider.native_tools import (
    GENERIC_NATIVE_TOOL_NAME,
    NATIVE_TOOLS,
    ORIGINAL_TYPE_ARG_KEY,
)

BYPASS = "--dangerously-bypass-approvals-and-sandbox"


def _argv(**kwargs) -> list[str]:
    runner = object.__new__(CodexExecRunner)
    args = ExecArgs(prompt="hi", **kwargs)
    return runner._build_argv(args)


class TestSpIsHonoured:
    def test_sp_bypasses_like_before(self) -> None:
        argv = _argv(auto_approve=True)
        assert BYPASS in argv

    def test_without_sp_there_is_no_bypass(self) -> None:
        """The whole point: the operator flag must MEAN something on codex
        too, exactly as it does on grok and claude-code."""
        argv = _argv(auto_approve=False)
        assert BYPASS not in argv

    def test_without_sp_codex_reviews_its_own_approvals(self) -> None:
        """``--approve-for-me``: codex auto-reviews inside its
        workspace-write sandbox, so deep-worker's own HITL interrupt is the
        only gate. Without ANY flag codex announces an MCP tool call and
        never sends the control request — every tool call then burns the
        full 120 s proxy bound and returns "not_started" (measured live:
        129.5 s for one ``execute``)."""
        argv = _argv(auto_approve=False)
        assert "--approve-for-me" in argv

    def test_approve_for_me_is_not_combined_with_sandbox_flag(self) -> None:
        """The CLI rejects the pair outright ("--approve-for-me cannot be
        used with --sandbox", exit 2), which took a live agent down."""
        argv = _argv(auto_approve=False)
        assert "-s" not in argv
        assert "approval_policy" not in " ".join(argv)

    def test_sp_path_still_pins_sandbox_and_policy(self) -> None:
        argv = _argv(auto_approve=True, sandbox_mode="workspace-write")
        assert "-s" in argv
        assert "workspace-write" in argv
        assert 'approval_policy="never"' in " ".join(argv)

    def test_default_is_fail_closed(self) -> None:
        """ExecArgs must not default to bypass — a caller that forgets the
        flag gets the SAFE mode, never the permissive one."""
        argv = _argv()
        assert BYPASS not in argv
        assert "--approve-for-me" in argv

    def test_chat_model_forwards_sp_to_the_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The constructor resolves the codex binary, which a CI box need
        not have; the wiring under test is the flag, not the binary."""
        from deep_worker.providers.codex_provider import chat_model as cm

        monkeypatch.setattr(cm, "CodexExecRunner", lambda *a, **k: object())
        assert cm.CodexChatModel(auto_approve=True)._auto_approve is True
        assert cm.CodexChatModel()._auto_approve is False

    def test_create_model_passes_sp_to_codex(self) -> None:
        import inspect

        from deep_worker_cli import config

        src = inspect.getsource(config.create_model)
        start = src.index("CodexChatModel(")
        call = src[start : src.index(")", start)]
        assert "auto_approve=auto_approve" in call, (
            "create_model does not forward -sp to codex — the flag would be "
            "decoration, which is what this test exists to prevent."
        )


class TestUnknownNativeToolsAreRecordedNotDropped:
    @pytest.mark.parametrize(
        "item_type", ["command_execution", "apply_patch", "web_search"]
    )
    def test_known_types_are_recordable(self, item_type: str) -> None:
        assert is_recordable_tool_item(item_type)

    @pytest.mark.parametrize(
        "item_type",
        [
            "dynamic_tool_call",  # codex 0.149 analytics: codex_dynamic_tool_call_event
            "collab_agent_tool_call",
            "control_tool_call",
            "some_tool_shipped_next_release",
        ],
    )
    def test_unknown_tool_types_are_recorded(self, item_type: str) -> None:
        """Not in the snapshot list, not informational → it must reach
        history through the generic recorder rather than vanish."""
        assert is_recordable_tool_item(item_type)
        assert item_type not in NATIVE_TOOL_ITEM_TYPES

    @pytest.mark.parametrize("item_type", sorted(INFORMATIONAL_ITEM_TYPES))
    def test_informational_types_stay_ignored(self, item_type: str) -> None:
        """reasoning/plan-style items carry no tool call; recording them
        would spam history with empty tool pairs."""
        assert not is_recordable_tool_item(item_type)

    def test_ignore_list_is_explicit_and_documented(self) -> None:
        """The ONLY way an item type gets dropped is by being named here —
        an owner decision (as with claude-code's idle-parking built-ins),
        never an accident of a stale allowlist."""
        assert "reasoning" in INFORMATIONAL_ITEM_TYPES
        assert not (INFORMATIONAL_ITEM_TYPES & NATIVE_TOOL_ITEM_TYPES)

    def test_generic_recorder_tool_is_bound(self) -> None:
        names = {t.name for t in NATIVE_TOOLS}
        assert GENERIC_NATIVE_TOOL_NAME in names, (
            "the generic recorder must be BOUND, otherwise emitting a "
            "tool_call under its name leaves LangGraph without a tool to run"
        )
        assert NATIVE_TOOL_ITEM_TYPES <= names


class TestChatModelSurfacesUnknownTools:
    """End-to-end on the mapping helper the ainvoke loop uses."""

    def test_known_type_keeps_its_own_name(self) -> None:
        tc = CodexChatModel._native_tool_call_for(
            item_id="i1", item_type="command_execution", args={"command": "ls"}
        )
        assert tc == {
            "id": "i1",
            "name": "command_execution",
            "args": {"command": "ls"},
            "type": "tool_call",
        }

    def test_unknown_type_routes_via_generic_recorder(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            tc = CodexChatModel._native_tool_call_for(
                item_id="i2",
                item_type="dynamic_tool_call",
                args={"payload": "x"},
            )
        assert tc is not None
        assert tc["name"] == GENERIC_NATIVE_TOOL_NAME
        assert tc["args"][ORIGINAL_TYPE_ARG_KEY] == "dynamic_tool_call"
        assert tc["args"]["payload"] == "x"
        # Visible, but not alarming: a new codex tool is expected traffic.
        assert "dynamic_tool_call" in caplog.text

    def test_informational_type_is_not_a_tool_call(self) -> None:
        assert (
            CodexChatModel._native_tool_call_for(
                item_id="i3", item_type="reasoning", args={}
            )
            is None
        )
