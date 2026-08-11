"""Regression test: 'ToolSearch' MUST be in BUILT_IN_TOOLS.

Background
==========
@anthropic-ai/claude-code 2.1.140+ enabled the Anthropic API beta feature
``tool_search_tool_bm25_20251119`` for opus/sonnet models. When that beta
is active, the CLI marks every MCP tool with ``defer_loading: true`` and
the model sees only tool *names* in the initial system prompt; full tool
schemas have to be retrieved on demand via the built-in ``ToolSearch``
tool. This breaks the contract of deep-worker, which expects its small,
fixed set of LangGraph-mediated tools to be fully known to the agent at
turn 1.

The CLI exposes a documented opt-out (visible in the strings table of
claude.exe):

    "Tool search disabled: ToolSearchTool is not available
     (may have been disallowed via disallowedTools)."

So adding the literal name ``"ToolSearch"`` to ``disallowedTools``
makes the CLI's ``isToolSearchEnabled`` short-circuit to ``false``,
which in turn sets ``deferLoading=false`` for every tool — schemas are
back in the initial prompt.

This is a strictly *local* fix: it becomes a ``--disallowedTools``
argv element of the subprocess we spawn (see
``claude_sdk/_internal/transport/subprocess_cli.py``). No environment
variable, no global config, no effect on other ``claude`` invocations.

These tests guard the fix against accidental removal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest


# Make claude_code_provider importable: this test file lives at
#   <TREE>/claude_code_provider/tests/test_tool_search_disallowed.py
# Adding parent x2 to sys.path puts <TREE>/ in front, which contains
# the ``claude_code_provider`` package.
_TREE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TREE_ROOT))


def test_tool_search_in_built_in_tools() -> None:
    """``ToolSearch`` must be present in BUILT_IN_TOOLS so it ends up
    in --disallowedTools when the agent spawns ``claude``."""
    from deep_worker.providers.claude_code_provider.client_manager import BUILT_IN_TOOLS

    assert "ToolSearch" in BUILT_IN_TOOLS, (
        "ToolSearch is missing from BUILT_IN_TOOLS. Without it the "
        "Anthropic tool-search beta will mark every MCP tool as "
        "defer_loading=True, and agents will lose immediate access "
        "to their tool schemas. See client_manager.py."
    )


# Claude Code built-ins that put the session into "ready while waiting"
# state — i.e. the model returns control with a final AIMessage while a
# real-world side effect is still pending. In interactive ``claude`` the
# CLI parks the session and re-activates it on the trigger (cron fire,
# new monitor output line, push reply, …). In deep-worker we have no
# such wait state: the LangGraph loop sees a final AIMessage and
# considers the turn complete. The visible symptom is "agent says
# ready, but it actually expected to be resumed later".
#
# Names taken verbatim from the official Claude Code tools reference:
# https://docs.claude.com/en/docs/claude-code/tools-reference
# Cross-checked against the v2.1.140 binary string table for accuracy.
_ASYNC_BUILTIN_TOOLS = (
    # Scheduled-task tools. Documented at
    # https://docs.claude.com/en/docs/claude-code/scheduled-tasks
    # The doc explicitly states: "Tasks only fire while Claude Code is
    # running AND IDLE. A scheduled prompt fires BETWEEN your turns,
    # not while Claude is mid-response." That idle-between-turns
    # contract is exactly what deep-worker has no analog for.
    "CronCreate",
    "CronDelete",
    "CronList",
    # Background-watch tool. Documented at
    # https://docs.claude.com/en/docs/claude-code/tools-reference#monitor-tool
    # "lets Claude watch something in the background and react when it
    # changes, without pausing the conversation. … Claude writes a
    # small script for the watch, runs it in the background, and
    # receives each output line as it arrives. You keep working in the
    # same session and Claude interjects when an event lands."
    "Monitor",
    # Push-notification tool. Documented in tools-reference as:
    # "Sends a desktop notification … so a long-running task or
    # scheduled task can reach you when you step away."
    # Tightly coupled with the scheduled-task / monitor model.
    "PushNotification",
    # Remote scheduling. Documented as: "Creates, updates, runs, and
    # lists Routines on claude.ai. Backs the /schedule command."
    # Routines run on Anthropic-managed infrastructure and call back
    # into the session asynchronously.
    "RemoteTrigger",
)


@pytest.mark.parametrize("tool_name", _ASYNC_BUILTIN_TOOLS)
def test_async_builtin_disallowed(tool_name: str) -> None:
    """Every Claude Code built-in that produces a "ready while waiting"
    state must be blocked at the CLI boundary. Otherwise the model
    can call it and the deep-worker LangGraph loop won't know to wait
    — the agent reports a finished turn while a side effect is still
    pending.

    The fix is the same mechanism as ToolSearch: list the name in
    BUILT_IN_TOOLS, which routes it into
    ClaudeAgentOptions.disallowed_tools -> --disallowedTools on the
    spawned ``claude`` subprocess.
    """
    from deep_worker.providers.claude_code_provider.client_manager import BUILT_IN_TOOLS

    assert tool_name in BUILT_IN_TOOLS, (
        f"{tool_name!r} is missing from BUILT_IN_TOOLS. Claude Code "
        f"exposes it as a built-in tool that schedules or watches for "
        f"events between turns. If the model can call it, deep-worker "
        f"will incorrectly report 'ready' while the pending side "
        f"effect never reaches the agent. See client_manager.py."
    )


# ─────────────────────────────────────────────────────────────────────────
# Audit-driven additions — see docs/CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md
# ─────────────────────────────────────────────────────────────────────────
# These names came out of the 2026-05-14 audit that compared our
# BUILT_IN_TOOLS list with the official Claude Code tools reference
# (https://docs.claude.com/en/docs/claude-code/tools-reference). Each of
# them either replaces a stale name we already block, or guards a feature
# that bypasses LangGraph if available. Grouped by audit category:
#
#   A) modern replacements for stale names we already block
#   B) the Task* family that supersedes our blocked TodoWrite
#   C) agent-teams tools — currently gated on
#      CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1, but we defend in depth
#
# Reasoning per name lives in the audit doc.
_AUDIT_TOOLS_GROUP_A = (
    "Agent",       # modern name for the subagent tool we used to block as "Task"
    "TaskStop",    # modern name for "KillShell"
    "TaskOutput",  # modern name for "BashOutput"
    "PowerShell",  # Bash sibling for Windows / opt-in Linux — same "execute" class
)
_AUDIT_TOOLS_GROUP_B = (
    # Task-list family that replaced TodoWrite. We already block
    # TodoWrite so the model can plan via LangGraph state; for
    # consistency we block these too.
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
)
_AUDIT_TOOLS_GROUP_C = (
    # Agent-teams tools. Inactive unless EXPERIMENTAL_AGENT_TEAMS=1 is
    # set, which we never set, but we still block them so the system
    # is safe under any future config drift.
    "SendMessage",
    "TeamCreate",
    "TeamDelete",
)
_AUDIT_NEW_TOOLS = (
    _AUDIT_TOOLS_GROUP_A + _AUDIT_TOOLS_GROUP_B + _AUDIT_TOOLS_GROUP_C
)


# ─────────────────────────────────────────────────────────────────────────
# 2026-05-15 audit additions — harness-level tools.
# ─────────────────────────────────────────────────────────────────────────
# These are "park the session" / "wait for external trigger" tools at the
# harness layer rather than the CLI's traditional tools layer. The 2026-05-14
# audit doc (CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md:139) originally claimed they
# never reach the SDK subprocess — that turned out to be wrong empirically:
# a deep-worker agent driven by claude-code SDK CAN invoke ``ScheduleWakeup``,
# which parks its turn waiting for a future re-invocation that LangGraph
# never delivers (same failure shape as ``CronCreate``/``Monitor``).
#
# Defence in depth: block the rest of the harness-level family too. Most are
# only available in specific Claude Code runtimes (claude.ai web, plan-mode
# enabled sessions, …), but blocking them is free and keeps the contract
# robust to future Claude Code releases re-routing them through the SDK.
_HARNESS_LEVEL_TOOLS = (
    # Wakeup / dynamic-loop tool. Parks the turn waiting for a delayed
    # re-invocation. Equivalent to ``CronCreate`` in our taxonomy but
    # with millisecond granularity. Empirically reachable via SDK.
    "ScheduleWakeup",
    # Plan-mode tools. Plan mode is a Claude Code feature where the
    # assistant proposes a plan and the harness gates execution on
    # user approval. There's no analog in the deep-worker LangGraph
    # loop — entering plan mode would just hang the turn.
    "EnterPlanMode",
    "ExitPlanMode",
    # Worktree tools. Tied to the Claude Code harness's git-worktree
    # feature — irrelevant in deep-worker and would create dangling
    # directories the agent has no mechanism to clean up.
    "EnterWorktree",
    "ExitWorktree",
    # claude.ai onboarding-guide upload tool. Pure UI feature, no place
    # in an autonomous agent.
    "ShareOnboardingGuide",
)


@pytest.mark.parametrize("tool_name", _HARNESS_LEVEL_TOOLS)
def test_harness_level_tool_disallowed(tool_name: str) -> None:
    """Each harness-level Claude Code tool must be present in
    BUILT_IN_TOOLS. These are tools that schedule work or pause the
    turn waiting for external input — deep-worker has no such wait
    state.

    Regression guard for the 2026-05-15 audit: ``ScheduleWakeup`` was
    initially documented as runtime-only-unreachable but is in fact
    invocable from a deep-worker agent. See
    ``CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md:139`` for the historical claim
    that we are now correcting.
    """
    from deep_worker.providers.claude_code_provider.client_manager import BUILT_IN_TOOLS

    assert tool_name in BUILT_IN_TOOLS, (
        f"{tool_name!r} is missing from BUILT_IN_TOOLS. It is a "
        f"harness-level Claude Code tool that either parks the turn "
        f"waiting for external input or controls Claude Code UI "
        f"features that have no deep-worker analog. If the model "
        f"calls it, deep-worker will incorrectly mark the turn done "
        f"while a side effect is still pending."
    )


@pytest.mark.parametrize("tool_name", _AUDIT_NEW_TOOLS)
def test_audit_new_tool_disallowed(tool_name: str) -> None:
    """Each Claude Code built-in identified by the 2026-05-14 audit
    must be present in BUILT_IN_TOOLS. See
    docs/CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md for the per-tool rationale.
    """
    from deep_worker.providers.claude_code_provider.client_manager import BUILT_IN_TOOLS

    assert tool_name in BUILT_IN_TOOLS, (
        f"{tool_name!r} is missing from BUILT_IN_TOOLS. The 2026-05-14 "
        f"tools audit flagged this name as a Claude Code built-in that "
        f"bypasses LangGraph if available. See "
        f"docs/CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md."
    )


@pytest.mark.asyncio
async def test_disallowed_tools_includes_tool_search() -> None:
    """End-to-end at the SDK boundary: the options ultimately handed
    to ClaudeSDKClient must list ``ToolSearch`` in disallowed_tools."""
    from claude_agent_sdk import ClaudeAgentOptions

    from deep_worker.providers.claude_code_provider.client_manager import ClientManager
    from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata

    tools_metadata: list[ToolMetadata] = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    base_options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    manager = ClientManager(base_options=base_options)

    captured: dict[str, Any] = {}

    def capture_options(*args: Any, **kwargs: Any) -> Mock:
        captured["options"] = kwargs.get("options")
        mock = Mock()

        async def _connect() -> None:
            return None

        mock.connect = _connect
        return mock

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        side_effect=capture_options,
    ):
        await manager.get_or_create_client(
            thread_id="test-tool-search",
            tools_metadata=tools_metadata,
        )

    options = captured["options"]
    assert options is not None
    disallowed = list(options.disallowed_tools or [])
    assert "ToolSearch" in disallowed, (
        f"ClaudeAgentOptions.disallowed_tools must contain 'ToolSearch' "
        f"to prevent the Anthropic tool-search beta from deferring tool "
        f"schemas. Got: {disallowed}"
    )
