"""TDD for the ACP (Agent Client Protocol) event taxonomy used by the
grok provider.

``acp_types`` is pure data + tiny classifier helpers — no I/O, no asyncio.
It exists so ``acp_client`` / ``chat_model`` can route a raw JSON-RPC
``session/update`` notification to the right handler without scattering
string literals across the codebase.

All tests are hermetic: no grok binary, no subprocess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from grok_provider import acp_types  # noqa: E402


class TestSessionUpdateClassification:
    def test_known_update_kinds_present(self) -> None:
        # The kinds we explicitly handle in §9.4.
        for kind in (
            "user_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
            "tool_call",
            "tool_call_update",
            "plan",
        ):
            assert kind in acp_types.HANDLED_SESSION_UPDATE_KINDS

    def test_session_update_kind_extracts_field(self) -> None:
        upd = {"sessionUpdate": "agent_message_chunk", "content": {"text": "hi"}}
        assert acp_types.session_update_kind(upd) == "agent_message_chunk"

    def test_session_update_kind_missing_returns_none(self) -> None:
        assert acp_types.session_update_kind({}) is None


class TestVendorNotificationFiltering:
    @pytest.mark.parametrize(
        "method",
        [
            "_x.ai/mcp_initialized",
            "_x.ai/git_head_changed",
            "_x.ai/queue/changed",
            "_x.ai/settings/update",
            "_x.ai/sessions/changed",
            "_x.ai/models/update",
            "_x.ai/session/prompt_complete",
            "_x.ai/announcements/update",
        ],
    )
    def test_vendor_methods_ignored(self, method: str) -> None:
        assert acp_types.is_ignored_vendor_notification(method) is True

    def test_session_update_not_ignored(self) -> None:
        assert acp_types.is_ignored_vendor_notification("session/update") is False


class TestBuiltinClassification:
    def test_known_builtins_include_core_tools(self) -> None:
        for t in ("Shell", "Read", "Write", "Edit", "Glob", "Grep"):
            assert t in acp_types.KNOWN_GROK_BUILTINS

    def test_call_mcp_tool_is_not_a_builtin(self) -> None:
        # CallMcpTool routes to channel A (proxy MCP), never passthrough.
        assert "CallMcpTool" not in acp_types.KNOWN_GROK_BUILTINS

    def test_imagine_tools_in_known_set(self) -> None:
        assert "GenerateImage" in acp_types.IMAGINE_GROK_BUILTINS
        assert "GenerateImage" in acp_types.KNOWN_GROK_BUILTINS
        for name in ("image_gen", "image_edit", "EditImage"):
            assert name in acp_types.IMAGINE_GROK_BUILTINS
            assert name in acp_types.KNOWN_GROK_BUILTINS

    def test_is_proxy_mcp_dispatch_true(self) -> None:
        upd = {
            "title": "CallMcpTool",
            "rawInput": {"server": "deep_worker_proxy", "toolName": "execute"},
        }
        assert acp_types.is_proxy_mcp_dispatch(upd, "deep_worker_proxy") is True

    def test_is_proxy_mcp_dispatch_other_server_false(self) -> None:
        upd = {
            "title": "CallMcpTool",
            "rawInput": {"server": "some_other", "toolName": "x"},
        }
        assert acp_types.is_proxy_mcp_dispatch(upd, "deep_worker_proxy") is False

    def test_is_proxy_mcp_dispatch_non_mcp_title_false(self) -> None:
        upd = {"title": "Shell", "rawInput": {"command": "ls"}}
        assert acp_types.is_proxy_mcp_dispatch(upd, "deep_worker_proxy") is False


class TestStopReason:
    @pytest.mark.parametrize(
        "reason",
        ["end_turn", "max_tokens", "refusal", "cancelled", "max_turn_requests"],
    )
    def test_protocol_stop_reasons_are_terminal(self, reason: str) -> None:
        # Per ACP, a stopReason in the session/prompt RESULT always ends the
        # turn (a tool call does NOT produce a stopReason — it keeps the same
        # prompt in-flight). So every documented stopReason is terminal.
        assert acp_types.is_terminal_stop_reason(reason) is True

    def test_none_is_not_terminal(self) -> None:
        assert acp_types.is_terminal_stop_reason(None) is False

    def test_unknown_reason_is_not_terminal(self) -> None:
        # Defensive: an unrecognized value should NOT silently end the turn.
        assert acp_types.is_terminal_stop_reason("something_new") is False
