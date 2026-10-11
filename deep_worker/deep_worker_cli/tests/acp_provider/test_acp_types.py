"""Unit tests for the pure ACP event-taxonomy classifiers (shared layer).

Covers ``deep_worker/providers/acp/acp_types.py``: session-update kind
routing, vendor-notification filtering, terminal stop reasons, and the
tool-name / MCP-dispatch classifiers that differ between providers
(grok's ``from_title`` + ``dispatch_builtin`` vs claude-agent-acp's
``from_claude_code_meta`` + ``mcp_name_prefix``).
"""

from __future__ import annotations

import pytest

from deep_worker.providers.acp import acp_types


class TestSessionUpdateKind:
    def test_kind_extracted_from_session_update(self) -> None:
        upd = {"sessionUpdate": "tool_call", "title": "Bash"}
        assert acp_types.session_update_kind(upd) == "tool_call"

    def test_missing_kind_returns_none(self) -> None:
        assert acp_types.session_update_kind({}) is None

    def test_non_dict_returns_none(self) -> None:
        assert acp_types.session_update_kind(None) is None
        assert acp_types.session_update_kind("nope") is None

    def test_non_string_kind_returns_none(self) -> None:
        assert acp_types.session_update_kind({"sessionUpdate": 42}) is None


class TestIsIgnoredVendorNotification:
    def test_vendor_prefixed_method_is_ignored(self) -> None:
        assert acp_types.is_ignored_vendor_notification("_claude/git_head_changed", "_claude/")

    def test_standard_acp_method_not_ignored(self) -> None:
        assert not acp_types.is_ignored_vendor_notification(
            "session/update", "_claude/"
        )
        assert not acp_types.is_ignored_vendor_notification(
            "session/request_permission", "_claude/"
        )

    def test_empty_method_or_prefix_not_ignored(self) -> None:
        assert not acp_types.is_ignored_vendor_notification(None, "_claude/")
        assert not acp_types.is_ignored_vendor_notification("x/y", "")


class TestIsTerminalStopReason:
    def test_documented_reasons_are_terminal(self) -> None:
        for reason in ("end_turn", "max_tokens", "refusal", "cancelled", "max_turn_requests"):
            assert acp_types.is_terminal_stop_reason(reason), reason

    def test_unknown_and_none_are_not_terminal(self) -> None:
        assert not acp_types.is_terminal_stop_reason("some_new_reason")
        assert not acp_types.is_terminal_stop_reason(None)


class TestToolNameFromUpdate:
    def test_from_title_default(self) -> None:
        upd = {"title": "Bash", "rawInput": {"command": "ls"}}
        assert acp_types.tool_name_from_update(upd) == "Bash"

    def test_from_title_missing_returns_none(self) -> None:
        assert acp_types.tool_name_from_update({"rawInput": {}}) is None

    def test_from_claude_code_meta(self) -> None:
        upd = {
            "title": "Read /etc/hosts (1-30)",
            "_meta": {"claudeCode": {"toolName": "Read"}},
        }
        assert (
            acp_types.tool_name_from_update(upd, "from_claude_code_meta") == "Read"
        )

    def test_from_claude_code_meta_falls_back_to_none(self) -> None:
        # title is prose, meta missing → no usable name (NOT the prose title).
        upd = {"title": "Read /etc/hosts (1-30)"}
        assert acp_types.tool_name_from_update(upd, "from_claude_code_meta") is None

    def test_from_claude_code_meta_empty_tool_name(self) -> None:
        upd = {"_meta": {"claudeCode": {"toolName": ""}}}
        assert acp_types.tool_name_from_update(upd, "from_claude_code_meta") is None


class TestMcpPrefixedToolName:
    def test_our_server_prefix_extracts_inner_tool(self) -> None:
        name = "mcp__deep_worker_acp_proxy_abc__probe_hello"
        assert (
            acp_types.mcp_prefixed_tool_name(name, "deep_worker_acp_proxy_abc")
            == "probe_hello"
        )

    def test_other_server_returns_none(self) -> None:
        name = "mcp__other_server__probe_hello"
        assert (
            acp_types.mcp_prefixed_tool_name(name, "deep_worker_acp_proxy_abc")
            is None
        )

    def test_plain_builtin_name_returns_none(self) -> None:
        assert acp_types.mcp_prefixed_tool_name("Bash", "proxy_x") is None

    def test_empty_inner_returns_none(self) -> None:
        assert (
            acp_types.mcp_prefixed_tool_name("mcp__proxy_x__", "proxy_x") is None
        )


class TestIsProxyMcpDispatch:
    def test_mcp_name_prefix_style(self) -> None:
        upd = {
            "title": "probe_hello",
            "_meta": {"claudeCode": {"toolName": "mcp__proxy_x__probe_hello"}},
        }
        assert acp_types.is_proxy_mcp_dispatch(upd, "proxy_x", "mcp_name_prefix")

    def test_mcp_name_prefix_other_server_false(self) -> None:
        upd = {
            "title": "probe_hello",
            "_meta": {"claudeCode": {"toolName": "mcp__other__probe_hello"}},
        }
        assert not acp_types.is_proxy_mcp_dispatch(upd, "proxy_x", "mcp_name_prefix")

    def test_mcp_name_prefix_plain_builtin_false(self) -> None:
        upd = {"title": "Bash", "_meta": {"claudeCode": {"toolName": "Bash"}}}
        assert not acp_types.is_proxy_mcp_dispatch(upd, "proxy_x", "mcp_name_prefix")

    def test_dispatch_builtin_style_matches_server(self) -> None:
        upd = {"rawInput": {"server": "proxy_x", "toolName": "read_file"}}
        assert acp_types.is_proxy_mcp_dispatch(upd, "proxy_x", "dispatch_builtin")

    def test_dispatch_builtin_style_other_server_false(self) -> None:
        upd = {"rawInput": {"server": "other"}}
        assert not acp_types.is_proxy_mcp_dispatch(upd, "proxy_x", "dispatch_builtin")

    def test_non_dict_update_false(self) -> None:
        assert not acp_types.is_proxy_mcp_dispatch(None, "proxy_x", "mcp_name_prefix")
