"""Unit tests for the permission policy (DESIGN §4.6).

Two states only: fail-closed without ``-sp``, BLANKET approval with it.
The per-tool allowlist this module used to exercise is GONE — owner
directive 2026-08-24, after a stale list cost grok four days of shell
access; see tests/acp_provider/test_sp_blanket_permission.py for the
invariants that keep it gone.

Still covered here: ``AcpClient._handle_permission_request`` option
selection (the CLI's allow/deny option KINDS remain per-CLI) and
``_permission_request_kind`` with the claude-agent-acp shapes —
``params.kind`` is NOT filled for tool asks, the tool name lives in
``params.toolCall._meta.claudeCode.toolName`` (claude) or ``title``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker.providers.acp.profile import PermissionPolicy


class TestPermissionPolicy:
    def test_default_is_deny_everything(self) -> None:
        policy = PermissionPolicy()
        assert not policy.auto_approve
        assert not policy.is_allowed_kind("Bash")

    def test_sp_approves_every_kind(self) -> None:
        """The whole contract: -sp approves whatever the CLI asks about.

        These three used to be three DIFFERENT answers (allowlisted -> yes,
        mcp__ -> yes, anything else -> no), which is how a tool the list had
        never heard of took an agent down."""
        policy = PermissionPolicy(auto_approve=True)
        assert policy.is_allowed_kind("Bash")
        assert policy.is_allowed_kind("mcp__proxy__read_file")
        assert policy.is_allowed_kind("a_tool_nobody_has_written_yet")

    def test_without_sp_nothing_is_approved(self) -> None:
        policy = PermissionPolicy()
        assert not policy.is_allowed_kind("Bash")
        assert not policy.is_allowed_kind("mcp__proxy__read_file")


class TestPermissionRequestKind:
    """Static fallback chain: params.kind → toolCall._meta.claudeCode.toolName
    → toolCall.title → "" (never raises)."""

    @staticmethod
    def _kind(params: dict) -> str:
        return AcpClient._permission_request_kind(params)

    def test_explicit_kind_wins(self) -> None:
        assert self._kind({"kind": "Read", "toolCall": {"title": "ignored"}}) == "Read"

    def test_non_string_kind_ignored(self) -> None:
        params = {
            "kind": 42,
            "toolCall": {"_meta": {"claudeCode": {"toolName": "Bash"}}},
        }
        assert self._kind(params) == "Bash"

    def test_claude_meta_tool_name(self) -> None:
        params = {"toolCall": {"_meta": {"claudeCode": {"toolName": "Read"}}}}
        assert self._kind(params) == "Read"

    def test_title_is_not_treated_as_a_tool_name(self) -> None:
        """``toolCall.title`` is the tool's CONTENT (the shell command, the
        file path), never its name — see ``toolInfoFromToolUse`` in the
        adapter. Consulting it made every top-level ask miss the allowlist
        and silently denied everything under ``-sp``."""
        assert self._kind({"toolCall": {"title": "Bash"}}) == ""
        assert self._kind({"toolCall": {"title": "rm -rf /"}}) == ""

    def test_empty_params_returns_empty_string(self) -> None:
        assert self._kind({}) == ""


class TestPermissionRequestKindRealAdapterShapes:
    """VERBATIM ``session/request_permission`` params from claude-agent-acp
    0.68.0 (read out of the installed adapter's ``dist/acp-agent.js`` +
    ``dist/tools.js``), NOT invented shapes.

    Two facts the previous fixtures got wrong:

    * ``_meta.claudeCode.toolName`` is attached ONLY when the tool use has a
      ``parentToolUseId`` (sub-agent call). For a top-level tool the spread
      is ``...(parentToolUseId ? { _meta: ... } : {})`` — i.e. NO ``_meta``.
    * ``toolCall.title`` is NOT the tool name. ``toolInfoFromToolUse`` sets
      it to the tool's *content*: the shell command for ``Bash``, the file
      path for ``Read``. ``toolCall.kind`` is the ACP category
      (``execute``/``read``/``edit``/...), never the built-in's name.

    So the tool name is simply NOT RECOVERABLE for a top-level ask, and the
    allowlist must not be consulted with a command string.
    """

    @staticmethod
    def _kind(params: dict) -> str:
        return AcpClient._permission_request_kind(params)

    def test_top_level_bash_is_not_reported_as_a_command_string(self) -> None:
        params = {
            "sessionId": "sess-1",
            "toolCall": {
                "toolCallId": "toolu_01",
                "rawInput": {"command": "ls -la"},
                "title": "ls -la",
                "kind": "execute",
                "content": [],
            },
        }
        assert self._kind(params) != "ls -la"

    def test_top_level_read_is_not_reported_as_a_file_path(self) -> None:
        params = {
            "sessionId": "sess-1",
            "toolCall": {
                "toolCallId": "toolu_02",
                "rawInput": {"file_path": "/etc/hosts"},
                "title": "/etc/hosts",
                "kind": "read",
                "content": [],
            },
        }
        assert self._kind(params) != "/etc/hosts"

    def test_top_level_ask_resolves_via_the_preceding_tool_call_notification(
        self,
    ) -> None:
        """END-TO-END for the live defect: the adapter emits ``tool_call``
        (which ALWAYS carries ``_meta.claudeCode.toolName``) and then asks
        for permission for the SAME ``toolCallId``. Under ``-sp`` the ask
        must be APPROVED. The kind still has to RESOLVE (it is what the
        approval/denial is logged under); before the fix it resolved to the
        shell command string instead of the tool name."""
        policy = PermissionPolicy(
            auto_approve=True,
            allow_kind_prefix="allow",
            deny_kind_prefix="reject",
        )
        client = AcpClient(
            cmd=("true",), permission_policy=policy, extra_limit_markers=()
        )
        client._write = MagicMock()  # type: ignore[method-assign]

        client._handle_notification(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-1",
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "toolu_01",
                        "status": "pending",
                        "title": "ls -la",
                        "kind": "execute",
                        "_meta": {"claudeCode": {"toolName": "Bash"}},
                    },
                },
            }
        )
        client._handle_permission_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "params": {
                    "sessionId": "sess-1",
                    "options": [
                        {"kind": "reject_once", "name": "Deny", "optionId": "reject"},
                        {"kind": "allow_once", "name": "Allow", "optionId": "allow"},
                    ],
                    "toolCall": {
                        "toolCallId": "toolu_01",
                        "rawInput": {"command": "ls -la"},
                        "title": "ls -la",
                        "kind": "execute",
                    },
                },
            }
        )

        sent = client._write.call_args[0][0]
        assert sent["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "allow",
        }, f"-sp did not approve a top-level Bash ask: {sent!r}"

    def test_completed_call_is_evicted_from_the_name_map(self) -> None:
        """The map must not grow for the life of the process."""
        client = AcpClient(cmd=("true",), extra_limit_markers=())
        base = {
            "sessionUpdate": "tool_call",
            "toolCallId": "toolu_09",
            "_meta": {"claudeCode": {"toolName": "Read"}},
        }
        client._handle_notification(
            {"method": "session/update", "params": {"sessionId": "s", "update": base}}
        )
        assert client._tool_name_by_call_id["toolu_09"] == "Read"
        client._handle_notification(
            {
                "method": "session/update",
                "params": {
                    "sessionId": "s",
                    "update": {**base, "sessionUpdate": "tool_call_update",
                               "status": "completed"},
                },
            }
        )
        assert "toolu_09" not in client._tool_name_by_call_id

    def test_name_map_is_capped_and_evicts_oldest_first(self) -> None:
        """Backstop for a CLI that asks for permission and then never reports
        the call terminal (wedged/killed mid-turn): the map must not grow
        without bound, and the NEWEST ids — the ones a live ask can still
        reference — must survive."""
        from deep_worker.providers.acp.custom_types import (
            MAX_TOOL_NAME_ENTRIES,
        )

        client = AcpClient(cmd=("true",), extra_limit_markers=())
        for i in range(MAX_TOOL_NAME_ENTRIES + 50):
            client._handle_notification(
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": "s",
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": f"toolu_{i}",
                            "status": "pending",
                            "_meta": {"claudeCode": {"toolName": "Bash"}},
                        },
                    },
                }
            )

        assert len(client._tool_name_by_call_id) <= MAX_TOOL_NAME_ENTRIES
        newest = f"toolu_{MAX_TOOL_NAME_ENTRIES + 49}"
        assert newest in client._tool_name_by_call_id, (
            "the newest id was evicted — a live permission ask would miss"
        )
        assert "toolu_0" not in client._tool_name_by_call_id

    def test_subagent_call_still_recovers_the_tool_name(self) -> None:
        params = {
            "sessionId": "sess-1",
            "toolCall": {
                "toolCallId": "toolu_03",
                "title": "ls -la",
                "kind": "execute",
                "_meta": {
                    "claudeCode": {"toolName": "Bash", "parentToolUseId": "p1"}
                },
            },
        }
        assert self._kind(params) == "Bash"


class TestHandlePermissionRequest:
    """Drive ``_handle_permission_request`` on a client with a mocked
    ``_write`` (no subprocess is started by the constructor)."""

    @staticmethod
    def _make_client(policy: PermissionPolicy) -> AcpClient:
        client = AcpClient(
            cmd=("true",),
            permission_policy=policy,
            extra_limit_markers=(),
        )
        client._write = MagicMock()  # type: ignore[method-assign]
        return client

    @staticmethod
    def _obj(kind: str = "Bash") -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 7,
            "params": {
                "sessionId": "s1",
                "toolCall": {
                    "toolCallId": "tc1",
                    "title": kind,
                    "_meta": {"claudeCode": {"toolName": kind}},
                },
                "options": [
                    {"kind": "reject_once", "optionId": "reject"},
                    {"kind": "allow_once", "optionId": "allow"},
                    {"kind": "allow_always", "optionId": "allow_always"},
                ],
            },
        }

    def _written(self, client: AcpClient) -> dict:
        assert client._write.call_count == 1
        return client._write.call_args.args[0]

    def test_no_policy_denies_with_reject_option(self) -> None:
        client = self._make_client(PermissionPolicy(deny_kind_prefix="reject"))
        client._handle_permission_request(self._obj())
        written = self._written(client)
        assert written["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "reject",
        }

    def test_auto_approve_selects_allow_option(self) -> None:
        client = self._make_client(PermissionPolicy(auto_approve=True))
        client._handle_permission_request(self._obj("Bash"))
        written = self._written(client)
        # First offered allow-* option wins.
        assert written["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "allow",
        }

    def test_a_kind_no_list_ever_mentioned_is_approved_under_sp(self) -> None:
        """The regression this file used to enshrine: an unlisted tool was
        denied and the CLI killed the turn."""
        client = self._make_client(
            PermissionPolicy(auto_approve=True, deny_kind_prefix="reject")
        )
        client._handle_permission_request(self._obj("Write"))
        written = self._written(client)
        assert written["result"]["outcome"]["optionId"] == "allow"

    def test_kind_is_recovered_from_claude_meta(self) -> None:
        """params.kind absent; the kind must come from toolCall meta (it is
        what the denial log names, so it still has to resolve)."""
        client = self._make_client(PermissionPolicy(auto_approve=True))
        obj = self._obj("Read")
        obj["params"].pop("kind", None)
        client._handle_permission_request(obj)
        written = self._written(client)
        assert written["result"]["outcome"]["optionId"] == "allow"

    def test_custom_deny_prefix_used(self) -> None:
        """claude-agent-acp profile: deny kind prefix is ``reject``."""
        client = self._make_client(
            PermissionPolicy(
                auto_approve=False,
                deny_kind_prefix="reject",
            )
        )
        client._handle_permission_request(self._obj())
        written = self._written(client)
        assert written["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "reject",
        }

    def test_no_deny_option_falls_back_to_cancelled(self) -> None:
        client = self._make_client(PermissionPolicy())
        obj = self._obj()
        obj["params"]["options"] = [{"kind": "allow_once", "optionId": "allow"}]
        client._handle_permission_request(obj)
        written = self._written(client)
        assert written["result"]["outcome"] == {"outcome": "cancelled"}

    def test_no_allow_option_denies(self) -> None:
        """Option ids are opaque and version-specific: if the CLI offered no
        allow option we deny rather than invent one (an invented id makes the
        CLI reject the reply and kill the turn)."""
        client = self._make_client(
            PermissionPolicy(auto_approve=True, deny_kind_prefix="reject")
        )
        obj = self._obj("Bash")
        obj["params"]["options"] = [{"kind": "reject_once", "optionId": "reject"}]
        client._handle_permission_request(obj)
        written = self._written(client)
        assert written["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "reject",
        }

    def test_malformed_options_are_skipped(self) -> None:
        client = self._make_client(PermissionPolicy(deny_kind_prefix="reject"))
        obj = self._obj()
        obj["params"]["options"] = [42, {"kind": "reject_once", "optionId": "reject"}]
        client._handle_permission_request(obj)
        written = self._written(client)
        assert written["result"]["outcome"]["optionId"] == "reject"


class TestBuildProfileUnderSp:
    """``build_profile`` must hand back a policy that approves everything
    under ``-sp`` and nothing without it — with no list in between.

    This class replaces ``TestBuildProfileAllowlist``, which asserted that
    the default allowlist equals ``KNOWN_BUILTINS``. That assertion was the
    defect written down: the built-ins known at authoring time are a
    snapshot, and the first tool the CLI adds afterwards is denied under a
    flag whose entire purpose is to skip permissions."""

    def _build(self, *, auto_approve: bool):
        from unittest.mock import patch

        from deep_worker.providers.claude_code_provider import profile as prof

        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "ADAPTER_ENTRY", entry):
            return prof.build_profile(auto_approve=auto_approve, verify=False)

    def test_sp_approves_known_and_unknown_alike(self) -> None:
        policy = self._build(auto_approve=True).permission_policy
        assert policy.auto_approve is True
        assert policy.is_allowed_kind("Bash")
        assert policy.is_allowed_kind("mcp__proxy__read_file")
        assert policy.is_allowed_kind("SomeToolShippedNextRelease")

    def test_fail_closed_without_auto_approve(self) -> None:
        policy = self._build(auto_approve=False).permission_policy
        assert not policy.is_allowed_kind("Bash")
        assert not policy.is_allowed_kind("mcp__proxy__read_file")
