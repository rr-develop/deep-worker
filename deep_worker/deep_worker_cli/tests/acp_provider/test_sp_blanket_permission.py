"""``-sp`` means BLANKET approval. No tool lists, ever.

Owner directive 2026-08-24, after a grok agent lost four days of shell
access to exactly this class of bug:

    "If the sp flag is set, no explicit declaration of tools should be
    required. Or ALL tools should be added there automatically. I never want
    to come back to this question in the future just because some tool got
    added somewhere and some agent forgot to add it to the allow list."

What actually happened (see a related internal incident analysis, not
included here): commit 3fe834d
moved grok from its fork — which auto-approved every permission request — to
the shared fail-closed policy, but ``config.py`` never passed ``-sp`` down
to ``GrokChatModel``. Permission requests were therefore denied on the
policy's FIRST branch, which had no logging at all, and grok terminates a
turn whose permission was rejected. Every shell/file-write turn died with
``AcpTurnCancelledError`` and nothing in any log said why. claude-code was
one CLI release away from the same fate: its allowlist defaulted to
``KNOWN_BUILTINS``, so the first built-in Anthropic adds would have been
denied under ``-sp``.

The invariants below are the guarantee that this cannot recur:

1. ``auto_approve`` (i.e. ``-sp``) approves EVERY kind — including kinds no
   list in this repo has ever heard of;
2. without ``-sp`` NOTHING is approved (fail-closed is preserved);
3. no per-tool allowlist knob may exist on the policy or on any provider's
   profile builder — enforced by introspection, so reintroducing the
   pattern breaks this test rather than an agent in production;
4. a denial is never silent.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker.providers.acp.profile import AcpProfile, PermissionPolicy

# Names a model could ask permission for that appear in NO list anywhere in
# this repository — today's unknown tool is tomorrow's shipped built-in.
NEVER_LISTED_KINDS = (
    "run_terminal_command",
    "some_tool_that_ships_in_2027",
    "mcp__foreign_server__write_everything",
    "Bash",
    "",
)


class TestBlanketApproval:
    @pytest.mark.parametrize("kind", NEVER_LISTED_KINDS)
    def test_sp_approves_any_kind(self, kind: str) -> None:
        assert PermissionPolicy(auto_approve=True).is_allowed_kind(kind)

    @pytest.mark.parametrize("kind", NEVER_LISTED_KINDS)
    def test_without_sp_nothing_is_approved(self, kind: str) -> None:
        assert not PermissionPolicy().is_allowed_kind(kind)


class TestNoListKnobsSurvive:
    """The anti-rot guard: the mechanism itself must be gone, not merely
    unused. A list that exists WILL eventually be relied on and then go
    stale — that is the whole failure mode being closed here."""

    def test_policy_has_no_per_tool_list_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(PermissionPolicy)}
        assert fields == {
            "auto_approve",
            "allow_kind_prefix",
            "deny_kind_prefix",
        }, (
            "PermissionPolicy grew a field. If it names a set of tools, the "
            "2026-08-24 outage is back: -sp must approve everything, and the "
            "only per-CLI knobs allowed here describe the CLI's OPTION kinds "
            "(allow-*/reject-*), never its tools."
        )

    @pytest.mark.parametrize(
        "module_name",
        [
            "deep_worker.providers.grok_provider.profile",
            "deep_worker.providers.claude_code_provider.profile",
        ],
    )
    def test_profile_builders_take_no_allowlist(self, module_name: str) -> None:
        module = __import__(module_name, fromlist=["build_profile"])
        params = inspect.signature(module.build_profile).parameters
        offenders = [
            name
            for name in params
            if "allow" in name.lower() or "whitelist" in name.lower()
        ]
        assert not offenders, (
            f"{module_name}.build_profile still takes {offenders} — a "
            f"per-tool list an operator has to maintain."
        )

    @pytest.mark.parametrize(
        "module_name",
        [
            "deep_worker.providers.grok_provider.chat_model",
            "deep_worker.providers.claude_code_provider.chat_model",
        ],
    )
    def test_chat_models_take_no_allowlist(self, module_name: str) -> None:
        module = __import__(module_name, fromlist=["*"])
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if not inspect.isclass(obj) or not obj_name.endswith("ChatModel"):
                continue
            if obj.__module__ != module_name:
                continue
            params = inspect.signature(obj.__init__).parameters
            offenders = [p for p in params if "allowlist" in p.lower()]
            assert not offenders, f"{obj_name}.__init__ still takes {offenders}"


class TestProviderProfilesUnderSp:
    """Both shipped ACP providers must approve an unheard-of tool under -sp
    — the profile is where the old defaults lived (grok: empty set → deny
    all; claude-code: KNOWN_BUILTINS → deny anything new)."""

    def test_grok_profile_approves_unknown_tool(self) -> None:
        from deep_worker.providers.grok_provider.profile import build_profile

        policy = build_profile(auto_approve=True).permission_policy
        assert policy.is_allowed_kind("some_tool_that_ships_in_2027")
        assert not build_profile().permission_policy.is_allowed_kind("Shell")

    def test_claude_code_profile_approves_unknown_tool(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from deep_worker.providers.claude_code_provider import profile as ccp

        # The adapter is not installed under the sandboxed HOME; only its
        # presence check stands between us and the profile, and this test is
        # about the permission policy, not about node packaging.
        entry = tmp_path / "index.js"
        entry.write_text("//", encoding="utf-8")
        monkeypatch.setattr(ccp, "ADAPTER_ENTRY", entry)

        policy = ccp.build_profile(
            auto_approve=True, verify=False
        ).permission_policy
        assert policy.is_allowed_kind("some_tool_that_ships_in_2027")
        assert not ccp.build_profile(verify=False).permission_policy.is_allowed_kind(
            "Bash"
        )

    def test_grok_chat_model_wires_sp_into_its_policy(self) -> None:
        from deep_worker.providers.grok_provider.chat_model import GrokChatModel

        model = GrokChatModel(auto_approve=True)
        assert model._profile.permission_policy.auto_approve is True
        assert model._profile.permission_policy.is_allowed_kind("anything_new")


class TestConfigWiring:
    """The 2026-08-24 defect was a WIRING omission, not a policy bug: the
    policy was fail-closed and correct, and ``-sp`` simply never reached it.
    Pin the wiring itself."""

    def test_create_model_passes_sp_to_every_acp_provider(self) -> None:
        import inspect as _inspect

        from deep_worker_cli import config

        src = _inspect.getsource(config.create_model)
        for ctor in ("GrokChatModel(", "ClaudeCodeAcpChatModel("):
            start = src.index(ctor)
            call = src[start : src.index(")", start)]
            assert "auto_approve=auto_approve" in call, (
                f"{ctor} in create_model does not forward -sp — a grok agent "
                f"silently lost every shell tool this way on 2026-08-24."
            )


class TestDenialIsNeverSilent:
    """The denial that broke production logged NOTHING, so the failure was
    invisible for a full day of debugging."""

    def _client(self, policy: PermissionPolicy | None) -> AcpClient:
        client = object.__new__(AcpClient)
        client._permission_policy = policy
        client._tool_name_by_call_id = {}
        client._written: list[dict] = []
        client._write = client._written.append  # type: ignore[assignment]
        return client

    def _request(self) -> dict:
        return {
            "id": 7,
            "params": {
                "toolCall": {"title": "run_terminal_command"},
                "options": [
                    {"kind": "allow-once", "optionId": "a1"},
                    {"kind": "deny", "optionId": "d1"},
                ],
            },
        }

    def test_denial_without_sp_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = self._client(PermissionPolicy())
        with caplog.at_level(logging.INFO):
            client._handle_permission_request(self._request())
        assert "denying" in caplog.text.lower()
        outcome = client._written[-1]["result"]["outcome"]
        assert outcome["optionId"] == "d1"

    def test_sp_selects_the_offered_allow_option_for_an_unlisted_tool(
        self,
    ) -> None:
        client = self._client(PermissionPolicy(auto_approve=True))
        client._handle_permission_request(self._request())
        outcome = client._written[-1]["result"]["outcome"]
        assert outcome == {"outcome": "selected", "optionId": "a1"}


def test_profile_default_policy_is_fail_closed() -> None:
    profile = AcpProfile(provider="x", cmd=("x",))
    assert profile.permission_policy.auto_approve is False
