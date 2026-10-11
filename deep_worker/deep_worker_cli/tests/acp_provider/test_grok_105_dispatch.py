"""grok CLI 1.0.5 MCP-dispatch contract: ``use_tool{tool_name, tool_input}``.

Background (a related internal incident analysis, not included here): grok
1.0.5 dispatches an MCP tool call through the ``use_tool`` built-in whose ``rawInput.tool_name``
is the MANGLED name ``<server>__<tool>`` (no separate ``server`` field). The
stack only recognised the ≤1.0.4 form ``CallMcpTool{server, toolName,
arguments}``, so every ``send_message_to_agent`` fell into the generic
passthrough, never executed, and the CLI ended the prompt
``stopReason="cancelled"``.

Also covered here:

* the 1.0.5 snake_case built-in vocabulary that COLLIDES with deep-worker
  tool names (``read_file``, ``grep``) must never be routed into our bound
  tools (double execution) nor into a same-named passthrough tool;
* the per-turn tool contract must name the provider-correct MANGLED call
  forms and must reach a RESUMED session too (the proxy server name embeds
  the PID, so it goes stale on every restart);
* an unknown built-in whose payload references OUR proxy server is the next
  contract drift — it must be an ERROR, not an INFO-level shrug.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from deep_worker.providers.acp import acp_types
from deep_worker.providers.acp.chat_model import (
    AcpChatModel,
    _build_tool_contract,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker.providers.grok_provider.acp_types import (
    GROK_COLLIDING_BUILTINS,
    KNOWN_GROK_BUILTINS,
)
from deep_worker.providers.grok_provider.builtin_tool_passthrough import (
    build_passthrough_tools,
)
from deep_worker.providers.grok_provider.profile import build_profile

PROXY = "deep_worker_acp_proxy_777"

# Deep-worker base tools whose names grok 1.0.5 reuses for its NATIVE
# built-ins (observed live: read_file, grep; the rest share the same
# vocabulary family and must stay collision-safe).
DEEP_WORKER_BASE_TOOLS = frozenset(
    {"read_file", "write_file", "edit_file", "execute", "grep", "ls"}
)


def _use_tool_update(
    tool_name: str,
    tool_input: dict | None = None,
    tcid: str = "call-ut-1",
) -> dict:
    """Verbatim shape of a grok 1.0.5 dispatch update (title == use_tool)."""
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": "use_tool",
        "rawInput": {
            "tool_name": tool_name,
            "tool_input": tool_input if tool_input is not None else {},
        },
    }


def _legacy_update(
    tool: str,
    arguments: dict | None = None,
    tcid: str = "call-legacy-1",
) -> dict:
    """The ≤1.0.4 form: CallMcpTool{server, toolName, arguments}."""
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": "CallMcpTool",
        "rawInput": {
            "server": PROXY,
            "toolName": tool,
            "arguments": arguments if arguments is not None else {},
        },
    }


def _builtin_update(title: str, raw_input: dict, tcid: str = "call-b-1") -> dict:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tcid,
        "title": title,
        "rawInput": raw_input,
    }


class _FakeProxy:
    def __init__(self, server_name: str = PROXY) -> None:
        self.server_name = server_name


def _model() -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._profile = build_profile(auto_approve=True)
    model._proxy_mcp = _FakeProxy()
    model._client_state = AcpClientState()
    model._tools_metadata = [
        {
            "name": "send_message_to_agent",
            "description": "send",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "message": {"type": "string"},
                    "reply_back": {"type": "boolean"},
                },
                "required": ["agent_name", "message"],
            },
        },
        {
            "name": "read_file",
            "description": "read",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    ]
    return model


SEND_ARGS = {
    "agent_name": "peer_agent",
    "message": "REVIEW: approve",
    "reply_back": True,
}


def _map_in_loop(model: AcpChatModel, state: ThreadState, upd: dict) -> dict | None:
    """Run _map_tool_call under a running loop — CASE B parks an
    asyncio.Future for the passthrough recorder."""

    async def _run() -> dict | None:
        return model._map_tool_call(state, upd)

    return asyncio.run(_run())


class TestDispatchPredicate:
    """acp_types.is_proxy_mcp_dispatch must accept BOTH wrapper forms."""

    def test_use_tool_with_our_prefix_is_dispatch(self) -> None:
        upd = _use_tool_update(f"{PROXY}__send_message_to_agent", SEND_ARGS)
        assert acp_types.is_proxy_mcp_dispatch(upd, PROXY, "dispatch_builtin")

    def test_use_tool_foreign_server_is_not_dispatch(self) -> None:
        upd = _use_tool_update("other_server_1__send_message_to_agent", SEND_ARGS)
        assert not acp_types.is_proxy_mcp_dispatch(
            upd, PROXY, "dispatch_builtin"
        )

    def test_use_tool_prefix_must_be_full_server_name(self) -> None:
        # A server whose name merely STARTS with ours must not match
        # (deep_worker_acp_proxy_77 vs deep_worker_acp_proxy_777).
        upd = _use_tool_update(f"{PROXY}7__send_message_to_agent", SEND_ARGS)
        assert not acp_types.is_proxy_mcp_dispatch(
            upd, PROXY, "dispatch_builtin"
        )

    def test_legacy_callmcptool_form_still_recognised(self) -> None:
        upd = _legacy_update("send_message_to_agent", SEND_ARGS)
        assert acp_types.is_proxy_mcp_dispatch(upd, PROXY, "dispatch_builtin")

    def test_local_builtin_is_not_dispatch(self) -> None:
        upd = _builtin_update("read_file", {"path": "/tmp/x"})
        assert not acp_types.is_proxy_mcp_dispatch(
            upd, PROXY, "dispatch_builtin"
        )


class TestDispatchInner:
    """acp_types.proxy_dispatch_inner extracts (inner_name, args) for both
    wrapper forms — the single seam _proxy_tool_from_update relies on."""

    def test_use_tool_inner(self) -> None:
        upd = _use_tool_update(f"{PROXY}__send_message_to_agent", SEND_ARGS)
        assert acp_types.proxy_dispatch_inner(upd, PROXY) == (
            "send_message_to_agent",
            SEND_ARGS,
        )

    def test_use_tool_missing_input_defaults_to_empty(self) -> None:
        upd = {
            "sessionUpdate": "tool_call",
            "toolCallId": "c",
            "title": "use_tool",
            "rawInput": {"tool_name": f"{PROXY}__restart_self"},
        }
        assert acp_types.proxy_dispatch_inner(upd, PROXY) == (
            "restart_self",
            {},
        )

    def test_legacy_inner(self) -> None:
        upd = _legacy_update("send_message_to_agent", SEND_ARGS)
        assert acp_types.proxy_dispatch_inner(upd, PROXY) == (
            "send_message_to_agent",
            SEND_ARGS,
        )

    def test_foreign_server_returns_none(self) -> None:
        upd = _use_tool_update("other__send_message_to_agent", SEND_ARGS)
        assert acp_types.proxy_dispatch_inner(upd, PROXY) is None

    def test_empty_inner_name_returns_none(self) -> None:
        upd = _use_tool_update(f"{PROXY}__", SEND_ARGS)
        assert acp_types.proxy_dispatch_inner(upd, PROXY) is None


class TestMapToolCall:
    """_map_tool_call end-to-end on the grok profile (dispatch_builtin)."""

    def test_use_tool_send_maps_to_bound_tool(self) -> None:
        model = _model()
        state = ThreadState()
        tc = model._map_tool_call(
            state,
            _use_tool_update(f"{PROXY}__send_message_to_agent", SEND_ARGS),
        )
        assert tc is not None
        assert tc["name"] == "send_message_to_agent"
        assert tc["args"] == SEND_ARGS

    def test_legacy_send_still_maps(self) -> None:
        model = _model()
        state = ThreadState()
        tc = model._map_tool_call(
            state, _legacy_update("send_message_to_agent", SEND_ARGS)
        )
        assert tc is not None
        assert tc["name"] == "send_message_to_agent"
        assert tc["args"] == SEND_ARGS

    def test_local_builtin_collision_never_routes_into_our_tool(self) -> None:
        """grok 1.0.5 runs its NATIVE read_file locally; the name collides
        with our bound read_file. Mapping it into our tool caused a false
        10s MCP-handler correlation per call and double execution with the
        CLI's arguments."""
        model = _model()
        state = ThreadState()
        upd = _builtin_update("read_file", {"path": "/etc/hosts"}, tcid="c-rf")
        tc = _map_in_loop(model, state, upd)
        assert tc is not None
        # Routed via the generic passthrough recorder, NOT our read_file
        # (and NOT a same-named passthrough that would shadow it).
        assert tc["name"] == "GrokBuiltin"
        assert tc["args"]["_grok_original_title"] == "read_file"
        assert "c-rf" in state.builtin_passthrough_futures

    def test_colliding_builtin_does_not_warn_unknown(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = _model()
        state = ThreadState()
        with caplog.at_level(logging.WARNING):
            _map_in_loop(
                model, state, _builtin_update("grep", {"pattern": "x"}, tcid="c-gr")
            )
        assert "unknown built-in" not in caplog.text

    def test_known_105_builtin_routes_as_itself(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = _model()
        state = ThreadState()
        with caplog.at_level(logging.WARNING):
            tc = _map_in_loop(
                model,
                state,
                _builtin_update(
                    "run_terminal_command", {"command": "true"}, tcid="c-sh"
                ),
            )
        assert tc is not None
        assert tc["name"] == "run_terminal_command"
        assert "unknown built-in" not in caplog.text

    def test_unknown_builtin_referencing_our_server_is_contract_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The NEXT dispatch drift (grok 1.0.6 renaming use_tool again) must
        be loud: an unknown built-in whose payload references our proxy
        server is a broken dispatch contract, not passthrough noise."""
        model = _model()
        state = ThreadState()
        upd = _builtin_update(
            "call_tool_v2",
            {"endpoint": f"{PROXY}__send_message_to_agent", "input": {}},
            tcid="c-drift",
        )
        with caplog.at_level(logging.ERROR):
            _map_in_loop(model, state, upd)
        assert any(
            r.levelno >= logging.ERROR and "dispatch" in r.getMessage().lower()
            for r in caplog.records
        )


class TestBuiltinVocabulary:
    """grok 1.0.5 snake_case vocabulary; collision safety by construction."""

    def test_105_builtins_are_known(self) -> None:
        for name in (
            "run_terminal_command",
            "list_dir",
            "search_tool",
            "search_replace",
            "todo_write",
            "get_command_or_subagent_output",
        ):
            assert name in KNOWN_GROK_BUILTINS, name

    def test_legacy_pascal_case_builtins_stay_known(self) -> None:
        # Checkpointed histories contain them; forgetting them would
        # re-route old records through the generic passthrough on replay.
        for name in ("Shell", "Read", "Write", "Edit", "Grep", "WebSearch"):
            assert name in KNOWN_GROK_BUILTINS, name

    def test_colliding_names_are_segregated(self) -> None:
        assert "read_file" in GROK_COLLIDING_BUILTINS
        assert "grep" in GROK_COLLIDING_BUILTINS
        assert not (GROK_COLLIDING_BUILTINS & KNOWN_GROK_BUILTINS)

    def test_use_tool_is_not_a_passthrough_builtin(self) -> None:
        # use_tool is the DISPATCH wrapper (channel A), never a local
        # built-in to record.
        assert "use_tool" not in KNOWN_GROK_BUILTINS

    def test_passthrough_tools_never_shadow_deep_worker_tools(self) -> None:
        names = {t.name for t in build_passthrough_tools()}
        assert not (names & DEEP_WORKER_BASE_TOOLS)


class TestToolContract:
    """The per-turn contract must name the provider-correct MANGLED forms."""

    def test_grok_contract_lists_mangled_call_names(self) -> None:
        contract = _build_tool_contract(
            ["send_message_to_agent", "read_file"],
            build_profile(auto_approve=True),
            server_name=PROXY,
        )
        assert f"{PROXY}__send_message_to_agent" in contract
        assert f"{PROXY}__read_file" in contract

    def test_grok_contract_does_not_claim_bare_names_work(self) -> None:
        contract = _build_tool_contract(
            ["send_message_to_agent"],
            build_profile(auto_approve=True),
            server_name=PROXY,
        )
        assert "callable directly by name" not in contract

    def test_mcp_name_prefix_contract_lists_prefixed_names(self) -> None:
        from deep_worker.providers.acp.profile import AcpProfile

        profile = AcpProfile(
            provider="claude-code",
            cmd=("node", "x"),
            mcp_dispatch_style="mcp_name_prefix",
            tool_name_from_update="from_claude_code_meta",
        )
        contract = _build_tool_contract(
            ["send_message_to_agent"], profile, server_name=PROXY
        )
        assert f"mcp__{PROXY}__send_message_to_agent" in contract


class TestContractReachesResumedSession:
    """The proxy server name embeds the PID: a session resumed after a
    restart holds a contract with the DEAD process's names. The first
    prompt of every process must re-deliver the contract."""

    def test_resumed_session_gets_contract_once(self) -> None:
        model = _model()
        state = ThreadState()
        block = model._system_block_for_prompt(
            state, is_first_prompt_in_session=False, system_prompt=None
        )
        assert block is not None
        assert f"{PROXY}__send_message_to_agent" in block
        # Second prompt of the same process: nothing to re-send.
        again = model._system_block_for_prompt(
            state, is_first_prompt_in_session=False, system_prompt=None
        )
        assert again is None

    def test_first_prompt_in_session_marks_contract_sent(self) -> None:
        model = _model()
        state = ThreadState()
        block = model._system_block_for_prompt(
            state, is_first_prompt_in_session=True, system_prompt="SYSTEM."
        )
        assert block is not None
        assert "SYSTEM." in block
        assert f"{PROXY}__send_message_to_agent" in block
        assert (
            model._system_block_for_prompt(
                state, is_first_prompt_in_session=False, system_prompt=None
            )
            is None
        )
