"""CLI wiring for ``--turn-ending-tool``.

See ``deep_worker_cli/docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`` §3.1–§3.3
and §3.5. Modelled 1-for-1 on ``--mcp-config`` / ``--mcp-server``, which
already survive ``--restart`` end-to-end.
"""

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli import agent_command
from deep_worker_cli.cli.agent_command import (
    build_launch_config,
    create_parser,
    merge_restart_config,
)
from deep_worker_cli.middleware.tool_completion import (
    DuplicateTurnEndingToolError,
    TurnEndingToolFlagError,
)

SPEC = "repo_graph__submit_fragments:document_closed"


def _base_namespace(**overrides) -> argparse.Namespace:
    """A namespace shaped like the one ``build_launch_config`` consumes."""
    ns = argparse.Namespace(
        name="a",
        provider="deepseek",
        model=None,
        dir=Path("/tmp"),
        skip_permissions=False,
        task_manager=False,
        memory_only=False,
        history_memory=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        dev_tree=False,
        web_search_policy="auto",
        preferred_provider="deepseek",
        preferred_model=None,
        mcp_config_paths=[],
        mcp_server_inline=[],
        turn_ending_tools=[],
        config=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


# ----------------------------------------------------------------------
# argparse surface
# ----------------------------------------------------------------------


def test_argparse_namespace_carries_raw_strings():
    """The namespace holds ``list[str]``, never parsed spec objects.

    Persistence depends on it: ``build_launch_config`` writes this value
    straight into ``session_metadata.json``.
    """
    parser = create_parser()
    ns = parser.parse_args(["a", "--turn-ending-tool", "foo:bar"])
    assert ns.turn_ending_tools == ["foo:bar"]
    assert all(isinstance(x, str) for x in ns.turn_ending_tools)


def test_argparse_flag_is_repeatable():
    parser = create_parser()
    ns = parser.parse_args(
        ["a", "--turn-ending-tool", "foo", "--turn-ending-tool", "bar:x"]
    )
    assert ns.turn_ending_tools == ["foo", "bar:x"]


def test_argparse_default_is_empty_list():
    parser = create_parser()
    assert create_parser().parse_args(["a"]).turn_ending_tools == []
    del parser


def test_argparse_rejects_malformed_value(capsys):
    """A grammar violation is a clean argparse usage error, exit code 2."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["a", "--turn-ending-tool", "foo:bar:baz"])
    assert exc.value.code == 2
    assert "turn-ending-tool" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Persistence: build_launch_config / merge_restart_config
# ----------------------------------------------------------------------


def test_flag_appears_in_build_launch_config():
    built = build_launch_config(_base_namespace(turn_ending_tools=[SPEC]))
    assert built["turn_ending_tools"] == [SPEC]


def test_omitted_flag_produces_empty_list_in_launch_config():
    ns = _base_namespace()
    del ns.turn_ending_tools
    assert build_launch_config(ns)["turn_ending_tools"] == []


def test_launch_config_is_json_serializable():
    ns = _base_namespace(turn_ending_tools=[SPEC])
    built = build_launch_config(ns)
    assert json.loads(json.dumps(built))["turn_ending_tools"] == [SPEC]


def test_merge_restart_inherits_saved_when_no_cli_override():
    saved = {"turn_ending_tools": [SPEC]}
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    assert effective.turn_ending_tools == [SPEC]


def test_merge_restart_cli_override_replaces_saved():
    saved = {"turn_ending_tools": [SPEC]}
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=["baz"]), explicit=set()
    )
    assert effective.turn_ending_tools == ["baz"]


def test_merge_restart_empty_cli_does_not_override():
    saved = {"turn_ending_tools": [SPEC]}
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    assert effective.turn_ending_tools == [SPEC]


def test_merge_restart_without_saved_key_yields_empty():
    effective = merge_restart_config(
        {}, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    assert effective.turn_ending_tools == []


def test_restart_round_trip_preserves_specs():
    """Executable round-trip: namespace -> saved -> merged -> saved."""
    original = _base_namespace(turn_ending_tools=[SPEC])
    saved = build_launch_config(original)
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    rebuilt = build_launch_config(effective)
    assert rebuilt["turn_ending_tools"] == original.turn_ending_tools


# ----------------------------------------------------------------------
# Fail-fast boundary in run_agent(): before AgentProcess construction
# ----------------------------------------------------------------------
#
# ``run_agent`` catches Exception at its top level and converts it into
# ``return 1`` + a stderr line (pre-existing behaviour, unchanged by this
# work). What these tests pin is the part that matters for the design
# contract: the failure happens BEFORE ``AgentProcess`` is constructed,
# so no ERROR-mode ghost process is registered — unlike the old path
# where the same collision surfaced inside ``create_cli_agent``, deep
# within the swallowing init ``try`` of ``AgentProcess.start()``.


def _run_agent_namespace(tmp_path: Path, specs: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        name="test_turn_ending_agent",
        dir=tmp_path,
        model="deepseek-chat",
        provider="deepseek",
        skip_permissions=True,
        task_manager=False,
        config=None,
        memory_only=True,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        restart=False,
        turn_ending_tools=specs,
        mcp_config_paths=[],
        mcp_server_inline=[],
        _resolved_startup_queries=(None, None),
        summarize=False,
    )


def _install_ctor_spy(monkeypatch) -> list[dict]:
    """Replace ``AgentProcess.__init__`` with a recorder; return its log."""
    calls: list[dict] = []

    def _spy(agent_self, *args, **kwargs):
        calls.append(kwargs)
        agent_self.name = kwargs.get("name")
        agent_self.running = False

    monkeypatch.setattr(agent_command.AgentProcess, "__init__", _spy)
    return calls


async def _run_agent_with_ctor_spy(monkeypatch, args) -> tuple[int, list[dict]]:
    calls = _install_ctor_spy(monkeypatch)

    async def _never_started(self):  # pragma: no cover - must not run
        raise AssertionError("agent.start() must not be reached")

    async def _cleanup(self):  # pragma: no cover - defensive
        return None

    monkeypatch.setattr(agent_command.AgentProcess, "start", _never_started)
    monkeypatch.setattr(agent_command.AgentProcess, "cleanup", _cleanup)
    return await agent_command.run_agent(args), calls


@pytest.mark.asyncio
async def test_restart_with_invalid_saved_spec_errors_before_agentprocess(
    monkeypatch, tmp_path, capsys
):
    """A hand-edited session_metadata.json with a bad spec fails early."""
    args = _run_agent_namespace(tmp_path, ["bad:spec:trap"])
    rc, calls = await _run_agent_with_ctor_spy(monkeypatch, args)
    assert rc == 1
    assert calls == []
    assert "bad:spec:trap" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_first_launch_duplicate_cli_names_errors_before_agentprocess(
    monkeypatch, tmp_path, capsys
):
    """``--turn-ending-tool foo --turn-ending-tool foo:done`` collides."""
    args = _run_agent_namespace(tmp_path, ["foo", "foo:done"])
    rc, calls = await _run_agent_with_ctor_spy(monkeypatch, args)
    assert rc == 1
    assert calls == []
    assert "foo" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_first_launch_builtin_name_errors_before_agentprocess(
    monkeypatch, tmp_path, capsys
):
    args = _run_agent_namespace(tmp_path, ["send_message_to_agent"])
    rc, calls = await _run_agent_with_ctor_spy(monkeypatch, args)
    assert rc == 1
    assert calls == []
    err = capsys.readouterr().err
    assert "send_message_to_agent" in err
    assert "built-in" in err


@pytest.mark.asyncio
async def test_restart_with_duplicate_saved_specs_errors_before_agentprocess(
    monkeypatch, tmp_path, capsys
):
    """Same collision, arriving from saved config rather than argv."""
    saved = {"turn_ending_tools": ["foo", "foo:done"]}
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    args = _run_agent_namespace(tmp_path, list(effective.turn_ending_tools))
    rc, calls = await _run_agent_with_ctor_spy(monkeypatch, args)
    assert rc == 1
    assert calls == []
    assert "foo" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_restart_with_builtin_saved_spec_errors_before_agentprocess(
    monkeypatch, tmp_path, capsys
):
    saved = {"turn_ending_tools": ["send_message_to_agent"]}
    effective = merge_restart_config(
        saved, _base_namespace(turn_ending_tools=[]), explicit=set()
    )
    args = _run_agent_namespace(tmp_path, list(effective.turn_ending_tools))
    rc, calls = await _run_agent_with_ctor_spy(monkeypatch, args)
    assert rc == 1
    assert calls == []
    assert "built-in" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_multiple_distinct_specs_pass_full_list_validation(
    monkeypatch, tmp_path
):
    """Positive case: the repeatable flag with DIFFERENT names works.

    Distinguishes "duplicates rejected" from "repeatable flag broken".
    """
    args = _run_agent_namespace(tmp_path, ["foo", "bar:x"])
    calls = _install_ctor_spy(monkeypatch)

    started = {"value": False}

    async def _start(self):
        started["value"] = True

    async def _wait(self):
        return None

    async def _cleanup(self):
        return None

    monkeypatch.setattr(agent_command.AgentProcess, "start", _start)
    monkeypatch.setattr(agent_command.AgentProcess, "cleanup", _cleanup)

    class _Event:
        async def wait(self):
            return None

    monkeypatch.setattr(
        agent_command.AgentProcess, "shutdown_event", _Event(), raising=False
    )

    rc = await agent_command.run_agent(args)
    assert rc == 0
    assert started["value"] is True
    assert len(calls) == 1
    assert calls[0]["turn_ending_tool_specs"] == ["foo", "bar:x"]


def test_grammar_errors_are_the_same_across_boundaries():
    """One grammar, three boundaries — identical exception type."""
    from deep_worker_cli.middleware.tool_completion import (
        _canonical_turn_ending_spec_str,
        _parse_turn_ending_tool_specs,
        _validate_turn_ending_spec_str,
    )

    bad = "foo:bar:baz"
    messages = []
    for fn, arg in (
        (_validate_turn_ending_spec_str, bad),
        (_canonical_turn_ending_spec_str, bad),
        (_parse_turn_ending_tool_specs, [bad]),
    ):
        with pytest.raises(TurnEndingToolFlagError) as exc:
            fn(arg)
        messages.append(str(exc.value))
    assert len(set(messages)) == 1


def test_duplicate_error_type_is_not_the_grammar_error():
    """Collisions and grammar errors stay distinguishable by type."""
    assert not issubclass(DuplicateTurnEndingToolError, TurnEndingToolFlagError)
    assert not issubclass(TurnEndingToolFlagError, DuplicateTurnEndingToolError)
