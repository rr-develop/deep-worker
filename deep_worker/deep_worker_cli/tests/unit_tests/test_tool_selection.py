"""Tool selection: the default must not move, and '-*' must mean none.

See ``docs/DESIGN_TOOL_SELECTION.md``. The tests that matter most here are
the ones about the *absent* selection: every existing agent depends on it
being indistinguishable from no feature at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deep_worker_cli.tool_selection import (
    EMPTY_SELECTION,
    ToolSelectionError,
    describe_selection,
    middleware_tool_names,
    native_tool_caveat,
    parse_tool_selection,
    parse_tool_selection_flag,
    parse_tool_selection_spec,
    tool_names,
)

TOOLS = [
    SimpleNamespace(name="read_file"),
    SimpleNamespace(name="execute"),
    SimpleNamespace(name="grep"),
    SimpleNamespace(name="send_message_to_agent"),
]


def _names(tools):
    return [t.name for t in tools]


# --- the default -----------------------------------------------------------


@pytest.mark.parametrize("spec", [None, [], ["", "  "]])
def test_absent_selection_restricts_nothing(spec) -> None:
    """The default must keep every tool AND advertise that it changes nothing.

    ``restricts`` is what ``create_cli_agent`` branches on to skip
    installing the middleware at all; if it ever returned True for an
    empty spec, every unconfigured agent would silently gain a wrapper.
    """
    selection = parse_tool_selection(spec)
    assert selection.restricts is False
    assert _names(selection.apply(TOOLS)) == _names(TOOLS)


# --- deny all, then re-admit ----------------------------------------------


def test_star_denies_every_tool() -> None:
    selection = parse_tool_selection(["-*"])
    assert selection.restricts is True
    assert selection.apply(TOOLS) == []


def test_star_plus_names_keeps_only_those_names() -> None:
    selection = parse_tool_selection(["-*", "read_file", "grep"])
    assert _names(selection.apply(TOOLS)) == ["read_file", "grep"]


def test_star_spares_no_infrastructure_tool() -> None:
    """'-*' means none — including the tools the runtime itself leans on.

    Quietly exempting ``send_message_to_agent`` would make the measured
    tool set differ from the stated one, which is the one thing this
    feature cannot afford.
    """
    assert parse_tool_selection(["-*"]).permits("send_message_to_agent") is False


def test_minus_removes_only_the_named_tool() -> None:
    selection = parse_tool_selection(["-execute"])
    assert _names(selection.apply(TOOLS)) == [
        "read_file",
        "grep",
        "send_message_to_agent",
    ]


# --- specs that cannot be honoured as written ------------------------------


def test_star_combined_with_minus_is_rejected() -> None:
    """'-*' already denies everything, so '-x' beside it states a contradiction."""
    with pytest.raises(ToolSelectionError, match="contradiction"):
        parse_tool_selection(["-*", "-execute"])


def test_bare_list_of_names_is_an_allow_list() -> None:
    """Listing names keeps only those — the star is a convenience, not a must.

    Getting *more* tools than you listed would be the surprising reading,
    and configs carrying a bare ``tools:`` list predate this feature, so
    they now mean what they always appeared to mean.
    """
    selection = parse_tool_selection(["read_file", "grep"])
    assert selection.restricts is True
    assert _names(selection.apply(TOOLS)) == ["read_file", "grep"]


def test_mixing_a_bare_allow_list_with_minus_is_rejected() -> None:
    """Keep-these and remove-those together state two different selections."""
    with pytest.raises(ToolSelectionError, match="one or the other"):
        parse_tool_selection(["read_file", "-execute"])


def test_lone_minus_is_rejected() -> None:
    with pytest.raises(ToolSelectionError, match="followed by a tool name"):
        parse_tool_selection(["-"])


def test_unknown_names_are_reported() -> None:
    selection = parse_tool_selection(["-*", "raed_file"])
    assert selection.unknown_names(["read_file", "execute"]) == ["raed_file"]


def test_known_names_are_not_reported() -> None:
    selection = parse_tool_selection(["-*", "read_file"])
    assert selection.unknown_names(["read_file", "execute"]) == []


# --- name collection -------------------------------------------------------


def test_tool_names_handles_objects_and_dicts() -> None:
    assert tool_names([SimpleNamespace(name="a"), {"name": "b"}]) == {"a", "b"}


def test_middleware_tool_names_asks_the_middleware() -> None:
    """Known names come from the middleware, not a hand-written constant.

    A constant would drift the moment a middleware renames a tool, and the
    failure mode is a correct spec rejected as a typo.
    """
    middleware = [
        SimpleNamespace(tools=[SimpleNamespace(name="project_memory")]),
        SimpleNamespace(tools=[]),
        SimpleNamespace(),  # no ``tools`` attribute at all
    ]
    assert middleware_tool_names(middleware) == {"project_memory"}


# --- CLI form --------------------------------------------------------------


def test_cli_flag_splits_on_commas() -> None:
    selection = parse_tool_selection_flag("-*,read_file,grep")
    assert _names(selection.apply(TOOLS)) == ["read_file", "grep"]


def test_cli_spec_round_trips_to_the_config_list_form() -> None:
    assert parse_tool_selection_spec("-*, read_file ") == ["-*", "read_file"]


def test_cli_spec_absent_is_the_empty_list() -> None:
    assert parse_tool_selection_spec(None) == []


def test_cli_spec_validates_eagerly() -> None:
    """A malformed --tools must fail at launch, not at agent construction."""
    with pytest.raises(ToolSelectionError):
        parse_tool_selection_spec("-*,-execute")


def test_config_list_of_names_restricts_to_those_names() -> None:
    """The shape a YAML ``tools:`` list has always had, now honoured."""
    selection = parse_tool_selection(["tool1", "tool2", "tool3"])
    assert selection.deny_all is True
    assert selection.permits("tool1") is True
    assert selection.permits("execute") is False


# --- honesty about native tools -------------------------------------------


@pytest.mark.parametrize("provider", ["claude-code", "grok", "codex"])
def test_providers_with_unsuppressable_natives_are_disclosed(provider) -> None:
    """A restricted run on these providers is not hermetic — say so.

    Letting the caller assume otherwise would poison the very experiment
    the selection exists to support.
    """
    caveat = native_tool_caveat(provider, parse_tool_selection(["-*"]))
    assert caveat is not None
    assert provider in caveat
    assert "claude-code-sdk" in caveat  # names the provider that IS hermetic


def test_fully_suppressed_provider_gets_no_caveat() -> None:
    assert native_tool_caveat("claude-code-sdk", parse_tool_selection(["-*"])) is None


def test_no_caveat_when_nothing_is_restricted() -> None:
    """The default path must stay silent — no warning on every agent start."""
    assert native_tool_caveat("grok", EMPTY_SELECTION) is None


def test_preflight_notice_is_visible_for_nonhermetic_provider(capsys) -> None:
    from deep_worker_cli.cli.agent_command import _emit_tool_selection_preflight

    _emit_tool_selection_preflight("grok", ["-*"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WARNING:" in captured.err
    assert "grok" in captured.err
    assert "claude-code-sdk" in captured.err


def test_preflight_default_is_completely_silent(capsys) -> None:
    from deep_worker_cli.cli.agent_command import _emit_tool_selection_preflight

    _emit_tool_selection_preflight("grok", [])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_restricted_config_preflight_is_visible_exactly_once(tmp_path, capsys) -> None:
    from deep_worker_cli.cli.agent_command import (
        _emit_effective_tool_selection_preflight,
    )

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        f"model: gpt-5\nworking_dir: {tmp_path}\nprovider: codex\ntools: ['-*']\n"
    )
    _emit_effective_tool_selection_preflight(
        SimpleNamespace(config=config_path, provider="deepseek", tools=None)
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("WARNING:") == 1
    assert "codex" in captured.err


def test_unrestricted_config_preflight_is_completely_silent(tmp_path, capsys) -> None:
    from deep_worker_cli.cli.agent_command import (
        _emit_effective_tool_selection_preflight,
    )

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        f"model: gpt-5\nworking_dir: {tmp_path}\nprovider: codex\ntools: []\n"
    )
    _emit_effective_tool_selection_preflight(
        SimpleNamespace(config=config_path, provider="deepseek", tools="-*")
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_config_preflight_runs_before_daemonize_and_not_again_in_run_agent() -> None:
    import inspect

    from deep_worker_cli.cli.agent_command import main, run_agent

    main_source = inspect.getsource(main)
    assert main_source.rindex("_emit_effective_tool_selection_preflight(args)") < (
        main_source.rindex("daemonize(args.name)")
    )
    assert "_emit_tool_selection_preflight" not in inspect.getsource(run_agent)


# --- description -----------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ([], "all tools"),
        (["-*"], "no tools"),
        (["-*", "read_file"], "only read_file"),
        (["-execute"], "all tools except execute"),
    ],
)
def test_description_states_what_is_left(spec, expected) -> None:
    assert describe_selection(parse_tool_selection(spec)) == expected


# --- restart override -------------------------------------------------------
#
# A restart must be able to CHANGE the selection, not just inherit it.
# Without an explicit-override branch the saved value always wins, so
# ``--restart A --tools '-*,read_file'`` is silently ignored and a restricted
# agent can never be un-restricted from the command line.


def _restart_args(**overrides):
    import argparse

    from deep_worker_cli.cli.agent_command import _PARSER_DEFAULTS

    defaults = dict(_PARSER_DEFAULTS)
    defaults.update(
        {
            "name": "test_agent",
            "config": None,
            "restart": True,
            "mcp_config_paths": [],
            "mcp_server_inline": [],
            "turn_ending_tools": [],
        }
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _saved(**overrides):
    saved = {
        "name": "test_agent",
        "provider": "grok",
        "model": "grok-4",
        "working_dir": "/tmp",
        "tools": ["-*", "read_file"],
    }
    saved.update(overrides)
    return saved


def test_restart_inherits_the_saved_selection_when_no_flag_is_given() -> None:
    """A restricted agent must come back restricted."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    merged = merge_restart_config(_saved(), _restart_args(tools=None), explicit=set())
    assert merged.tools == "-*,read_file"


def test_restart_explicit_tools_replaces_the_saved_selection() -> None:
    """``--restart A --tools '-*,grep'`` must win over the saved value."""
    from deep_worker_cli.cli.agent_command import merge_restart_config

    merged = merge_restart_config(
        _saved(), _restart_args(tools="-*,grep"), explicit={"tools"}
    )
    assert merged.tools == "-*,grep"


def test_restart_explicit_empty_tools_clears_the_saved_selection() -> None:
    """``--tools ''`` is the only way back to an unrestricted agent.

    The saved spec must not resurrect itself: an explicit empty string is a
    deliberate CLEAR, not an absent flag.
    """
    from deep_worker_cli.cli.agent_command import merge_restart_config

    merged = merge_restart_config(
        _saved(), _restart_args(tools=""), explicit={"tools"}
    )
    assert not merged.tools


def test_restart_detector_sees_the_tools_flag() -> None:
    """The explicit-flag detector must recognise ``--tools``.

    Override detection cannot fall back to "differs from default" here,
    because an explicit ``--tools ''`` is indistinguishable from an absent
    flag by value alone.
    """
    from deep_worker_cli.cli.agent_command import _explicit_cli_flags

    assert "tools" in _explicit_cli_flags(["a", "--restart", "--tools", "none"])
    assert "tools" in _explicit_cli_flags(["a", "--restart", "--tools=-*"])
    assert "tools" in _explicit_cli_flags(["a", "--restart", "--tools", ""])
    assert "tools" not in _explicit_cli_flags(["a", "--restart"])


# --- SDK-built tools ---------------------------------------------------------
#
# Regression: ``create_deep_agent`` builds FilesystemMiddleware and the
# subagent middleware AFTER the selection is validated, so validating
# against ``agent_middleware`` alone rejected ``read_file``/``task`` — the
# very tools a tool-effectiveness experiment is most likely to name — as
# typos. Found by a live create_cli_agent build; a test over the assembled
# list could not have seen it.


def test_sdk_tool_names_cover_the_filesystem_and_task_tools() -> None:
    from deep_worker_cli.tool_selection import sdk_tool_names

    names = sdk_tool_names()
    for expected in (
        "read_file",
        "write_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "execute",
        "task",
    ):
        assert expected in names, f"{expected} must not look like a typo"


def test_sdk_tool_names_are_read_from_the_real_middleware() -> None:
    """Not a hand-written constant: renaming upstream must not go stale."""
    from deepagents.middleware.filesystem import FilesystemMiddleware

    from deep_worker_cli.tool_selection import sdk_tool_names, tool_names

    real = tool_names(FilesystemMiddleware().tools)
    assert real, "FilesystemMiddleware should expose tools"
    assert real <= sdk_tool_names()


def test_sdk_tool_names_refuses_to_return_an_empty_set(monkeypatch) -> None:
    """An empty set must raise, not silently reclassify tools as typos.

    A swallowed failure here is invisible: validation would reject
    ``read_file`` with a "typo" message that names the wrong cause, and the
    launch would abort for a reason the user cannot act on.
    """
    import deepagents.middleware.filesystem as fsmod

    from deep_worker_cli import tool_selection as ts

    class _Toolless(fsmod.FilesystemMiddleware):
        def __init__(self, *args, **kwargs):
            self.tools = []

    monkeypatch.setattr(fsmod, "FilesystemMiddleware", _Toolless)
    with pytest.raises(RuntimeError, match="silently reject"):
        ts.sdk_tool_names()


def test_sdk_tool_names_is_a_superset_after_patching() -> None:
    """Patching only ADDS names, so ordering cannot reject a real tool.

    ``patch_filesystem_middleware()`` swaps in a subclass carrying
    ``wait_for_job``/``kill_job``. Whether it has run depends on test
    order, so the invariant that matters is monotonicity, not equality.
    """
    from deep_worker_cli.agent import patch_filesystem_middleware
    from deep_worker_cli.tool_selection import sdk_tool_names

    before = sdk_tool_names()
    patch_filesystem_middleware()
    after = sdk_tool_names()
    assert before <= after
    for expected in ("read_file", "execute", "task"):
        assert expected in after


def test_config_launch_discloses_natives_before_daemonize(tmp_path, capsys) -> None:
    """``--config`` must disclose too — it bypasses the --tools flag path."""
    import argparse
    from pathlib import Path

    from deep_worker_cli.cli.agent_command import (
        _emit_effective_tool_selection_preflight,
    )

    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "model: grok-4\nprovider: grok\n"
        f"working_dir: {tmp_path}\ntools: ['-*', 'read_file']\n"
    )
    args = argparse.Namespace(config=Path(cfg), provider="grok", tools=None)
    _emit_effective_tool_selection_preflight(args)

    err = capsys.readouterr().err
    assert "WARNING:" in err
    assert "grok" in err
    assert "claude-code-sdk" in err


def test_config_launch_without_a_selection_is_silent(tmp_path, capsys) -> None:
    import argparse
    from pathlib import Path

    from deep_worker_cli.cli.agent_command import (
        _emit_effective_tool_selection_preflight,
    )

    cfg = tmp_path / "agent.yaml"
    cfg.write_text(f"model: grok-4\nprovider: grok\nworking_dir: {tmp_path}\n")
    args = argparse.Namespace(config=Path(cfg), provider="grok", tools=None)
    _emit_effective_tool_selection_preflight(args)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- The star means EVERY tool -----------------------------------------------
#
# ``*`` originally spelled deny-all, which read as "all tools are available"
# and meant the opposite. The spelling is now ``-*``, and the old one is a
# launch error rather than a silent re-interpretation: the same spec would
# now WIDEN a restricted agent, and specs live in session_metadata.json where
# they survive a restart.


def test_none_is_an_alias_for_deny_all() -> None:
    """argparse eats ``-*``, so the common case must not need the '=' form."""
    assert parse_tool_selection(["none"]) == parse_tool_selection(["-*"])
    assert parse_tool_selection(["none"]).apply(TOOLS) == []


def test_none_plus_names_keeps_only_those_names() -> None:
    kept = parse_tool_selection(["none", "read_file"]).apply(TOOLS)
    assert tool_names(kept) == {"read_file"}


def test_a_bare_star_is_refused() -> None:
    """``["*"]`` meant "no tools at all"; it must not become "every tool".

    This is the larger of the two legacy widenings — the whole registry
    handed back to an agent that had none — and it is the form the shipped
    documentation taught, so configs in the wild carry it.
    """
    with pytest.raises(ToolSelectionError) as excinfo:
        parse_tool_selection(["*"])
    message = str(excinfo.value)
    assert "whole registry" in message
    assert "'none'" in message, "the error must carry the rewrite"


def test_refusing_a_bare_star_costs_the_grammar_nothing() -> None:
    """"Every tool" is what an empty spec already says."""
    assert parse_tool_selection([]).restricts is False
    assert tool_names(parse_tool_selection([]).apply(TOOLS)) == tool_names(TOOLS)


def test_star_with_minus_reads_as_everything_except() -> None:
    """``["*", "-execute"]`` says out loud what ``["-execute"]`` says tersely."""
    spelled_out = parse_tool_selection(["*", "-execute"])
    terse = parse_tool_selection(["-execute"])
    assert spelled_out == terse
    assert "execute" not in tool_names(spelled_out.apply(TOOLS))
    assert "read_file" in tool_names(spelled_out.apply(TOOLS))


def test_the_old_deny_all_spelling_is_refused() -> None:
    """``["*", "read_file"]`` used to mean "only read_file"; it must not run.

    Under the current grammar it would mean "every tool, plus read_file" —
    a silent widening of an agent that was deliberately confined.
    """
    with pytest.raises(ToolSelectionError) as excinfo:
        parse_tool_selection(["*", "read_file"])
    message = str(excinfo.value)
    assert "-*, read_file" in message, "the error must carry the rewrite"
    assert "EVERY tool" in message


def test_the_refusal_names_every_tool_it_would_have_kept() -> None:
    with pytest.raises(ToolSelectionError) as excinfo:
        parse_tool_selection(["*", "grep", "read_file"])
    message = str(excinfo.value)
    assert "-*, grep, read_file" in message


def test_a_config_written_in_the_new_spelling_is_not_refused() -> None:
    """The rejection must be narrow: ``-*`` beside names is the correct form."""
    kept = parse_tool_selection(["-*", "read_file"]).apply(TOOLS)
    assert tool_names(kept) == {"read_file"}
