"""Startup diagnostics for CLI-registered turn-ending tools.

See ``deep_worker_cli/docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`` §3.4.

Two machine-greppable events, one per declared spec:

    turn_ending_tool_registered name=<mangled>
    turn_ending_tool_not_found name=<mangled> candidates=[...]

Both are emitted unconditionally so a launcher can PROVE from the log
that its flag reached ``AgentProcess`` — the absence of a WARNING is
otherwise indistinguishable from "the flag never arrived".

A miss is a WARNING, never a hard failure: an external MCP endpoint may
come up after the agent, and turning "the optimisation did not apply
this run" into "the run did not start" is strictly worse.
"""

import ast
import logging
from pathlib import Path

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess

SPEC = "repo_graph__submit_fragments:document_closed"
FOUND = "repo_graph__submit_fragments"


class _Meta:
    """Stand-in for a ``ToolMetadata`` entry (claude-code-sdk shape)."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make(tmp_path: Path, specs: list[str]) -> AgentProcess:
    return AgentProcess(
        name="test_turn_ending_diag",
        config=AgentConfig(model="deepseek-chat", working_dir=tmp_path),
        rpyc_port=False,
        memory_only=True,
        data_dir=tmp_path / "data",
        pid_file=tmp_path / "agent.pid",
        turn_ending_tool_specs=specs,
    )


def _messages(caplog, level=None) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("turn_ending_tool_")
        and (level is None or r.levelno == level)
    ]


def test_registered_and_not_found_events_format(tmp_path, caplog):
    agent = _make(tmp_path, [SPEC, "typo__submit"])
    inventory = [_Meta(FOUND), _Meta("read_file")]

    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics(inventory)

    info = _messages(caplog, logging.INFO)
    warning = _messages(caplog, logging.WARNING)
    assert info == [f"turn_ending_tool_registered name={FOUND}"]
    assert len(warning) == 1
    assert warning[0].startswith("turn_ending_tool_not_found name=typo__submit ")
    assert "candidates=[" in warning[0]
    # The closest real name must lead the candidate list.
    assert warning[0].endswith(f"candidates=['{FOUND}', 'read_file']")


def test_no_events_when_no_specs(tmp_path, caplog):
    agent = _make(tmp_path, [])
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics([_Meta("read_file")])
    assert _messages(caplog) == []


def test_metadata_may_be_dicts(tmp_path, caplog):
    """Non-claude providers return ``{"name": ...}`` dicts, not objects."""
    agent = _make(tmp_path, ["read_file"])
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics([{"name": "read_file"}])
    assert _messages(caplog, logging.INFO) == [
        "turn_ending_tool_registered name=read_file"
    ]


def test_empty_inventory_yields_empty_candidates(tmp_path, caplog):
    agent = _make(tmp_path, ["foo"])
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics([])
    assert _messages(caplog, logging.WARNING) == [
        "turn_ending_tool_not_found name=foo candidates=[]"
    ]


def test_candidates_ranked_by_similarity(tmp_path, caplog):
    agent = _make(tmp_path, ["repo_graph__submit_fragment"])
    inventory = [_Meta("other__unrelated"), _Meta(FOUND)]
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics(inventory)
    (line,) = _messages(caplog, logging.WARNING)
    assert line.split("candidates=")[1].startswith(f"['{FOUND}'")


def test_candidates_deterministic_tie_break(tmp_path, caplog):
    """Equal-similarity candidates are ordered by ascending name."""
    agent = _make(tmp_path, ["axx"])
    inventory = [_Meta("bca"), _Meta("acb"), _Meta("abc")]

    outputs = []
    for _ in range(5):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            agent._log_turn_ending_tool_diagnostics(inventory)
        outputs.append(_messages(caplog, logging.WARNING)[0])

    assert len(set(outputs)) == 1
    # 'abc' and 'acb' both score 1/3 against 'axx'; 'bca' scores 0.
    assert outputs[0] == (
        "turn_ending_tool_not_found name=axx candidates=['abc', 'acb', 'bca']"
    )


def test_candidates_capped_at_eight(tmp_path, caplog):
    agent = _make(tmp_path, ["foo"])
    inventory = [_Meta(f"tool_{i:02d}") for i in range(20)]
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics(inventory)
    (line,) = _messages(caplog, logging.WARNING)
    names = line.split("candidates=")[1]
    assert names.count("'") == 16  # 8 quoted names


def test_diagnostics_uses_metadata_from_extract_helper_return(tmp_path, caplog):
    """The behavioural seam: diagnostics classify against the SAME
    inventory the REAL ``_extract_final_tool_metadata`` returns — not a
    test-local list, and not a second independent lookup.

    The metadata is produced by running the actual helper over a fake
    compiled graph, so the shape it emits (dicts with a ``"name"`` key
    for non-claude providers) is the shape the diagnostic must handle.
    """
    from unittest.mock import MagicMock

    from langchain_core.tools import StructuredTool

    def _tool(name: str) -> StructuredTool:
        return StructuredTool.from_function(
            lambda value="": value, name=name, description=f"desc:{name}"
        )

    tool_node = MagicMock()
    tool_node.tools_by_name = {n: _tool(n) for n in (FOUND, "read_file")}
    compiled = MagicMock()
    compiled.nodes = {"tools": tool_node}

    metadata = AgentProcess._extract_final_tool_metadata(
        compiled, provider="deepseek"
    )
    assert metadata, "the real extraction helper returned nothing"

    agent = _make(tmp_path, [SPEC, "typo__submit"])
    with caplog.at_level(logging.INFO):
        agent._log_turn_ending_tool_diagnostics(metadata)

    known = {m["name"] for m in metadata}
    assert known == {FOUND, "read_file"}

    info = _messages(caplog, logging.INFO)
    assert info, "no registered event emitted for a name that IS present"
    for line in info:
        assert line.split("name=")[1] in known

    warnings = _messages(caplog, logging.WARNING)
    assert warnings, "no not_found event emitted for an absent name"
    for line in warnings:
        head, _, tail = line.partition(" candidates=")
        assert head.split("name=")[1] not in known
        for candidate in ast.literal_eval(tail):
            assert candidate in known


@pytest.mark.asyncio
async def test_diagnostics_run_between_inventory_and_prewarm(tmp_path, monkeypatch):
    """Call ORDER, observed behaviourally: the diagnostic runs after the
    inventory is finalised and before the provider prewarm.

    Emitting earlier would report false misses while external MCP is
    still connecting; emitting after prewarm would report a state the
    prewarm already consumed.
    """
    order: list[str] = []

    def _extract(compiled_agent, provider=None):
        order.append("extract")
        return [{"name": FOUND}]

    def _diag(self, metadata):
        order.append("diagnostics")
        assert [m["name"] for m in metadata] == [FOUND]

    async def _prewarm(**kwargs):
        order.append("prewarm")

    monkeypatch.setattr(
        AgentProcess, "_extract_final_tool_metadata", staticmethod(_extract)
    )
    monkeypatch.setattr(
        AgentProcess, "_log_turn_ending_tool_diagnostics", _diag
    )
    monkeypatch.setattr(
        AgentProcess, "_prewarm_client_manager", staticmethod(_prewarm)
    )

    # Replay the exact fragment start() runs, against the patched methods.
    agent = _make(tmp_path, [SPEC])
    agent.agent = object()
    final_metadata = AgentProcess._extract_final_tool_metadata(
        agent.agent, provider=None
    )
    agent._log_turn_ending_tool_diagnostics(final_metadata)
    await AgentProcess._prewarm_client_manager(
        client_manager=object(),
        thread_id=agent.name,
        tools_metadata=final_metadata,
        working_dir=None,
    )

    assert order == ["extract", "diagnostics", "prewarm"]


def _start_ast():
    import ast as ast_mod
    import inspect
    import textwrap

    return ast_mod, ast_mod.parse(
        textwrap.dedent(inspect.getsource(AgentProcess.start))
    )


def _calls_named(ast_mod, tree, attr: str) -> list:
    return [
        node
        for node in ast_mod.walk(tree)
        if isinstance(node, ast_mod.Call)
        and isinstance(node.func, ast_mod.Attribute)
        and node.func.attr == attr
    ]


def test_start_calls_diagnostics_with_the_extracted_inventory():
    """The diagnostic consumes the extracted inventory, not its own.

    Checked on the AST so formatting churn cannot break it and a
    renamed variable cannot slip past.
    """
    ast_mod, tree = _start_ast()

    diag_calls = _calls_named(ast_mod, tree, "_log_turn_ending_tool_diagnostics")
    assert len(diag_calls) == 1, "expected exactly one diagnostics call site"

    (arg,) = diag_calls[0].args
    assert isinstance(arg, ast_mod.Name), (
        "diagnostics must consume the extracted inventory variable, "
        "not an inline expression"
    )
    assigned_from_extract = {
        target.id
        for node in ast_mod.walk(tree)
        if isinstance(node, ast_mod.Assign)
        and isinstance(node.value, ast_mod.Call)
        and isinstance(node.value.func, ast_mod.Attribute)
        and node.value.func.attr == "_extract_final_tool_metadata"
        for target in node.targets
        if isinstance(target, ast_mod.Name)
    }
    assert arg.id in assigned_from_extract, (
        f"{arg.id!r} does not hold _extract_final_tool_metadata's return value"
    )


def test_start_orders_extract_then_diagnostics_then_prewarm():
    """PRODUCTION ordering, pinned on real source positions.

    The behavioural test above replays the three calls in an order the
    test itself chooses, so it would stay green if ``start()`` moved the
    diagnostic after the prewarm. This one reads the ordering out of
    ``AgentProcess.start()`` itself: emitting before the inventory is
    finalised would report false misses while external MCP is still
    connecting, and emitting after prewarm would describe a state the
    prewarm already consumed.

    Structural rather than end-to-end because reaching this region of
    ``start()`` for real needs a daemonised process, a registry, an RPyC
    server and a live model — the same constraint documented in
    ``test_provider_capability_guards_in_start.py``.
    """
    ast_mod, tree = _start_ast()

    extract_assigns = [
        node
        for node in ast_mod.walk(tree)
        if isinstance(node, ast_mod.Assign)
        and isinstance(node.value, ast_mod.Call)
        and isinstance(node.value.func, ast_mod.Attribute)
        and node.value.func.attr == "_extract_final_tool_metadata"
    ]
    diag_calls = _calls_named(ast_mod, tree, "_log_turn_ending_tool_diagnostics")
    prewarm_calls = _calls_named(ast_mod, tree, "_prewarm_client_manager")

    assert len(extract_assigns) == 1, (
        "expected exactly one inventory extraction assignment in start()"
    )
    assert len(diag_calls) == 1, "expected exactly one diagnostics call site"
    assert len(prewarm_calls) == 1, "expected exactly one prewarm call site"

    extract_line = extract_assigns[0].lineno
    diag_line = diag_calls[0].lineno
    prewarm_line = prewarm_calls[0].lineno

    assert extract_line < diag_line < prewarm_line, (
        "start() must emit turn-ending diagnostics AFTER the final tool "
        "inventory is extracted and BEFORE the provider prewarm; got "
        f"extract@{extract_line}, diagnostics@{diag_line}, "
        f"prewarm@{prewarm_line}"
    )
