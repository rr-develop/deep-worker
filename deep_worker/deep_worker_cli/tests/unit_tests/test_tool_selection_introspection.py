"""What the agent REPORTS it can do must match what it can do.

``_handle_get_tools`` answers the Full Dialog "Available Tools" view. It
reads the inventory out of the compiled graph's ``tools`` node — which is
the FULL registry, because tool selection is enforced one layer above it,
on ``request.tools`` inside ``wrap_model_call``. The graph deliberately
keeps every tool so that ``ToolNode`` can refuse a denied call by name;
that design is correct and must not change.

The consequence is that the introspection view reported all 23 tools for an
agent launched with ``--tools '*,project_memory'`` (the spelling of the
time) — observed live on
2026-08-30 against agent ``tsel_probe``, whose ``session_metadata.json``
carried ``"tools": ["-*", "project_memory"]`` and which could in fact call
exactly one. The neighbouring handler ``_handle_get_system_prompt`` already
passes ``self._tool_selection`` through; this one did not.

A wrong inventory is not cosmetic. The view exists to answer "what can this
agent do", and an operator reading 23 where the answer is 1 draws a false
conclusion about a restricted agent — the precise measurement the feature
was built to support.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import tool

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.tool_selection import parse_tool_selection


@tool
def ls(path: str = ".") -> str:
    """List a directory."""
    return "listing"


@tool
def project_memory(query: str) -> str:
    """Recall knowledge from an indexed repository."""
    return "recalled"


@tool
def execute(command: str) -> str:
    """Execute a shell command."""
    return "ran"


class _ToolNode:
    """Stands in for the graph's ToolNode, which holds the FULL registry."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools_by_name = {t.name: t for t in tools}


class _Agent:
    def __init__(self, tools: list[Any]) -> None:
        self.nodes = {"tools": _ToolNode(tools)}


class _Config:
    provider = "minimax"
    model = None


def _process(selection_spec: list[str] | None) -> AgentProcess:
    """An AgentProcess wired to a compiled-agent stub, without booting one.

    ``_handle_get_tools`` only reads ``self.agent``, ``self.config`` and the
    parsed selection, so constructing the real process is unnecessary — and
    would need a provider, a checkpointer and a working directory.
    """
    proc = AgentProcess.__new__(AgentProcess)
    proc.agent = _Agent([ls, project_memory, execute])
    proc.config = _Config()
    proc._tool_selection = parse_tool_selection(selection_spec)
    return proc


async def _names(proc: AgentProcess) -> list[str]:
    payload = await proc._handle_get_tools()
    assert payload["success"] is True, payload
    return [t["name"] for t in payload["tools"]]


@pytest.mark.asyncio
async def test_deny_all_but_one_reports_only_that_one() -> None:
    """The live defect: deny-all-but-one reported the whole registry."""
    names = await _names(_process(["-*", "project_memory"]))
    assert names == ["project_memory"], (
        "the reported inventory must be the tools the model is offered, "
        f"not the graph's full registry; got {names}"
    )


@pytest.mark.asyncio
async def test_a_denied_tool_is_not_reported() -> None:
    names = await _names(_process(["-execute"]))
    assert "execute" not in names
    assert {"ls", "project_memory"} <= set(names)


@pytest.mark.asyncio
async def test_deny_all_reports_nothing() -> None:
    assert await _names(_process(["-*"])) == []


@pytest.mark.asyncio
async def test_no_selection_reports_every_tool() -> None:
    """The unrestricted path must stay exactly as it was."""
    names = await _names(_process(None))
    assert set(names) == {"ls", "project_memory", "execute"}


@pytest.mark.asyncio
async def test_a_process_without_the_attribute_still_answers() -> None:
    """A process predating the feature has no ``_tool_selection``.

    It must report the full inventory rather than raising — the same
    tolerance ``_handle_get_system_prompt`` shows via ``getattr``.
    """
    proc = _process(None)
    del proc._tool_selection
    assert set(await _names(proc)) == {"ls", "project_memory", "execute"}
