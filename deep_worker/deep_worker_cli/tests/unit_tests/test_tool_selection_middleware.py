"""The filter must act where the model reads tools, not where lists are built.

Tools reach the model from two directions — the list ``create_cli_agent``
assembles, and the tools middleware contribute. ``wrap_model_call`` sees
the union, so a selection enforced there is the only one that can honestly
claim "no tools".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage

from deep_worker_cli.middleware.tool_selection import ToolSelectionMiddleware
from deep_worker_cli.tool_selection import parse_tool_selection


class _Request:
    """Minimal stand-in for ModelRequest: tools plus ``override``."""

    def __init__(self, tools, system_message=None):
        self.tools = list(tools)
        self.system_message = system_message or SystemMessage(content="")

    def override(self, **kwargs):
        replacement = _Request(self.tools, self.system_message)
        for key, value in kwargs.items():
            setattr(replacement, key, value)
        return replacement


def _request(*names):
    return _Request([SimpleNamespace(name=n) for n in names])


def _system_text(request):
    content = request.system_message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def _seen_by_model(selection_spec, request):
    """Return the tool names the handler is actually offered."""
    middleware = ToolSelectionMiddleware(parse_tool_selection(selection_spec))
    captured = {}

    def handler(req):
        captured["names"] = [t.name for t in req.tools]
        return "response"

    assert middleware.wrap_model_call(request, handler) == "response"
    return captured["names"]


def test_star_leaves_the_model_with_nothing() -> None:
    assert _seen_by_model(["-*"], _request("read_file", "execute")) == []


def test_star_plus_name_leaves_only_that_name() -> None:
    seen = _seen_by_model(["-*", "read_file"], _request("read_file", "execute"))
    assert seen == ["read_file"]


def test_minus_removes_only_the_named_tool() -> None:
    seen = _seen_by_model(["-execute"], _request("read_file", "execute"))
    assert seen == ["read_file"]


def test_middleware_contributed_tools_are_filtered_too() -> None:
    """The whole point of filtering here rather than at assembly time.

    ``project_memory`` never appears in the assembled ``tools`` list — it
    is added by ProjectMemoryMiddleware. A filter applied earlier would
    leave it callable while reporting that all tools were off.
    """
    request = _request("read_file", "project_memory", "task", "execute")
    assert _seen_by_model(["-*"], request) == []


def test_request_is_not_mutated_in_place() -> None:
    """Filtering must go through ``override``, leaving the original intact."""
    request = _request("read_file", "execute")
    _seen_by_model(["-*"], request)
    assert [t.name for t in request.tools] == ["read_file", "execute"]


def test_restricted_request_always_receives_the_selection_instruction() -> None:
    middleware = ToolSelectionMiddleware(parse_tool_selection(["-absent_tool"]))
    request = _request("read_file")
    seen = {}

    def handler(req):
        seen["request"] = req
        return "response"

    middleware.wrap_model_call(request, handler)
    assert seen["request"] is not request
    assert "Enabled deep-worker tools: read_file" in _system_text(seen["request"])


def test_selection_states_what_is_enabled_and_forbids_the_rest() -> None:
    """The instruction names the survivors and forbids everything else.

    Free-form prose an agent's own author wrote is NOT a middleware section
    and is deliberately left alone (see
    ``test_real_chain_unowned_prompt_text_is_never_touched``); the
    instruction is what covers it.
    """
    middleware = ToolSelectionMiddleware(parse_tool_selection(["-*"]))
    request = _Request(
        [SimpleNamespace(name="project_memory")],
        SystemMessage(content="You have project_memory and should use it first."),
    )
    seen = {}

    def handler(req):
        seen["request"] = req
        return "response"

    middleware.wrap_model_call(request, handler)
    filtered = seen["request"]
    assert filtered.tools == []
    content = _system_text(filtered)
    assert "Enabled deep-worker tools: none" in content
    assert "Do not use or claim access" in content


def test_native_caveat_is_marked_as_preference_not_enforcement() -> None:
    middleware = ToolSelectionMiddleware(
        parse_tool_selection(["-*"]), native_caveat="grok native tools remain"
    )
    seen = {}

    def handler(req):
        seen["request"] = req
        return "response"

    middleware.wrap_model_call(_request("read_file"), handler)
    content = _system_text(seen["request"])
    assert "grok native tools remain" in content
    assert "preference, not enforcement" in content


def test_caveat_path_does_not_crash_on_a_restricted_acp_agent() -> None:
    """The warning branch must survive ``agent.py``'s shadowed ``logging``.

    ``create_cli_agent`` binds ``logging`` locally further down its body,
    which makes the name local to the entire function — the first version
    of the caveat branch raised UnboundLocalError instead of warning, and
    only a real agent build surfaced it.
    """
    from pathlib import Path

    from deep_worker_cli.agent import create_cli_agent

    agent, _backend = create_cli_agent(
        model="claude-sonnet-4-5-20250929",
        assistant_id="test_tool_selection_caveat",
        project_memory_dir=Path(__file__).resolve().parent,
        provider="grok",
        tool_selection=parse_tool_selection(["-*"]),
    )
    assert agent is not None


@pytest.mark.asyncio
async def test_async_path_filters_identically() -> None:
    """The async wrapper is the one the agent actually runs."""
    middleware = ToolSelectionMiddleware(parse_tool_selection(["-*", "grep"]))
    request = _request("read_file", "grep", "execute")
    captured = {}

    async def handler(req):
        captured["names"] = [t.name for t in req.tools]
        return "response"

    assert await middleware.awrap_model_call(request, handler) == "response"
    assert captured["names"] == ["grep"]


# ---------------------------------------------------------------------------
# Real-chain regressions.
#
# The tests above drive the middleware through a hand-built request. That
# proves the filter's arithmetic but not the thing that actually reaches the
# model: by the time this middleware runs, the OTHER middleware have already
# appended their advertisements to the system prompt. A filter that removes
# schemas while leaving "You have access to an `execute` tool" in the prompt
# produces an agent that is told it has tools it cannot call — which corrupts
# the measurement the feature exists to support.
#
# So these run the genuine middleware chain, in the genuine order, and assert
# on the ModelRequest the model layer is handed.
# ---------------------------------------------------------------------------


def _markers():
    from deepagents.middleware.filesystem import (
        EXECUTION_SYSTEM_PROMPT,
        FILESYSTEM_SYSTEM_PROMPT,
    )
    from deepagents.middleware.project_memory import COGNITIVE_HINT as PROJECT_HINT
    from deepagents.middleware.skills import SKILLS_SYSTEM_PROMPT
    from deepagents.middleware.subagents import TASK_SYSTEM_PROMPT
    from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT

    from deep_worker_cli.middleware.history_memory import (
        COGNITIVE_HINT as HISTORY_HINT,
    )

    return {
        "execute": EXECUTION_SYSTEM_PROMPT,
        "filesystem": FILESYSTEM_SYSTEM_PROMPT,
        "project_memory": PROJECT_HINT,
        "skills": SKILLS_SYSTEM_PROMPT,
        "task": TASK_SYSTEM_PROMPT,
        "history_memory": HISTORY_HINT,
        "write_todos": WRITE_TODOS_SYSTEM_PROMPT,
    }


def test_every_section_marker_matches_the_real_prompt_constant():
    """The markers are matched against real text, not a copy of themselves.

    If a middleware's heading drifts, its section silently becomes
    un-droppable: the schema goes but the advertisement stays. This test is
    the tripwire for that drift.
    """
    from deep_worker_cli.middleware.tool_selection import ToolSelectionMiddleware

    expected = {
        "execute": {"execute"},
        "filesystem": {"ls", "read_file", "write_file", "edit_file", "glob", "grep"},
        "project_memory": {"project_memory"},
        "skills": {"read_file"},
        "task": {"task"},
        "history_memory": {"history_memory"},
        "write_todos": {"write_todos"},
    }
    for key, prompt in _markers().items():
        owners = ToolSelectionMiddleware._section_owners(prompt)
        assert owners is not None, f"{key} prompt no longer matches any marker"
        assert set(owners) == expected[key], key


def test_write_todos_section_goes_with_its_schema():
    """The loop that cost ``test_lsp`` a whole turn (2026-09-04).

    A selection that drops ``write_todos`` must drop its instructions too.
    The negative control matters as much as the positive one: with the
    tool KEPT the section has to survive, or the fix would just be
    "delete this section always" rather than "bind it to its schema".
    """
    from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT

    from deep_worker_cli.middleware.tool_selection import ToolSelectionMiddleware
    from deep_worker_cli.tool_selection import parse_tool_selection

    prompt = SystemMessage(
        content=[{"type": "text", "text": WRITE_TODOS_SYSTEM_PROMPT}]
    )

    def _surviving_text(selection_spec, kept_names):
        middleware = ToolSelectionMiddleware(parse_tool_selection(selection_spec))
        pruned = middleware._strip_dead_sections(prompt, kept_names)
        return "".join(
            block.get("text", "")
            for block in pruned.content
            if isinstance(block, dict)
        )

    dropped = _surviving_text(["none", "+serena_*"], {"serena__find_symbol"})
    assert "`write_todos`" not in dropped, (
        "the prompt still advertises write_todos after its schema was "
        "removed — the model will call it, be refused, and loop"
    )

    kept = _surviving_text(["none", "+write_todos"], {"write_todos"})
    assert "`write_todos`" in kept, (
        "the section was dropped even though the tool survived the "
        "selection — the fix must bind the section to its schema"
    )


def _run_real_chain(selection_spec, middlewares, tools):
    """Drive a real middleware chain and capture the final ModelRequest."""
    from langchain.agents.middleware.types import AgentMiddleware

    from deep_worker_cli.middleware.tool_selection import ToolSelectionMiddleware
    from deep_worker_cli.tool_selection import parse_tool_selection

    captured = {}

    class _Capture(AgentMiddleware):
        tools: list = []

        def wrap_model_call(self, request, handler):
            captured["tools"] = [
                getattr(t, "name", None) for t in request.tools
            ]
            captured["system"] = _system_text(request)
            return "done"

    chain = list(middlewares) + [
        ToolSelectionMiddleware(parse_tool_selection(selection_spec)),
        _Capture(),
    ]

    from langchain.agents.middleware.types import ModelRequest

    # Production hands the model the union of the assembled tool list and
    # every middleware's own ``.tools``; reproduce that union here, or the
    # filter would be measured against a list far smaller than the real one.
    union = list(tools)
    seen_names = {getattr(t, "name", None) for t in union}
    for middleware in chain:
        for tool in getattr(middleware, "tools", None) or []:
            name = getattr(tool, "name", None)
            if name not in seen_names:
                seen_names.add(name)
                union.append(tool)

    request = ModelRequest(
        model=None,
        tools=union,
        system_prompt=None,
        messages=[],
        tool_choice=None,
        state={"messages": []},
        runtime=None,
    )

    def invoke(index, req):
        if index == len(chain):
            return "done"
        return chain[index].wrap_model_call(
            req, lambda r, i=index: invoke(i + 1, r)
        )

    invoke(0, request)
    return captured


def _filesystem_and_subagent_middleware():
    """Tool-owning middleware that inject the REAL production prompt text.

    ``FilesystemMiddleware`` is the genuine article — it contributes both
    the filesystem and the execute sections, and it is the middleware the
    design cites as precedent.

    ``SubAgentMiddleware`` cannot be constructed without a live backend and
    at least one subagent, which would drag a model into a unit test. What
    matters here is the SECTION it appends, so this stand-in appends the
    real ``TASK_SYSTEM_PROMPT`` constant (not a copy of it) through the
    real ``append_to_system_message`` helper. If that constant's heading
    drifts, ``test_every_section_marker_matches_the_real_prompt_constant``
    fails.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    from deepagents.middleware._utils import append_to_system_message
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import TASK_SYSTEM_PROMPT

    from langchain_core.tools import tool as _tool_decorator

    @_tool_decorator
    def task(description: str) -> str:
        """Spawn a subagent."""
        return description

    class _TaskPrompt(AgentMiddleware):
        tools: list = [task]

        def wrap_model_call(self, request, handler):
            return handler(
                request.override(
                    system_message=append_to_system_message(
                        request.system_message, TASK_SYSTEM_PROMPT
                    )
                )
            )

    return [FilesystemMiddleware(), _TaskPrompt()]


def test_real_chain_deny_all_leaves_no_tools_and_no_advertisements():
    """`*` must strip the schemas AND the prose promising those tools."""
    captured = _run_real_chain(
        ["-*"], _filesystem_and_subagent_middleware(), tools=[]
    )

    assert captured["tools"] == []

    system = captured["system"]
    for label, prompt in _markers().items():
        if label == "history_memory":
            continue  # not installed in this chain
        heading = prompt.lstrip().splitlines()[0]
        assert heading not in system, (
            f"prompt still advertises {label!r} after the selection removed it"
        )
    assert "Enabled deep-worker tools: none." in system


def test_real_chain_allowlist_keeps_only_the_named_tool_and_its_section():
    """An allow-list keeps the surviving tool's instructions, drops the rest."""
    captured = _run_real_chain(
        ["-*", "read_file"], _filesystem_and_subagent_middleware(), tools=[]
    )

    assert captured["tools"] == ["read_file"]

    system = captured["system"]
    from deepagents.middleware.filesystem import (
        EXECUTION_SYSTEM_PROMPT,
        FILESYSTEM_SYSTEM_PROMPT,
    )
    from deepagents.middleware.subagents import TASK_SYSTEM_PROMPT

    # read_file survives, so the filesystem section stays — the model still
    # needs to know how to call the tool it DOES have.
    assert FILESYSTEM_SYSTEM_PROMPT.lstrip().splitlines()[0] in system
    # execute and task are gone; their prose must be gone too.
    assert EXECUTION_SYSTEM_PROMPT.lstrip().splitlines()[0] not in system
    assert TASK_SYSTEM_PROMPT.lstrip().splitlines()[0] not in system
    assert "Enabled deep-worker tools: read_file." in system


def test_real_chain_unowned_prompt_text_is_never_touched():
    """Only tool-owned sections are removable; the agent's own prompt stays."""
    from langchain.agents.middleware.types import AgentMiddleware

    from deepagents.middleware._utils import append_to_system_message

    class _AgentPrompt(AgentMiddleware):
        tools: list = []

        def wrap_model_call(self, request, handler):
            return handler(
                request.override(
                    system_message=append_to_system_message(
                        request.system_message,
                        "You are agent_1. Follow the owner's instructions.",
                    )
                )
            )

    captured = _run_real_chain(
        ["-*"],
        [_AgentPrompt()] + _filesystem_and_subagent_middleware(),
        tools=[],
    )

    assert "You are agent_1." in captured["system"]
