"""Withholding a schema is not enforcement — the dispatch must refuse too.

The tool-selection feature originally filtered ``request.tools`` inside
``wrap_model_call`` and stopped there. That is a statement about what the
model is OFFERED, and a model can call a tool it was never offered: from
memory, from a prompt hint, or because it saw the name in an earlier turn.
``ToolNode`` resolves such a call against the graph's OWN tool registry —
not against the filtered request — and runs the real implementation.

Observed live on 2026-08-30: an agent launched with
``--tools '*,project_memory'`` (the spelling of the time) logged
"1 of 23 tools offered to the model"
and then successfully executed ``ls`` twice and reached ``read_file``.

These tests drive a REAL compiled agent with a REAL ToolNode, because a
middleware exercised in isolation cannot see that gap: the isolated filter
was correct the whole time the live agent was bypassing it.
"""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.graph import create_deep_agent
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deep_worker_cli.middleware.tool_selection import ToolSelectionMiddleware
from deep_worker_cli.tool_selection import parse_tool_selection


# Side effects are recorded here so a test can prove a refused tool did not
# merely return an error message — it never ran at all.
_RAN: list[str] = []


@tool
def ls(path: str = ".") -> str:
    """List a directory."""
    _RAN.append("ls")
    return f"listing of {path}"


@tool
def project_memory(query: str) -> str:
    """Search the project index."""
    _RAN.append("project_memory")
    return f"hits for {query}"


class _ModelCallingTool(GenericFakeChatModel):
    """A model that calls ``tool_name`` regardless of what it was offered.

    This is the whole point: the real model did exactly this. Binding is
    accepted and ignored, so the fake cannot be "fixed" by the filter.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ModelCallingTool":
        return self


def _agent_calling(
    tool_name: str, selection_spec: list[str], args: dict | None = None
):
    """Compile a real agent whose model always calls ``tool_name`` once."""
    messages = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": args or {}, "id": "call_1"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    return create_agent(
        model=_ModelCallingTool(messages=messages),
        tools=[ls, project_memory],
        middleware=[
            ToolSelectionMiddleware(parse_tool_selection(selection_spec))
        ],
    )


def _tool_messages(result: dict) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


@pytest.fixture(autouse=True)
def _clear_ran():
    _RAN.clear()
    yield
    _RAN.clear()


def test_disallowed_tool_is_refused_by_a_real_graph() -> None:
    """The regression: ``ls`` ran despite a deny-all-but-project_memory spec."""
    result = _agent_calling("ls", ["-*", "project_memory"]).invoke(
        {"messages": [("user", "look around")]}
    )

    refusals = _tool_messages(result)
    assert refusals, "the tool call produced no ToolMessage at all"
    assert refusals[0].status == "error"
    assert "disabled" in refusals[0].content
    # The load-bearing assertion: not "it returned an error" but "the tool
    # body never executed". An error-shaped result from a tool that still
    # ran would leave the measurement just as corrupted.
    assert "ls" not in _RAN


def test_allowed_tool_still_runs_through_the_same_graph() -> None:
    """The barrier must not swallow the tools the selection kept."""
    result = _agent_calling(
        "project_memory", ["-*", "project_memory"], {"query": "anything"}
    ).invoke({"messages": [("user", "search")]})

    messages = _tool_messages(result)
    assert messages
    assert messages[0].status != "error"
    assert "project_memory" in _RAN


def test_deny_list_form_refuses_the_named_tool() -> None:
    result = _agent_calling("ls", ["-ls"]).invoke(
        {"messages": [("user", "look around")]}
    )

    assert _tool_messages(result)[0].status == "error"
    assert "ls" not in _RAN


def test_unrestricted_selection_runs_everything() -> None:
    """An empty selection must not turn into a refusal.

    ``create_cli_agent`` does not even install the middleware in this case,
    but a middleware that refused here would be a landmine for any caller
    that did install it.
    """
    result = _agent_calling("ls", []).invoke(
        {"messages": [("user", "look around")]}
    )

    assert _tool_messages(result)[0].status != "error"
    assert "ls" in _RAN


def test_subagent_cannot_run_what_the_selection_denied() -> None:
    """Delegation must not hand the tools back (review finding B4).

    A subagent is a SEPARATE graph with its own middleware stack and its
    own ``FilesystemMiddleware``. Before the fix, ``--tools '-*,task'``
    produced a main agent that correctly refused ``ls``/``read_file`` and
    a subagent that ran them: the restriction stopped at the delegation
    boundary, which is exactly where a tool-starved agent goes next.

    ``create_deep_agent`` therefore propagates any middleware declaring
    ``applies_to_subagents`` into every subagent graph it builds.
    """
    messages = iter(
        [
            # main agent delegates...
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "do it",
                            "subagent_type": "general-purpose",
                        },
                        "id": "call_task",
                    }
                ],
            ),
            # ...and the SUBAGENT reaches for a denied tool.
            AIMessage(
                content="",
                tool_calls=[{"name": "ls", "args": {}, "id": "call_ls"}],
            ),
            AIMessage(content="subagent done"),
            AIMessage(content="main done"),
        ]
    )
    agent = create_deep_agent(
        model=_ModelCallingTool(messages=messages),
        tools=[ls],
        middleware=[
            ToolSelectionMiddleware(parse_tool_selection(["-*", "task"]))
        ],
    )

    agent.invoke({"messages": [("user", "delegate please")]})

    assert "ls" not in _RAN, (
        "the subagent executed a tool the selection denied — the "
        "restriction did not survive delegation"
    )


def test_subagent_from_a_custom_spec_is_covered_too() -> None:
    """Custom subagent specs build their own stack; same rule applies."""
    messages = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "go",
                            "subagent_type": "researcher",
                        },
                        "id": "call_task",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "ls", "args": {}, "id": "call_ls"}],
            ),
            AIMessage(content="subagent done"),
            AIMessage(content="main done"),
        ]
    )
    agent = create_deep_agent(
        model=_ModelCallingTool(messages=messages),
        tools=[ls],
        middleware=[
            ToolSelectionMiddleware(parse_tool_selection(["-*", "task"]))
        ],
        subagents=[
            {
                "name": "researcher",
                "description": "researches",
                "system_prompt": "you research",
            }
        ],
    )

    agent.invoke({"messages": [("user", "delegate please")]})

    assert "ls" not in _RAN


def test_subagent_is_not_offered_the_denied_schemas_either() -> None:
    """Enforcement and advertisement must agree inside the subagent too."""
    offered: list[list[str]] = []

    class _Spy(_ModelCallingTool):
        def bind_tools(self, tools, **kwargs):
            offered.append(
                sorted(getattr(t, "name", str(t)) for t in tools)
            )
            return self

    messages = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "go",
                            "subagent_type": "general-purpose",
                        },
                        "id": "call_task",
                    }
                ],
            ),
            AIMessage(content="subagent done"),
            AIMessage(content="main done"),
        ]
    )
    agent = create_deep_agent(
        model=_Spy(messages=messages),
        tools=[ls],
        middleware=[
            ToolSelectionMiddleware(parse_tool_selection(["-*", "task"]))
        ],
    )

    agent.invoke({"messages": [("user", "delegate")]})

    # Every binding made from inside the graph offers only the kept tool;
    # notably none of them includes the filesystem tools the subagent's
    # own FilesystemMiddleware contributes.
    in_graph = [names for names in offered if names != ["ls"]]
    assert in_graph, "the model was never bound inside the graph"
    for names in in_graph:
        assert names == ["task"], f"subagent was offered {names}"


def test_refusal_names_the_tool_and_forbids_a_workaround() -> None:
    """The refusal has to be actionable, or the model retries forever."""
    result = _agent_calling("ls", ["-*", "project_memory"]).invoke(
        {"messages": [("user", "look around")]}
    )

    content = _tool_messages(result)[0].content
    assert "'ls'" in content
    assert "not retry" in content.lower()
    # An agent told only "no" tends to look for another route to the same
    # capability; the text must close that door explicitly.
    assert "another route" in content.lower()


def test_denied_call_never_reaches_the_approval_prompt() -> None:
    """A forbidden tool must be refused outright, never negotiated (B6).

    ``HumanInTheLoopMiddleware`` works in an ``after_model`` GRAPH NODE,
    which runs before the whole ``wrap_tool_call`` chain the dispatch
    barrier lives in. So with HITL configured for a DENIED tool, the user
    was asked to authorise something the agent is not allowed to have —
    and a user who approves is entitled to expect the call to happen.

    The selection's own ``after_model`` cuts the denied call first. The
    two hooks order oppositely (``wrap_tool_call`` nests in list order,
    ``after_model`` nodes chain in REVERSE list order), which is why the
    selection is appended AFTER HITL in ``create_deep_agent``.
    """
    messages = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "ls", "args": {}, "id": "call_ls"}],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(
        model=_ModelCallingTool(messages=messages),
        tools=[ls],
        middleware=[
            ToolSelectionMiddleware(parse_tool_selection(["-*", "task"]))
        ],
        interrupt_on={"ls": True},
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [("user", "look around")]},
        {"configurable": {"thread_id": "b6"}},
    )

    assert not result.get("__interrupt__"), (
        "HITL raised an approval prompt for a tool the selection denied"
    )
    assert "ls" not in _RAN


def test_compiled_subagent_is_rejected_when_a_selection_is_active() -> None:
    """Fail closed rather than silently un-restricting (review finding B5).

    A ``CompiledSubAgent`` arrives as an already-built graph, so there is
    no middleware stack left to propagate the selection into. Accepting it
    would produce exactly the escape B4 closed, only quieter. The error
    has to name the offender and say what to do instead.
    """
    inner = create_agent(model=_ModelCallingTool(messages=iter([])), tools=[ls])

    with pytest.raises(ValueError) as excinfo:
        create_deep_agent(
            model=_ModelCallingTool(messages=iter([])),
            tools=[ls],
            subagents=[
                {
                    "name": "prebuilt",
                    "description": "a pre-compiled helper",
                    "runnable": inner,
                }
            ],
            middleware=[
                ToolSelectionMiddleware(parse_tool_selection(["-*", "task"]))
            ],
        )

    assert "prebuilt" in str(excinfo.value)


def test_compiled_subagent_still_builds_without_a_selection() -> None:
    """The fail-closed check must not break unrestricted agents."""
    inner = create_agent(model=_ModelCallingTool(messages=iter([])), tools=[ls])

    agent = create_deep_agent(
        model=_ModelCallingTool(messages=iter([])),
        tools=[ls],
        subagents=[
            {
                "name": "prebuilt",
                "description": "a pre-compiled helper",
                "runnable": inner,
            }
        ],
    )

    assert agent is not None
