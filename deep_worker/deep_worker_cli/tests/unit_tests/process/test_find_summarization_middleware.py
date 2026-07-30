"""Test AgentProcess._find_summarization_middleware against a REAL compiled graph.

See DESIGN_STARTUP_QUERY_OPTIONS.md §4.4. The startup pseudo-summary path must
be able to recover the live SummarizationMiddleware instance embedded in the
compiled LangGraph agent. If the graph has none, it must hard-fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from deep_worker_cli.process.agent_process import AgentProcess
from deepagents.middleware.summarization import SummarizationMiddleware


def _fake(agent: object) -> MagicMock:
    fake = MagicMock()
    fake.name = "ag"
    fake.agent = agent
    return fake


def test_finds_instance_in_real_compiled_graph() -> None:
    from langchain.agents import create_agent

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    sm = SummarizationMiddleware(
        model=model,
        backend=None,
        trigger=("messages", 5),
        keep=("messages", 2),
    )
    agent = create_agent(model, middleware=[sm])

    found = AgentProcess._find_summarization_middleware(_fake(agent))
    assert found is sm


def test_hard_fails_when_no_summarization_middleware() -> None:
    from langchain.agents import create_agent

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    agent = create_agent(model, middleware=[])  # no SummarizationMiddleware

    with pytest.raises(RuntimeError):
        AgentProcess._find_summarization_middleware(_fake(agent))


def test_hard_fails_when_agent_none() -> None:
    fake = _fake(None)
    with pytest.raises(RuntimeError):
        AgentProcess._find_summarization_middleware(fake)


def test_finds_instance_via_async_only_node() -> None:
    """Reviewer concern: an async-only graph build may expose the middleware
    only through ``afunc`` (abefore_model), not ``func``. The bounded walk
    must still reach the instance via ``afunc.__self__``."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    sm = SummarizationMiddleware(
        model=model, backend=None, trigger=("messages", 5), keep=("messages", 2)
    )

    class _FakeRunnableCallable:
        def __init__(self, func: object, afunc: object) -> None:
            self.func = func
            self.afunc = afunc

    class _FakeNode:
        def __init__(self, bound: object) -> None:
            self.bound = bound

    # func is None → only the async handler is bound to the instance.
    node = _FakeNode(_FakeRunnableCallable(func=None, afunc=sm.abefore_model))
    fake_agent = MagicMock()
    fake_agent.nodes = {"SummarizationMiddleware.before_model": node}

    found = AgentProcess._find_summarization_middleware(_fake(fake_agent))
    assert found is sm
