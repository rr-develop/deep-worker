from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool

from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.sessions import get_checkpointer

from deep_worker_cli.storage_paths import deep_worker_home
class _ToolAwareFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@tool
def timestamp_echo(value: str) -> str:
    """Echo a value for timestamp integration tests."""
    return f"echo:{value}"


def _created_at(message: BaseMessage) -> str:
    value = message.additional_kwargs.get("created_at")
    assert isinstance(value, str), f"missing created_at on {message!r}"
    return value


@pytest.mark.asyncio
async def test_agent_persists_monotonic_created_at_for_all_message_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_name = "timestamp_integ"
    thread_id = agent_name
    model = _ToolAwareFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_echo",
                            "name": "timestamp_echo",
                            "args": {"value": "abc"},
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    async with get_checkpointer(agent_name) as checkpointer:
        agent, _backend = create_cli_agent(
            model=model,
            assistant_id=agent_name,
            tools=[timestamp_echo],
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
            checkpointer=checkpointer,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "run echo"}]},
            {
                "configurable": {"thread_id": thread_id},
                "metadata": {"agent_name": agent_name, "assistant_id": agent_name},
            },
        )
        assert result["messages"][-1].content == "done"

    async with get_checkpointer(agent_name) as checkpointer:
        cp_tuple = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )

    assert cp_tuple is not None
    messages = cp_tuple.checkpoint["channel_values"]["messages"]
    assert len(messages) >= 4
    assert all(isinstance(message, BaseMessage) for message in messages)

    timestamps = [_created_at(message) for message in messages]
    parsed = [datetime.fromisoformat(value) for value in timestamps]
    assert parsed == sorted(parsed)
    assert len(set(timestamps)) == len(timestamps)
    assert all(value.endswith("+00:00") for value in timestamps)

    ai_with_tool = next(
        message for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
    )
    call_map = ai_with_tool.additional_kwargs.get("tool_call_created_at")
    assert isinstance(call_map, dict)
    assert call_map["call_echo"] == ai_with_tool.additional_kwargs["created_at"]

    tool_message = next(message for message in messages if isinstance(message, ToolMessage))
    assert tool_message.tool_call_id == "call_echo"
    assert tool_message.additional_kwargs["created_at"] > ai_with_tool.additional_kwargs["created_at"]


@pytest.mark.asyncio
async def test_summarization_markdown_contains_per_message_created_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_name = "timestamp_summary_integ"
    thread_id = "summary-thread"
    model = _ToolAwareFakeModel(
        messages=iter([AIMessage(content="summary"), AIMessage(content="final")])
    )
    model.profile = {"max_input_tokens": 200_000}

    async with get_checkpointer(agent_name) as checkpointer:
        agent, _backend = create_cli_agent(
            model=model,
            assistant_id=agent_name,
            tools=[],
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
            checkpointer=checkpointer,
        )
        long_user = "u" * 10_000 * 4
        long_ai = "a" * 50_000 * 4
        await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": long_user},
                    AIMessage(content=long_ai),
                    {"role": "user", "content": long_user},
                    AIMessage(content=long_ai),
                    {"role": "user", "content": long_user},
                    AIMessage(content=long_ai),
                    {"role": "user", "content": "finish"},
                ]
            },
            {
                "configurable": {"thread_id": thread_id},
                "metadata": {"agent_name": agent_name, "assistant_id": agent_name},
            },
        )

    summary_file = deep_worker_home(tmp_path) / agent_name / "summaries" / f"{thread_id}.md"
    assert summary_file.exists()
    content = summary_file.read_text()

    assert "## Summarized at " in content
    assert "+00:00] Human:" in content
    assert "+00:00] AI:" in content
