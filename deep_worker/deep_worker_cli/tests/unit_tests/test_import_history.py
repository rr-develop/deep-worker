"""TDD for sessions.import_history — write an imported message list into the
agent's per-agent SQLite DB (DESIGN_IMPORT_CLAUDE_SESSION.md §4).

Acceptance: after import, the EXISTING offline read path
``get_thread_history(agent)`` returns the same messages — i.e. they actually
landed in the agent's own ``threads.db``. We isolate HOME so no real DB
is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from deep_worker_cli import sessions


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))


def _sample() -> list[BaseMessage]:
    return [
        HumanMessage(content="remember: the port is 8472"),
        AIMessage(content="noted, port 8472"),
    ]


async def test_import_writes_messages_to_sqlite_db() -> None:
    agent = "import_test_agent"
    await sessions.import_history(agent, _sample(), replace=False)

    # DB file exists where the agent expects it.
    assert sessions.get_db_path(agent).exists()

    # The existing offline read path sees the imported turns.
    result = await sessions.get_thread_history(agent)
    assert result["success"] is True, result
    contents = [str(getattr(m, "content", m)) for m in result["messages"]]
    assert any("8472" in c for c in contents)
    assert len(result["messages"]) == 2


async def test_import_with_tool_messages_roundtrips() -> None:
    agent = "import_tool_agent"
    msgs: list[BaseMessage] = [
        HumanMessage(content="read it"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "read_file", "args": {}}]),
        ToolMessage(content="file body", tool_call_id="t1"),
        AIMessage(content="done"),
    ]
    await sessions.import_history(agent, msgs, replace=False)

    result = await sessions.get_thread_history(agent)
    assert result["success"] is True
    assert len(result["messages"]) == 4


async def test_refuses_when_history_exists_without_replace() -> None:
    agent = "import_guard_agent"
    await sessions.import_history(agent, _sample(), replace=False)
    # Second import without replace must refuse (don't splice two convos).
    with pytest.raises(Exception):
        await sessions.import_history(agent, _sample(), replace=False)


async def test_replace_overwrites_existing_history() -> None:
    agent = "import_replace_agent"
    await sessions.import_history(
        agent, [HumanMessage(content="old conversation")], replace=False
    )
    await sessions.import_history(
        agent, [HumanMessage(content="brand new conversation")], replace=True
    )
    result = await sessions.get_thread_history(agent)
    assert result["success"] is True
    contents = [str(getattr(m, "content", m)) for m in result["messages"]]
    assert any("brand new" in c for c in contents)
    assert not any("old conversation" in c for c in contents)


async def test_empty_messages_is_noop_or_error_not_crash() -> None:
    agent = "import_empty_agent"
    # Importing nothing should not crash; either no-op or a clear error.
    try:
        await sessions.import_history(agent, [], replace=False)
    except ValueError:
        pass  # acceptable: refuse empty import


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
