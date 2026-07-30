"""Integration test for apply_pseudo_summary with a REAL filesystem backend.

See DESIGN_STARTUP_QUERY_OPTIONS.md §6.4. No mocks: a real FilesystemBackend
rooted at tmp_path, a real SummarizationMiddleware, real JSONL serialization.

Verifies Blocker-1 end-to-end: the offload lands in
``<prefix>/<agent>.jsonl`` (driven by the EXPLICIT thread_id argument, with
NO graph run / NO config-context), and the returned summary-shape message
references that exact path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AnyMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware


def _middleware(backend: FilesystemBackend) -> SummarizationMiddleware:
    model = __import__(
        "langchain_core.language_models.fake_chat_models",
        fromlist=["GenericFakeChatModel"],
    ).GenericFakeChatModel(messages=iter([HumanMessage(content="x")]))
    return SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
        history_path_prefix="/conversation_history",
    )


@pytest.mark.asyncio
async def test_apply_pseudo_summary_writes_agent_jsonl_no_graph(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    mw = _middleware(backend)

    messages: list[AnyMessage] = [
        HumanMessage(content="user 1", id="h1"),
        HumanMessage(content="user 2", id="h2"),
        HumanMessage(content="user 3", id="h3"),
    ]

    new_msgs, file_path = await mw.apply_pseudo_summary(
        "myagent", "brand new task", messages
    )

    # Blocker-1: the offload targeted <agent>.jsonl, NOT session_<uuid>.jsonl.
    assert file_path == "/conversation_history/myagent.jsonl"
    on_disk = tmp_path / "conversation_history" / "myagent.jsonl"
    assert on_disk.exists(), "JSONL history must be written under the agent name"
    content = on_disk.read_text(encoding="utf-8")
    assert content.strip(), "history file must not be empty"
    # The evicted user messages are serialized into the JSONL.
    assert "user 1" in content
    assert "user 3" in content

    # The summary-shape message references the exact JSONL path and carries
    # the sum-query text and the summarization marker.
    assert len(new_msgs) == 1
    assert "/conversation_history/myagent.jsonl" in new_msgs[0].content
    assert "brand new task" in new_msgs[0].content
    assert new_msgs[0].additional_kwargs.get("lc_source") == "summarization"


@pytest.mark.asyncio
async def test_apply_pseudo_summary_empty_history_writes_nothing(tmp_path: Path) -> None:
    """Empty history (e.g. after --clean): no JSONL written, no file reference."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    mw = _middleware(backend)

    new_msgs, file_path = await mw.apply_pseudo_summary("myagent", "hello", [])

    assert file_path is None
    assert not (tmp_path / "conversation_history" / "myagent.jsonl").exists()
    assert ".jsonl" not in new_msgs[0].content
    assert "hello" in new_msgs[0].content
