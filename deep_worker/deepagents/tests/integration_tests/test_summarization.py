import json
import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit_tests.chat_model import GenericFakeChatModel

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware.summarization_history import (
    KIND,
    SEGMENT_KIND,
    deserialize_summarized_messages_jsonl,
)

# URL for a large file that will trigger summarization
LARGE_FILE_URL = "https://raw.githubusercontent.com/langchain-ai/langchain/3356d0555725c3e0bbb9408c2b3f554cad2a6ee2/libs/partners/openai/langchain_openai/chat_models/base.py"

SYSTEM_PROMPT = dedent(
    """
    ## File Reading Best Practices

    When exploring codebases or reading multiple files, use pagination to prevent context overflow.

    **Pattern for codebase exploration:**
    1. First scan: `read_file(path, limit=100)` - See file structure and key sections
    2. Targeted read: `read_file(path, offset=100, limit=200)` - Read specific sections if needed
    3. Full read: Only use `read_file(path)` without limit when necessary for editing

    **When to paginate:**
    - Reading any file >500 lines
    - Exploring unfamiliar codebases (always start with limit=100)
    - Reading multiple files in sequence

    **When full read is OK:**
    - Small files (<500 lines)
    - Files you need to edit immediately after reading
    """
)


def _write_file(p: Path, content: str) -> None:
    """Helper to write a file, creating parent directories."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _setup_summarization_test(tmp_path: Path, model_name: str) -> tuple[Any, FilesystemBackend, Path, dict[str, Any]]:
    """Common setup for summarization tests.

    Returns:
        Tuple of `(agent, backend, root_path, config)`
    """
    response = requests.get(LARGE_FILE_URL, timeout=30)
    response.raise_for_status()

    root = tmp_path
    fp = root / "base.py"
    _write_file(fp, response.text)

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    checkpointer = InMemorySaver()

    model = init_chat_model(model_name)
    if model.profile is None:
        model.profile = {}
    # Lower artificially to trigger summarization more easily
    model.profile["max_input_tokens"] = 30_000

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        backend=backend,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}

    return agent, backend, root, config


class StableThreadSummarizationMiddleware(SummarizationMiddleware):
    def _get_thread_id(self) -> str:
        return "integration-jsonl-thread"


def test_jsonl_history_offload_appends_to_real_filesystem_backend(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    middleware = StableThreadSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="unused")])),
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 1),
    )

    first_batch = [
        HumanMessage(
            content="first user message",
            id="h-jsonl-1",
            additional_kwargs={"provider_seen_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)},
        ),
        AIMessage(
            content="I will call a tool",
            id="a-jsonl-1",
            tool_calls=[{"id": "call-jsonl-1", "name": "lookup", "args": {"needle": "alpha"}}],
            additional_kwargs={"tool_call_created_at": {"call-jsonl-1": "2026-01-02T03:04:05.000001+00:00"}},
            response_metadata={"cost": Decimal("1.25")},
        ),
    ]
    second_batch = [
        ToolMessage(content="tool result", tool_call_id="call-jsonl-1", id="t-jsonl-1"),
    ]

    path = middleware._offload_to_backend(backend, first_batch)
    assert path == "/conversation_history/integration-jsonl-thread.jsonl"

    same_path = middleware._offload_to_backend(backend, second_batch)
    assert same_path == path

    history_file = tmp_path / "conversation_history" / "integration-jsonl-thread.jsonl"
    content = history_file.read_text()
    records = [json.loads(line) for line in content.splitlines() if line.strip()]

    # Two offloads append two segment headers plus the three messages. Count the
    # kinds separately: asserting a bare line total silently breaks whenever the
    # envelope gains a record, which is exactly how this test rotted unnoticed
    # after 07354c3 introduced segment headers.
    headers = [r for r in records if r.get("kind") == SEGMENT_KIND]
    message_lines = [r for r in records if r.get("kind") == KIND]
    assert len(headers) == 2
    assert len(message_lines) == 3
    assert len(headers) + len(message_lines) == len(records)

    restored = deserialize_summarized_messages_jsonl(content)
    assert [message.id for message in restored] == ["h-jsonl-1", "a-jsonl-1", "t-jsonl-1"]
    # Plain JSON serialization: datetime → ISO string via str(), Decimal → string
    assert restored[0].additional_kwargs["provider_seen_at"] == "2026-01-02 03:04:05+00:00"
    assert restored[1].response_metadata["cost"] == "1.25"
    assert restored[1].tool_calls[0]["id"] == "call-jsonl-1"
    assert restored[1].additional_kwargs["tool_call_created_at"]["call-jsonl-1"] == "2026-01-02T03:04:05.000001+00:00"
    assert restored[2].tool_call_id == "call-jsonl-1"


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("anthropic:claude-sonnet-4-5-20250929", id="claude-sonnet"),
    ],
)
def test_summarize_continues_task(tmp_path: Path, model_name: str) -> None:
    """Test that summarization triggers and the agent can continue reading a large file."""
    agent, _, _, config = _setup_summarization_test(tmp_path, model_name)

    input_message = {
        "role": "user",
        "content": "Can you read the entirety of base.py and summarize it?",
    }
    result = agent.invoke({"messages": [input_message]}, config)

    # Check we summarized
    assert result["messages"][0].additional_kwargs["lc_source"] == "summarization"

    # Verify the agent made substantial progress reading the file after summarization.
    # We check the highest line number seen across all tool messages to confirm
    # the agent continued working after context was summarized.
    max_line_seen = 0
    reached_eof = False

    for message in result["messages"]:
        if message.type == "tool":
            # Check for EOF error (indicates agent tried to read past end)
            if "exceeds file length" in message.content:
                reached_eof = True
            # Extract line numbers from formatted output (e.g., "4609\t    )")
            line_numbers = re.findall(r"^\s*(\d+)\t", message.content, re.MULTILINE)
            if line_numbers:
                max_line_seen = max(max_line_seen, *[int(n) for n in line_numbers])

    assert max_line_seen >= 4609 or reached_eof, (
        f"Expected agent to make substantial progress reading file. Max line seen: {max_line_seen}, reached EOF: {reached_eof}"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("anthropic:claude-sonnet-4-5-20250929", id="claude-sonnet"),
    ],
)
def test_summarization_offloads_to_filesystem(tmp_path: Path, model_name: str) -> None:
    """Test that conversation history is offloaded to filesystem during summarization.

    This verifies the summarization middleware correctly writes conversation history
    as lossless JSONL to the backend at /conversation_history/{thread_id}.jsonl.
    """
    agent, _, root, config = _setup_summarization_test(tmp_path, model_name)

    input_message = {
        "role": "user",
        "content": "Can you read the entirety of base.py and summarize it?",
    }
    result = agent.invoke({"messages": [input_message]}, config)

    # Check we summarized
    assert result["messages"][0].additional_kwargs["lc_source"] == "summarization"

    # Verify conversation history was offloaded to filesystem
    conversation_history_root = root / "conversation_history"
    assert conversation_history_root.exists(), f"Conversation history root directory not found at {conversation_history_root}"

    # Verify the JSONL file exists for thread_id
    thread_id = config["configurable"]["thread_id"]
    history_file = conversation_history_root / f"{thread_id}.jsonl"
    assert history_file.exists(), f"Expected JSONL file at {history_file}"

    # Read and verify JSONL content
    content = history_file.read_text()
    lines = [line for line in content.splitlines() if line.strip()]
    assert lines, "Expected at least one JSONL history line"

    restored_messages = deserialize_summarized_messages_jsonl(content)
    assert len(restored_messages) == len(lines)
    assert all("created_at" in message.additional_kwargs for message in restored_messages)
    assert any(message.type == "human" for message in restored_messages)
    assert any(message.type == "ai" for message in restored_messages)

    # Verify the summary message references the conversation_history path
    summary_message = result["messages"][0]
    assert "conversation_history" in summary_message.content
    assert f"{thread_id}.jsonl" in summary_message.content

    # The persisted JSONL is machine-restorable as plain JSON (no base64/msgpack).
    assert any("base.py" in str(message.content) for message in restored_messages)
