"""Unit tests for `SummarizationMiddleware` with backend offloading."""

import json
from collections.abc import Generator
from datetime import UTC, datetime
from decimal import Decimal
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from deepagents.backends.protocol import BackendProtocol, EditResult, FileDownloadResponse, WriteResult
from deepagents.middleware.summarization import HistoryOffloadError, SummarizationMiddleware
from deepagents.middleware.summarization_history import (
    deserialize_summarized_messages_jsonl,
    serialize_summarized_messages_jsonl,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


def make_conversation_messages(
    num_old: int = 6,
    num_recent: int = 3,
    *,
    include_previous_summary: bool = False,
) -> list:
    """Create a realistic conversation message sequence.

    Args:
        num_old: Number of "old" messages that will be summarized
        num_recent: Number of "recent" messages to preserve
        include_previous_summary: If `True`, start with a summary `HumanMessage`
            containing placeholder text.

    Returns:
        List of messages simulating a conversation
    """
    messages: list[BaseMessage] = []

    if include_previous_summary:
        messages.append(
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nPrevious summary content...",
                additional_kwargs={"lc_source": "summarization"},
                id="summary-msg-0",
            )
        )

    # Add old messages (to be summarized)
    for i in range(num_old):
        if i % 3 == 0:
            messages.append(HumanMessage(content=f"User message {i}", id=f"human-{i}"))
        elif i % 3 == 1:
            messages.append(
                AIMessage(
                    content=f"AI response {i}",
                    id=f"ai-{i}",
                    tool_calls=[{"id": f"tool-call-{i}", "name": "test_tool", "args": {}}],
                )
            )
        else:
            messages.append(
                ToolMessage(
                    content=f"Tool result {i}",
                    tool_call_id=f"tool-call-{i - 1}",
                    id=f"tool-{i}",
                )
            )

    # Add recent messages (to be preserved)
    for i in range(num_recent):
        idx = num_old + i
        messages.append(HumanMessage(content=f"Recent message {idx}", id=f"recent-{idx}"))

    return messages


class MockBackend(BackendProtocol):
    """A mock backend that records read/write calls and can simulate failures."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        error_message: str | None = None,
        existing_content: str | None = None,
        download_raises: bool = False,
        write_raises: bool = False,
    ) -> None:
        """Initialize the mock backend.

        Args:
            should_fail: If `True`, write operations will simulate a failure.
            error_message: The error message to return on failure.
            existing_content: Initialize the backend with existing content for reads.
            download_raises: If `True`, `download_files` will raise an exception.
            write_raises: If `True`, `write`/`edit` will raise an exception.
        """
        self.write_calls: list[tuple[str, str]] = []
        self.edit_calls: list[tuple[str, str, str]] = []
        self.read_calls: list[str] = []
        self.download_files_calls: list[list[str]] = []
        self.should_fail = should_fail
        self.error_message = error_message
        self.existing_content = existing_content
        self.download_raises = download_raises
        self.write_raises = write_raises

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        self.read_calls.append(path)
        if self.existing_content is not None:
            return self.existing_content
        return ""

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(path, offset, limit)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files - returns raw content as bytes."""
        self.download_files_calls.append(paths)
        if self.download_raises:
            msg = "Mock download_files exception"
            raise RuntimeError(msg)
        responses = []
        for path in paths:
            if self.existing_content is not None:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=self.existing_content.encode("utf-8"),
                        error=None,
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",
                    )
                )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        if self.download_raises:
            msg = "Mock adownload_files exception"
            raise RuntimeError(msg)
        return self.download_files(paths)

    def write(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        if self.write_raises:
            msg = "Mock write exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return WriteResult(error=self.error_message or "Mock write failure")
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        if self.write_raises:
            msg = "Mock awrite exception"
            raise RuntimeError(msg)
        return self.write(path, content)

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Edit a file by replacing string occurrences."""
        self.edit_calls.append((path, old_string, new_string))
        if self.write_raises:
            msg = "Mock edit exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return EditResult(error=self.error_message or "Mock edit failure")
        return EditResult(path=path, occurrences=1)

    async def aedit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Async version of edit."""
        if self.write_raises:
            msg = "Mock aedit exception"
            raise RuntimeError(msg)
        return self.edit(path, old_string, new_string, replace_all)


def make_mock_runtime() -> MagicMock:
    """Create a mock `Runtime`.

    Note: `Runtime` does not have a `config` attribute. Config is accessed
    via `get_config()` from langgraph's contextvar. Use `mock_get_config()`
    to control thread_id in tests.
    """
    runtime = MagicMock()
    runtime.context = {}
    runtime.stream_writer = MagicMock()
    runtime.store = None
    # Explicitly don't set runtime.config - it doesn't exist on real Runtime
    del runtime.config
    return runtime


@contextmanager
def mock_get_config(thread_id: str | None = "test-thread-123") -> Generator[None, None, None]:
    """Context manager to mock `get_config()` with a specific `thread_id`.

    Args:
        thread_id: The `thread_id` to return, or `None` to simulate missing config.

    Yields:
        `None` - use as a context manager around test code.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id is not None else {"configurable": {}}

    with patch("deepagents.middleware.summarization.get_config", return_value=config):
        yield


def make_mock_model(summary_response: str = "This is a test summary.") -> MagicMock:
    """Create a mock LLM model for summarization.

    Args:
        summary_response: The text to return as the summary for testing purposes.
    """
    model = MagicMock()
    model.invoke.return_value = MagicMock(text=summary_response)
    model._llm_type = "test-model"
    model.profile = {"max_input_tokens": 100000}
    model._get_ls_params.return_value = {"ls_provider": "test"}
    return model


@pytest.mark.asyncio
async def test_requested_summary_forces_real_compaction_below_trigger() -> None:
    backend = MockBackend()
    model = make_mock_model()
    model.ainvoke = AsyncMock(return_value=MagicMock(text="Requested real summary"))
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 2),
    )
    messages = make_conversation_messages(num_old=6, num_recent=2)

    result = await middleware.apply_requested_summary(
        "requested-thread",
        messages,
    )

    assert result.summarized is True
    assert result.history_path == "/conversation_history/requested-thread.jsonl"
    assert result.replacement_messages is not None
    assert result.replacement_messages[0].additional_kwargs["lc_source"] == "summarization"
    assert "Requested real summary" in result.replacement_messages[0].content
    assert result.replacement_messages[-2:] == messages[-2:]
    assert backend.write_calls[0][0] == result.history_path
    assert model.ainvoke.await_args.kwargs["config"] == {
        "configurable": {"thread_id": "requested-thread"},
        "metadata": {"lc_source": "summarization"},
    }


@pytest.mark.asyncio
async def test_requested_summary_below_cutoff_reports_no_change() -> None:
    middleware = SummarizationMiddleware(
        model=make_mock_model(),
        backend=MockBackend(),
        trigger=("messages", 100),
        keep=("messages", 20),
    )

    result = await middleware.apply_requested_summary(
        "requested-thread",
        [HumanMessage(content="Only message", id="h1")],
    )

    assert result.summarized is False
    assert result.replacement_messages is None
    assert result.history_path is None


@pytest.mark.asyncio
async def test_requested_summary_uses_custom_prompt_without_mutating_default() -> None:
    backend = MockBackend()
    model = make_mock_model()
    model.ainvoke = AsyncMock(return_value=MagicMock(text="Focused summary"))
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 2),
    )
    default_prompt = middleware.summary_prompt

    await middleware.apply_requested_summary(
        "custom-thread",
        make_conversation_messages(num_old=6, num_recent=2),
        "Focus on decisions",
    )

    invoked_prompt = model.ainvoke.await_args.args[0]
    assert invoked_prompt.startswith("Focus on decisions\n\n")
    assert "User message 0" in invoked_prompt
    assert middleware.summary_prompt == default_prompt


@pytest.mark.asyncio
async def test_requested_summary_rejects_factory_backend() -> None:
    middleware = SummarizationMiddleware(
        model=make_mock_model(),
        backend=lambda runtime: MockBackend(),
        trigger=("messages", 100),
        keep=("messages", 2),
    )

    with pytest.raises(RuntimeError, match="concrete backend"):
        await middleware.apply_requested_summary(
            "explicit-thread",
            make_conversation_messages(num_old=6, num_recent=2),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    ["Broken {", "Unknown {topic}", "{messages} and {messages}"],
)
async def test_requested_summary_rejects_invalid_custom_prompt(prompt: str) -> None:
    middleware = SummarizationMiddleware(
        model=make_mock_model(),
        backend=MockBackend(),
        trigger=("messages", 100),
        keep=("messages", 2),
    )

    with pytest.raises((ValueError, KeyError)):
        await middleware.apply_requested_summary(
            "prompt-thread",
            make_conversation_messages(num_old=6, num_recent=2),
            prompt,
        )


@pytest.mark.asyncio
async def test_automatic_model_failure_soft_errors_after_offload() -> None:
    messages = make_conversation_messages(num_old=6, num_recent=2)
    backend = MockBackend()
    model = make_mock_model()
    model.ainvoke = AsyncMock(side_effect=RuntimeError("model down"))
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )

    with mock_get_config("auto-fail-thread"):
        result = await middleware.abefore_model(
            cast("AgentState[Any]", {"messages": messages}),
            make_mock_runtime(),
        )

    assert result is not None
    replacement = result["messages"][1:]
    assert "Error generating summary: model down" in replacement[0].content
    assert backend.write_calls, "offload still happens before model failure"


@pytest.mark.asyncio
async def test_requested_model_failure_hard_fails_after_offload() -> None:
    messages = make_conversation_messages(num_old=6, num_recent=2)
    backend = MockBackend()
    model = make_mock_model()
    model.ainvoke = AsyncMock(side_effect=RuntimeError("model down"))
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 2),
    )

    with pytest.raises(RuntimeError, match="model down"):
        await middleware.apply_requested_summary("requested-fail-thread", messages)
    assert backend.write_calls, "offload still happens before model failure"


@pytest.mark.asyncio
async def test_requested_and_automatic_async_compaction_are_equivalent() -> None:
    messages = make_conversation_messages(num_old=6, num_recent=2)
    automatic_backend = MockBackend()
    requested_backend = MockBackend()
    automatic_model = make_mock_model()
    requested_model = make_mock_model()
    automatic_model.ainvoke = AsyncMock(
        return_value=MagicMock(text="Equivalent summary")
    )
    requested_model.ainvoke = AsyncMock(
        return_value=MagicMock(text="Equivalent summary")
    )
    automatic = SummarizationMiddleware(
        model=automatic_model,
        backend=automatic_backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )
    requested = SummarizationMiddleware(
        model=requested_model,
        backend=requested_backend,
        trigger=("messages", 100),
        keep=("messages", 2),
    )

    with mock_get_config("equivalent-thread"):
        auto_result = await automatic.abefore_model(
            cast("AgentState[Any]", {"messages": messages}),
            make_mock_runtime(),
        )
    requested_result = await requested.apply_requested_summary(
        "equivalent-thread",
        messages,
    )

    assert auto_result is not None
    auto_messages = auto_result["messages"][1:]
    requested_messages = requested_result.replacement_messages
    assert requested_messages is not None
    assert [message.content for message in auto_messages] == [
        message.content for message in requested_messages
    ]
    assert automatic_backend.write_calls[0][0] == requested_backend.write_calls[0][0]
    assert deserialize_summarized_messages_jsonl(
        automatic_backend.write_calls[0][1]
    ) == deserialize_summarized_messages_jsonl(
        requested_backend.write_calls[0][1]
    )
    assert automatic_model.ainvoke.await_args.args == requested_model.ainvoke.await_args.args
    assert automatic_model.ainvoke.await_args.kwargs["config"] == {
        "metadata": {"lc_source": "summarization"}
    }
    assert requested_model.ainvoke.await_args.kwargs["config"] == {
        "configurable": {"thread_id": "equivalent-thread"},
        "metadata": {"lc_source": "summarization"},
    }


# -----------------------------------------------------------------------------


class TestSummarizationHistoryJsonl:
    def test_jsonl_has_one_object_per_message_and_restores_losslessly(self) -> None:
        messages = [
            HumanMessage(
                content="hello",
                id="h1",
                additional_kwargs={"created_at": "2026-05-29T10:00:00.000001+00:00"},
            ),
            AIMessage(
                content="using a tool",
                id="a1",
                tool_calls=[{"id": "call-1", "name": "lookup", "args": {"query": "x"}}],
                additional_kwargs={
                    "created_at": "2026-05-29T10:00:00.000002+00:00",
                    "tool_call_created_at": {"call-1": "2026-05-29T10:00:00.000002+00:00"},
                },
                response_metadata={"provider": "test"},
            ),
            ToolMessage(
                content="tool result",
                tool_call_id="call-1",
                id="t1",
                additional_kwargs={"created_at": "2026-05-29T10:00:00.000003+00:00"},
            ),
        ]

        content = serialize_summarized_messages_jsonl(messages, summarized_at="2026-05-29T10:01:00.000000+00:00")
        lines = content.splitlines()

        assert len(lines) == len(messages)
        entries = [json.loads(line) for line in lines]
        assert [entry["message_index"] for entry in entries] == [0, 1, 2]
        assert all(entry["schema_version"] == 4 for entry in entries)
        # message is plain JSON, not msgpack/base64
        assert all(isinstance(entry["message"], dict) for entry in entries)
        assert all("type" in entry["message"] and "data" in entry["message"] for entry in entries)
        assert all(isinstance(entry["message"]["data"]["content"], str) for entry in entries)

        restored = deserialize_summarized_messages_jsonl(content)
        assert [message.type for message in restored] == ["human", "ai", "tool"]
        assert restored[0].id == "h1"
        assert restored[1].id == "a1"
        assert restored[1].tool_calls == messages[1].tool_calls
        assert restored[1].additional_kwargs["tool_call_created_at"] == {"call-1": "2026-05-29T10:00:00.000002+00:00"}
        assert restored[2].tool_call_id == "call-1"

    def test_jsonplus_payload_preserves_non_json_metadata(self) -> None:
        created = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
        message = AIMessage(
            content="metadata",
            id="a-non-json",
            additional_kwargs={"created_at": "2026-05-29T10:00:00.000001+00:00", "dt": created},
            response_metadata={"cost": Decimal("1.25")},
        )

        content = serialize_summarized_messages_jsonl([message])
        restored = deserialize_summarized_messages_jsonl(content)

        # Plain JSON serialization converts datetime to ISO string and Decimal to string
        # Note: str(datetime) uses space, not T separator: "2026-05-29 10:00:00+00:00"
        assert restored[0].additional_kwargs["dt"] == "2026-05-29 10:00:00+00:00"
        assert restored[0].response_metadata["cost"] == "1.25"
        # created_at was already a string, preserved as-is
        assert restored[0].additional_kwargs["created_at"] == "2026-05-29T10:00:00.000001+00:00"


class TestSummarizationMiddlewareInit:
    """Tests for middleware initialization."""

    def test_init_with_backend(self) -> None:
        """Test initialization with a backend instance."""
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert middleware._backend is backend
        assert middleware._history_path_prefix == "/conversation_history"

    def test_init_with_backend_factory(self) -> None:
        """Test initialization with a backend factory function."""
        backend = MockBackend()
        factory = lambda _rt: backend  # noqa: E731

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert callable(middleware._backend)


class TestOffloadingBasic:
    """Tests for basic offloading behavior."""

    def test_offload_writes_to_backend(self) -> None:
        """Test that summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            result = middleware.before_model(state, runtime)

        # Should have triggered summarization
        assert result is not None
        assert len(backend.write_calls) == 1

        path, content = backend.write_calls[0]
        assert path == "/conversation_history/test-thread-123.jsonl"

        lines = content.splitlines()
        assert len(lines) == 6
        restored = deserialize_summarized_messages_jsonl(content)
        assert [message.id for message in restored] == ["human-0", "ai-1", "tool-2", "human-3", "ai-4", "tool-5"]

    def test_offload_appends_to_existing_content(self) -> None:
        """Test that second summarization appends JSONL lines to existing file."""
        existing = serialize_summarized_messages_jsonl(
            [HumanMessage(content="Previous message", id="prev-human")],
            summarized_at="2024-01-01T00:00:00.000000+00:00",
        )
        backend = MockBackend(existing_content=existing)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

        assert len(backend.edit_calls) == 1
        _, old_string, new_string = backend.edit_calls[0]

        # old_string should be the existing content
        assert old_string == existing

        restored = deserialize_summarized_messages_jsonl(new_string)
        assert restored[0].id == "prev-human"
        assert len(restored) == 7
        assert [message.id for message in restored[1:]] == ["human-0", "ai-1", "tool-2", "human-3", "ai-4", "tool-5"]
        assert len(new_string.splitlines()) == 7

    def test_typical_tool_heavy_conversation(self) -> None:
        """Test with a realistic tool-heavy conversation pattern.

        Simulates:

        ```txt
        HumanMessage -> AIMessage(tool_calls) -> ToolMessage -> ToolMessage ->
        ToolMessage -> AIMessage -> HumanMessage -> AIMessage -> ToolMessage (trigger)
        ```
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 8),
            keep=("messages", 3),
        )

        messages = [
            HumanMessage(content="Search for Python tutorials", id="h1"),
            AIMessage(
                content="I'll search for Python tutorials.",
                id="a1",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "python"}}],
            ),
            ToolMessage(content="Result 1: Python basics", tool_call_id="tc1", id="t1"),
            ToolMessage(content="Result 2: Advanced Python", tool_call_id="tc1", id="t2"),
            ToolMessage(content="Result 3: Python projects", tool_call_id="tc1", id="t3"),
            AIMessage(content="Here are some Python tutorials I found...", id="a2"),
            HumanMessage(content="Show me the first one", id="h2"),
            AIMessage(
                content="Let me get that for you.",
                id="a3",
                tool_calls=[{"id": "tc2", "name": "fetch", "args": {"url": "..."}}],
            ),
            ToolMessage(content="Tutorial content...", tool_call_id="tc2", id="t4"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        assert result is not None
        assert len(backend.write_calls) == 1

        _, content = backend.write_calls[0]

        restored = deserialize_summarized_messages_jsonl(content)
        assert restored[0].content == "Search for Python tutorials"
        assert restored[1].tool_calls[0]["id"] == "tc1"

    def test_second_summarization_after_first(self) -> None:
        """Test a second summarization event after an initial one.

        Ensures the chained summarization correctly handles the existing summary message.
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization
        messages = [
            # Previous summary from first summarization
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        assert result is not None

        _, content = backend.write_calls[0]

        # The previous summary message (marked with lc_source: "summarization") should NOT
        # be offloaded—it's a synthetic message, and the original messages it summarized
        # are already stored in the backend file from the first summarization
        restored = deserialize_summarized_messages_jsonl(content)
        assert all(message.id != "prev-summary" for message in restored), "Previous summary should be filtered from offload"
        # But the new conversation messages should be offloaded
        assert any(message.content == "New question 1" for message in restored)

    def test_filters_previous_summary_messages(self) -> None:
        """Test that previous summary `HumanMessage` objects are NOT included in offload.

        When a second summarization occurs, the previous summary message should be
        filtered out since we already have the original messages stored.
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # Create messages that include a previous summary
        messages = make_conversation_messages(
            num_old=6,
            num_recent=2,
            include_previous_summary=True,
        )
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

        _, content = backend.write_calls[0]

        # Check that the offloaded content doesn't include "Previous summary content"
        # (which is the content of the summary message added by include_previous_summary)
        assert "Previous summary content" not in content, "Previous summary message should be filtered from offload"


class TestSummaryMessageFormat:
    """Tests for the summary message format with file path reference."""

    def test_summary_includes_file_path(self) -> None:
        """Test that summary message includes the file path reference."""
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Test summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            result = middleware.before_model(state, runtime)
        assert result is not None

        # Get the summary message (second in list, after RemoveMessage)
        summary_msg = result["messages"][1]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/test-thread.jsonl" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Test summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

    def test_summary_has_lc_source_marker(self) -> None:
        """Test that summary message has `lc_source=summarization` marker."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        assert result is not None
        summary_msg = result["messages"][1]

        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"

    def test_summarization_aborts_on_backend_failure(self) -> None:
        """Test that summarization raises HistoryOffloadError when backend write fails.

        Fail-loud contract per DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md:
        continuing summarization would silently discard the evicted history,
        so we raise instead and let the runtime catch-all transition the
        agent to ``AgentStatus.ERROR``.
        """
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model(summary_response="Unused summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError):
            middleware.before_model(state, runtime)

    def test_summary_includes_file_path_after_second_summarization(self) -> None:
        """Test that summary message includes file path reference after multiple summarizations.

        This ensures the path reference is present even when a previous summary message
        exists in the conversation (i.e., chained summarization).
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization - starts with a summary message
        messages = [
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary that trigger second summarization
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="multi-summarize-thread"):
            result = middleware.before_model(state, runtime)

        assert result is not None

        # The summary message should be at index 1 (after RemoveMessage)
        summary_msg = result["messages"][1]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/multi-summarize-thread.jsonl" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Second summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

        # Should have lc_source marker
        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"


class TestNoSummarizationTriggered:
    """Tests for when summarization threshold is not met."""

    def test_no_offload_when_below_threshold(self) -> None:
        """Test that no offload occurs when message count is below trigger."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        # Should return None (no summarization)
        assert result is None

        # No writes should have occurred
        assert len(backend.write_calls) == 0


class TestBackendFailureHandling:
    """Tests for backend failure handling — summarization raises HistoryOffloadError.

    See DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md.
    """

    def test_summarization_aborts_on_write_failure(self) -> None:
        """Backend returning ``WriteResult(error=...)`` raises HistoryOffloadError with actionable message."""
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Storage unavailable"):
            middleware.before_model(state, runtime)

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Backend ``write`` raising bubbles up as HistoryOffloadError with __cause__ chain."""
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Mock write exception") as exc:
            middleware.before_model(state, runtime)
        # The original backend exception is chained as __cause__.
        assert isinstance(exc.value.__cause__, RuntimeError)


class TestThreadIdExtraction:
    """Tests for thread ID extraction via `get_config()`."""

    def test_thread_id_from_config(self) -> None:
        """Test that `thread_id` is correctly extracted from `get_config()`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="custom-thread-456"):
            middleware.before_model(state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/conversation_history/custom-thread-456.jsonl"

    def test_fallback_thread_id_when_missing(self) -> None:
        """Test that a fallback ID is generated when `thread_id` is not in config."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id=None):
            middleware.before_model(state, runtime)

        path, _ = backend.write_calls[0]

        # Should have a generated session ID in the path
        assert "session_" in path
        assert path.endswith(".jsonl")


class TestAsyncBehavior:
    """Tests for async version of `before_model`."""

    @pytest.mark.anyio
    async def test_async_offload_writes_to_backend(self) -> None:
        """Test that async summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()
        # Mock the async create summary
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_aborts_on_failure(self) -> None:
        """Async summarization raises HistoryOffloadError on backend failure."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError):
            await middleware.abefore_model(state, runtime)


class TestBackendFactoryInvocation:
    """Tests for backend factory invocation during summarization."""

    def test_backend_factory_invoked_during_summarization(self) -> None:
        """Test that backend factory is called with `ToolRuntime` during summarization."""
        backend = MockBackend()
        factory_called_with: list = []

        def factory(tool_runtime: object) -> MockBackend:
            factory_called_with.append(tool_runtime)
            return backend

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

        # Factory should have been called once
        assert len(factory_called_with) == 1
        # Backend should have received write call
        assert len(backend.write_calls) == 1


class TestCustomHistoryPathPrefix:
    """Tests for custom `history_path_prefix` configuration."""

    def test_custom_history_path_prefix(self) -> None:
        """Test that custom `history_path_prefix` is used in file paths."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            history_path_prefix="/custom/path",
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            middleware.before_model(state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/custom/path/test-thread.jsonl"


class TestJsonlFormatting:
    """Tests for JSONL message formatting."""

    def test_jsonl_format_restores_message_content_and_timestamps(self) -> None:
        """Test that JSONL format restores message content and timestamps."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)
        assert result is not None

        _, content = backend.write_calls[0]

        restored = deserialize_summarized_messages_jsonl(content)
        assert restored[0].content == "User message 0"
        assert restored[0].additional_kwargs["created_at"].endswith("+00:00")
        assert len(content.splitlines()) == len(restored)


class TestDownloadFilesException:
    """Tests for exception handling when download_files raises."""

    def test_summarization_continues_on_download_files_exception(self) -> None:
        """Test that summarization continues when download_files raises an exception."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result = middleware.before_model(state, runtime)

        assert result is not None
        assert "messages" in result
        # download_files was called (and raised)
        assert len(backend.download_files_calls) == 1
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_summarization_continues_on_download_files_exception(self) -> None:
        """Test that async summarization continues when adownload_files raises."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert "messages" in result
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1


class TestWriteEditException:
    """Tests for exception handling when `write`/`edit` raises — summarization raises HistoryOffloadError.

    See DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md.
    """

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Exception from ``write`` bubbles up wrapped in HistoryOffloadError."""
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Mock write exception"):
            middleware.before_model(state, runtime)

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_write_exception(self) -> None:
        """Async: exception from ``awrite`` bubbles up wrapped in HistoryOffloadError."""
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Mock awrite exception"):
            await middleware.abefore_model(state, runtime)

    def test_summarization_aborts_on_edit_exception(self) -> None:
        """Exception from ``edit`` (with existing content) bubbles up wrapped in HistoryOffloadError."""
        existing = serialize_summarized_messages_jsonl([HumanMessage(content="Previous message", id="prev-human")])
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Mock edit exception"):
            middleware.before_model(state, runtime)

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_edit_exception(self) -> None:
        """Async: exception from ``aedit`` (with existing content) bubbles up wrapped."""
        existing = serialize_summarized_messages_jsonl([HumanMessage(content="Previous message", id="prev-human")])
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.raises(HistoryOffloadError, match="Mock aedit exception"):
            await middleware.abefore_model(state, runtime)


class TestCutoffIndexEdgeCases:
    """Tests for edge cases where `cutoff_index <= 0`."""

    def test_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that no summarization occurs when `cutoff_index` is `0`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        # Should return None because cutoff_index would be 0 or negative
        assert result is None
        # No writes should occur
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_not_triggered(self) -> None:
        """Test that async `abefore_model` returns `None` when summarization not triggered."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = await middleware.abefore_model(state, runtime)

        # Should return None (no summarization)
        assert result is None
        # No writes should have occurred
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that async `abefore_model` returns `None` when `cutoff_index <= 0`."""
        backend = MockBackend()
        mock_model = make_mock_model()
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result = await middleware.abefore_model(state, runtime)

        # Should return None because cutoff_index would be 0 or negative
        assert result is None
        # No writes should occur
        assert len(backend.write_calls) == 0


# -----------------------------------------------------------------------------
# Argument truncation tests
# -----------------------------------------------------------------------------


def test_no_truncation_when_trigger_is_none() -> None:
    """Test that no truncation occurs when truncate_args_settings is None."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings=None,  # Truncation disabled
    )

    # Create messages with large tool calls
    large_content = "x" * 200
    messages = [
        HumanMessage(content="Write a file", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    # Should return None (no truncation, no summarization)
    assert result is None


def test_truncate_old_write_file_tool_call() -> None:
    """Test that old write_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message with write_file tool call (will be cleaned)
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    # Check that the old tool call was cleaned
    first_ai_msg = cleaned_messages[0]
    assert isinstance(first_ai_msg, AIMessage)
    assert len(first_ai_msg.tool_calls) == 1
    assert first_ai_msg.tool_calls[0]["name"] == "write_file"
    # Content should be first 20 chars + truncation text
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_old_edit_file_tool_call() -> None:
    """Test that old edit_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_old_string = "a" * 100
    large_new_string = "b" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "edit_file",
                    "args": {
                        "file_path": "/test.py",
                        "old_string": large_old_string,
                        "new_string": large_new_string,
                    },
                }
            ],
        ),
        ToolMessage(content="File edited", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["name"] == "edit_file"
    assert first_ai_msg.tool_calls[0]["args"]["old_string"] == "a" * 20 + "...(argument truncated)"
    assert first_ai_msg.tool_calls[0]["args"]["new_string"] == "b" * 20 + "...(argument truncated)"


def test_truncate_ignores_other_tool_calls() -> None:
    """Test that tool calls other than write_file and edit_file are not affected."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    # Should return None since read_file is not cleaned
    assert result is None


def test_truncate_respects_recent_messages() -> None:
    """Test that recent messages are not cleaned."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 4),  # Keep last 4 messages
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a1"),
        # Recent message with write_file (should NOT be cleaned - it's in the last 4)
        AIMessage(
            content="",
            id="a2",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    # No truncation should happen since the tool call is in the keep window (last 4 messages)
    assert result is None


def test_truncate_with_token_keep_policy() -> None:
    """Test truncation with token-based keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()

    # Custom token counter that returns predictable counts
    def simple_token_counter(msgs: list) -> int:
        return len(msgs) * 100  # 100 tokens per message

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("tokens", 250),  # Keep ~2-3 messages
            "max_length": 100,
        },
        token_counter=simple_token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    # First message should be cleaned since it's outside the token window
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_with_fraction_trigger_and_keep() -> None:
    """Test truncation with fraction-based trigger and keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.profile = {"max_input_tokens": 1000}

    # Custom token counter: 200 tokens per message
    def token_counter(msgs: list) -> int:
        return len(msgs) * 200

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold for summarization
        truncate_args_settings={
            "trigger": ("fraction", 0.5),  # Trigger at 50% of 1000 = 500 tokens
            "keep": ("fraction", 0.2),  # Keep 20% of 1000 = 200 tokens (~1 message)
            "max_length": 100,
        },
        token_counter=token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Message 1", id="h1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    # Should trigger truncation: 3 messages * 200 = 600 tokens > 500 threshold
    # Should keep only ~200 tokens (1 message) from the end
    # So first 2 messages should be in truncation zone
    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_before_summarization() -> None:
    """Test that truncation happens before summarization."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Test summary")

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 10),  # Trigger summarization
        keep=("messages", 2),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 3),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message that will be cleaned and summarized
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ] + [HumanMessage(content=f"Message {i}", id=f"h{i}") for i in range(10)]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    with mock_get_config(thread_id="test-thread"):
        result = middleware.before_model(state, runtime)

    assert result is not None

    # Should have triggered both truncation and summarization
    # Backend should have received a write call for offloading
    assert len(backend.write_calls) == 1

    # Result should contain summary message (skip RemoveMessage at index 0)
    new_messages = result["messages"][1:]
    assert any("summary" in str(msg.content).lower() for msg in new_messages)


def test_truncate_without_summarization() -> None:
    """Test that truncation can happen independently of summarization."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None

    # No backend write (no summarization)
    assert len(backend.write_calls) == 0

    # But truncation should have happened
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_preserves_small_arguments() -> None:
    """Test that small arguments are not truncated even in old messages."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    small_content = "short"

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": small_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    # No modification should happen since content is small
    assert result is None


def test_truncate_mixed_tool_calls() -> None:
    """Test that only write_file and edit_file are cleaned in a message with multiple tool calls."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt"},
                },
                {
                    "id": "tc2",
                    "name": "write_file",
                    "args": {"file_path": "/output.txt", "content": large_content},
                },
                {
                    "id": "tc3",
                    "name": "shell",
                    "args": {"command": "ls -la"},
                },
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        ToolMessage(content="File written", tool_call_id="tc2", id="t2"),
        ToolMessage(content="Output", tool_call_id="tc3", id="t3"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    first_ai_msg = cleaned_messages[0]
    assert len(first_ai_msg.tool_calls) == 3

    # read_file should be unchanged
    assert first_ai_msg.tool_calls[0]["name"] == "read_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.txt"

    # write_file should be cleaned
    assert first_ai_msg.tool_calls[1]["name"] == "write_file"
    assert first_ai_msg.tool_calls[1]["args"]["content"] == "x" * 20 + "...(argument truncated)"

    # shell should be unchanged
    assert first_ai_msg.tool_calls[2]["name"] == "shell"
    assert first_ai_msg.tool_calls[2]["args"]["command"] == "ls -la"


def test_truncate_custom_truncation_text() -> None:
    """Test that custom truncation text is used."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
            "truncation_text": "[TRUNCATED]",
        },
    )

    large_content = "y" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = middleware.before_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "y" * 20 + "[TRUNCATED]"


@pytest.mark.anyio
async def test_truncate_async_works() -> None:
    """Test that async argument truncation works correctly."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result = await middleware.abefore_model(state, runtime)

    assert result is not None
    # Skip RemoveMessage at index 0, actual messages start at index 1
    cleaned_messages = result["messages"][1:]

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"
