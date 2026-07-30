"""Summarization middleware for offloading conversation history.

Persists conversation history to a backend prior to summarization, enabling retrieval of
full context if needed later by an agent.

## Usage

```python
from deepagents import create_deep_agent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/data")

middleware = SummarizationMiddleware(
    model="gpt-4o-mini",
    backend=backend,
    trigger=("fraction", 0.85),
    keep=("fraction", 0.10),
)

agent = create_deep_agent(middleware=[middleware])
```

## Storage

Offloaded messages are stored as JSONL at `/conversation_history/{thread_id}.jsonl`.

Each summarization event appends one JSON object per evicted message to this file,
creating a lossless structured log of summarized history.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from string import Formatter
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.summarization import (
    _DEFAULT_FALLBACK_MESSAGE_COUNT,
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    DEFAULT_SUMMARY_PROMPT,
    ContextSize,
    SummarizationMiddleware as BaseSummarizationMiddleware,
    TokenCounter,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string, trim_messages
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import TypedDict, override

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.message_timestamps import MonotonicUtcClock, ensure_message_created_at, ensure_messages_created_at
from deepagents.middleware.summarization_history import serialize_summarized_messages_jsonl

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState
    from langchain.chat_models import BaseChatModel
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES

logger = logging.getLogger(__name__)


class HistoryOffloadError(RuntimeError):
    """Raised when the pre-summarization history offload cannot be persisted.

    The summarization middleware evicts old messages from the active thread
    and writes them to a JSONL history file BEFORE producing the summary
    message. If that write fails (backend returns an error, exception during
    write, permission denied, disk full, etc.), continuing summarization
    would silently discard the history that is about to be dropped from the
    active thread — a data-loss failure mode we explicitly forbid.

    Fatal — the agent runtime's fatal-exception catch-all matches the
    class name ``HistoryOffloadError`` in
    :data:`deep_worker_cli.exceptions.FATAL_EXCEPTION_TYPES` and transitions
    the agent into ``AgentStatus.ERROR`` with the error message.

    See ``deep_worker/docs/DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md``.
    """


class TruncateArgsSettings(TypedDict, total=False):
    """Settings for truncating large tool arguments in old messages.

    Attributes:
        trigger: Threshold to trigger argument truncation. If None, truncation is disabled.
        keep: Context retention policy for message truncation (defaults to last 20 messages).
        max_length: Maximum character length for tool arguments before truncation (defaults to 2000).
        truncation_text: Text to replace truncated arguments with (defaults to "...(argument truncated)").
    """

    trigger: ContextSize | None
    keep: ContextSize
    max_length: int
    truncation_text: str


class SummarizationDefaults(TypedDict):
    """Default settings computed from model profile."""

    trigger: ContextSize
    keep: ContextSize
    truncate_args_settings: TruncateArgsSettings


@dataclass(frozen=True, slots=True)
class RequestedSummaryResult:
    summarized: bool
    replacement_messages: list[AnyMessage] | None
    history_path: str | None


def _compute_summarization_defaults(model: BaseChatModel) -> SummarizationDefaults:
    """Compute default summarization settings based on model profile.

    This is an internal helper function used by middleware implementations.

    Args:
        model: A resolved chat model instance.

    Returns:
        Default settings for trigger, keep, and truncate_args_settings.
        If the model has a profile with max_input_tokens, uses fraction-based
        settings. Otherwise, uses fixed token/message counts.
    """
    has_profile = (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    )

    if has_profile:
        return {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }
    return {
        "trigger": ("tokens", 170000),
        "keep": ("messages", 6),
        "truncate_args_settings": {
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        },
    }


class SummarizationMiddleware(BaseSummarizationMiddleware):
    """Summarization middleware with backend for conversation history offloading."""

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        history_path_prefix: str = "/conversation_history",
        truncate_args_settings: TruncateArgsSettings | None = None,
        on_summarized: Callable[[str], None] | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware with backend support.

        Args:
            model: The language model to use for generating summaries.
            backend: Backend instance or factory for persisting conversation history.
            trigger: Threshold(s) that trigger summarization.
            keep: Context retention policy after summarization.

                Defaults to keeping last 20 messages.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Max tokens to include when generating summary.

                Defaults to 4000.
            truncate_args_settings: Settings for truncating large tool arguments in old messages.

                Provide a [`TruncateArgsSettings`][deepagents.middleware.summarization.TruncateArgsSettings]
                dictionary to configure when and how to truncate tool arguments. If `None`,
                argument truncation is disabled.

                !!! example

                    ```python
                    # Truncate when 50 messages is reached, ignoring the last 20 messages
                    {"trigger": ("messages", 50), "keep": ("messages", 20), "max_length": 2000, "truncation_text": "...(truncated)"}

                    # Truncate when 50% of context window reached, ignoring messages in last 10% of window
                    {"trigger": ("fraction", 0.5), "keep": ("fraction", 0.1), "max_length": 2000, "truncation_text": "...(truncated)"}
            history_path_prefix: Path prefix for storing conversation history.

        Example:
            ```python
            from deepagents.middleware.summarization import SummarizationMiddleware
            from deepagents.backends import StateBackend

            middleware = SummarizationMiddleware(
                model="gpt-4o-mini",
                backend=lambda tool_runtime: StateBackend(tool_runtime),
                trigger=("tokens", 100000),
                keep=("messages", 20),
            )
            ```
        """
        super().__init__(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )
        self._backend = backend
        self._history_path_prefix = history_path_prefix
        self._on_summarized = on_summarized

        # Parse truncate_args_settings
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep: ContextSize = ("messages", 20)
            self._max_arg_length = 2000
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 2000)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

        # ═══════════════════════════════════════════════════════════════
        # Token-count cache (DESIGN_CONTEXT_DELTA_CACHE.md §4.6).
        #
        # before_model() runs on every agent turn and needs the total
        # token count to decide whether to summarize. ``token_counter``
        # over all messages is O(messages × tokens) — that's the bulk
        # of the per-turn overhead on a long conversation.
        #
        # Invariant used: if the messages-list is the SAME list (by
        # identity of message ids, in the same order, with the same
        # length), the token count cannot have changed. The cache
        # remembers the id-sequence signature of the last list it
        # counted and returns the cached total on an exact match.
        #
        # We intentionally do NOT try to add tokens of appended
        # messages incrementally: user-supplied token_counter callables
        # may not be strictly additive (they may account for
        # separators, prefixes, role tokens), so only exact-list
        # equality is safe to memoize.
        # ═══════════════════════════════════════════════════════════════
        self._cached_token_signature: tuple[Any, ...] | None = None
        self._cached_token_total: int = 0

    def _cached_total_tokens(self, messages: list[Any]) -> int:
        """Return total tokens for ``messages``, caching the last result.

        See the comment above ``_cached_token_signature``.
        """
        signature = tuple(getattr(m, "id", id(m)) for m in messages)
        if signature == self._cached_token_signature:
            return self._cached_token_total
        total = int(self.token_counter(messages))
        self._cached_token_signature = signature
        self._cached_token_total = total
        return total

    @override
    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits, accounting for prompt size.

        This override fixes a bug where the summary prompt itself could exceed context limits.
        The base implementation trims messages to `trim_tokens_to_summarize` tokens but then
        adds the summary_prompt on top, potentially exceeding the model's context limit.

        This implementation:
        1. Calculates the token count of the summary_prompt
        2. Subtracts that from the trim limit to get an adjusted limit
        3. Uses the adjusted limit when trimming messages

        Args:
            messages: Messages to trim for summary generation.

        Returns:
            Trimmed messages that, when combined with summary_prompt, fit within context limits.
        """
        try:
            if self.trim_tokens_to_summarize is None:
                return messages

            # Calculate prompt token count using a dummy HumanMessage
            # (token_counter expects a list of messages)
            prompt_tokens = self.token_counter([HumanMessage(content=self.summary_prompt)])

            # Adjust limit to account for prompt, with minimum safety buffer of 1000 tokens
            adjusted_limit = max(1000, self.trim_tokens_to_summarize - prompt_tokens)

            logger.debug(
                f"Trimming messages for summary: prompt_tokens={prompt_tokens}, "
                f"trim_limit={self.trim_tokens_to_summarize}, adjusted_limit={adjusted_limit}"
            )

            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=adjusted_limit,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception as e:
            logger.warning(f"Error trimming messages for summary: {e}, using fallback")
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.

        Returns:
            Resolved backend instance.
        """
        # `self._backend` is BackendProtocol | BackendFactory. ty can't
        # narrow `callable(...)`, so use isinstance — factories are plain
        # Callables, not BackendProtocol instances.
        if isinstance(self._backend, BackendProtocol):
            return self._backend

        # Because we're using `before_model`, which doesn't receive `config` as a
        # parameter, we access it via `runtime.config` instead.
        # Cast is safe: empty dict `{}` is a valid `RunnableConfig` (all fields are
        # optional in TypedDict).
        config = cast("RunnableConfig", getattr(runtime, "config", {}))

        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            stream_writer=runtime.stream_writer,
            store=runtime.store,
            config=config,
            tool_call_id=None,
        )
        return self._backend(cast(Any, tool_runtime))

    def _get_thread_id(self) -> str:
        """Extract `thread_id` from langgraph config.

        Uses `get_config()` to access the `RunnableConfig` from langgraph's
        `contextvar`. Falls back to a generated session ID if not available.

        Returns:
            Thread ID string from config, or a generated session ID
                (e.g., `'session_a1b2c3d4'`) if not in a runnable context.
        """
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        except RuntimeError:
            # Not in a runnable context
            pass

        # Fallback: generate session ID
        generated_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.debug("No thread_id found, using generated session ID: %s", generated_id)
        return generated_id

    def _get_history_path(self, thread_id: str | None = None) -> str:
        """Generate path for storing conversation history.

        Returns a single file per thread that gets appended to over time.

        Args:
            thread_id: Explicit thread id to target. When ``None`` (the
                default, used by ``before_model`` / ``abefore_model``), the
                thread id is resolved from langgraph's ``get_config()``
                contextvar via :meth:`_get_thread_id`. An explicit value is
                passed by :meth:`apply_pseudo_summary`, which runs OUTSIDE a
                graph run where the contextvar is unavailable (see
                ``DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.4 Blocker-1, option B).

        Returns:
            Path string like `'/conversation_history/{thread_id}.jsonl'`
        """
        tid = thread_id if thread_id is not None else self._get_thread_id()
        return f"{self._history_path_prefix}/{tid}.jsonl"

    def _is_summary_message(self, msg: AnyMessage) -> bool:
        """Check if a message is a previous summarization message.

        Summary messages are `HumanMessage` objects with `lc_source='summarization'` in
        `additional_kwargs`. These should be filtered from offloads to avoid redundant
        storage during chained summarization.

        Args:
            msg: Message to check.

        Returns:
            Whether this is a summary `HumanMessage` from a previous summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the previous
        summary `HumanMessage` since the original messages are already stored in the
        backend.

        Args:
            messages: List of messages to filter.

        Returns:
            Messages without previous summary `HumanMessage` objects.
        """
        return [msg for msg in messages if not self._is_summary_message(msg)]

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The generated summary text.
            file_path: Path where conversation history was stored, or `None`.

                Optional since offloading may fail.

        Returns:
            List containing the summary `HumanMessage`.
        """
        if file_path is not None:
            content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        summary_message = HumanMessage(
            content=content,
            additional_kwargs={"lc_source": "summarization"},
        )
        ensure_message_created_at(summary_message, MonotonicUtcClock())
        return [summary_message]

    def _should_truncate_args(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Check if argument truncation should be triggered.

        Args:
            messages: Current message history.
            total_tokens: Total token count of messages.

        Returns:
            True if truncation should occur, False otherwise.
        """
        if self._truncate_args_trigger is None:
            return False

        trigger_type, trigger_value = self._truncate_args_trigger

        if trigger_type == "messages":
            return len(messages) >= trigger_value
        if trigger_type == "tokens":
            return total_tokens >= trigger_value
        if trigger_type == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * trigger_value)
            if threshold <= 0:
                threshold = 1
            return total_tokens >= threshold

        return False

    def _determine_truncate_cutoff_index(self, messages: list[AnyMessage]) -> int:  # noqa: PLR0911
        """Determine the cutoff index for argument truncation based on keep policy.

        Messages at index >= cutoff should be preserved without truncation.
        Messages at index < cutoff can have their tool args truncated.

        Args:
            messages: Current message history.

        Returns:
            Index where truncation cutoff occurs. Messages before this index
            should have args truncated, messages at/after should be preserved.
        """
        keep_type, keep_value = self._truncate_args_keep

        if keep_type == "messages":
            # Keep the most recent N messages
            if len(messages) <= keep_value:
                return len(messages)  # All messages are recent
            return int(len(messages) - keep_value)

        if keep_type in {"tokens", "fraction"}:
            # Calculate target token count
            if keep_type == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None:
                    # Fallback to message count if profile not available
                    messages_to_keep = 20
                    if len(messages) <= messages_to_keep:
                        return len(messages)
                    return len(messages) - messages_to_keep
                target_token_count = int(max_input_tokens * keep_value)
            else:
                target_token_count = int(keep_value)

            if target_token_count <= 0:
                target_token_count = 1

            # Keep recent messages up to token limit
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = self.token_counter([messages[i]])
                if tokens_kept + msg_tokens > target_token_count:
                    return i + 1
                tokens_kept += msg_tokens
            return 0  # All messages are within token limit

        return len(messages)

    def _truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Truncate large arguments in a single tool call.

        Args:
            tool_call: The tool call dictionary to truncate.

        Returns:
            A copy of the tool call with large arguments truncated.
        """
        args = tool_call.get("args", {})

        truncated_args = {}
        modified = False

        for key, value in args.items():
            if isinstance(value, str) and len(value) > self._max_arg_length:
                truncated_args[key] = value[:20] + self._truncation_text
                modified = True
            else:
                truncated_args[key] = value

        if modified:
            return {
                **tool_call,
                "args": truncated_args,
            }
        return tool_call

    def _truncate_args(self, messages: list[AnyMessage]) -> tuple[list[AnyMessage], bool]:
        """Truncate large tool call arguments in old messages.

        Args:
            messages: Messages to potentially truncate.

        Returns:
            Tuple of (truncated_messages, modified). If modified is False,
            truncated_messages is the same as input messages.
        """
        total_tokens = self._cached_total_tokens(messages)
        if not self._should_truncate_args(messages, total_tokens):
            return messages, False

        cutoff_index = self._determine_truncate_cutoff_index(messages)
        if cutoff_index >= len(messages):
            return messages, False

        # Process messages before the cutoff
        truncated_messages = []
        modified = False

        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
                # Check if this AIMessage has tool calls we need to truncate
                truncated_tool_calls = []
                msg_modified = False

                for tool_call in msg.tool_calls:
                    if tool_call["name"] in {"write_file", "edit_file"}:
                        truncated_call = self._truncate_tool_call(cast(dict[str, Any], tool_call))
                        if truncated_call != tool_call:
                            msg_modified = True
                        truncated_tool_calls.append(truncated_call)
                    else:
                        truncated_tool_calls.append(tool_call)

                if msg_modified:
                    # Create a new AIMessage with truncated tool calls
                    truncated_msg = msg.model_copy()
                    truncated_msg.tool_calls = truncated_tool_calls
                    truncated_messages.append(truncated_msg)
                    modified = True
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)

        return truncated_messages, modified

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        thread_id: str | None = None,
    ) -> str:
        """Persist messages to backend before summarization.

        Appends evicted messages to a single JSONL file per thread. Each
        line stores one losslessly serialized original message.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            thread_id: Explicit target thread id (additive, default ``None``).
                When ``None`` the path is resolved via the langgraph config
                contextvar (existing ``before_model`` behaviour). An explicit
                value is forwarded by :meth:`apply_pseudo_summary` so the
                offload lands in ``<agent>.jsonl`` outside a graph run
                (``DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.4 Blocker-1).

        Returns:
            The file path where history was stored.

        Raises:
            HistoryOffloadError: When the backend fails to persist the history
                (returns ``None``, an error result, or raises). Fatal per the
                No-Agent-Error-State Policy — the agent runtime catch-all
                matches this class name in ``FATAL_EXCEPTION_TYPES`` and
                transitions the agent to ``AgentStatus.ERROR``.
                See ``DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md``.
        """
        path = self._get_history_path(thread_id)

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
        ensure_messages_created_at(filtered_messages, MonotonicUtcClock())
        new_section = serialize_summarized_messages_jsonl(
            filtered_messages,
            summarized_at=timestamp,
        )

        # Read existing content (if any) and append
        # Note: We use download_files() instead of read() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        # A read failure here is NOT fatal — the file likely doesn't exist yet.
        # Only the write path below is fail-loud.
        existing_content = ""
        try:
            responses = backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = backend.edit(path, existing_content, combined_content) if existing_content else backend.write(path, combined_content)
        except Exception as e:  # noqa: BLE001
            # Fail loud: raising HistoryOffloadError bubbles up through
            # before_model into the runtime catch-all → AgentStatus.ERROR.
            msg = (
                f"Exception writing conversation history to {path} "
                f"({len(filtered_messages)} messages): {type(e).__name__}: {e}"
            )
            logger.error(msg)
            raise HistoryOffloadError(msg) from e

        if result is None or result.error:
            error_msg = result.error if result else "backend returned None"
            msg = (
                f"Backend failed to write conversation history to {path} "
                f"({len(filtered_messages)} messages): {error_msg}"
            )
            logger.error(msg)
            raise HistoryOffloadError(msg)

        logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
        if self._on_summarized:
            self._on_summarized(path)
        return path

    async def _aoffload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        thread_id: str | None = None,
    ) -> str:
        """Persist messages to backend before summarization (async).

        Appends evicted messages to a single JSONL file per thread. Each
        line stores one losslessly serialized original message.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            thread_id: Explicit target thread id (additive, default ``None``).
                Symmetric with sync :meth:`_offload_to_backend`. When ``None``
                the path is resolved via the langgraph config contextvar;
                :meth:`apply_pseudo_summary` forwards an explicit value so the
                offload lands in ``<agent>.jsonl`` outside a graph run
                (``DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.4 Blocker-1).

        Returns:
            The file path where history was stored.

        Raises:
            HistoryOffloadError: When the backend fails to persist the history.
                See sync :meth:`_offload_to_backend` for the rationale.
        """
        path = self._get_history_path(thread_id)

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
        ensure_messages_created_at(filtered_messages, MonotonicUtcClock())
        new_section = serialize_summarized_messages_jsonl(
            filtered_messages,
            summarized_at=timestamp,
        )

        # Read existing content (if any) and append
        # Note: We use adownload_files() instead of aread() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        # A read failure here is NOT fatal — the file likely doesn't exist yet.
        # Only the write path below is fail-loud.
        existing_content = ""
        try:
            responses = await backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = (
                await backend.aedit(path, existing_content, combined_content) if existing_content else await backend.awrite(path, combined_content)
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                f"Exception writing conversation history to {path} "
                f"({len(filtered_messages)} messages): {type(e).__name__}: {e}"
            )
            logger.error(msg)
            raise HistoryOffloadError(msg) from e

        if result is None or result.error:
            error_msg = result.error if result else "backend returned None"
            msg = (
                f"Backend failed to write conversation history to {path} "
                f"({len(filtered_messages)} messages): {error_msg}"
            )
            logger.error(msg)
            raise HistoryOffloadError(msg)

        logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
        if self._on_summarized:
            # ⚠️ CRITICAL — DO NOT call ``self._on_summarized(path)``
            # inline here. We are inside an ``async`` method
            # (``_aoffload_to_backend``) running on the AGENT'S
            # event-loop thread. The production callback is
            # ``index_all_history_files`` — a FULLY SYNCHRONOUS LanceDB
            # WRITE (vectors.lance) + (cold) embedding-model load.
            #
            # LanceDB bridges to its OWN background event loop via
            # ``run_coroutine_threadsafe(...).result()`` (see
            # ``lancedb/background_loop.py``). Run inline, that blocks
            # THIS loop thread WHILE HOLDING THE GIL; LanceDB's
            # background loop then can never acquire the GIL to finish
            # → permanent GIL deadlock, the agent's loop stops ticking
            # forever (status stuck "running", RPyC dead). This is the
            # live mj1_init deadlock (2026-06-15).
            #
            # Offloading to a worker thread makes the blocking
            # ``.result()`` wait there RELEASE the GIL, so LanceDB's
            # loop runs and our loop keeps ticking. NEVER revert to an
            # inline call. See
            # docs/DESIGN_NO_LOOP_BLOCKING_TOOLS.md and
            # tests/test_no_loop_blocking_summarization.py.
            await asyncio.to_thread(self._on_summarized, path)
        return path

    async def apply_pseudo_summary(
        self,
        thread_id: str,
        summary_text: str,
        current_messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], str | None]:
        """Evict ``current_messages`` to JSONL history and build the
        replacement summary-shape message list for a startup pseudo-summary.

        Pure producer: this method offloads the supplied messages and RETURNS
        ``(new_messages, file_path)``. It does NOT write to any checkpointer
        and does NOT emit ``RemoveMessage`` — the caller owns the state wipe
        (see ``DESIGN_STARTUP_QUERY_OPTIONS.md`` §4.6, Blocker-2).

        Delegates to the existing private helpers with one additive change:
        ``_aoffload_to_backend`` is called with an explicit ``thread_id`` so
        the JSONL lands in ``<agent>.jsonl`` even though this runs OUTSIDE a
        graph run where langgraph's config contextvar is unavailable
        (Blocker-1, option B). Serialization, append, fail-loud, and
        summary-message building stay entirely inside those helpers.

        Args:
            thread_id: The thread whose JSONL file the offload appends to.
                Callers pass the agent name (matching the real invoke's
                checkpointer thread_id).
            summary_text: Text for the summary-shape ``HumanMessage``.
            current_messages: The active-context messages to evict.

        Returns:
            ``(new_messages, file_path)`` where ``new_messages`` is exactly
            one summary-shape ``HumanMessage`` (``lc_source='summarization'``)
            — with a JSONL reference line when an offload write occurred, and
            without one when the (filtered) history was empty. ``file_path``
            is the JSONL path when a write occurred, else ``None``.

        Raises:
            RuntimeError: When the middleware holds a factory backend rather
                than a concrete ``BackendProtocol`` instance — a factory needs
                a ``ToolRuntime`` unavailable outside a graph run (Gap-4).
            HistoryOffloadError: When the backend fails to persist history —
                same fail-loud contract as ``before_model`` / ``abefore_model``.
        """
        # Gap-4: backend must be a concrete instance; factory backends need a
        # ToolRuntime that does not exist outside a graph run.
        if not isinstance(self._backend, BackendProtocol):
            msg = (
                "apply_pseudo_summary requires SummarizationMiddleware to hold "
                "a concrete backend instance; factory backends need a "
                "ToolRuntime that is unavailable outside a graph run."
            )
            raise RuntimeError(msg)
        backend = self._backend

        # Empty-history invariant: filter previous summary messages first; if
        # nothing remains, skip the offload entirely and build a summary-shape
        # message WITHOUT a JSONL reference (no dangling file link).
        filtered = self._filter_summary_messages(current_messages)
        if not filtered:
            return self._build_new_messages_with_path(summary_text, None), None

        # Offload the SAME filtered list we emptiness-checked (NIT-1: "what we
        # checked is what we write" — one filtration, no ambiguity).
        # ``_aoffload_to_backend`` re-applies ``_filter_summary_messages``
        # internally; on an already-filtered list that is an idempotent no-op.
        file_path = await self._aoffload_to_backend(
            backend, filtered, thread_id
        )
        new_messages = self._build_new_messages_with_path(summary_text, file_path)
        return new_messages, file_path

    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, with history offloading and arg truncation.

        First truncates large tool arguments in old messages if configured.
        Then offloads messages to backend before summarization if thresholds are met.
        The summary message includes a reference to the file path where the full
        conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with truncated/summarized messages if processing was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        # Step 1: Truncate args if configured
        truncated_messages, args_were_truncated = self._truncate_args(messages)

        # Step 2: Check if summarization should happen.
        # ``_cached_total_tokens`` memoizes the last-counted list by
        # message-id signature so a repeat before_model call on the
        # same history skips the O(messages × tokens) pass.
        total_tokens = self._cached_total_tokens(truncated_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If only truncation happened (no summarization)
        if args_were_truncated and not should_summarize:
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *truncated_messages,
                ]
            }

        # If no truncation and no summarization
        if not should_summarize:
            return None

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # If truncation happened but we can't summarize, still return truncated messages
            if args_were_truncated:
                return {
                    "messages": [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        *truncated_messages,
                    ]
                }
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend FIRST — abort summarization if this fails to prevent data loss.
        # ``_offload_to_backend`` now raises ``HistoryOffloadError`` on any failure; the raise
        # propagates through this method into the agent runtime catch-all, which matches
        # the class name in ``FATAL_EXCEPTION_TYPES`` and transitions the agent to
        # ``AgentStatus.ERROR``. See ``DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md``.
        backend = self._get_backend(state, runtime)
        file_path = self._offload_to_backend(backend, messages_to_summarize)

        # Generate summary — only reached when the offload succeeded.
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _format_requested_summary_prompt(
        self,
        prompt: str,
        messages: list[AnyMessage],
    ) -> str:
        fields = [
            field_name
            for _, field_name, _, _ in Formatter().parse(prompt)
            if field_name is not None
        ]
        if any(field != "messages" for field in fields) or fields.count("messages") > 1:
            raise ValueError(
                "summary prompt may contain at most one {messages} field"
            )
        template = prompt if fields else f"{prompt}\n\n{{messages}}"
        return template.format(messages=get_buffer_string(messages)).rstrip()

    async def _acreate_summary_for_prompt(
        self,
        messages_to_summarize: list[AnyMessage],
        summary_prompt: str | None,
        thread_id: str | None,
        *,
        fail_loud: bool = False,
    ) -> str:
        if not messages_to_summarize:
            return "No previous conversation history."
        if summary_prompt is None:
            trimmed_messages = self._trim_messages_for_summary(
                messages_to_summarize
            )
            if not trimmed_messages:
                return "Previous conversation was too long to summarize."
            formatted_prompt = self.summary_prompt.format(
                messages=get_buffer_string(trimmed_messages)
            ).rstrip()
            config: RunnableConfig = {
                "metadata": {"lc_source": "summarization"},
            }
            if thread_id is not None:
                config["configurable"] = {"thread_id": thread_id}
            try:
                response = await self.model.ainvoke(
                    formatted_prompt, config=config
                )
                return response.text.strip()
            except Exception as e:
                if fail_loud:
                    raise
                return f"Error generating summary: {e!s}"
        prompt_for_budget = (
            summary_prompt
            if "{messages}" in summary_prompt
            else f"{summary_prompt}\n\n{{messages}}"
        )
        if self.trim_tokens_to_summarize is None:
            trimmed_messages = messages_to_summarize
        else:
            prompt_tokens = self.token_counter(
                [HumanMessage(content=prompt_for_budget)]
            )
            adjusted_limit = max(
                1000,
                self.trim_tokens_to_summarize - prompt_tokens,
            )
            trimmed_messages = cast(
                "list[AnyMessage]",
                trim_messages(
                    messages_to_summarize,
                    max_tokens=adjusted_limit,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."
        formatted_prompt = self._format_requested_summary_prompt(
            summary_prompt,
            trimmed_messages,
        )
        custom_config: RunnableConfig = {
            "metadata": {"lc_source": "summarization"},
        }
        if thread_id is not None:
            custom_config["configurable"] = {"thread_id": thread_id}
        try:
            response = await self.model.ainvoke(
                formatted_prompt, config=custom_config
            )
            return response.text.strip()
        except Exception as e:
            if fail_loud:
                raise
            return f"Error generating summary: {e!s}"

    async def _arun_compaction(
        self,
        messages: list[AnyMessage],
        runtime: Runtime | None,
        *,
        force: bool,
        summary_prompt: str | None = None,
        backend: BackendProtocol | None = None,
        thread_id: str | None = None,
    ) -> RequestedSummaryResult:
        self._ensure_message_ids(messages)
        truncated_messages, args_were_truncated = self._truncate_args(messages)
        total_tokens = self._cached_total_tokens(truncated_messages)
        should_summarize = force or self._should_summarize(
            truncated_messages,
            total_tokens,
        )
        if not should_summarize:
            replacement = truncated_messages if args_were_truncated else None
            return RequestedSummaryResult(False, replacement, None)

        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            replacement = truncated_messages if args_were_truncated else None
            return RequestedSummaryResult(False, replacement, None)

        messages_to_summarize, preserved_messages = self._partition_messages(
            truncated_messages,
            cutoff_index,
        )
        resolved_backend = backend
        if resolved_backend is None:
            if runtime is None:
                raise RuntimeError(
                    "requested summarization requires a concrete backend"
                )
            state = cast("AgentState[Any]", {"messages": messages})
            resolved_backend = self._get_backend(state, runtime)
        file_path = await self._aoffload_to_backend(
            resolved_backend,
            messages_to_summarize,
            thread_id,
        )
        summary = await self._acreate_summary_for_prompt(
            messages_to_summarize,
            summary_prompt,
            thread_id,
            fail_loud=force,
        )
        new_messages = self._build_new_messages_with_path(summary, file_path)
        return RequestedSummaryResult(
            True,
            [*new_messages, *preserved_messages],
            file_path,
        )

    async def apply_requested_summary(
        self,
        thread_id: str,
        current_messages: list[AnyMessage],
        summary_prompt: str | None = None,
    ) -> RequestedSummaryResult:
        if not isinstance(self._backend, BackendProtocol):
            raise RuntimeError(
                "requested summarization requires a concrete backend"
            )
        return await self._arun_compaction(
            current_messages,
            None,
            force=True,
            summary_prompt=summary_prompt,
            backend=self._backend,
            thread_id=thread_id,
        )

    @override
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, with history offloading and arg truncation (async).

        First truncates large tool arguments in old messages if configured.
        Then offloads messages to backend before summarization if thresholds are met.
        The summary message includes a reference to the file path where the
        full conversation history was stored.

        The summary message includes a reference to the file path where the full
        conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with truncated/summarized messages if processing was performed.
        """
        result = await self._arun_compaction(
            state["messages"],
            runtime,
            force=False,
        )
        if result.replacement_messages is None:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *result.replacement_messages,
            ]
        }
