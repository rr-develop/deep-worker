from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

CREATED_AT_KEY = "created_at"
TOOL_CALL_CREATED_AT_KEY = "tool_call_created_at"


class MonotonicUtcClock:
    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._last: datetime | None = None

    def next(self) -> str:
        current = self._now().astimezone(UTC)
        if self._last is not None and current <= self._last:
            current = self._last + timedelta(microseconds=1)
        self._last = current
        return _format_utc(current)

    def observe(self, value: str) -> bool:
        parsed = _parse_utc(value)
        if parsed is None:
            return False
        if self._last is None or parsed > self._last:
            self._last = parsed
            return True
        return False


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def ensure_message_created_at(
    message: AnyMessage,
    clock: MonotonicUtcClock,
) -> AnyMessage:
    existing = message.additional_kwargs.get(CREATED_AT_KEY)
    if isinstance(existing, str) and clock.observe(existing):
        created_at = existing
    else:
        created_at = clock.next()
        message.additional_kwargs[CREATED_AT_KEY] = created_at

    if isinstance(message, AIMessage) and message.tool_calls:
        raw_map = message.additional_kwargs.get(TOOL_CALL_CREATED_AT_KEY)
        if not isinstance(raw_map, dict):
            raw_map = {}
        tool_call_created_at = dict(raw_map)
        for tool_call in message.tool_calls:
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                tool_call_created_at.setdefault(call_id, created_at)
        message.additional_kwargs[TOOL_CALL_CREATED_AT_KEY] = tool_call_created_at

    return message


def ensure_messages_created_at(
    messages: list[AnyMessage],
    clock: MonotonicUtcClock,
) -> list[AnyMessage]:
    for message in messages:
        ensure_message_created_at(message, clock)
    return messages


def format_message_with_timestamp(message: AnyMessage, clock: MonotonicUtcClock) -> str:
    ensure_message_created_at(message, clock)
    created_at = message.additional_kwargs[CREATED_AT_KEY]

    lines: list[str] = []
    if isinstance(message, HumanMessage):
        lines.append(f"[{created_at}] Human: {_content_text(message.content)}")
    elif isinstance(message, AIMessage):
        lines.append(f"[{created_at}] AI: {_content_text(message.content)}")
        tool_call_created_at = message.additional_kwargs.get(TOOL_CALL_CREATED_AT_KEY)
        if not isinstance(tool_call_created_at, dict):
            tool_call_created_at = {}
        for tool_call in message.tool_calls or []:
            call_id = str(tool_call.get("id") or "")
            name = str(tool_call.get("name") or "unknown")
            args = tool_call.get("args", {})
            call_created_at = tool_call_created_at.get(call_id, created_at)
            lines.append(f"[{call_created_at}] Tool call {name} ({call_id}): {args}")
    elif isinstance(message, ToolMessage):
        name = getattr(message, "name", None) or "unknown"
        call_id = getattr(message, "tool_call_id", None) or ""
        lines.append(
            f"[{created_at}] Tool result {name} ({call_id}): {_content_text(message.content)}"
        )
    else:
        lines.append(
            f"[{created_at}] {message.__class__.__name__}: {_content_text(getattr(message, 'content', ''))}"
        )
    return "\n".join(lines)


def format_messages_with_timestamps(
    messages: list[AnyMessage],
    clock: MonotonicUtcClock | None = None,
) -> str:
    active_clock = clock or MonotonicUtcClock()
    return "\n".join(format_message_with_timestamp(message, active_clock) for message in messages)


def ensure_message_created_at_after(
    prior_messages: list[AnyMessage],
    message: AnyMessage,
) -> AnyMessage:
    clock = MonotonicUtcClock()
    for prior in prior_messages:
        existing = prior.additional_kwargs.get(CREATED_AT_KEY)
        if isinstance(existing, str):
            clock.observe(existing)
    return ensure_message_created_at(message, clock)


def _stamp_response(value: Any, clock: MonotonicUtcClock) -> Any:
    if isinstance(value, BaseMessage):
        ensure_message_created_at(value, clock)
        return value
    if isinstance(value, list):
        for item in value:
            _stamp_response(item, clock)
        return value
    result = getattr(value, "result", None)
    if result is not None:
        _stamp_response(result, clock)
    return value


def _stamp_tool_result(value: Any, clock: MonotonicUtcClock) -> Any:
    if isinstance(value, ToolMessage):
        ensure_message_created_at(value, clock)
        return value
    if isinstance(value, Command):
        update = value.update
        if isinstance(update, dict):
            messages = update.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, BaseMessage):
                        ensure_message_created_at(message, clock)
        return value
    return _stamp_response(value, clock)


class MessageTimestampMiddleware(AgentMiddleware):
    def __init__(self, clock: MonotonicUtcClock | None = None) -> None:
        super().__init__()
        self.clock = clock or MonotonicUtcClock()

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        # Use a per-state clock to order restored messages, then sync the persistent clock for new outputs.
        state_clock = MonotonicUtcClock()
        for message in messages:
            if isinstance(message, BaseMessage):
                ensure_message_created_at(message, state_clock)
                created_at = message.additional_kwargs.get(CREATED_AT_KEY)
                if isinstance(created_at, str):
                    self.clock.observe(created_at)
        return None

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return _stamp_response(handler(request), self.clock)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return _stamp_response(await handler(request), self.clock)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return _stamp_tool_result(handler(request), self.clock)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return _stamp_tool_result(await handler(request), self.clock)
