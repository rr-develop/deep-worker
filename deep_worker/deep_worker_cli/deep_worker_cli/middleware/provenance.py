"""Stamp provider/model/usage onto AIMessages (DESIGN_SUMMARIZATION_HOT_COLD §3).

Stamps live in ``response_metadata`` / ``additional_kwargs`` and are never
sent to the provider. Set-if-missing for ``created_at`` is owned by
``MessageTimestampMiddleware``; this middleware is the single writer of
``deep_worker_origin`` (delta) and ``deep_worker_usage`` (every AIMessage).
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage

from deep_worker_cli.context_limit.hot_cache import (
    ORIGIN_KEY,
    USAGE_KEY,
    effective_origin,
)

CONTEXT_KEY = "deep_worker_context"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _usage_from_message(message: AIMessage) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "received_at": None,
        "request_started_at": None,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "context_used": None,
        "context_size": None,
        "provider_session_id": None,
        "provider_turn_id": None,
        "source": None,
    }
    meta = getattr(message, "usage_metadata", None) or {}
    if isinstance(meta, dict) and meta:
        usage["input_tokens"] = meta.get("input_tokens")
        usage["output_tokens"] = meta.get("output_tokens")
        details = meta.get("input_token_details") or {}
        if isinstance(details, dict):
            usage["cache_read_tokens"] = details.get("cache_read") or details.get(
                "cache_read_tokens"
            )
            usage["cache_write_tokens"] = details.get("cache_creation") or details.get(
                "cache_write"
            )
        if usage["input_tokens"] is not None:
            usage["source"] = "api_usage_metadata"
    rm = getattr(message, "response_metadata", None) or {}
    ctx = rm.get(CONTEXT_KEY) if isinstance(rm, dict) else None
    if isinstance(ctx, dict):
        usage["context_used"] = ctx.get("used")
        usage["context_size"] = ctx.get("size")
        if usage["source"] is None:
            usage["source"] = "acp_usage_update"
    return usage


def resolve_provider_name(*candidates: Any) -> str | None:
    """First usable CLI/provider name. Never returns ``unknown``."""
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        if text and text != "unknown":
            return text
    return None


def _current_origin(
    *,
    provider_name: str,
    model_name: str | None,
    model_id: str | None,
    adapter: str | None,
) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "model": model_name,
        "model_id": model_id,
        "adapter": adapter,
        "since": _now_iso(),
    }


def _origin_tuple(origin: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    if not origin:
        return (None, None, None)
    return (origin.get("provider"), origin.get("model"), origin.get("model_id"))


def stamp_ai_message(
    message: AIMessage,
    *,
    request_messages: list[Any],
    provider_name: str,
    model_name: str | None,
    model_id: str | None = None,
    adapter: str | None = None,
    request_started_at: str,
    latency_ms: int,
    provider_session_id: str | None = None,
) -> AIMessage:
    """Mutate ``message`` in place with origin (delta) and usage. Return it."""
    usage = _usage_from_message(message)
    usage["received_at"] = _now_iso()
    usage["request_started_at"] = request_started_at
    usage["latency_ms"] = latency_ms
    usage["provider_session_id"] = provider_session_id
    rm = dict(getattr(message, "response_metadata", None) or {})
    rm[USAGE_KEY] = usage
    resolved = resolve_provider_name(provider_name)
    if resolved is not None:
        current = _current_origin(
            provider_name=resolved,
            model_name=model_name,
            model_id=model_id,
            adapter=adapter,
        )
        prior = effective_origin(
            [m for m in request_messages if isinstance(m, BaseMessage)]
        )
        if _origin_tuple(prior) != _origin_tuple(current):
            rm[ORIGIN_KEY] = current
    message.response_metadata = rm
    return message


def _iter_ai_messages(value: Any) -> list[AIMessage]:
    found: list[AIMessage] = []
    if isinstance(value, AIMessage):
        found.append(value)
    elif isinstance(value, list):
        for item in value:
            found.extend(_iter_ai_messages(item))
    else:
        result = getattr(value, "result", None)
        if result is not None:
            found.extend(_iter_ai_messages(result))
    return found


class ProvenanceMiddleware(AgentMiddleware):
    """Write ``deep_worker_origin`` / ``deep_worker_usage`` after each model call."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str | None,
        adapter: str | None = None,
        model_id_getter: Callable[[], str | None] | None = None,
        session_id_getter: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.model_name = model_name
        self.adapter = adapter
        self.model_id_getter = model_id_getter
        self.session_id_getter = session_id_getter

    def _stamp_response(self, request: ModelRequest, response: Any, started: str, t0: float) -> Any:
        latency_ms = int((perf_counter() - t0) * 1000)
        model_id = None
        if self.model_id_getter is not None:
            try:
                model_id = self.model_id_getter()
            except Exception:
                model_id = None
        session_id = None
        if self.session_id_getter is not None:
            try:
                session_id = self.session_id_getter()
            except Exception:
                session_id = None
        messages = list(getattr(request, "messages", None) or [])
        for ai in _iter_ai_messages(response):
            stamp_ai_message(
                ai,
                request_messages=messages,
                provider_name=self.provider_name,
                model_name=self.model_name,
                model_id=model_id,
                adapter=self.adapter,
                request_started_at=started,
                latency_ms=latency_ms,
                provider_session_id=session_id,
            )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        started = _now_iso()
        t0 = perf_counter()
        response = handler(request)
        return self._stamp_response(request, response, started, t0)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        started = _now_iso()
        t0 = perf_counter()
        response = await handler(request)
        return self._stamp_response(request, response, started, t0)
