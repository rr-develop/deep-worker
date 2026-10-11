"""TokenStatsCallbackHandler — per-invocation usage recording for
LangChain-backed providers (DESIGN_AGENT_TOKEN_STATS.md §5.1).

Attached to the model instance in :func:`deep_worker_cli.config.create_model`
for the standard LangChain providers (deepseek/openai/anthropic/google/vertexai).
Custom providers (claude-code-sdk/codex/grok) never receive this handler — they use
the state-machine event callback (see ``UsageEventCallback``).

Contract (§5.1):
- ``run_id`` is the invocation key: ``on_llm_start`` records it in a pending
  set; ``on_llm_end`` / ``on_llm_error`` close exactly one record per run_id.
- ``on_llm_end`` does not return before the append finished; recorder errors
  are logged, never raised (they must not break the LLM result).
- ``on_llm_error`` distinguishes ``asyncio.CancelledError`` (status
  "cancelled") from other failures (status "failed"); the pending run_id is
  removed on BOTH paths.
- There is no buffer: every record is appended (and awaited) immediately, so
  no flush-on-shutdown is needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from deep_worker_cli.stats.token_stats import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_NO_USAGE,
    STATUS_OK,
    TokenUsageRecorder,
    UsageRecord,
    build_record,
    normalize_usage,
)

_logger = logging.getLogger(__name__)


class TokenStatsCallbackHandler(AsyncCallbackHandler):
    """Async callback that records one UsageRecord per observed model run.

    Provider and kind are fixed at construction (create_model knows the
    provider it just resolved; after hot-failover a new model + new handler
    are created with the new provider). The model name is captured from the
    ``serialized`` kwargs on ``on_llm_start``, falling back to the name
    passed at construction.
    """

    def __init__(
        self,
        recorder: TokenUsageRecorder,
        *,
        provider: str,
        kind: str,
        model_name: str | None = None,
    ) -> None:
        self._recorder = recorder
        self._provider = provider
        self._kind = kind
        self._model_name = model_name
        self._pending: set[UUID] = set()

    # -- helpers ------------------------------------------------------------

    def _capture_model_name(self, serialized: dict[str, Any] | None) -> None:
        if self._model_name is not None:
            return
        try:
            kwargs = (serialized or {}).get("kwargs") or {}
            name = kwargs.get("model") or kwargs.get("model_name")
            if name:
                self._model_name = str(name)
        except Exception:  # noqa: BLE001 — best-effort, never break callbacks
            _logger.exception("token-stats: failed to capture model name")

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        """Extract the raw provider usage dict from an LLMResult."""
        try:
            generations = getattr(response, "generations", None)
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                md = getattr(message, "response_metadata", None) or {}
                usage = md.get("token_usage")
                if usage is None:
                    usage = md.get("usage_metadata")
                if usage is None:
                    usage = getattr(message, "usage_metadata", None)
                if isinstance(usage, dict):
                    return usage
            llm_output = getattr(response, "llm_output", None)
            if isinstance(llm_output, dict):
                usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
                if isinstance(usage, dict):
                    return usage
        except Exception:  # noqa: BLE001
            _logger.exception("token-stats: failed to extract usage from LLMResult")
        return None

    def _build_record(
        self, *, run_id: UUID, status: str, usage: dict[str, Any] | None
    ) -> UsageRecord:
        norm = None
        if status == STATUS_OK and usage is not None:
            norm = normalize_usage(self._provider, usage)
        if norm is None and status == STATUS_OK:
            status = STATUS_NO_USAGE
        return build_record(
            provider=self._provider,
            model=self._model_name or "",
            kind=self._kind,
            status=status,
            usage=norm,
            event_key=f"langchain|{run_id}",
        )

    async def _record_safe(self, record: UsageRecord) -> None:
        try:
            await self._recorder.record(record)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort, never break LLM result
            _logger.exception("token-stats: recorder failed, record dropped")

    # -- callbacks ----------------------------------------------------------

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pending.add(run_id)
        self._capture_model_name(serialized)

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id not in self._pending:
            return  # not an invocation we observed (e.g. foreign run)
        self._pending.discard(run_id)
        usage = self._extract_usage(response)
        await self._record_safe(self._build_record(run_id=run_id, status=STATUS_OK, usage=usage))

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id not in self._pending:
            return
        self._pending.discard(run_id)
        status = (
            STATUS_CANCELLED
            if isinstance(error, asyncio.CancelledError)
            else STATUS_FAILED
        )
        await self._record_safe(self._build_record(run_id=run_id, status=status, usage=None))


def make_stats_callback_factory(recorder: TokenUsageRecorder) -> Any:
    """Build the ``stats_callback_factory`` for :func:`create_model`.

    Returns a factory callable as ``factory(kind, *, provider, model_name)``
    producing a :class:`TokenStatsCallbackHandler` bound to ``recorder``
    (DESIGN_AGENT_TOKEN_STATS.md §5.1). Used by the agent-process and
    TaskManager call sites so standard LangChain providers record
    per-invocation usage into one shared file.
    """

    def _factory(kind: str, *, provider: str, model_name: str) -> Any:
        return TokenStatsCallbackHandler(
            recorder,
            provider=provider,
            kind=kind,
            model_name=model_name or "",
        )

    return _factory


def make_usage_event_callback_factory(recorder: TokenUsageRecorder) -> Any:
    """Build the ``usage_event_callback_factory`` for :func:`create_model`.

    Returns a factory callable as ``factory(kind, *, provider, model_name)``
    producing the custom-provider state-machine callback bound to ``recorder``
    (DESIGN_AGENT_TOKEN_STATS.md §5.2). Installed by the provider via
    ``model.set_usage_event_callback(cb)``.
    """

    def _factory(kind: str, *, provider: str, model_name: str) -> Any:
        return recorder.make_usage_event_callback(
            provider=provider,
            kind=kind,
            model_name=model_name or "",
        )

    return _factory
