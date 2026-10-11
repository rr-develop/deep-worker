"""Tests for TokenStatsCallbackHandler and create_model wiring (§5.1)."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_worker_cli.config import _attach_token_stats
from deep_worker_cli.stats.callback import TokenStatsCallbackHandler
from deep_worker_cli.stats.token_stats import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    TokenUsageRecorder,
    UsageEventCallback,
    read_stats_file,
)


def _llm_result(metadata: dict) -> LLMResult:
    msg = AIMessage(content="ok", response_metadata=metadata)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


class TestTokenStatsCallbackHandler:
    @pytest.mark.asyncio
    async def test_every_record_pushes_one_model_context_event(self, tmp_path):
        events = []

        async def on_record(event):
            events.append(event)

        rec = TokenUsageRecorder(tmp_path, on_record=on_record)
        handler = TokenStatsCallbackHandler(rec, provider="deepseek", kind="agent")
        for prompt_tokens in (100, 125):
            run_id = uuid4()
            await handler.on_llm_start(
                {"kwargs": {"model": "deepseek-v4-flash"}}, ["hi"], run_id=run_id
            )
            await handler.on_llm_end(
                _llm_result({
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 5,
                        "total_tokens": prompt_tokens + 5,
                    }
                }),
                run_id=run_id,
            )

        assert len(events) == 2
        assert all(event["type"] == "model_context_changed" for event in events)
        assert [event["prompt_tokens"] for event in events] == [100, 125]

    @pytest.mark.asyncio
    async def test_on_llm_end_records_one_record(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="deepseek", kind="agent")
        run_id = uuid4()
        await handler.on_llm_start({"kwargs": {"model": "deepseek-v4-flash"}}, ["hi"], run_id=run_id)
        resp = _llm_result(
            {
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_cache_hit_tokens": 60,
                    "prompt_cache_miss_tokens": 40,
                    "total_tokens": 150,
                }
            }
        )
        await handler.on_llm_end(resp, run_id=run_id)

        read = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(read.records) == 1
        r = read.records[0]
        assert r.provider == "deepseek"
        assert r.kind == "agent"
        assert r.model == "deepseek-v4-flash"
        assert r.status == STATUS_OK
        assert r.prompt_tokens == 100
        assert r.completion_tokens == 50
        assert r.total_tokens == 150
        assert r.cache_hit_tokens == 60
        assert r.cache_miss_tokens == 40
        assert r.event_key == f"langchain|{run_id}"
        assert r.provider_session_id is None

    @pytest.mark.asyncio
    async def test_unknown_run_id_is_ignored(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="openai", kind="agent")
        run_id = uuid4()
        # No on_llm_start observed → end is a no-op.
        await handler.on_llm_end(_llm_result({"token_usage": {}}), run_id=run_id)
        read = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(read.records) == 0

    @pytest.mark.asyncio
    async def test_error_statuses_and_pending_cleanup(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="anthropic", kind="tm")
        run_id = uuid4()
        await handler.on_llm_start({}, ["hi"], run_id=run_id)
        await handler.on_llm_error(RuntimeError("boom"), run_id=run_id)
        # Pending was removed: a late on_llm_end for the same run_id writes nothing.
        await handler.on_llm_end(_llm_result({"token_usage": {}}), run_id=run_id)

        read = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(read.records) == 1
        assert read.records[0].status == STATUS_FAILED
        assert read.records[0].kind == "tm"
        assert read.records[0].prompt_tokens is None

    @pytest.mark.asyncio
    async def test_cancelled_error_status(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="deepseek", kind="agent")
        run_id = uuid4()
        await handler.on_llm_start({}, ["hi"], run_id=run_id)
        await handler.on_llm_error(asyncio.CancelledError(), run_id=run_id)
        read = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(read.records) == 1
        assert read.records[0].status == STATUS_CANCELLED

    @pytest.mark.asyncio
    async def test_recorder_error_does_not_break_llm_end(self, tmp_path, monkeypatch):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="deepseek", kind="agent")
        run_id = uuid4()
        await handler.on_llm_start({}, ["hi"], run_id=run_id)

        async def boom(record):
            raise OSError("disk full")

        monkeypatch.setattr(rec, "record", boom)
        # Must complete without raising (errors are logged, never raised).
        await handler.on_llm_end(_llm_result({"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}}), run_id=run_id)

    @pytest.mark.asyncio
    async def test_no_usage_records_no_usage(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        handler = TokenStatsCallbackHandler(rec, provider="openai", kind="agent")
        run_id = uuid4()
        await handler.on_llm_start({}, ["hi"], run_id=run_id)
        await handler.on_llm_end(_llm_result({"token_usage": {"bogus": 1}}), run_id=run_id)
        read = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(read.records) == 1
        assert read.records[0].status == "no_usage"
        assert read.records[0].prompt_tokens is None


class TestAttachTokenStats:
    def _handler_factory(self, rec):
        def factory(kind, *, provider, model_name):
            return TokenStatsCallbackHandler(
                rec, provider=provider, kind=kind, model_name=model_name
            )

        return factory

    def _event_factory(self, rec):
        def factory(kind, *, provider, model_name):
            return UsageEventCallback(rec, provider=provider, kind=kind, model_name=model_name)

        return factory

    def test_standard_provider_gets_handler_once(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        model = SimpleNamespace(callbacks=[])
        _attach_token_stats(
            model,
            provider="deepseek",
            model_name="deepseek-v4-flash",
            kind="agent",
            stats_callback_factory=self._handler_factory(rec),
            usage_event_callback_factory=None,
        )
        assert len(model.callbacks) == 1
        assert isinstance(model.callbacks[0], TokenStatsCallbackHandler)
        # Idempotent: attaching again does not duplicate.
        _attach_token_stats(
            model,
            provider="deepseek",
            model_name="deepseek-v4-flash",
            kind="agent",
            stats_callback_factory=self._handler_factory(rec),
            usage_event_callback_factory=None,
        )
        assert len(model.callbacks) == 1

    def test_custom_provider_gets_event_callback_only(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)

        class FakeCustom:
            def __init__(self):
                self.cb = None

            def set_usage_event_callback(self, cb):
                self.cb = cb

        model = FakeCustom()
        _attach_token_stats(
            model,
            provider="claude-code-sdk",
            model_name="sonnet",
            kind="agent",
            stats_callback_factory=self._handler_factory(rec),
            usage_event_callback_factory=self._event_factory(rec),
        )
        # Custom providers NEVER receive the LangChain handler (structural
        # single source); only set_usage_event_callback is installed.
        assert not hasattr(model, "callbacks")
        assert isinstance(model.cb, UsageEventCallback)

    def test_unknown_provider_with_factory_is_noop(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        model = SimpleNamespace(callbacks=[])
        _attach_token_stats(
            model,
            provider="unknown",
            model_name="x",
            kind="agent",
            stats_callback_factory=self._handler_factory(rec),
            usage_event_callback_factory=self._event_factory(rec),
        )
        assert model.callbacks == []
