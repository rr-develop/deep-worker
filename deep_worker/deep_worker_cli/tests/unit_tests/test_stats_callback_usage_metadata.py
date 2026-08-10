"""Extraction seam guard: ``message.usage_metadata`` → record with tokens.

Incident 2026-08-03: ``token_stats.jsonl`` held 196
records, ALL ``status="no_usage"`` with null tokens. Root cause was a parser
gap — the streaming path hands the callback an ``AIMessage`` whose
``response_metadata`` carries NO ``token_usage`` at all; the usage lives only
in the cross-provider aggregate ``AIMessage.usage_metadata``
(``input_tokens``/``output_tokens`` + ``input_token_details.cache_read``), and
``normalize_usage`` did not understand that shape, so every record degraded.

``TestNormalizeUsage`` pins the parser alone. These tests pin the whole seam
``LLMResult → _extract_usage → normalize_usage → recorder`` with a REAL
``TokenUsageRecorder`` (no mocks), so a regression anywhere along it fails
here — including a future change that only fixes the parser but breaks
extraction, or vice versa.

The payloads are verbatim from live DeepSeek ``astream`` probes (2026-08-03,
deepseek-v4-flash, with and without tool calls).
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_worker_cli.stats.callback import TokenStatsCallbackHandler
from deep_worker_cli.stats.token_stats import STATUS_OK, TokenUsageRecorder

# Verbatim live payload: streamed turn WITH tool calls (deepseek-v4-flash).
_LIVE_STREAM_USAGE_METADATA = {
    "input_tokens": 359,
    "output_tokens": 66,
    "total_tokens": 425,
    "input_token_details": {"cache_read": 256},
    "output_token_details": {"reasoning": 21},
}


def _streamed_llm_result(usage_metadata: dict | None) -> LLMResult:
    """LLMResult exactly as the streaming path produces it.

    ``response_metadata`` deliberately carries NO ``token_usage`` — that is the
    whole point of the regression: only ``usage_metadata`` is populated.
    """
    message = AIMessage(content="OK")
    message.response_metadata = {
        "model_provider": "deepseek",
        "finish_reason": "stop",
        "model_name": "deepseek-v4-flash",
    }
    if usage_metadata is not None:
        message.usage_metadata = usage_metadata  # type: ignore[assignment]
    return LLMResult(generations=[[ChatGeneration(message=message)]])


async def _run_turn(recorder: TokenUsageRecorder, result: LLMResult) -> dict:
    handler = TokenStatsCallbackHandler(
        recorder, provider="deepseek", kind="agent", model_name="deepseek-v4-flash"
    )
    run_id = uuid4()
    await handler.on_llm_start({}, ["prompt"], run_id=run_id)
    await handler.on_llm_end(result, run_id=run_id)
    lines = recorder.path.read_text().strip().split("\n")
    assert len(lines) == 1, f"expected exactly one record, got: {lines}"
    return json.loads(lines[0])


class TestStreamingUsageMetadataSeam:
    @pytest.mark.asyncio
    async def test_streamed_usage_metadata_is_recorded_with_tokens(self, tmp_path):
        """The exact shape that produced 196/196 ``no_usage`` must record ok."""
        recorder = TokenUsageRecorder(tmp_path)
        rec = await _run_turn(
            recorder, _streamed_llm_result(_LIVE_STREAM_USAGE_METADATA)
        )

        assert rec["status"] == STATUS_OK, (
            "record degraded — the usage_metadata→record seam is broken"
        )
        assert rec["prompt_tokens"] == 359
        assert rec["completion_tokens"] == 66
        assert rec["total_tokens"] == 425
        assert rec["cache_hit_tokens"] == 256
        assert rec["cache_miss_tokens"] == 103
        assert rec["cache_hit_tokens"] + rec["cache_miss_tokens"] == rec["prompt_tokens"]
        assert rec["provider"] == "deepseek"
        assert rec["model"] == "deepseek-v4-flash"
        assert rec["kind"] == "agent"

    @pytest.mark.asyncio
    async def test_streamed_usage_metadata_without_cache_details(self, tmp_path):
        """No ``input_token_details`` → tokens recorded, cache stays null."""
        recorder = TokenUsageRecorder(tmp_path)
        rec = await _run_turn(
            recorder,
            _streamed_llm_result(
                {"input_tokens": 85, "output_tokens": 16, "total_tokens": 101}
            ),
        )

        assert rec["status"] == STATUS_OK
        assert rec["prompt_tokens"] == 85
        assert rec["completion_tokens"] == 16
        assert rec["cache_hit_tokens"] is None
        assert rec["cache_miss_tokens"] is None

    @pytest.mark.asyncio
    async def test_token_usage_shape_still_recorded(self, tmp_path):
        """Non-streaming ``response_metadata.token_usage`` keeps working."""
        message = AIMessage(content="OK")
        message.response_metadata = {
            "token_usage": {
                "prompt_tokens": 347,
                "completion_tokens": 26,
                "total_tokens": 373,
                "prompt_cache_hit_tokens": 256,
                "prompt_cache_miss_tokens": 91,
            }
        }
        recorder = TokenUsageRecorder(tmp_path)
        rec = await _run_turn(
            recorder, LLMResult(generations=[[ChatGeneration(message=message)]])
        )

        assert rec["status"] == STATUS_OK
        assert rec["prompt_tokens"] == 347
        assert rec["cache_hit_tokens"] == 256
        assert rec["cache_miss_tokens"] == 91

    @pytest.mark.asyncio
    async def test_no_usage_at_all_still_degrades(self, tmp_path):
        """A turn that truly carries no usage must stay ``no_usage``."""
        recorder = TokenUsageRecorder(tmp_path)
        rec = await _run_turn(recorder, _streamed_llm_result(None))

        assert rec["status"] == "no_usage"
        assert rec["prompt_tokens"] is None
        assert rec["total_tokens"] is None
