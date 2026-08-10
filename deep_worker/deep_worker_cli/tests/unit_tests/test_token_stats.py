"""Tests for the stats package: schema, normalize, recorder, aggregator.

Covers DESIGN_AGENT_TOKEN_STATS.md §5.2/§5.3/§4.2/§4.3/§6.1:
- normalize_usage per provider format (incl. canonical Anthropic prompt and
  the hit+miss=prompt invariant).
- Recorder append contract (0600 perms, O_APPEND, short-write, lock).
- UsageEventCallback state machine (at-most-once, event_key rules,
  failed/no_usage/snapshot, in-process seen-set).
- Reader: skipped_records / duplicate_records / partial_tail, missing file.
- Aggregator: ratios (observed/lower/upper/coverage), null-on-zero-division,
  context snapshots (latest by ts, max), totals + by_provider + by_model.
"""

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from deep_worker_cli.stats.token_stats import (
    CUSTOM_PROVIDERS,
    SCHEMA_VERSION,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_NO_USAGE,
    STATUS_OK,
    USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
    USAGE_SEMANTICS_TOKENS,
    TokenUsageRecorder,
    UsageEventCallback,
    UsageRecord,
    aggregate_records,
    aggregate_stats,
    build_record,
    normalize_usage,
    read_stats_file,
    stats_for_agent,
)


def _rec(*, provider="deepseek", model="m", kind="agent", status=STATUS_OK,
         prompt=None, completion=None, hit=None, miss=None, total=None,
         ts="2026-08-01T12:00:00.000Z", event_key=None, session_id=None,
         context_used=None, context_size=None, semantics=None) -> UsageRecord:
    if semantics is None:
        semantics = (
            USAGE_SEMANTICS_CONTEXT_SNAPSHOT
            if context_used is not None or context_size is not None
            else USAGE_SEMANTICS_TOKENS
        )
    return UsageRecord(
        provider=provider, model=model, kind=kind, status=status,
        usage_semantics=semantics, ts=ts, event_key=event_key,
        provider_session_id=session_id, prompt_tokens=prompt,
        completion_tokens=completion, cache_hit_tokens=hit,
        cache_miss_tokens=miss, total_tokens=total,
        context_used=context_used, context_size=context_size,
    )


class TestNormalizeUsage:
    def test_anthropic_canonical_prompt(self):
        u = normalize_usage("anthropic", {
            "input_tokens": 300,
            "cache_read_input_tokens": 2500,
            "cache_creation_input_tokens": 200,
            "output_tokens": 100,
        })
        assert u is not None
        assert u.prompt_tokens == 3000  # input + read + creation
        assert u.completion_tokens == 100
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens == 500  # input + creation
        assert u.total_tokens == 3100
        # invariant hit + miss = prompt
        assert u.cache_hit_tokens + u.cache_miss_tokens == u.prompt_tokens

    def test_claude_code_uses_anthropic_format(self):
        # Raw ResultMessage.usage (live probe shape).
        u = normalize_usage("claude-code", {
            "input_tokens": 2,
            "cache_creation_input_tokens": 8200,
            "cache_read_input_tokens": 20399,
            "output_tokens": 5,
            "iterations": [],
        })
        assert u is not None
        assert u.prompt_tokens == 28601
        assert u.cache_hit_tokens == 20399
        assert u.cache_miss_tokens == 8202
        assert u.cache_hit_tokens + u.cache_miss_tokens == u.prompt_tokens

    def test_langchain_normalized_deepseek(self):
        u = normalize_usage("deepseek", {
            "prompt_tokens": 3000,
            "prompt_cache_hit_tokens": 2500,
            "prompt_cache_miss_tokens": 500,
            "completion_tokens": 100,
            "total_tokens": 3100,
        })
        assert u is not None
        assert u.prompt_tokens == 3000
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens == 500
        assert u.total_tokens == 3100

    def test_openai_cached_tokens_fallback(self):
        u = normalize_usage("openai", {
            "prompt_tokens": 3000,
            "completion_tokens": 100,
            "total_tokens": 3100,
            "prompt_tokens_details": {"cached_tokens": 2500},
        })
        assert u is not None
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens == 500

    def test_openai_no_cache(self):
        u = normalize_usage("openai", {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        })
        assert u is not None
        assert u.cache_hit_tokens is None
        assert u.cache_miss_tokens is None

    def test_codex_cached_input(self):
        u = normalize_usage("codex", {
            "input_tokens": 1000,
            "output_tokens": 50,
            "cached_input_tokens": 600,
        })
        assert u is not None
        assert u.prompt_tokens == 1000
        assert u.cache_hit_tokens == 600
        assert u.cache_miss_tokens == 400
        assert u.total_tokens == 1050

    def test_gemini_usage_metadata(self):
        u = normalize_usage("google", {
            "prompt_token_count": 3000,
            "candidates_token_count": 100,
            "total_token_count": 3100,
            "cached_content_token_count": 2500,
        })
        assert u is not None
        assert u.prompt_tokens == 3000
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens == 500

    def test_langchain_usage_metadata_streaming_shape(self):
        """Streaming path yields ``usage_metadata``, not ``token_usage``.

        Verbatim payload captured live (2026-08-03, deepseek-chat, real
        ``astream``). ``response_metadata`` carries no ``token_usage`` at all
        on the streamed AIMessage, so the aggregate ``usage_metadata`` shape
        (``input_tokens``/``output_tokens`` with
        ``input_token_details.cache_read``) MUST be understood — otherwise
        usage arriving in that shape is dropped and the record degrades to
        ``no_usage``.
        """
        u = normalize_usage("deepseek", {
            "input_tokens": 6,
            "output_tokens": 1,
            "total_tokens": 7,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {},
        })
        assert u is not None
        assert u.prompt_tokens == 6
        assert u.completion_tokens == 1
        assert u.total_tokens == 7
        assert u.cache_hit_tokens == 0
        assert u.cache_miss_tokens == 6

    def test_langchain_usage_metadata_with_cache_read(self):
        """Warm prefix cache: cache_read splits prompt into hit/miss."""
        u = normalize_usage("deepseek", {
            "input_tokens": 3000,
            "output_tokens": 100,
            "total_tokens": 3100,
            "input_token_details": {"cache_read": 2500},
        })
        assert u is not None
        assert u.prompt_tokens == 3000
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens == 500
        assert u.cache_hit_tokens + u.cache_miss_tokens == u.prompt_tokens

    def test_langchain_usage_metadata_without_cache_details(self):
        """No cache details → no invented cache split."""
        u = normalize_usage("openai", {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        })
        assert u is not None
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 20
        assert u.cache_hit_tokens is None
        assert u.cache_miss_tokens is None

    def test_token_usage_shape_wins_over_usage_metadata_keys(self):
        """A dict carrying both shapes uses the explicit prompt/completion keys."""
        u = normalize_usage("deepseek", {
            "prompt_tokens": 3000,
            "completion_tokens": 100,
            "total_tokens": 3100,
            "prompt_cache_hit_tokens": 2500,
            "prompt_cache_miss_tokens": 500,
            "input_tokens": 999,
            "output_tokens": 999,
        })
        assert u is not None
        assert u.prompt_tokens == 3000
        assert u.completion_tokens == 100
        assert u.cache_hit_tokens == 2500

    def test_unknown_or_empty(self):
        assert normalize_usage("deepseek", None) is None
        assert normalize_usage("deepseek", {}) is None
        assert normalize_usage("anthropic", {"foo": 1}) is None
        assert normalize_usage("deepseek", "not-a-dict") is None

    def test_generic_none_provider_preserves_single_cache_field(self):
        # Byte-parity with legacy log_token_usage: hit present, miss absent →
        # miss must stay None (log line prints "miss=0"), NOT invented.
        u = normalize_usage(None, {
            "prompt_tokens": 3000,
            "prompt_cache_hit_tokens": 2500,
            "completion_tokens": 100,
            "total_tokens": 3100,
        })
        assert u is not None
        assert u.cache_hit_tokens == 2500
        assert u.cache_miss_tokens is None


class TestRecorder:
    @pytest.mark.asyncio
    async def test_append_creates_0600_file(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        await rec.record(_rec(prompt=100, completion=10, total=110))
        path = tmp_path / "token_stats.jsonl"
        assert path.exists()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600
        assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700
        data = json.loads(path.read_text().strip())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_append_serialized_with_lock(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        async def _w(i: int) -> None:
            await rec.record(_rec(prompt=i, completion=0, total=i))
        await asyncio.gather(*(_w(i) for i in range(50)))
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 50
        prompts = sorted(json.loads(l)["prompt_tokens"] for l in lines)
        assert prompts == list(range(50))

    @pytest.mark.asyncio
    async def test_append_repairs_existing_file_mode(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        path.write_text("")
        os.chmod(path, 0o644)
        rec = TokenUsageRecorder(tmp_path)
        await rec.record(_rec(prompt=1, completion=0, total=1))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_short_write_detected(self, tmp_path, monkeypatch):
        rec = TokenUsageRecorder(tmp_path)
        real_write = os.write
        calls = {"n": 0}

        def fake_write(fd, data):
            calls["n"] += 1
            if calls["n"] == 1:
                # Real partial write: actually persist the first N-1 bytes
                # (a corrupted tail) and report the short count.
                return real_write(fd, data[:-1])
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", fake_write)
        with pytest.raises(OSError):
            asyncio.run(rec.record(_rec(prompt=7, completion=0, total=7)))
        # The partial line (missing its trailing newline) is what a real
        # short write leaves behind; reader flags it as partial_tail.
        content = (tmp_path / "token_stats.jsonl").read_bytes()
        assert content and not content.endswith(b"\n")


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_state_machine_writes_once_per_terminal(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="claude-code", kind="agent")
        await cb(event="invocation_start")
        await cb(event="terminal", status=STATUS_OK, usage={
            "input_tokens": 2, "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 10, "output_tokens": 5,
        }, session_id="S1", uuid="U1", model="sonnet")
        # Defensive duplicate terminal for the same event — must not write twice.
        await cb(event="terminal", status=STATUS_OK, usage={
            "input_tokens": 2, "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 10, "output_tokens": 5,
        }, session_id="S1", uuid="U1", model="sonnet")
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_key"] == "claude-code|S1|U1"
        assert data["provider_session_id"] == "S1"
        assert data["status"] == STATUS_OK
        assert data["prompt_tokens"] == 112  # 2 + 100 + 10
        assert data["cache_hit_tokens"] == 100
        assert not cb.pending

    @pytest.mark.asyncio
    async def test_no_usage_terminal(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="codex", kind="tm")
        await cb(event="terminal", status=STATUS_OK, usage={},
                 session_id="T1", turn_id="M1", model="gpt-5")
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["status"] == STATUS_NO_USAGE
        assert data["event_key"] == "codex|T1|M1"
        assert data["prompt_tokens"] is None

    @pytest.mark.asyncio
    async def test_failed_terminal_writes_failed(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="codex", kind="agent")
        await cb(event="invocation_start")
        await cb(event="terminal", status=STATUS_FAILED,
                 session_id="T1", turn_id="M1", model="gpt-5")
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["status"] == STATUS_FAILED

    @pytest.mark.asyncio
    async def test_grok_snapshot_never_deduped(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="grok", kind="agent")
        await cb(event="terminal", status=STATUS_OK, model="grok-x",
                 context_used=5000, context_size=10000)
        await cb(event="terminal", status=STATUS_OK, model="grok-x",
                 context_used=6000, context_size=10000)
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert data["event_key"] is None
            assert data["usage_semantics"] == USAGE_SEMANTICS_CONTEXT_SNAPSHOT
            assert data["context_size"] == 10000
        # Token aggregates must exclude snapshots entirely (see aggregator tests).

    @pytest.mark.asyncio
    async def test_cancelled_propagation_and_no_write(self, tmp_path, monkeypatch):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="claude-code", kind="agent")
        await cb(event="invocation_start")

        async def boom(record):
            raise asyncio.CancelledError()

        monkeypatch.setattr(rec, "record", boom)
        with pytest.raises(asyncio.CancelledError):
            await cb(event="terminal", status=STATUS_OK, usage={
                "input_tokens": 1, "output_tokens": 1,
            }, session_id="S", uuid="U", model="m")
        # Cancellation of the provider turn (no terminal event delivered) →
        # nothing written.
        path = tmp_path / "token_stats.jsonl"
        assert not path.exists() or path.read_text() == ""

    @pytest.mark.asyncio
    async def test_errors_are_logged_not_raised(self, tmp_path, monkeypatch):
        rec = TokenUsageRecorder(tmp_path)

        def boom(line):
            raise OSError("disk full")

        monkeypatch.setattr(rec, "_append_line", boom)
        cb = UsageEventCallback(rec, provider="grok", kind="agent")
        # Must not raise — best-effort contract.
        await cb(event="terminal", status=STATUS_OK, model="grok-x",
                 context_used=1, context_size=2)

    @pytest.mark.asyncio
    async def test_claude_uuid_none_never_deduped(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        cb = UsageEventCallback(rec, provider="claude-code", kind="agent")
        await cb(event="terminal", status=STATUS_OK, usage={
            "input_tokens": 1, "output_tokens": 1,
        }, session_id="S", uuid=None, model="m")
        await cb(event="terminal", status=STATUS_OK, usage={
            "input_tokens": 1, "output_tokens": 1,
        }, session_id="S", uuid=None, model="m")
        lines = (tmp_path / "token_stats.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2  # event_key=null → both unique, both kept
        for line in lines:
            assert json.loads(line)["event_key"] is None
            assert json.loads(line)["provider_session_id"] == "S"


class TestReader:
    def test_missing_file(self, tmp_path):
        result = read_stats_file(tmp_path / "nope.jsonl")
        assert result.records == []
        assert result.skipped_records == 0
        assert result.duplicate_records == 0
        assert result.partial_tail is False

    def test_skips_malformed_and_unknown_version(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        good = _rec(prompt=1, completion=1, total=2)
        path.write_text(
            "not-json\n"
            + json.dumps({"schema_version": 99, "provider": "x"}) + "\n"
            + json.dumps({"schema_version": 1, "provider": 5}) + "\n"
            + json.dumps(good.to_dict()) + "\n"
        )
        result = read_stats_file(path)
        assert len(result.records) == 1
        assert result.skipped_records == 3
        assert result.partial_tail is False

    def test_partial_tail(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        path.write_text(json.dumps(_rec(prompt=1, completion=1, total=2).to_dict()))
        result = read_stats_file(path)
        assert result.partial_tail is True
        assert result.records == []

    def test_duplicate_event_key(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        rec = _rec(prompt=1, completion=1, total=2, event_key="langchain|run1")
        path.write_text(
            json.dumps(rec.to_dict()) + "\n" + json.dumps(rec.to_dict()) + "\n"
        )
        result = read_stats_file(path)
        assert len(result.records) == 1
        assert result.duplicate_records == 1

    def test_null_event_key_never_deduped(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        rec1 = _rec(prompt=1, completion=1, total=2)
        rec2 = _rec(prompt=2, completion=2, total=4)
        path.write_text(
            json.dumps(rec1.to_dict()) + "\n" + json.dumps(rec2.to_dict()) + "\n"
        )
        result = read_stats_file(path)
        assert len(result.records) == 2
        assert result.duplicate_records == 0

    def test_roundtrip_through_recorder_and_reader(self, tmp_path):
        rec = TokenUsageRecorder(tmp_path)
        asyncio.run(rec.record(_rec(
            provider="deepseek", model="deepseek-v4-flash", kind="agent",
            prompt=3000, completion=100, hit=2500, miss=500, total=3100,
            event_key="langchain|run1",
        )))
        result = read_stats_file(tmp_path / "token_stats.jsonl")
        assert len(result.records) == 1
        r = result.records[0]
        assert r.prompt_tokens == 3000
        assert r.cache_hit_tokens == 2500
        assert r.event_key == "langchain|run1"


class TestAggregator:
    def test_empty(self):
        t = aggregate_records([])
        assert t.requests == 0
        assert t.requests_with_usage == 0
        assert t.usage_coverage is None
        assert t.cache_ratio_total_observed is None
        assert t.cache_ratio_total_lower_bound is None
        assert t.cache_ratio_total_upper_bound is None
        assert t.cache_ratio_prompt_observed is None
        assert t.cache_coverage is None
        assert t.avg_tokens_per_usage_observed_request is None
        assert t.has_cache is False

    def test_ratios_math(self):
        # 3 records with cache; 1 record without cache (P_unknown=1000).
        records = [
            _rec(prompt=3000, completion=100, hit=2500, miss=500, total=3100),
            _rec(prompt=2000, completion=50, hit=1500, miss=500, total=2050),
            _rec(prompt=1000, completion=25, hit=1000, miss=0, total=1025),
            _rec(prompt=1000, completion=10, total=1010),  # no cache fields
        ]
        t = aggregate_records(records)
        assert t.requests == 4
        assert t.requests_with_usage == 4
        assert t.prompt_tokens == 7000
        assert t.total_tokens == 7185
        assert t.cache_hit_tokens == 5000
        assert t.cache_miss_tokens == 1000
        assert t.n_cache_observed == 3
        assert t.has_cache is True
        # H_known=5000, T_known=3100+2050+1025=6175
        assert t.cache_ratio_total_observed == pytest.approx(5000 / 6175)
        # lower bound: H_known / T_all = 5000/7185
        assert t.cache_ratio_total_lower_bound == pytest.approx(5000 / 7185)
        # upper bound: (H_known + P_unknown)/T_all = (5000+1000)/7185
        assert t.cache_ratio_total_upper_bound == pytest.approx(6000 / 7185)
        # prompt observed: 5000 / (3000+2000+1000) = 5000/6000
        assert t.cache_ratio_prompt_observed == pytest.approx(5000 / 6000)
        # coverage: P_known/P_all = 6000/7000
        assert t.cache_coverage == pytest.approx(6000 / 7000)
        assert t.usage_coverage == pytest.approx(1.0)
        # avg request = Σ prompt / requests_with_usage = 7000/4 (NOT total)
        assert t.avg_tokens_per_usage_observed_request == pytest.approx(7000 / 4)
        assert t.avg_tokens_per_usage_observed_response == pytest.approx(185 / 4)

    def test_requests_includes_no_usage_and_failed(self):
        records = [
            _rec(prompt=100, completion=10, hit=50, miss=50, total=110),
            _rec(status=STATUS_NO_USAGE),
            _rec(status=STATUS_FAILED),
            _rec(status=STATUS_CANCELLED),
        ]
        t = aggregate_records(records)
        assert t.requests == 4
        assert t.requests_with_usage == 1
        assert t.total_tokens == 110
        assert t.usage_coverage == pytest.approx(1 / 4)

    def test_zero_division_null(self):
        # no_usage records only → token ratios None, but usage_coverage is
        # a real 0.0 (requests > 0), NOT null (null only when requests == 0).
        t = aggregate_records([_rec(status=STATUS_NO_USAGE), _rec(status=STATUS_FAILED)])
        assert t.requests == 2
        assert t.requests_with_usage == 0
        assert t.usage_coverage == 0.0
        assert t.cache_ratio_total_observed is None
        assert t.cache_coverage is None
        assert t.avg_tokens_per_usage_observed_request is None
        assert t.cache_hit_tokens == 0

    def test_snapshot_only_requests_coverage_zero(self):
        # context snapshots are requests without token usage (grok fallback):
        # they count towards requests, so usage_coverage is 0.0, not null.
        t = aggregate_records([
            _rec(context_used=5000, context_size=10000, ts="2026-08-01T12:00:00.000Z"),
            _rec(context_used=6000, context_size=10000, ts="2026-08-01T12:00:01.000Z"),
        ])
        assert t.requests == 2
        assert t.requests_with_usage == 0
        assert t.usage_coverage == 0.0
        assert t.avg_tokens_per_usage_observed_request is None
        assert t.context.snapshots == 2

    def test_context_snapshots_latest_and_max(self):
        records = [
            _rec(context_used=5000, context_size=10000, ts="2026-08-01T12:00:00.000Z"),
            _rec(context_used=6000, context_size=10000, ts="2026-08-01T12:00:01.000Z"),
            _rec(context_used=9000, context_size=20000, ts="2026-08-01T11:00:00.000Z"),
        ]
        t = aggregate_records(records)
        assert t.context.snapshots == 3
        assert t.context.latest_context_used == 6000
        assert t.context.latest_context_size == 10000
        assert t.context.max_context_used == 9000
        assert t.context.max_context_size == 20000
        # token aggregates untouched by snapshots
        assert t.requests == 3
        assert t.requests_with_usage == 0
        assert t.prompt_tokens == 0

    def test_by_provider_and_by_model(self):
        records = [
            _rec(provider="deepseek", model="deepseek-v4-flash",
                 prompt=100, completion=10, hit=50, miss=50, total=110),
            _rec(provider="deepseek", model="deepseek-v4-flash",
                 prompt=200, completion=20, total=220),
            _rec(provider="claude-code", model="sonnet",
                 prompt=300, completion=30, hit=100, miss=200, total=330),
            _rec(provider="claude-code", model="haiku",
                 prompt=50, completion=5, total=55),
        ]
        resp = aggregate_stats(records, agent="alice")
        assert resp.totals.requests == 4
        assert [p.provider for p in resp.by_provider] == ["claude-code", "deepseek"]
        deepseek = next(p for p in resp.by_provider if p.provider == "deepseek")
        assert deepseek.totals.requests == 2
        assert [m.model for m in deepseek.by_model] == ["deepseek-v4-flash"]
        assert deepseek.by_model[0].totals.prompt_tokens == 300
        claude = next(p for p in resp.by_provider if p.provider == "claude-code")
        assert [m.model for m in claude.by_model] == ["haiku", "sonnet"]

    def test_stats_for_agent_missing_file_zeros(self, tmp_path):
        resp = stats_for_agent("alice", tmp_path)
        assert resp.agent == "alice"
        assert resp.totals.requests == 0
        assert resp.by_provider == []
        assert resp.skipped_records == 0
        # no file created
        assert not (tmp_path / "token_stats.jsonl").exists()

    def test_stats_for_agent_wires_reader_meta(self, tmp_path):
        path = tmp_path / "token_stats.jsonl"
        # first line malformed (skipped), trailing line without \n (partial tail)
        path.write_text("garbage\ntruncated")
        resp = stats_for_agent("alice", tmp_path)
        assert resp.skipped_records == 1
        assert resp.partial_tail is True

    def test_build_record_defaults(self):
        r = build_record(provider="deepseek", model="m", kind="agent", status=STATUS_OK)
        assert r.usage_semantics == USAGE_SEMANTICS_TOKENS
        assert r.prompt_tokens is None
        assert r.record_id

    def test_custom_providers_set(self):
        assert CUSTOM_PROVIDERS == {"claude-code", "codex", "grok"}
