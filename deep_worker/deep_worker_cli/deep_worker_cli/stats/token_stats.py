"""Token/cache usage statistics: schema, normalization, recorder, reader, aggregator.

Single source of truth (DESIGN_AGENT_TOKEN_STATS.md §5.1, MAJOR-1):
``deep_worker_cli`` writes records into
``<AgentProcess.data_dir>/token_stats.jsonl`` and both ``deep_worker_cli`` and
``deep_worker_web`` aggregate them with the functions defined here.

The module is deliberately **dependency-free** (stdlib only) so that
``deep_worker_web`` can import it without pulling in langchain — same
convention as ``deep_worker_cli.model_predicate``.

Key contracts (see design §§4-6):

- One JSONL line = one unit of account (schema_version 1).
- ``event_key`` (non-null) is the stable dedup key; ``null`` records are
  always unique. Aggregation dedups by ``event_key`` and counts duplicates
  separately in ``duplicate_records``.
- Append path: ``os.open(O_WRONLY|O_APPEND|O_CREAT)`` + ``os.fchmod(0600)`` +
  one ``os.write`` with short-write detection, serialized per-process by an
  ``asyncio.Lock`` and executed in a worker thread via ``asyncio.to_thread``
  (no extra threads are created). No fsync — documented crash semantics below.
- Custom providers (claude-code-sdk/codex/grok) drive the recorder through the
  async ``UsageEventCallback`` state machine (§5.2): a terminal event is
  written **at most once** per observed invocation; errors are logged and
  never raised (best-effort), except ``asyncio.CancelledError`` which
  propagates (provider turn cancellation).

Crash semantics: a crash between a terminal event arriving and our append =
one lost record (at-most-once, documented). A crash after the append = the
record is already on disk and the event will not be replayed → no duplicate.
No fsync is performed; on OS crash the last appends may be lost (acceptable
for statistics).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

STATS_FILE_NAME = "token_stats.jsonl"

# LangChain-backed providers: usage flows through TokenStatsCallbackHandler
# attached to the model instance in create_model (§5.1).
STANDARD_PROVIDERS = frozenset({"deepseek", "openai", "anthropic", "google", "vertexai"})

# Custom providers: usage flows through the async UsageEventCallback state
# machine (§5.2) installed via provider.set_usage_event_callback().
CUSTOM_PROVIDERS = frozenset({"claude-code-sdk", "claude-code", "codex", "grok"})


def is_standard_provider(provider: str) -> bool:
    """True for providers whose usage flows through the LangChain callback.

    Covers the built-in ``STANDARD_PROVIDERS`` plus any provider declared in
    ``~/.deep_worker/providers.json``: those are constructed as LangChain
    chat models (OpenAI-compatible dialect), so their usage arrives through
    the same handler. Without this, a config-added provider would silently
    record no token statistics.

    ``CUSTOM_PROVIDERS`` (CLI/ACP) keep their own event-callback path and are
    never treated as standard.
    """
    if provider in STANDARD_PROVIDERS:
        return True
    if provider in CUSTOM_PROVIDERS:
        return False
    try:
        from deep_worker_cli.runtime.provider_registry import load_provider_specs

        return any(spec.name == provider for spec in load_provider_specs())
    except Exception:  # noqa: BLE001 — statistics must never break a turn
        return False

STATUS_OK = "ok"
STATUS_NO_USAGE = "no_usage"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

USAGE_SEMANTICS_TOKENS = "tokens"
USAGE_SEMANTICS_CONTEXT_SNAPSHOT = "context_snapshot"

KIND_AGENT = "agent"
KIND_TM = "tm"
KIND_OTHER = "other"

_KNOWN_STATUSES = frozenset({STATUS_OK, STATUS_NO_USAGE, STATUS_FAILED, STATUS_CANCELLED})
_KNOWN_SEMANTICS = frozenset({USAGE_SEMANTICS_TOKENS, USAGE_SEMANTICS_CONTEXT_SNAPSHOT})
_KNOWN_KINDS = frozenset({KIND_AGENT, KIND_TM, KIND_OTHER})

_UTC = UTC


# ---------------------------------------------------------------------------
# Normalization (§5.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedUsage:
    """Provider-agnostic token/cache usage (§3)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _from_anthropic(usage: dict[str, Any]) -> NormalizedUsage | None:
    """Raw Anthropic/Claude usage (input/cache_read/cache_creation/output).

    ⚠ DO NOT UNIFY with ``_from_codex``. The miss formula differs because
    ``input_tokens`` means different things: here it EXCLUDES the cached
    portions, so prompt = ``input + cache_read + cache_creation`` and miss =
    ``input + cache_creation``. In Codex ``input_tokens`` INCLUDES the cached
    part, so miss = ``input - cached``. The two look mergeable and are not —
    merging silently corrupts one provider's cache accounting.
    """
    inp = _to_int(usage.get("input_tokens"))
    out = _to_int(usage.get("output_tokens"))
    if inp is None and out is None:
        return None
    inp = inp or 0
    out = out or 0
    read = _to_int(usage.get("cache_read_input_tokens"))
    creation = _to_int(usage.get("cache_creation_input_tokens"))
    hit = read
    miss = inp + (creation or 0)
    prompt = inp + (read or 0) + (creation or 0)
    total = _to_int(usage.get("total_tokens"))
    if total is None:
        total = prompt + out
    return NormalizedUsage(
        prompt_tokens=prompt,
        completion_tokens=out,
        total_tokens=total,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _from_codex(usage: dict[str, Any]) -> NormalizedUsage | None:
    """Codex turn usage (input/output/cached_input).

    ⚠ DO NOT UNIFY with ``_from_anthropic``. ``input_tokens`` here INCLUDES
    the cached prefix, so prompt = ``input`` and miss = ``input - cached``;
    Anthropic's excludes it and adds the cache fields on top. See the note
    there.

    ⚠ This normaliser assumes it is handed THIS TURN's usage. It cannot
    detect the exec-process TOTAL — feeding it ``turn.completed.usage``
    directly produced a live ``prompt=3736695`` row against a real spend of
    ``113457``. The per-turn source is selected upstream by
    ``last_token_usage_from_rollout`` (``DESIGN_PER_TURN_USAGE_MAPPING.md``
    §2.1-ter); do not "fix" inflation by adding arithmetic here.
    """
    inp = _to_int(usage.get("input_tokens"))
    out = _to_int(usage.get("output_tokens"))
    if inp is None and out is None:
        return None
    inp = inp or 0
    out = out or 0
    cached = _to_int(usage.get("cached_input_tokens"))
    hit = cached
    miss = inp - cached if cached is not None else None
    total = _to_int(usage.get("total_tokens"))
    if total is None:
        total = inp + out
    return NormalizedUsage(
        prompt_tokens=inp,
        completion_tokens=out,
        total_tokens=total,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _from_google(usage: dict[str, Any]) -> NormalizedUsage | None:
    """Gemini usage_metadata (prompt/candidates/total/cached_content counts)."""
    prompt = _to_int(usage.get("prompt_token_count"))
    completion = _to_int(usage.get("candidates_token_count"))
    if prompt is None and completion is None:
        return None
    prompt = prompt or 0
    completion = completion or 0
    cached = _to_int(usage.get("cached_content_token_count"))
    hit = cached
    miss = prompt - cached if cached is not None else None
    total = _to_int(usage.get("total_token_count"))
    if total is None:
        total = prompt + completion
    return NormalizedUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _from_langchain_usage_metadata(usage: dict[str, Any]) -> NormalizedUsage | None:
    """LangChain aggregate ``usage_metadata`` (the STREAMING shape).

    On the streaming path (live-verified 2026-08-03, deepseek-chat and
    deepseek-v4-flash via ``astream``, with and without tool calls) the final
    AIMessage carries NO ``response_metadata.token_usage`` at all — only the
    cross-provider ``usage_metadata``: ``input_tokens`` / ``output_tokens`` /
    ``total_tokens`` with an optional ``input_token_details.cache_read``.
    ``_from_langchain_normalized`` only understands the non-streaming
    ``token_usage`` shape, so without this parser any usage that arrives in
    the streaming shape is dropped and the record degrades to ``no_usage``.

    Cache semantics match the rest of §4.1: ``cache_read`` is the hit count and
    the miss is the remainder of the prompt, so ``hit + miss == prompt`` holds.
    """
    prompt = _to_int(usage.get("input_tokens"))
    completion = _to_int(usage.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    prompt = prompt or 0
    completion = completion or 0
    hit: int | None = None
    miss: int | None = None
    details = usage.get("input_token_details")
    if isinstance(details, dict):
        cached = _to_int(details.get("cache_read"))
        if cached is not None:
            hit = cached
            miss = prompt - cached if prompt else 0
    total = _to_int(usage.get("total_tokens"))
    if total is None:
        total = prompt + completion
    return NormalizedUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _from_langchain_normalized(usage: dict[str, Any]) -> NormalizedUsage | None:
    """LangChain-normalized ``token_usage`` (deepseek/openai/generic).

    Preserves the exact semantics of the legacy ``log_token_usage`` parser
    (utils.py): top-level ``prompt_cache_hit_tokens``/``prompt_cache_miss_tokens``
    first, falling back to ``prompt_tokens_details.cached_tokens``. When only
    one cache field is present it is kept as-is (no invented counterpart) —
    byte-level parity for the log line.
    """
    prompt = _to_int(usage.get("prompt_tokens"))
    completion = _to_int(usage.get("completion_tokens"))
    if prompt is None and completion is None:
        return _from_langchain_usage_metadata(usage)
    prompt = prompt or 0
    completion = completion or 0
    hit = _to_int(usage.get("prompt_cache_hit_tokens"))
    miss = _to_int(usage.get("prompt_cache_miss_tokens"))
    if hit is None and miss is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = _to_int(details.get("cached_tokens"))
            if cached is not None:
                hit = cached
                miss = prompt - cached if prompt else 0
    total = _to_int(usage.get("total_tokens"))
    if total is None:
        total = prompt + completion
    return NormalizedUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def normalize_usage(provider: str | None, usage: Any) -> NormalizedUsage | None:
    """Normalize a provider usage dict into :class:`NormalizedUsage`.

    Args:
        provider: Provider name, or ``None`` for the generic
            LangChain-normalized ``token_usage`` shape (used by
            ``log_token_usage`` for byte-level parity).
        usage: Plain dict with provider usage. Anything else (``None``, empty
            dict, unknown schema) yields ``None`` (recorded as ``no_usage``).

    Returns:
        Normalized usage, or ``None`` when the dict carries no recognizable
        token fields.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    if provider in ("anthropic", "claude-code-sdk"):
        return _from_anthropic(usage)
    if provider in ("codex", "grok", "claude-code"):
        # grok (live-verified 0.2.73): session/prompt RESULT._meta mapped to
        # the codex shape (input_tokens/output_tokens/cached_input_tokens).
        #
        # ``claude-code`` is the ACP provider since ``b164482`` (the rename
        # moved this NAME off the SDK provider, which kept the raw Anthropic
        # shape under ``claude-code-sdk`` above). ACP builds the SAME codex
        # shape from ``session/prompt`` RESULT ``_meta``
        # (``acp/chat_model.py:1544-1557``: cachedReadTokens →
        # cached_input_tokens), so it normalises here — NOT via
        # ``_from_anthropic``, whose ``input_tokens`` excludes the cache.
        # Routing it to the anthropic branch yields ``miss == prompt`` and
        # drops the hit; leaving it on the LangChain fallback (the bug this
        # replaces) silently dropped BOTH cache fields to ``None``.
        return _from_codex(usage)
    if provider in ("google", "vertexai"):
        return _from_google(usage)
    return _from_langchain_normalized(usage)


# ---------------------------------------------------------------------------
# UsageRecord schema (§4.1)
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """One unit of account — one JSONL line (schema_version 1)."""

    provider: str
    model: str
    kind: str
    status: str
    usage_semantics: str
    ts: str
    record_id: str = field(default_factory=lambda: uuid_module.uuid4().hex)
    event_key: str | None = None
    session_cumulative: dict[str, Any] | None = None
    provider_turn_id: str | None = None
    provider_session_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    total_tokens: int | None = None
    context_used: int | None = None
    context_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "event_key": self.event_key,
            "session_cumulative": self.session_cumulative,
            "ts": self.ts,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "status": self.status,
            "usage_semantics": self.usage_semantics,
            "provider_turn_id": self.provider_turn_id,
            "provider_session_id": self.provider_session_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "total_tokens": self.total_tokens,
            "context_used": self.context_used,
            "context_size": self.context_size,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageRecord | None":
        """Parse a parsed-JSON line. Returns ``None`` for malformed records."""
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        provider = data.get("provider")
        if not isinstance(provider, str):
            return None
        kind = data.get("kind")
        if kind not in _KNOWN_KINDS:
            return None
        status = data.get("status")
        if status not in _KNOWN_STATUSES:
            return None
        semantics = data.get("usage_semantics")
        if semantics not in _KNOWN_SEMANTICS:
            return None
        ts = data.get("ts")
        if not isinstance(ts, str):
            return None
        model = data.get("model")
        if not isinstance(model, str):
            model = ""
        return cls(
            record_id=str(data.get("record_id") or uuid_module.uuid4().hex),
            event_key=data.get("event_key"),
            session_cumulative=data.get("session_cumulative"),
            ts=ts,
            provider=provider,
            model=model,
            kind=kind,
            status=status,
            usage_semantics=semantics,
            provider_turn_id=data.get("provider_turn_id"),
            provider_session_id=data.get("provider_session_id"),
            prompt_tokens=_to_int(data.get("prompt_tokens")),
            completion_tokens=_to_int(data.get("completion_tokens")),
            cache_hit_tokens=_to_int(data.get("cache_hit_tokens")),
            cache_miss_tokens=_to_int(data.get("cache_miss_tokens")),
            total_tokens=_to_int(data.get("total_tokens")),
            context_used=_to_int(data.get("context_used")),
            context_size=_to_int(data.get("context_size")),
        )


def _now_utc() -> str:
    return datetime.now(_UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_record(
    *,
    provider: str,
    model: str,
    kind: str,
    status: str,
    usage: NormalizedUsage | None = None,
    event_key: str | None = None,
    provider_turn_id: str | None = None,
    provider_session_id: str | None = None,
    context_used: int | None = None,
    context_size: int | None = None,
) -> UsageRecord:
    """Build a UsageRecord from a normalized usage (or none)."""
    semantics = (
        USAGE_SEMANTICS_CONTEXT_SNAPSHOT
        if (context_used is not None or context_size is not None)
        else USAGE_SEMANTICS_TOKENS
    )
    return UsageRecord(
        provider=provider,
        model=model,
        kind=kind,
        status=status,
        usage_semantics=semantics,
        ts=_now_utc(),
        event_key=event_key,
        provider_turn_id=provider_turn_id,
        provider_session_id=provider_session_id,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        cache_hit_tokens=usage.cache_hit_tokens if usage else None,
        cache_miss_tokens=usage.cache_miss_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        context_used=context_used,
        context_size=context_size,
    )


# ---------------------------------------------------------------------------
# Recorder (§4.2)
# ---------------------------------------------------------------------------


class TokenUsageRecorder:
    """Append-only recorder for ``token_stats.jsonl`` (§4.2).

    - One ``asyncio.Lock`` per process serializes appends; the actual write
      runs in a worker thread via ``asyncio.to_thread`` (no new threads).
    - ``os.open(O_WRONLY|O_APPEND|O_CREAT, 0o600)`` + ``os.fchmod(0600)``
      (also repairs the mode of an existing file); the directory is chmod'ed
      to ``0o700``.
    - One ``os.write``; a short write is a corrupted line — the fd is closed
      and the line is NOT re-appended (documented behavior). A short write
      raises ``OSError`` so the caller sees the dropped record.
    - No fsync (see module docstring for crash semantics).
    """

    def __init__(self, data_dir: str | Path, *, on_record: Any = None) -> None:
        self.path: Path = Path(data_dir) / STATS_FILE_NAME
        self._lock = asyncio.Lock()
        self._on_record = on_record

    def _append_line(self, line: str) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            data = line.encode("utf-8")
            written = os.write(fd, data)
            if written != len(data):
                # Short write: the line would be corrupted. Close the fd and
                # raise so best-effort callers actually see the failure.
                _logger.error(
                    "token-stats: short write (%d of %d bytes) to %s — record dropped",
                    written,
                    len(data),
                    self.path,
                )
                raise OSError(
                    f"token-stats: short write to {self.path} "
                    f"({written} of {len(data)} bytes)"
                )
        finally:
            os.close(fd)

    async def record(self, record: UsageRecord) -> None:
        """Append one record. Raises on failure (callers must not propagate)."""
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)
        if self._on_record is not None:
            await self._on_record({"type": "model_context_changed", **record.to_dict()})

    def make_usage_event_callback(
        self,
        *,
        provider: str,
        kind: str,
        model_name: str | None = None,
    ) -> "UsageEventCallback":
        """Create the custom-provider state-machine callback (§5.2).

        The provider installs it via ``model.set_usage_event_callback(cb)``.
        """
        return UsageEventCallback(
            self,
            provider=provider,
            kind=kind,
            model_name=model_name or "",
        )


class UsageEventCallback:
    """Async state machine for custom-provider usage (§5.2).

    Installed on claude-code-sdk/codex/grok chat models via
    ``set_usage_event_callback(cb)``. Contract:

    - ``cb(event="invocation_start")`` — marks a pending invocation (no record).
    - ``cb(event="terminal", status=..., usage=..., ...)`` — writes **at most
      once** per observed terminal event (status ``ok``/``no_usage``/``failed``).
      Tool sub-turns between two terminal events are packed into the record.
    - Cancellation of the provider turn (no terminal event) writes nothing.
    - Best-effort: exceptions are logged and never raised, except
      ``asyncio.CancelledError`` which propagates.
    """

    def __init__(
        self,
        recorder: TokenUsageRecorder,
        *,
        provider: str,
        kind: str,
        model_name: str = "",
    ) -> None:
        self._recorder = recorder
        self._provider = provider
        self._kind = kind
        self._model_name = model_name
        self._pending = False
        self._seen_keys: set[str] = set()

    @property
    def pending(self) -> bool:
        return self._pending

    def _event_key(
        self,
        *,
        session_id: str | None,
        uuid: str | None,
        turn_id: str | None,
    ) -> str | None:
        if self._provider == "claude-code-sdk":
            if uuid:
                return f"claude-code-sdk|{session_id}|{uuid}"
            return None
        if self._provider == "codex":
            if session_id and turn_id:
                return f"codex|{session_id}|{turn_id}"
            return None
        # grok: no stable provider id → never deduplicated.
        return None

    async def __call__(
        self,
        *,
        event: str,
        status: str = STATUS_OK,
        usage: dict[str, Any] | None = None,
        model: str | None = None,
        session_id: str | None = None,
        uuid: str | None = None,
        turn_id: str | None = None,
        context_used: int | None = None,
        context_size: int | None = None,
    ) -> None:
        try:
            if event == "invocation_start":
                self._pending = True
                return
            if event != "terminal":
                _logger.warning("token-stats: unknown callback event %r", event)
                return
            # Terminal event.
            self._pending = False
            event_key = self._event_key(
                session_id=session_id, uuid=uuid, turn_id=turn_id
            )
            if event_key is not None:
                if event_key in self._seen_keys:
                    return  # in-process duplicate — never reaches disk
                self._seen_keys.add(event_key)

            is_snapshot = context_used is not None or context_size is not None
            model = model or self._model_name or ""
            if status == STATUS_FAILED:
                record = build_record(
                    provider=self._provider,
                    model=model,
                    kind=self._kind,
                    status=STATUS_FAILED,
                    event_key=event_key,
                    provider_session_id=session_id,
                    provider_turn_id=turn_id or uuid,
                )
            elif is_snapshot:
                # Hybrid record: when the provider hands over BOTH a
                # snapshot AND per-turn usage in the same terminal event
                # (e.g. ACP claude-code: usage_update → usage_snapshot;
                # session/prompt reply _meta → per-turn totalTokens),
                # persist both on ONE row.  ``build_record`` promotes
                # ``usage_semantics=context_snapshot`` whenever
                # ``context_used``/``context_size`` is set, so the UI's
                # Context field renders the honest observed number (X/Y),
                # while the per-turn ``total_tokens``/``prompt_tokens``/
                # ``completion_tokens`` stay on the same record for the
                # "Last turn" UI row and for the aggregate summary.
                #
                # Without this the ACP finalize path emitted the snapshot
                # only when ``_meta.totalTokens`` was ABSENT (rare), and
                # the UI showed "Context: 2265.4k tokens" against
                # "Limit: 400k" — see agent_a 2026-08-22 and the
                # the code-graph server extractor.
                norm = normalize_usage(self._provider, usage) if usage else None
                record = build_record(
                    provider=self._provider,
                    model=model,
                    kind=self._kind,
                    status=STATUS_OK,
                    usage=norm,
                    event_key=event_key,
                    provider_session_id=session_id,
                    provider_turn_id=turn_id or uuid,
                    context_used=context_used,
                    context_size=context_size,
                )
            else:
                norm = normalize_usage(self._provider, usage)
                record = build_record(
                    provider=self._provider,
                    model=model,
                    kind=self._kind,
                    status=STATUS_OK if norm is not None else STATUS_NO_USAGE,
                    usage=norm,
                    event_key=event_key,
                    provider_session_id=session_id,
                    provider_turn_id=turn_id or uuid,
                )
            await self._recorder.record(record)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — best-effort, never breaks a provider turn
            _logger.exception(
                "token-stats: event callback failed (provider=%s, event=%s)",
                self._provider,
                event,
            )


# ---------------------------------------------------------------------------
# Reader + aggregator (§4.3, §6.1)
# ---------------------------------------------------------------------------


@dataclass
class ReadStats:
    """Result of reading a stats file."""

    records: list[UsageRecord] = field(default_factory=list)
    skipped_records: int = 0
    duplicate_records: int = 0
    partial_tail: bool = False


def read_stats_file(path: str | Path) -> ReadStats:
    """Read and parse a token_stats.jsonl file (idempotent aggregation).

    - Unknown ``schema_version`` or malformed lines → ``skipped_records``.
    - A trailing partial line without ``\\n`` → ``partial_tail=True`` (not counted).
    - Duplicates by non-null ``event_key`` → ``duplicate_records`` (excluded
      from ``records``); ``event_key=null`` records are always unique.
    - Missing file → empty result (no error).
    """
    result = ReadStats()
    p = Path(path)
    if not p.exists():
        return result
    try:
        raw = p.read_bytes()
    except OSError:
        _logger.exception("token-stats: failed to read %s", p)
        return result
    if not raw:
        return result
    if not raw.endswith(b"\n"):
        result.partial_tail = True
        raw = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    seen: set[str] = set()
    for line in raw.split(b"\n"):
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            result.skipped_records += 1
            continue
        record = UsageRecord.from_dict(data)
        if record is None:
            result.skipped_records += 1
            continue
        if record.event_key is not None:
            if record.event_key in seen:
                result.duplicate_records += 1
                continue
            seen.add(record.event_key)
        result.records.append(record)
    return result


@dataclass
class ContextAggregate:
    """Grok context-snapshot gauge aggregate (latest + max, §4.3)."""

    snapshots: int = 0
    latest_context_used: int | None = None
    latest_context_size: int | None = None
    max_context_used: int | None = None
    max_context_size: int | None = None


@dataclass(frozen=True)
class ContextSnapshot:
    """Per-kind context snapshot for the UI status bar (DESIGN_AGENT_CONTEXT_STATUS_BAR.md).

    Carries the four fields the UI needs:

      * ``context_used`` / ``context_size``: the values (one or both may be
        None — see ``formatAgentContext`` semantics).
      * ``usage_semantics``: ``"context_snapshot"`` or ``"tokens"`` — lets the
        UI choose between ``"X/Y"`` (provider snapshot with both fields)
        and ``"X tokens"`` (provider-reported token count, exact from the
        provider's tokenizer — no ``"(est.)"`` qualifier).
      * ``ts``: original record timestamp (informational; the UI uses
        in-memory state, not this field).
    """

    context_used: int | None
    context_size: int | None
    usage_semantics: str
    ts: str
    # Per-turn total tokens from the SAME record — populated for the
    # ACP hybrid path (context_snapshot + usage in one row) and for any
    # pure-tokens record that carries ``total_tokens``.  The UI surfaces
    # this in a separate ``Last turn: Xk`` row so it is never compared
    # against ``context_limit``.  ``None`` when the record has no
    # ``total_tokens``.
    total_tokens: int | None = None


def latest_context_by_kind(records: list[UsageRecord]) -> dict[str, ContextSnapshot]:
    """Return the latest context snapshot per ``kind ∈ {"agent", "tm"}``.

    Filter (DESIGN_AGENT_CONTEXT_STATUS_BAR §1):

      * Only ``kind="agent"`` and ``kind="tm"`` records are kept — ``kind="other"``
        (sub-agent calls) is excluded.
      * ``usage_semantics="context_snapshot"`` records with BOTH
        ``context_used`` and ``context_size`` set to ``None`` (negative
        malformed snapshot) are excluded.
      * Token-semantics records are kept only when they contribute context data
        (i.e. ``context_used`` or ``context_size`` is non-None) — UNLESS the
        record carries per-turn token counts, in which case we expose the
        last turn's INPUT size as ``context_used``, preferring
        ``prompt_tokens`` (the full blob the provider read: system + tools
        + history + the new user turn) and falling back to ``total_tokens``
        only for adapters that never report a prompt count.  Note that
        ``total_tokens`` is prompt + completion — a transport metric, NOT a
        context size — so it overstates by ``completion_tokens`` and is a
        last resort, not the primary source.  Either number is EXACT from
        the provider's tokenizer — the web UI does NOT label it ``"(est.)"``
        because it is provider data, not an estimate.  Without this fallback
        every provider that never reports a snapshot (e.g. the
        OpenAI-compatible ``minimax`` adapter) would show ``-`` forever.

    Tie-break (DESIGN §1, mutations M2 / M2.5):

      * Records with a later parsed UTC timestamp replace earlier ones.
      * On equal ts (or both unparsable), the **last** record in the input
        list wins (file-order tie-break — this matches how the records are
        read by ``read_stats_file``, with newer entries appended last).

    Empty input → ``{}``.
    """
    out: dict[str, ContextSnapshot] = {}
    for rec in records:
        if rec.kind not in (KIND_AGENT, KIND_TM):
            continue
        snap: ContextSnapshot | None = None
        if rec.context_used is None and rec.context_size is None:
            # No explicit context data — fall back to per-turn tokens
            # ONLY for tokens semantics.  Snapshot semantics with both
            # fields None is the documented "negative malformed
            # snapshot" case and stays excluded.
            #
            # Priority: ``prompt_tokens`` (the FULL context the provider
            # read on the last turn — system + tools + full history +
            # new user turn shipped as one blob) STRICTLY BEATS
            # ``total_tokens`` (prompt + completion, a transport metric).
            # ``prompt_tokens`` is the number an operator can meaningfully
            # compare with ``-l`` in the status bar (the guard itself uses
            # its own estimator, not this field);
            # ``total_tokens`` overstates by ``completion_tokens`` and
            # for tokens-only providers the two used to collide in the
            # UI Context field (agent_a 2026-08-22, "507.3k tokens
            # vs -l 250k" on codex — completion_tokens was 1498 there,
            # so the diff was small, but the source was still wrong).
            # ``total_tokens`` remains the fallback of last resort for
            # future adapters that only surface totals (minimax-style
            # OpenAI-compat wrappers) so ``Context: -`` does not
            # regress (2026-08-17 incident).
            if rec.usage_semantics == USAGE_SEMANTICS_TOKENS:
                fallback_used: int | None = None
                if rec.prompt_tokens is not None and rec.prompt_tokens > 0:
                    fallback_used = rec.prompt_tokens
                elif rec.total_tokens is not None:
                    fallback_used = rec.total_tokens
                if fallback_used is not None:
                    snap = ContextSnapshot(
                        context_used=fallback_used,
                        context_size=None,
                        usage_semantics=USAGE_SEMANTICS_TOKENS,
                        ts=rec.ts,
                        total_tokens=rec.total_tokens,
                    )
            if snap is None:
                continue
        else:
            snap = ContextSnapshot(
                context_used=rec.context_used,
                context_size=rec.context_size,
                usage_semantics=rec.usage_semantics,
                ts=rec.ts,
                total_tokens=rec.total_tokens,
            )
        existing = out.get(rec.kind)
        if existing is None:
            out[rec.kind] = snap
            continue
        dt_existing = _parse_ts(existing.ts)
        dt_new = _parse_ts(rec.ts)
        # None of the new ts → keep existing (file order, unparsable).
        if dt_new is None:
            continue
        # None of the existing ts → new wins (replaces unparsable).
        if dt_existing is None:
            out[rec.kind] = snap
            continue
        # Strictly later ts → new wins; equal ts → file order wins (new wins).
        if dt_new >= dt_existing:
            out[rec.kind] = snap
    return out


@dataclass
class BucketTotals:
    """Aggregate over a bucket of records (agent, agent×provider, ×model)."""

    requests: int = 0
    requests_with_usage: int = 0
    usage_coverage: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_ratio_total_observed: float | None = None
    cache_ratio_total_lower_bound: float | None = None
    cache_ratio_total_upper_bound: float | None = None
    cache_ratio_prompt_observed: float | None = None
    cache_coverage: float | None = None
    avg_tokens_per_usage_observed_request: float | None = None
    avg_tokens_per_usage_observed_response: float | None = None
    has_cache: bool = False
    n_usage: int = 0
    n_cache_observed: int = 0
    context: ContextAggregate = field(default_factory=ContextAggregate)
    # Per-kind latest context snapshots for the UI status bar.
    # Populated by ``aggregate_records`` via ``latest_context_by_kind``;
    # ``kind`` keys are restricted to ``{"agent", "tm"}`` (design §1).
    latest_by_kind: dict[str, ContextSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "requests_with_usage": self.requests_with_usage,
            "usage_coverage": self.usage_coverage,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "cache_ratio_total_observed": self.cache_ratio_total_observed,
            "cache_ratio_total_lower_bound": self.cache_ratio_total_lower_bound,
            "cache_ratio_total_upper_bound": self.cache_ratio_total_upper_bound,
            "cache_ratio_prompt_observed": self.cache_ratio_prompt_observed,
            "cache_coverage": self.cache_coverage,
            "avg_tokens_per_usage_observed_request": self.avg_tokens_per_usage_observed_request,
            "avg_tokens_per_usage_observed_response": self.avg_tokens_per_usage_observed_response,
            "has_cache": self.has_cache,
            "n_usage": self.n_usage,
            "n_cache_observed": self.n_cache_observed,
            "context": {
                "snapshots": self.context.snapshots,
                "latest_context_used": self.context.latest_context_used,
                "latest_context_size": self.context.latest_context_size,
                "max_context_used": self.context.max_context_used,
                "max_context_size": self.context.max_context_size,
            },
            "latest_by_kind": {
                kind: {
                    "context_used": snap.context_used,
                    "context_size": snap.context_size,
                    "usage_semantics": snap.usage_semantics,
                    "ts": snap.ts,
                }
                for kind, snap in self.latest_by_kind.items()
            },
        }


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def aggregate_records(records: list[UsageRecord]) -> BucketTotals:
    """Aggregate a list of records into :class:`BucketTotals` (§4.3, §6.1).

    - ``requests`` counts every record (incl. no_usage/failed/cancelled and
      context snapshots).
    - ``requests_with_usage`` counts token-semantics records with non-null
      total tokens; sums cover those same records.
    - Cache math over records with non-null ``cache_hit_tokens`` (H_known,
      P_known, T_known); P_unknown = prompt of token records without cache.
      Ratios are ``None`` when their denominator is 0; sums default to 0;
      averages are ``None`` when ``requests_with_usage == 0``.
    - Context snapshots aggregate separately (latest by parsed UTC ts, tie-break
      by file order; max of used/size) and never enter token aggregates.
    """
    totals = BucketTotals()
    totals.requests = len(records)

    token_records: list[UsageRecord] = []
    snapshot_records: list[UsageRecord] = []
    for rec in records:
        if rec.usage_semantics == USAGE_SEMANTICS_CONTEXT_SNAPSHOT:
            snapshot_records.append(rec)
        else:
            token_records.append(rec)

    # Token aggregates.
    with_usage = [r for r in token_records if r.total_tokens is not None]
    totals.requests_with_usage = len(with_usage)
    totals.prompt_tokens = sum(r.prompt_tokens or 0 for r in with_usage)
    totals.completion_tokens = sum(r.completion_tokens or 0 for r in with_usage)
    totals.total_tokens = sum(r.total_tokens or 0 for r in with_usage)
    totals.n_usage = len(with_usage)

    cache_known = [r for r in with_usage if r.cache_hit_tokens is not None]
    totals.n_cache_observed = len(cache_known)
    totals.has_cache = bool(cache_known)
    totals.cache_hit_tokens = sum(r.cache_hit_tokens or 0 for r in cache_known)
    totals.cache_miss_tokens = sum(r.cache_miss_tokens or 0 for r in cache_known)

    h_known = sum(r.cache_hit_tokens or 0 for r in cache_known)
    p_known = sum(r.prompt_tokens or 0 for r in cache_known)
    t_known = sum(r.total_tokens or 0 for r in cache_known)
    t_all = totals.total_tokens
    p_all = totals.prompt_tokens
    p_unknown = p_all - p_known

    def ratio(num: float, den: float) -> float | None:
        return num / den if den else None

    totals.cache_ratio_total_observed = ratio(h_known, t_known)
    totals.cache_ratio_total_lower_bound = ratio(h_known, t_all)
    totals.cache_ratio_total_upper_bound = ratio(h_known + p_unknown, t_all)
    totals.cache_ratio_prompt_observed = ratio(h_known, p_known)
    totals.cache_coverage = ratio(p_known, p_all)
    if totals.requests > 0:
        # Usage coverage: share of terminal requests that carried usage.
        # null only when there are no requests at all (denominator zero).
        totals.usage_coverage = ratio(
            float(totals.requests_with_usage), float(totals.requests)
        )
    if totals.requests_with_usage:
        totals.avg_tokens_per_usage_observed_request = ratio(
            float(totals.prompt_tokens), float(totals.requests_with_usage)
        )
        totals.avg_tokens_per_usage_observed_response = ratio(
            float(totals.completion_tokens), float(totals.requests_with_usage)
        )

    # Context snapshots: latest by parsed UTC ts (tie-break by file order),
    # max of used/size. Snapshots are never summed.
    if snapshot_records:
        latest: UsageRecord | None = None
        latest_dt: datetime | None = None
        max_used: int | None = None
        max_size: int | None = None
        for rec in snapshot_records:
            dt = _parse_ts(rec.ts)
            if dt is not None and (latest_dt is None or dt >= latest_dt):
                latest_dt = dt
                latest = rec
            used = rec.context_used
            size = rec.context_size
            if used is not None:
                max_used = used if max_used is None else max(max_used, used)
            if size is not None:
                max_size = size if max_size is None else max(max_size, size)
        totals.context.snapshots = len(snapshot_records)
        totals.context.latest_context_used = latest.context_used if latest else None
        totals.context.latest_context_size = latest.context_size if latest else None
        totals.context.max_context_used = max_used
        totals.context.max_context_size = max_size

    # Per-kind latest context snapshots (UI status bar).
    # Computed over ALL records (incl. tokens-semantics) so tokens-only
    # providers get ``usage_semantics="tokens"`` to drive the
    # ``"X tokens"`` rendering on the UI side (exact from the provider's
    # tokenizer — no ``"(est.)"`` qualifier).
    totals.latest_by_kind = latest_context_by_kind(records)

    return totals


@dataclass
class ModelAggregate:
    model: str
    totals: BucketTotals

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "totals": self.totals.to_dict()}


@dataclass
class ProviderAggregate:
    provider: str
    totals: BucketTotals
    by_model: list[ModelAggregate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "totals": self.totals.to_dict(),
            "by_model": [m.to_dict() for m in self.by_model],
        }


@dataclass
class StatsResponse:
    """API response shape (§6.1)."""

    agent: str
    generated_at: str
    skipped_records: int
    duplicate_records: int
    partial_tail: bool
    totals: BucketTotals
    by_provider: list[ProviderAggregate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "generated_at": self.generated_at,
            "skipped_records": self.skipped_records,
            "duplicate_records": self.duplicate_records,
            "partial_tail": self.partial_tail,
            "totals": self.totals.to_dict(),
            "by_provider": [p.to_dict() for p in self.by_provider],
        }


def aggregate_stats(
    records: list[UsageRecord],
    *,
    agent: str,
    generated_at: str | None = None,
) -> StatsResponse:
    """Aggregate records into the full API response (totals + by_provider)."""
    totals = aggregate_records(records)
    by_provider: list[ProviderAggregate] = []
    providers: dict[str, list[UsageRecord]] = {}
    for rec in records:
        providers.setdefault(rec.provider, []).append(rec)
    for provider in sorted(providers):
        provider_records = providers[provider]
        p_agg = ProviderAggregate(
            provider=provider,
            totals=aggregate_records(provider_records),
        )
        models: dict[str, list[UsageRecord]] = {}
        for rec in provider_records:
            models.setdefault(rec.model or "", []).append(rec)
        for model in sorted(models):
            p_agg.by_model.append(
                ModelAggregate(
                    model=model,
                    totals=aggregate_records(models[model]),
                )
            )
        by_provider.append(p_agg)
    return StatsResponse(
        agent=agent,
        generated_at=generated_at or _now_utc(),
        skipped_records=0,
        duplicate_records=0,
        partial_tail=False,
        totals=totals,
        by_provider=by_provider,
    )


def stats_for_agent(
    agent_name: str,
    data_dir: str | Path,
    *,
    generated_at: str | None = None,
) -> StatsResponse:
    """Read + aggregate the stats file of one agent (web endpoint core).

    A missing file yields a zero-filled response (no file is created).
    """
    read = read_stats_file(Path(data_dir) / STATS_FILE_NAME)
    response = aggregate_stats(
        read.records,
        agent=agent_name,
        generated_at=generated_at,
    )
    response.skipped_records = read.skipped_records
    response.duplicate_records = read.duplicate_records
    response.partial_tail = read.partial_tail
    return response
