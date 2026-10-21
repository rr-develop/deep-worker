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
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
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


class SummarizationFailedError(RuntimeError):
    """Raised when producing the summary itself fails.

    Sibling of :class:`HistoryOffloadError`: that one covers a failure to
    PERSIST the evicted history, this one covers a failure to PRODUCE the
    summary that replaces it. They are kept distinct so the operator can tell
    which stage lost data — by the time this is raised the offload has already
    succeeded, so the history is safe on disk.

    The failure text must never be substituted for the summary, and the
    history must not be evicted behind it: an agent that wakes up holding
    ``Error generating summary: …`` in place of its context, with the real
    messages already deleted, cannot recover and cannot even tell that
    anything went wrong.

    Fatal — the agent runtime's fatal-exception catch-all matches the class
    name ``SummarizationFailedError`` in
    :data:`deep_worker_cli.exceptions.FATAL_EXCEPTION_TYPES` and transitions
    the agent into ``AgentStatus.ERROR`` with the error message.

    See ``docs/POLICIES.md`` §12 and ``docs/DESIGN_SUMMARIZATION_HOT_COLD.md``.
    """


# The langchain base class swallows summarization failures and returns this
# prefix as if it were the summary. We detect it in ``_create_summary`` and
# convert it back into a raised ``SummarizationFailedError`` (POLICIES.md §12).
# Kept as a constant so the string exists in exactly one place; the guard in
# ``test_summarization_fatal_visible.py`` allows it here and nowhere else.
_SUMMARY_ERROR_MARKER = "Error generating summary"


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

    trigger: ContextSize | list[ContextSize]
    keep: ContextSize
    truncate_args_settings: TruncateArgsSettings


@dataclass(frozen=True, slots=True)
class RequestedSummaryResult:
    summarized: bool
    replacement_messages: list[AnyMessage] | None
    history_path: str | None


def _instruction_from_prompt(prompt: str | None, default: str) -> str:
    raw = prompt if prompt else default
    if "{messages}" in raw:
        raw = raw.replace("{messages}", "").rstrip()
    return raw


def _utc_now_iso() -> str:
    """Current time as ISO-8601 with an explicit ``Z``.

    UTC is a hard project rule for anything that lands in a report or a log:
    a downstream consumer correlates this against its own run timeline, and
    a local-time stamp would silently shift an incident by the host's offset.
    """
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _describe_trigger(trigger: Any) -> str:
    """Render the configured trigger as ``kind:value`` for the incident report.

    Purely descriptive — the operator reading the incident needs to know WHICH
    threshold fired (``fraction:0.85`` vs ``tokens:170000``), and a trigger may
    be a single spec or a list of them.
    """
    def _one(spec: Any) -> str:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return f"{spec[0]}:{spec[1]}"
        return str(spec)

    if isinstance(trigger, list):
        return ",".join(_one(spec) for spec in trigger)
    return _one(trigger)


def _has_summarizer_port(model: Any) -> bool:
    """Whether ``model`` implements the SummarizerPort.

    Delegates to the port's own predicate so "what counts as implementing it"
    has ONE definition. The local fallback keeps deepagents importable without
    the CLI package, and matches the delegate's semantics: check the TYPE (an
    instance probe reads truthy on any ``MagicMock``) and require BOTH halves
    (``asummarize`` without ``areset_after_summary`` is a half-migrated
    provider — claude-code-sdk sat in exactly that state until 2026-08-24,
    unnoticed precisely because this check only looked at one half).
    """
    try:
        from deep_worker_cli.context_limit.summarizer_port import (
            implements_summarizer_port,
        )
    except Exception:  # noqa: BLE001 — deepagents may run standalone
        model_type = type(model)
        return all(
            asyncio.iscoroutinefunction(getattr(model_type, name, None))
            for name in ("asummarize", "areset_after_summary")
        )
    return implements_summarizer_port(model)


def _is_session_stateful(model: Any) -> bool:
    """Whether the provider keeps conversation state on ITS side.

    For such providers (ACP/CLI-backed) re-invoking the model replays the
    whole history into a new provider session, so losing the summarizer
    port must fail loudly instead of falling through. Reuses
    ``hot_cache.SESSION_STATEFUL_PROVIDERS`` — the single source of truth
    for this classification; do not duplicate the list here.

    Deliberately an ALLOW-LIST, and deliberately string-only. Answering
    "is it NOT stateless?" made every unknown provider AND every
    ``MagicMock`` (whose attribute access yields a truthy mock) look
    session-stateful, which broke seven unrelated summarization tests
    (reviewer B2, 2026-08-21). Anything we cannot positively identify as a
    CLI-session provider takes the normal path.
    """
    try:
        from deep_worker_cli.context_limit.hot_cache import (
            SESSION_STATEFUL_PROVIDERS,
        )
    except Exception:  # noqa: BLE001 — deepagents may run standalone
        return False
    provider, _model_id, _adapter = _current_provider_model(model)
    # Must be a real string: a mock/sentinel is not a provider identity.
    if not isinstance(provider, str) or not provider.strip():
        return False
    return provider.strip().lower() in SESSION_STATEFUL_PROVIDERS


def _current_provider_model(model: Any) -> tuple[str, str | None, str | None]:
    # ``_profile`` FIRST, and it is load-bearing.
    #
    # ``AcpChatModel`` exposes the profile as a ``@property``, but it is a
    # pydantic model: pydantic strips the property off the class and the
    # instance ``__dict__`` shadows it, so ``getattr(model, "profile")`` hands
    # back the raw ``property`` OBJECT rather than the profile. Reading
    # ``.provider`` off that yields None, so EVERY ACP provider — claude-code
    # included — resolved to "unknown" here, and ``_is_session_stateful`` was
    # therefore always False for them.
    #
    # That made the session-stateful guard dead code for the exact providers it
    # was written to protect. It looked fine only because those providers also
    # happened to implement the summarizer port, so the dump never triggered.
    # Verified directly: ``_is_session_stateful`` returned False for
    # AcpChatModel, ClaudeCodeAcpChatModel and GrokChatModel alike.
    profile = getattr(model, "_profile", None) or getattr(model, "profile", None)
    provider = (
        getattr(model, "_deep_worker_provider", None)
        or getattr(model, "provider", None)
        or getattr(profile, "provider", None)
    )
    if isinstance(profile, dict):
        provider = provider or profile.get("provider")
    if not provider:
        hint = str(
            getattr(model, "model_name", None) or getattr(model, "model", None) or ""
        ).lower()
        if "deepseek" in hint:
            provider = "deepseek"
        elif "grok" in hint:
            provider = "grok"
    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "_model_name", None)
        or getattr(profile, "model", None)
    )
    model_id = getattr(model, "resolved_model_name", None)
    if provider:
        provider = str(provider).strip().lower()
        if provider == "unknown":
            provider = None
    return str(provider or "unknown"), (str(model_name) if model_name else None), (
        str(model_id) if model_id else None
    )


def _session_id_for(model: Any, thread_id: str | None) -> str | None:
    if not thread_id:
        return None
    client_state = getattr(model, "_client_state", None)
    getter = getattr(client_state, "get", None)
    if getter is None:
        return None
    try:
        slot = getter(thread_id)
    except Exception:  # noqa: BLE001
        return None
    session_id = getattr(slot, "session_id", None)
    return session_id if isinstance(session_id, str) else None


def _hot_cold_decision(
    model: Any,
    messages: list[AnyMessage],
    thread_id: str | None,
    ttl_overrides: dict[str, int] | None = None,
) -> Any:
    try:
        from deep_worker_cli.context_limit.hot_cache import decide
    except ImportError:
        return None
    provider, model_name, model_id = _current_provider_model(model)
    state_last = None
    client_state = getattr(model, "_client_state", None)
    if client_state is not None and thread_id:
        slot = getattr(client_state, "get", lambda _tid: None)(thread_id)
        state_last = getattr(slot, "last_response_at", None) if slot is not None else None
    overrides = ttl_overrides if ttl_overrides is not None else getattr(
        model, "_hot_cache_ttl_overrides", None
    )
    return decide(
        current_provider=provider,
        current_model=model_name,
        current_model_id=model_id,
        messages=messages,
        state_last_response_at=state_last,
        session_id=_session_id_for(model, thread_id),
        ttl_overrides=overrides,
    )


def _compute_summarization_defaults(
    model: BaseChatModel,
    context_limit: int | None = None,
) -> SummarizationDefaults:
    """Compute default summarization settings based on model profile.

    This is an internal helper function used by middleware implementations.

    Args:
        model: A resolved chat model instance.
        context_limit: Optional operator-set effective window in tokens.
            When set, trigger is the OR of ``int(L * 0.85)`` tokens and the
            model default; keep and truncate_args are derived from ``L``.

    Returns:
        Default settings for trigger, keep, and truncate_args_settings.
        If the model has a profile with max_input_tokens, uses fraction-based
        settings. Otherwise, uses fixed token/message counts.
    """
    raw_profile = getattr(model, "profile", None)
    max_input_tokens: int | None = None
    if isinstance(raw_profile, dict):
        candidate = raw_profile.get("max_input_tokens")
        if isinstance(candidate, int):
            max_input_tokens = candidate
    else:
        candidate = getattr(raw_profile, "max_input_tokens", None)
        if isinstance(candidate, int):
            max_input_tokens = candidate
    has_profile = max_input_tokens is not None

    if has_profile:
        model_trigger: ContextSize = ("fraction", 0.85)
    else:
        model_trigger = ("tokens", 170000)

    if context_limit:
        limit_trigger: ContextSize = ("tokens", int(context_limit * 0.85))
        limit_keep: ContextSize = ("tokens", int(context_limit * 0.10))
        # When ``-l`` is set the fixed 170k fallback MUST NOT fire (an
        # earlier incident: AcpProfile is a dataclass so has_profile was False and
        # 170k OR-ed with 340k, causing a second summarization of a 224k
        # dump-session). Keep the model fraction trigger only when the
        # profile actually has max_input_tokens.
        trigger: ContextSize | list[ContextSize]
        if has_profile:
            trigger = [limit_trigger, model_trigger]
        else:
            trigger = limit_trigger
        limited: SummarizationDefaults = {
            "trigger": trigger,
            "keep": limit_keep,
            "truncate_args_settings": {
                "trigger": limit_trigger,
                "keep": limit_keep,
            },
        }
        return limited

    if has_profile:
        profile_defaults: SummarizationDefaults = {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }
        return profile_defaults

    no_profile: SummarizationDefaults = {
        "trigger": ("tokens", 170000),
        "keep": ("messages", 6),
        "truncate_args_settings": {
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        },
    }
    return no_profile


def _apply_keep_override(
    defaults: SummarizationDefaults,
    keep_override: tuple[str, int | float],
) -> SummarizationDefaults:
    """Fold an operator ``--summary-keep`` spec into computed defaults.

    A token spec is a CEILING: it may only lower the computed share, never
    raise it. ``("tokens", 0)`` is exact (empty tail) and is never min()-ed.
    Fraction and message specs replace the computed value outright.

    ``truncate_args_settings.keep`` deliberately keeps its own computed value:
    it governs tool-argument trimming of *older* messages, not the preserved
    tail, and zeroing it would truncate arguments everywhere.

    ``kind`` is one of ``fraction`` / ``messages`` / ``tokens`` — the only
    three ``parse_summary_keep`` emits. Anything else is treated as a token
    spec rather than silently widening ``ContextSize``.
    """
    kind, value = keep_override
    computed = defaults["keep"]
    resolved: ContextSize
    if kind == "fraction":
        resolved = ("fraction", value)
    elif kind == "messages":
        resolved = ("messages", int(value))
    else:
        # ``0`` needs no special case: min(0, N) == 0 against a token default,
        # and the pass-through branch yields 0 against a fraction/message
        # default. Both land on ("tokens", 0) — an empty tail.
        computed_kind, computed_value = computed
        if computed_kind == "tokens":
            resolved = ("tokens", min(int(value), int(computed_value)))
        else:
            resolved = ("tokens", int(value))
    out: SummarizationDefaults = {
        "trigger": defaults["trigger"],
        "keep": resolved,
        "truncate_args_settings": defaults["truncate_args_settings"],
    }
    return out


class SummarizationMiddleware(BaseSummarizationMiddleware):
    """Summarization middleware with backend for conversation history offloading."""

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        # NOTE: deep_worker ALWAYS passes an explicit ``token_counter``
        # (``provider_context_token_counter``) so that this middleware and
        # the context-limit guard decide on the SAME number — the one the
        # PROVIDER reported. This langchain default counts characters; using
        # it inside deep_worker means summarizing on a number nobody
        # measured, which is exactly what the project owner forbade
        # (measured cost: 214613 estimated vs 90k real).
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        history_path_prefix: str = "/conversation_history",
        truncate_args_settings: TruncateArgsSettings | None = None,
        on_summarized: Callable[[str], None] | None = None,
        hot_cache_ttl_overrides: dict[str, int] | None = None,
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
        # ``keep=("tokens", 0)`` means "no tail at all — summary only".
        # The upstream base class cannot represent it: ``_validate_context_size``
        # rejects non-positive values outright, and even if it did not,
        # ``_find_token_based_cutoff`` coerces a non-positive budget to 1 token
        # while ``_find_safe_cutoff_point`` walks the boundary back to keep an
        # AIMessage(tool_calls)/ToolMessage pair together. Both would leave at
        # least one message behind. Hand the base class a benign placeholder and
        # honour the real intent in ``_determine_cutoff_index``.
        self._keep_nothing = keep == ("tokens", 0)
        super().__init__(
            model=model,
            trigger=trigger,
            keep=("messages", _DEFAULT_MESSAGES_TO_KEEP) if self._keep_nothing else keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )
        if self._keep_nothing:
            # Restore the operator-visible value so ``self.keep`` never lies.
            self.keep = ("tokens", 0)
        self._backend = backend
        self._history_path_prefix = history_path_prefix
        self._on_summarized = on_summarized
        # Per-turn observability (downstream requirement REQ-90): set by ``abefore_model``
        # when a REAL size-based compaction fired during the turn, read and
        # cleared by the turn owner (agent_process). Deliberately NOT set by
        # ``apply_pseudo_summary`` (the operator's per-document context reset,
        # a normal mode) nor by ``apply_requested_summary`` (force=True, an
        # explicit operator command with its own report). A flag on either of
        # those would fail EVERY document of EVERY run downstream.
        self.last_turn_compaction: dict[str, Any] | None = None
        self._hot_cache_ttl_overrides = (
            dict(hot_cache_ttl_overrides) if hot_cache_ttl_overrides else None
        )
        if self._hot_cache_ttl_overrides:
            setattr(self.model, "_hot_cache_ttl_overrides", self._hot_cache_ttl_overrides)

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

    def _origin_for_offload(
        self,
        messages: list[AnyMessage],
        filtered_messages: list[AnyMessage],
    ) -> dict[str, Any] | None:
        """Resolve v5 segment origin, including previous-summary stamps.

        Filtering summaries before lookup dropped origin on chained
        offloads (observed in live chained offloads).
        """
        try:
            from deep_worker_cli.context_limit.hot_cache import effective_origin
        except ImportError:
            return None
        origin = effective_origin(messages)
        if origin is None:
            origin = effective_origin(filtered_messages)
        if origin is None:
            provider, model_name, model_id = _current_provider_model(self.model)
            if provider and provider != "unknown":
                origin = {
                    "provider": provider,
                    "model": model_name,
                    "model_id": model_id,
                    "adapter": None,
                    "since": datetime.now(UTC).isoformat(timespec="microseconds"),
                }
        return origin

    def _stamp_preserved_origin(
        self,
        preserved: list[AnyMessage],
        origin: dict[str, Any] | None,
    ) -> list[AnyMessage]:
        """Copy effective origin onto the first AIMessage of the kept tail."""
        if not origin:
            return preserved
        out = list(preserved)
        for i, msg in enumerate(out):
            if not isinstance(msg, AIMessage):
                continue
            rm = dict(getattr(msg, "response_metadata", None) or {})
            existing = rm.get("deep_worker_origin")
            if isinstance(existing, dict) and existing.get("provider"):
                break
            stamped = msg.model_copy()
            rm["deep_worker_origin"] = dict(origin)
            stamped.response_metadata = rm
            out[i] = stamped
            break
        return out

    @staticmethod
    def _drop_pre_summary_usage(preserved: list[AnyMessage]) -> list[AnyMessage]:
        """Strip ``usage_metadata`` from the tail we are about to keep.

        Those numbers measure the prompt that produced the message — a
        prompt built from the history summarization is discarding. Kept on
        a preserved message they describe a conversation that no longer
        exists, and ``provider_context_tokens`` reads
        ``usage_metadata.input_tokens`` as "the context the provider
        reported for THIS history".

        One agent deadlocked on exactly that: a preserved
        AIMessage reporting 450132 input tokens against a six-message
        state. The guard blocked every turn, so no model call ever
        produced a fresh measurement to replace it — sixteen identical
        failures, the number frozen while the history grew.

        The stamped ``deep_worker_context`` observation does not need this
        treatment: it carries a ``first_id`` validity key and is rejected
        on its own once the history it measured is gone. ``usage_metadata``
        has no such key, so the association has to be broken where the tail
        is chosen. That happens in TWO places — the automatic trigger in
        ``before_model`` and the requested one in ``_arun_compaction`` — and
        both must call this. Cleaning only one leaves the same deadlock
        reachable through the other entrance (review finding, 2026-08-30).

        Per-turn accounting keeps working — ``ProvenanceMiddleware`` copies
        these numbers into ``response_metadata['deep_worker_usage']`` as
        the model returns, which is what the web UI and token stats read.
        Every other reader (stream processor, stats callback, textual
        adapter) takes them from the live stream, never from history.
        """
        out = list(preserved)
        for i, msg in enumerate(out):
            if getattr(msg, "usage_metadata", None) is None:
                continue
            # ``update=`` rather than attribute assignment: ``preserved`` is
            # typed as the union of message classes and only ``AIMessage``
            # declares ``usage_metadata``, so assigning through the union is
            # a type error even though the value is a legal ``None``.
            out[i] = msg.model_copy(update={"usage_metadata": None})
        return out

    def _build_new_messages_with_path(
        self,
        summary: str,
        file_path: str | None,
        *,
        origin: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        verbatim: bool = False,
    ) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The summary text.
            file_path: JSONL path to reference, or ``None``.
            origin: Optional provenance stamp.
            usage: Optional per-turn usage stamp.
            verbatim: When True, ``content`` is EXACTLY ``summary`` — no
                framing, no ``<summary>`` tags, no history-file reference.
                Used by the pseudo-summary path, where ``summary`` is not a
                recap of the past at all but the agent's actual task, and it
                becomes the model's FIRST user request. Framing it as "a
                conversation that has been summarized" made deepseek read the
                task as history and spend its whole turn trying to inspect the
                JSONL file (observed in a downstream pipeline: 20 consecutive
                `execute` calls, zero links added, and the same in another
                pipeline). The offload still happens — the model is simply
                not told about it. Real size-based summarization keeps the
                framing and MUST NOT pass this flag.
        """
        if verbatim:
            content = summary
        elif file_path is not None:
            content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        extra: dict[str, Any] = {"lc_source": "summarization"}
        if origin:
            extra["deep_worker_origin"] = origin
        metadata: dict[str, Any] = {}
        if usage:
            metadata["deep_worker_usage"] = usage
        summary_message = HumanMessage(
            content=content,
            additional_kwargs=extra,
            response_metadata=metadata,
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

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Cutoff honouring ``keep=("tokens", 0)`` — an empty preserved tail.

        For every other retention policy this defers to the base class, which
        preserves AIMessage(tool_calls)/ToolMessage pairs. With ``keep=0`` there
        is no tail to keep pairs *in*: the whole history is summarized and the
        resulting state is ``[summary]`` alone.
        """
        if self._keep_nothing:
            return len(messages)
        return super()._determine_cutoff_index(messages)

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
        origin = self._origin_for_offload(messages, filtered_messages)
        new_section = serialize_summarized_messages_jsonl(
            filtered_messages,
            summarized_at=timestamp,
            thread_id=thread_id,
            origin=origin,
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
        origin = self._origin_for_offload(messages, filtered_messages)
        new_section = serialize_summarized_messages_jsonl(
            filtered_messages,
            summarized_at=timestamp,
            thread_id=thread_id,
            origin=origin,
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
            # live agent_c deadlock (2026-06-15).
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
            whose content is ``summary_text`` VERBATIM — no framing, no
            ``<summary>`` tags and no history-file reference, on both the
            offloaded and the empty-history branch. ``file_path`` is the JSONL
            path when a write occurred, else ``None``; it is reported to the
            caller but never shown to the model.

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
            return (
                self._build_new_messages_with_path(
                    summary_text, None, verbatim=True
                ),
                None,
            )

        # Offload the SAME filtered list we emptiness-checked (NIT-1: "what we
        # checked is what we write" — one filtration, no ambiguity).
        # ``_aoffload_to_backend`` re-applies ``_filter_summary_messages``
        # internally; on an already-filtered list that is an idempotent no-op.
        file_path = await self._aoffload_to_backend(
            backend, filtered, thread_id
        )
        # ``file_path`` is still returned to the caller (the offload is real and
        # worth logging), but it is deliberately NOT woven into the message:
        # the model must receive the sum-query byte for byte.
        new_messages = self._build_new_messages_with_path(
            summary_text, file_path, verbatim=True
        )
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
        # Both paths that choose a tail must drop the pre-summary measurement
        # (see ``_drop_pre_summary_usage``); this is the automatic trigger,
        # ``_arun_compaction`` is the requested one. Cleaning only one of them
        # leaves the same deadlock reachable through the other entrance.
        preserved_messages = self._drop_pre_summary_usage(preserved_messages)

        # Offload to backend FIRST — abort summarization if this fails to prevent data loss.
        # ``_offload_to_backend`` now raises ``HistoryOffloadError`` on any failure; the raise
        # propagates through this method into the agent runtime catch-all, which matches
        # the class name in ``FATAL_EXCEPTION_TYPES`` and transitions the agent to
        # ``AgentStatus.ERROR``. See ``DESIGN_FAIL_LOUD_SUMMARIZATION_OFFLOAD.md``.
        backend = self._get_backend(state, runtime)
        file_path = self._offload_to_backend(backend, messages_to_summarize)

        # Generate summary — only reached when the offload succeeded.
        # ``_create_summary`` is overridden below to raise instead of returning
        # the base class's ``"Error generating summary: …"`` string, so a
        # failure here never reaches ``_build_new_messages_with_path`` and the
        # ``RemoveMessage`` below is never constructed.
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        # REQ-90, same contract as ``abefore_model``. Production always runs the
        # async hook, but a size-based firing on THIS path is the same incident,
        # and leaving it unstamped would make the gap silent if it ever runs.
        replacement = [*new_messages, *preserved_messages]
        self.last_turn_compaction = {
            "occurred": True,
            "observable": True,
            "reason": "size_trigger",
            "trigger": _describe_trigger(self.trigger),
            "tokens_before": total_tokens,
            "tokens_after": self._cached_total_tokens(replacement),
            "messages_before": len(messages),
            "messages_after": len(replacement),
            "history_path": file_path,
            "at_utc": _utc_now_iso(),
        }

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Summarise ``messages_to_summarize``, raising instead of swallowing.

        DELIBERATE DIVERGENCE FROM THE LANGCHAIN BASE CLASS (POLICIES.md §12).

        The base implementation ends with::

            except Exception as e:
                return f"Error generating summary: {e!s}"

        The caller cannot tell that string from a real summary, so it gets
        wrapped by ``_build_new_messages_with_path`` and returned alongside
        ``RemoveMessage(REMOVE_ALL_MESSAGES)``: the agent wakes up holding the
        error text as its entire context, status ``IDLE``, nothing reporting a
        fault. That is the 2026-08-23 incident, and it is the reason the async
        path was fixed — this override closes the same hole on the sync path.

        Matching upstream is not a reason to lose user data. We diverge, and
        this docstring is the divergence site the policy asks for.

        Overriding the helper rather than patching the one call site in
        ``before_model`` means any current or future caller of
        ``_create_summary`` inherits the guarantee, including one added
        upstream in a base-class method we do not override.

        Raises:
            SummarizationFailedError: if the underlying model call fails, or if
                the base class swallowed the failure into its marker string.
        """
        try:
            summary = super()._create_summary(messages_to_summarize)
        except SummarizationFailedError:
            raise
        except Exception as e:
            msg = f"Failed to generate conversation summary: {type(e).__name__}: {e}"
            logger.error(msg)
            raise SummarizationFailedError(msg) from e

        # Belt and braces: the base class catches internally, so a failure
        # arrives as a RETURN VALUE, not an exception. Detect the marker and
        # convert it back into the fault it always was. Pinned by
        # ``test_sync_before_model_raises_instead_of_summarising_the_error``.
        if isinstance(summary, str) and summary.startswith(_SUMMARY_ERROR_MARKER):
            msg = f"Failed to generate conversation summary: {summary}"
            logger.error(msg)
            raise SummarizationFailedError(msg)
        return summary

    def _format_requested_summary_prompt(
        self,
        prompt: str,
        messages: list[AnyMessage],  # noqa: ARG002 — signature kept for callers
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
        return prompt if fields else prompt

    def _trim_for_prompt(
        self,
        messages_to_summarize: list[AnyMessage],
        summary_prompt: str | None,
    ) -> list[AnyMessage]:
        if summary_prompt is None:
            return self._trim_messages_for_summary(messages_to_summarize)
        if self.trim_tokens_to_summarize is None:
            return messages_to_summarize
        prompt_for_budget = (
            summary_prompt
            if "{messages}" in summary_prompt
            else f"{summary_prompt}\n\n{{messages}}"
        )
        prompt_tokens = self.token_counter([HumanMessage(content=prompt_for_budget)])
        adjusted_limit = max(1000, self.trim_tokens_to_summarize - prompt_tokens)
        return cast(
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

    async def _ainvoke_identical_prefix(
        self,
        trimmed_messages: list[AnyMessage],
        instruction: str,
        thread_id: str | None,
    ) -> str:
        # Fail loud rather than dump. This path re-``ainvoke``s the model on
        # the AGENT's own thread_id. For a session-stateful provider (ACP)
        # that means the 4b net re-sends the whole history into a fresh CLI
        # session — the very dump the summarizer port exists to avoid. Today
        # ``AcpChatModel.bind_tools`` returns ``self`` so the port survives
        # binding, but any wrapper that does not preserve ``asummarize``
        # would silently regress to dumping. Refuse instead.
        if _is_session_stateful(self.model) and not _has_summarizer_port(self.model):
            raise RuntimeError(
                "summarization port unavailable on a session-stateful "
                f"provider ({type(self.model).__name__}); refusing to "
                "re-invoke the agent session, which would dump the full "
                "history. The model wrapper must preserve `asummarize`."
            )
        payload: list[AnyMessage] = [*trimmed_messages, HumanMessage(content=instruction)]
        config: RunnableConfig = {"metadata": {"lc_source": "summarization"}}
        if thread_id is not None:
            config["configurable"] = {"thread_id": thread_id}
        response = await self.model.ainvoke(payload, config=config)
        return response.text.strip()

    async def _acreate_summary_for_prompt(
        self,
        messages_to_summarize: list[AnyMessage],
        summary_prompt: str | None,
        thread_id: str | None,
        *,
        decision: Any = None,
    ) -> tuple[str, Any]:
        if not messages_to_summarize:
            return "No previous conversation history.", None
        if summary_prompt is not None:
            self._format_requested_summary_prompt(summary_prompt, messages_to_summarize)
        trimmed_messages = self._trim_for_prompt(messages_to_summarize, summary_prompt)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize.", None
        instruction = _instruction_from_prompt(summary_prompt, self.summary_prompt)
        if decision is None:
            decision = _hot_cold_decision(
                self.model,
                messages_to_summarize,
                thread_id,
                ttl_overrides=self._hot_cache_ttl_overrides,
            )
        # Hot API path must keep the live prefix (no trim). Cold may trim.
        prefix_messages = (
            messages_to_summarize
            if decision is not None and getattr(decision, "hot", False)
            else trimmed_messages
        )
        if decision is not None:
            logger.info(
                "summarization.decision reason=%s elapsed=%s N=%s",
                decision.reason,
                decision.elapsed_sec,
                decision.n_sec,
            )
            logger.info("summarization.decision hot=%s", decision.hot)

        async def _via_port(chosen: Any) -> Any:
            # Call through the unbound attribute, matching how
            # ``_has_summarizer_port`` probes for it: ``asummarize`` is an
            # optional port, absent from BaseChatModel, so a direct
            # ``self.model.asummarize(...)`` is an unresolved-attribute error.
            summarize = getattr(type(self.model), "asummarize")  # noqa: B009
            return await summarize(
                self.model,
                thread_id=thread_id or "",
                messages=prefix_messages,
                instruction=instruction,
                decision=chosen,
            )

        try:
            if _has_summarizer_port(self.model) and decision is not None:
                logger.info(
                    "summarization.path=port hot=%s reason=%s",
                    getattr(decision, "hot", None),
                    getattr(decision, "reason", None),
                )
                try:
                    result = await _via_port(decision)
                    if not str(getattr(result, "text", "")).strip():
                        raise RuntimeError("empty summary")
                except Exception as exc:
                    if not getattr(decision, "hot", False):
                        raise
                    logger.warning("summarization.hot_path_failed reason=%s", exc)
                    from deep_worker_cli.context_limit.hot_cache import HotColdDecision

                    decision = HotColdDecision(
                        False,
                        f"hot_path_failed:{exc}",
                        decision.elapsed_sec,
                        decision.n_sec,
                        decision.last_origin,
                    )
                    result = await _via_port(decision)
                    if not str(getattr(result, "text", "")).strip():
                        raise RuntimeError("empty summary") from exc
                return result.text.strip(), result
            logger.info(
                "summarization.path=identical_prefix hot=%s reason=%s",
                getattr(decision, "hot", None),
                getattr(decision, "reason", None),
            )
            text = await self._ainvoke_identical_prefix(
                prefix_messages, instruction, thread_id
            )
            return text, None
        except SummarizationFailedError:
            raise
        except Exception as e:
            # Fail loud on EVERY path, automatic included (POLICIES.md §12).
            #
            # This used to soft-return f"Error generating summary: {e}" when
            # ``fail_loud`` was False — i.e. on the automatic ``abefore_model``
            # path, the one that runs unattended. The caller cannot distinguish
            # that string from a real summary, so it was written into state and
            # the real history was evicted alongside it. Agents woke up after a
            # restart holding the error text instead of their context, status
            # IDLE, with nothing reporting a fault.
            #
            # ``SummarizationFailedError`` is in FATAL_EXCEPTION_TYPES, so this
            # bubbles through ``abefore_model`` into the runtime catch-all →
            # ``AgentStatus.ERROR`` with the message visible in the CLI/web UI.
            msg = f"Failed to generate conversation summary: {type(e).__name__}: {e}"
            logger.error(msg)
            raise SummarizationFailedError(msg) from e

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
        decision = _hot_cold_decision(
            self.model,
            messages,
            thread_id,
            ttl_overrides=self._hot_cache_ttl_overrides,
        )
        summary, port_result = await self._acreate_summary_for_prompt(
            messages_to_summarize,
            summary_prompt,
            thread_id,
            decision=decision,
        )
        origin = None
        usage = None
        if port_result is not None:
            usage = getattr(port_result, "usage", None)
        if decision is not None:
            origin = getattr(decision, "last_origin", None)
        new_messages = self._build_new_messages_with_path(
            summary, file_path, origin=origin, usage=usage
        )
        preserved_messages = self._stamp_preserved_origin(preserved_messages, origin)
        preserved_messages = self._drop_pre_summary_usage(preserved_messages)
        reset = getattr(type(self.model), "areset_after_summary", None)
        if asyncio.iscoroutinefunction(reset) and thread_id:
            # Call through the bound attribute: the port is optional and not on
            # BaseChatModel, so a direct self.model.areset_after_summary(...)
            # is an unresolved-attribute error for the type checker.
            await reset(self.model, thread_id=thread_id)
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
        thread_id = None
        try:
            cfg = get_config()
            thread_id = (cfg.get("configurable") or {}).get("thread_id")
        except Exception:  # noqa: BLE001
            thread_id = None
        incoming = state["messages"]
        tokens_before = self._cached_total_tokens(incoming)
        result = await self._arun_compaction(
            incoming,
            runtime,
            force=False,
            thread_id=thread_id,
        )
        if result.summarized:
            # REQ-90: a REAL size-based compaction fired mid-turn. Record it on
            # the instance so the turn owner can report it alongside the turn's
            # result; the caller cannot see it otherwise, because this hook's
            # return value goes into the graph state, not into the turn reply.
            #
            # Stamped here rather than inside _arun_compaction on purpose: that
            # body is shared with apply_requested_summary (force=True), which is
            # an explicit operator command and must NOT look like an incident.
            replacement = result.replacement_messages or []
            self.last_turn_compaction = {
                "occurred": True,
                "observable": True,
                "reason": "size_trigger",
                "trigger": _describe_trigger(self.trigger),
                "tokens_before": tokens_before,
                "tokens_after": self._cached_total_tokens(replacement),
                "messages_before": len(incoming),
                "messages_after": len(replacement),
                "history_path": result.history_path,
                "at_utc": _utc_now_iso(),
            }
        if result.replacement_messages is None:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *result.replacement_messages,
            ]
        }
