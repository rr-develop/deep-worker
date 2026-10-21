"""The SummarizerPort — the single interface every session-stateful provider implements.

WHY THIS EXISTS
---------------
A *session-stateful* provider keeps the conversation on ITS side: a CLI
subprocess (ACP, codex app-server, claude-code-sdk) holds the history and we
send only the unsent delta (``messages[sent_count:]``). Re-``ainvoke``-ing such
a model on the agent's own thread replays the ENTIRE history into a fresh
provider session — the exact history dump summarization exists to avoid, at
full re-tokenization cost.

So these providers cannot be summarized the generic way. Each must expose a
way to summarize *in place*. Before this module that requirement was folk
knowledge:

  * ``SummarizerPort`` was named in four docstrings but was not a type. Nothing
    declared it, nothing enforced it, nothing could be checked against it.
  * Both call sites duck-typed it — ``getattr(model, "areset_after_summary")``
    (agent_process.py) and ``getattr(type(model), ...)`` (summarization.py).
    A provider that misspells a method degrades SILENTLY to "no port".
  * The drift is not hypothetical: ``AcpChatModel`` spells the halves
    ``_summarize_hot``/``_summarize_cold`` while ``ClaudeCodeChatModel`` spells
    them ``_asummarize_hot``/``_asummarize_cold``, and the latter never grew an
    ``areset_after_summary`` at all.
  * ``CodexChatModel`` and ``GrokChatModel`` implement neither half. For codex
    that is a loud daily crash (``SummarizationFailedError``, agent_a
    2026-08-23). For grok it was worse — misclassified as stateless, it sailed
    past the guard into the silent full-history dump.

This module makes the contract explicit, typed and checkable, so adding the
next CLI provider is "implement two methods" rather than "rediscover the
protocol and add another branch".

THE CONTRACT
------------
``asummarize`` returns a :class:`SummaryResult` and MUST NOT touch the agent's
live session — not its ``sent_count``, not its session id, not its epoch.

    hot   the provider-side prefix is still warm. Summarize WITHOUT resending
          history: prompt in-session, fork-resume, or ask the CLI to compact
          itself. Cheap, no re-tokenization.
    cold  the prefix is gone. Use an EPHEMERAL provider session and send the
          instruction plus the already-trimmed messages. The live session is
          still untouched.

Which one to take is not the provider's choice — the middleware decides via
``hot_cache.decide()`` and passes it in ``decision.hot``.

Failure is FATAL and LOUD (POLICIES §12): raise. Never return an error string
and never return an empty summary — both are indistinguishable from success to
the caller, and the caller is about to delete the history the summary replaces.

``areset_after_summary`` runs after the summary is installed, to drop any
provider-side state that outlived the wipe. It must be idempotent.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from .hot_cache import SummaryResult

__all__ = [
    "SummarizerPort",
    "implements_summarizer_port",
]


@runtime_checkable
class SummarizerPort(Protocol):
    """Structural interface for session-stateful providers.

    A ``Protocol`` rather than a base class on purpose: the concrete models
    already inherit ``BaseChatModel`` (a pydantic model with its own metaclass),
    and threading a second ABC through that hierarchy buys nothing here. What we
    need is a *checkable declaration* of the contract — which a Protocol gives
    us statically, plus :func:`implements_summarizer_port` at runtime.
    """

    @abstractmethod
    async def asummarize(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        instruction: str,
        decision: Any,
    ) -> SummaryResult:
        """Summarize without dumping history into the live session.

        Args:
            thread_id: the agent's thread id (its slot key, NOT the provider's
                session id — the provider maps it to its own).
            messages: already trimmed to budget by the middleware. Used by the
                cold path only; the hot path must ignore them.
            instruction: the summarization prompt.
            decision: carries ``.hot`` (bool). Take the hot path only when true.

        Returns:
            A populated :class:`SummaryResult`. ``text`` must be non-empty.

        Raises:
            Exception: on any failure. Never encode failure in the return value.
        """
        ...

    @abstractmethod
    async def areset_after_summary(self, *, thread_id: str) -> None:
        """Drop provider-side state that must not outlive the summary wipe.

        Called after the new summary-shaped history is installed. Must be
        idempotent: both the middleware and ``sum_query`` may invoke it.
        """
        ...


def implements_summarizer_port(model: Any) -> bool:
    """Whether ``model`` genuinely implements the port.

    Checks the TYPE, and checks that both halves are ``async def``.

    Looking at the type (not the instance) is deliberate and load-bearing: a
    ``MagicMock`` yields a truthy child attribute for any name, so an
    instance-level probe reports a working port on any test double and drags
    unrelated tests into the stateful path. This is the same failure that broke
    seven summarization tests in reviewer round B2 (2026-08-21); do not
    "simplify" it back to ``getattr(model, ...)``.

    Requiring BOTH halves is also deliberate. A provider with ``asummarize`` and
    no ``areset_after_summary`` is a half-migrated provider, and the silent
    ``getattr`` probes are precisely what let that state persist unnoticed.
    """
    import asyncio  # noqa: PLC0415 — keep module import-light for deepagents

    model_type = type(model)
    return all(
        asyncio.iscoroutinefunction(getattr(model_type, name, None))
        for name in ("asummarize", "areset_after_summary")
    )
