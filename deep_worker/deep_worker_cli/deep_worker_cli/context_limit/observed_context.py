"""The PROVIDER-reported context size — the only number that may decide.

Owner directive 2026-08-25:

    "I HAVE ALWAYS DEMANDED HONEST DATA FROM THE PROVIDER, AND THAT IS
    EXACTLY WHAT MUST BE USED. AT THE MOMENT THERE ARE NO EXCEPTIONS. Any
    fallback requires my direct and explicit consent."

Why the rule is absolute. Both consumers — ``SummarizationMiddleware`` and
the pre-call guard — used to derive their number from a chars-per-token
table over the LangChain message list. That number cannot see what is not
in the list (the CLI's own prompt and tool schemas, extended-thinking
blocks, full-size built-in results) and cannot know a provider's
tokenizer. It was wrong in both directions, and both directions cost:

* rg1_worker 2026-08-19 — table said ~150k while the CLI held 468k against
  ``-l 400k``: nothing fired, and the next turn loaded 466k;
* rg1_worker 2026-08-25 — table said 214613 against a real 90k (``-l``
  250k): the guard's amber hint fired on fiction, and the extra message it
  injected tripped the codex provider's shrink detector, which dropped a
  warm session and restarted its context from zero.

Two shapes count as a provider report, and nothing else does:

* **the stamp** — a chat model writes the CLI's own gauge onto the returned
  ``AIMessage`` via :func:`annotate_context_observation` (ACP
  ``usage_update``, codex ``last_token_usage``, claude-code-sdk ``usage``).
  It carries the id of the first NON-system message of the history it
  measured, so a stamp from a pre-summarization conversation is never
  reused;
* **``usage_metadata.input_tokens``** — what a standard API provider says
  it read for that call. Same class of fact, different transport.

No report means UNKNOWN, and UNKNOWN stays unknown: see
:func:`provider_context_tokens`. A provider that never reports is a defect
in that provider, not a licence to estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

# ``response_metadata`` key carrying the observation. Namespaced so generic
# consumers (token-stats ``_extract_usage`` reads ``token_usage`` /
# ``usage_metadata``) never mistake it for per-call usage.
CONTEXT_OBSERVATION_KEY = "deep_worker_context"

TokenCounter = Callable[[Iterable[Any]], int]


def first_state_message_id(messages: Iterable[Any]) -> str | None:
    """Return the id of the first non-``SystemMessage`` in ``messages``.

    Both producer and consumer use THIS as the identity of "the history
    this observation belongs to". The LangGraph factory prepends the
    system prompt, the guard may prepend a hint — neither is state, so
    leading ``SystemMessage``s are skipped on both sides.
    """
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        mid = getattr(m, "id", None)
        return str(mid) if mid is not None else None
    return None


def annotate_context_observation(
    message: BaseMessage,
    *,
    used: int,
    size: int | None,
    first_id: str | None,
) -> BaseMessage:
    """Stamp ``message`` (in place) with an observed context size.

    Args:
        message: The ``AIMessage`` the provider is about to return.
        used: Provider-reported context tokens AFTER this call.
        size: Provider-reported context window (informational, may be
            ``None``).
        first_id: :func:`first_state_message_id` of the request that
            produced the message — the validity key.
    """
    message.response_metadata[CONTEXT_OBSERVATION_KEY] = {
        "used": int(used),
        "size": int(size) if isinstance(size, int) else None,
        "first_id": first_id,
    }
    return message


def context_observation(message: Any) -> dict[str, Any] | None:
    """Return the raw observation dict stamped on ``message`` or ``None``."""
    md = getattr(message, "response_metadata", None)
    if not isinstance(md, dict):
        return None
    obs = md.get(CONTEXT_OBSERVATION_KEY)
    if not isinstance(obs, dict):
        return None
    used = obs.get("used")
    if isinstance(used, bool) or not isinstance(used, int) or used <= 0:
        return None
    return obs


def find_context_anchor(messages: Sequence[Any]) -> tuple[int, int] | None:
    """Return ``(index, used)`` of the latest VALID observation in ``messages``.

    An observation is valid iff its ``first_id`` equals the id of the first
    state message of ``messages`` (see module docstring). The LAST valid
    one wins — it is the most recent measurement of the same history.
    """
    first_id = first_state_message_id(messages)
    if first_id is None:
        return None
    anchor: tuple[int, int] | None = None
    for i, m in enumerate(messages):
        obs = context_observation(m)
        if obs is None:
            continue
        if obs.get("first_id") != first_id:
            continue
        anchor = (i, int(obs["used"]))
    return anchor


def provider_context_tokens(messages: Sequence[Any]) -> int | None:
    """The context the PROVIDER reported for this history, or ``None``.

    Two shapes count as a provider report, and nothing else does:

    * a ``deep_worker_context`` stamp written by a chat model from the
      CLI's own gauge (ACP ``usage_update``, codex ``last_token_usage``,
      claude-code-sdk ``usage``) — validated against ``first_id`` so a
      stamp measuring a pre-summarization history is never reused;
    * ``usage_metadata.input_tokens`` on an ``AIMessage`` — what a standard
      API provider says it read for that call. It is the provider's own
      tokenizer output, i.e. the same class of fact as the stamp.

    ``None`` means UNKNOWN, and UNKNOWN must stay unknown: estimating here
    is what let a 2.4x fiction (214613 claimed vs 90k real, rg1_worker
    2026-08-25) drive summarization and guard decisions. Owner directive
    2026-08-25: provider numbers only, no fallback without explicit
    approval, and no provider is approved for one today.
    """
    msgs = list(messages)
    anchor = find_context_anchor(msgs)
    if anchor is not None:
        return anchor[1]
    for msg in reversed(msgs):
        used = _usage_metadata_input_tokens(msg)
        if used is not None:
            return used
    return None


def _usage_metadata_input_tokens(message: Any) -> int | None:
    """``usage_metadata.input_tokens`` when it is a real, positive count."""
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict):
        return None
    used = meta.get("input_tokens")
    if isinstance(used, bool) or not isinstance(used, int) or used <= 0:
        return None
    return used


def provider_context_token_counter(
    floor: Callable[[], int | None] | None = None,
) -> TokenCounter:
    """``token_counter`` for ``SummarizationMiddleware`` — provider data only.

    Returns the reported context size, or ``0`` when nothing has been
    reported. Zero keeps the trigger silent: a summarization decision taken
    on a number nobody measured is exactly the failure being removed.

    ``floor`` supplies a PERSISTED provider reading (the provider's own
    session state, or the last ``context_snapshot`` in ``token_stats``) for
    the first call after a restart, when the checkpoint carries no stamp
    yet — rg1_worker 2026-08-19 went blind precisely there while the CLI
    held 468k. A persisted provider number is still the provider's number,
    not an estimate; a LIVE report always wins over it.
    """

    def _counter(messages: Iterable[Any]) -> int:
        reported = provider_context_tokens(list(messages))
        if reported is not None:
            return reported
        if floor is None:
            return 0
        try:
            persisted = floor()
        except Exception:  # noqa: BLE001 — a broken floor must not decide
            return 0
        if isinstance(persisted, int) and not isinstance(persisted, bool) and persisted > 0:
            return persisted
        return 0

    _counter.__name__ = "provider_context_token_counter"
    return _counter
