"""Live context gauge for codex — driven by filesystem events, never polled.

WHY THIS EXISTS
───────────────────────────────────────────────────────────────────────────
The web UI's ``Context`` field sat frozen for minutes while a codex agent
worked and then jumped (owner report 2026-08-25, rg1_worker: 40.9k → 71.7k,
9.5 minutes apart — real records, ``prompt_tokens`` 40884 then 71716).

The cause is not a bug in the gauge but the shape of the transport. The
provider emits usage in exactly one place, ``TurnCompletedEvent``, and a
codex *turn* is NOT a model call: it is one prompt → final answer, covering
the entire internal loop. Measured on rg1_worker's own rollouts: 48
``token_count`` events across 4 ``task_complete`` — roughly a dozen model
calls per turn, whose context growth (18k → 21k → 26k → 27k → 28k …) the UI
never saw. And the stream cannot help: on codex-cli 0.149.1 both ``--json``
and ``--experimental-json`` carry usage ONLY in ``turn.completed`` (verified
live: a three-step turn produced three ``item.started`` / five
``item.completed`` and zero token-bearing events until the end).

WHERE THE LIVE NUMBER COMES FROM
───────────────────────────────────────────────────────────────────────────
Codex writes a rollout JSONL for the session as it goes and appends one
``token_count`` entry per model call:

    payload.info.last_token_usage.input_tokens   ← THIS call's context
    payload.info.total_token_usage.input_tokens  ← cumulative spend

``last_token_usage`` is the number an operator can compare with ``-l``.
``total_token_usage`` must never be shown as context: it is the sum across
calls (``turn.completed`` reports the same sum — five calls of ~21.5k came
back as 108058), and mistaking one for the other is the 2026-08-22
"507.3k vs -l 250k" defect.

The file is appended DURING the turn — verified live on rg1_worker: +1664
bytes and one new ``token_count`` within 45 s of watching.

WHY A WATCHER AND NOT A TIMER
───────────────────────────────────────────────────────────────────────────
Owner constraint 2026-08-25: event-driven, no polling. The kernel already
knows when the file grows; ``watchfiles`` (a declared dependency, and the
same mechanism the LSP change-detector uses under
POLICY_NO_INPROCESS_HEAVY_MCP §3.5) turns inotify events into an async
iterator. Only the APPENDED bytes are parsed — a long session's rollout is
megabytes (rg1_worker: 906 KB and growing), so re-reading it per event
would be the polling cost we are avoiding, merely hidden.

Codex's app-server protocol does expose a first-class
``thread/tokenUsage/updated`` notification carrying ``last``/``total`` and
``modelContextWindow``. That is the better long-term source, but it is a
different transport (``codex app-server`` instead of ``codex exec``); the
migration is tracked separately and is not worth taking for one gauge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

_logger = logging.getLogger(__name__)

# Change-event debounce for the rollout watcher. Small enough that the gauge
# tracks the agent, large enough that a burst of appends is one wake-up.
_DEBOUNCE_MS = 200
_STEP_MS = 50

OnContext = Callable[[int, "int | None"], Awaitable[None]]
ChangesFactory = Callable[[Path], AsyncIterator[Any]]


@dataclass(frozen=True)
class TokenCount:
    """One ``token_count`` entry: what a single model call read."""

    context_used: int
    total_input: int | None


def parse_token_counts(chunk: str) -> list[TokenCount]:
    """Every ``token_count`` entry in ``chunk``, oldest first.

    Tolerant by construction: the tail of a file being appended can be half
    a line, and a rollout carries many entry types we do not care about.
    Anything unparsable is skipped — never raised, never guessed at.
    """
    out: list[TokenCount] = []
    for raw in chunk.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # partial tail or noise
        if not isinstance(entry, dict):
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        last = info.get("last_token_usage")
        if not isinstance(last, dict):
            continue
        used = last.get("input_tokens")
        if not isinstance(used, int) or isinstance(used, bool):
            continue
        total_blob = info.get("total_token_usage")
        total = None
        if isinstance(total_blob, dict):
            candidate = total_blob.get("input_tokens")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                total = candidate
        out.append(TokenCount(context_used=used, total_input=total))
    return out


def latest_context_used(chunk: str) -> int | None:
    """The most recent per-call context size in ``chunk``, or ``None``."""
    counts = parse_token_counts(chunk)
    return counts[-1].context_used if counts else None


from deep_worker_cli.context_limit.live_gauge import LiveContextWatcher


def _default_changes(path: Path) -> AsyncIterator[Any]:
    """inotify-backed change stream for ``path`` (lazy import).

    Watches the file when it exists and its directory otherwise, so a
    rollout created moments after the turn starts is still picked up.
    """
    from watchfiles import awatch  # noqa: PLC0415 — optional at import time

    target = path if path.exists() else path.parent
    return awatch(target, debounce=_DEBOUNCE_MS, step=_STEP_MS)


class RolloutContextWatcher(LiveContextWatcher):
    """Codex's rollout, read through the ONE shared gauge implementation.

    This used to be a second, hand-written copy of ``LiveContextWatcher``,
    and the copy drifted exactly where copies do. Two defects lived only
    here and were found live on 2026-08-26:

    * ``stop`` released the inotify handle without a final drain, so a turn
      that ended within the change-stream's debounce reported NOTHING. A
      one-word prompt through the provider produced zero readings and an
      unstamped message — and because the codex stamp is built from
      ``last_provider_context`` after the ``finally`` that stops the watch,
      the guard and the summarizer saw UNKNOWN for the whole turn;
    * a drain emitted only ``counts[-1]``, so two model calls landing
      between two wake-ups were reported as one. Tool round-trips are fast
      and the stream is debounced at 200 ms, so that is a routine burst, not
      a corner case — and it is invisible to any fixture that signals after
      every single append.

    Subclassing the shared watcher is therefore the fix AND the guarantee:
    there is no longer a second implementation to drift. The only codex
    specific left is how a chunk is parsed.

    ``total_token_usage`` is deliberately dropped on the way through: it is
    the cumulative spend across calls, never a context size (the 2026-08-22
    "507.3k vs -l 250k" defect), and every consumer already ignores it.
    """

    def __init__(
        self,
        *,
        path: Path,
        on_context: Callable[[int, "int | None"], Awaitable[None]],
        changes_factory: Callable[[Path], AsyncIterator[Any]] | None = None,
    ) -> None:
        super().__init__(
            path=path,
            parse=lambda chunk: [c.context_used for c in parse_token_counts(chunk)],
            on_context=on_context,
            changes_factory=changes_factory or _default_changes,
        )


def rollout_path_for(session_uuid: str, codex_home: Path) -> Path | None:
    """Newest rollout file for ``session_uuid``, or ``None``.

    Same glob-by-uuid strategy as ``last_token_usage_from_rollout``: the
    path is date-sharded and a turn can cross midnight, so today's date must
    never be constructed.
    """
    if not session_uuid:
        return None
    sessions_dir = codex_home / "sessions"
    try:
        candidates = sorted(
            sessions_dir.rglob(f"*{session_uuid}.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None
