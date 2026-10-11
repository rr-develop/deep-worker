"""Live context gauge: one reading per model call, driven by file events.

Owner directive 2026-08-25: "All providers must track live data, and
not only at the very end but after every call to the model."

Why a file watcher at all. A CLI-backed provider runs a whole agentic loop
inside ONE deep-worker turn — a dozen model calls between two LangGraph
invocations — and the wire protocols report usage only at the turn
boundary: codex carries it in ``turn.completed``, grok in the
``session/prompt`` reply's ``_meta``, and grok sends no ``usage_update`` at
all. Measured on a production agent: eight model calls inside one turn while the
recorded context did not move, climbing 260740 → 274277 unseen — past the
operator's ``-l 250000``.

Both CLIs, however, write their own telemetry as they go: codex appends
``token_count`` per model call to the session rollout, grok appends
``shell.turn.inference_done`` to ``~/.grok/logs/unified.jsonl``. Those are
the provider's own numbers (POLICIES §13), just in a different transport.

The kernel already knows when a file grows, so this is event-driven —
``watchfiles``, the same mechanism the LSP change-detector uses, never a
timer. Only the APPENDED bytes are parsed: these logs are megabytes and
re-reading one per event would be the polling cost we are avoiding, merely
hidden.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypeAlias

_logger = logging.getLogger(__name__)

# Change-event debounce: small enough that the gauge tracks the agent, large
# enough that a burst of appends is one wake-up.
_DEBOUNCE_MS = 200
_STEP_MS = 50

OnContext: TypeAlias = Callable[[int, "int | None"], Awaitable[None]]
ParseReadings: TypeAlias = Callable[[str], "list[int]"]
ChangesFactory: TypeAlias = Callable[[Path], AsyncIterator[Any]]


def default_changes(path: Path) -> AsyncIterator[Any]:
    """inotify-backed change stream for ``path`` (lazy import).

    Watches the file when it exists and its directory otherwise, so a log
    created moments after the turn starts is still picked up.
    """
    from watchfiles import awatch  # noqa: PLC0415

    target = path if path.exists() else path.parent
    return awatch(target, debounce=_DEBOUNCE_MS, step=_STEP_MS)


class LiveContextWatcher:
    """Report a context reading per model call, from a provider's own log.

    ``parse`` receives the newly appended text and returns the readings it
    contains, oldest first — it is where a provider's log format (and, for a
    host-wide log, its session filter) lives.

    Best-effort telemetry: every failure path degrades to "no live gauge"
    and NEVER touches the turn. ``start`` takes the current end of file as
    the baseline, so history already on disk — for a host-wide log, every
    other agent's traffic — is not replayed as if it were happening now.
    """

    def __init__(
        self,
        *,
        path: Path,
        parse: ParseReadings,
        on_context: OnContext,
        changes_factory: ChangesFactory | None = None,
    ) -> None:
        self._path = Path(path)
        self._parse = parse
        self._on_context = on_context
        self._changes_factory = changes_factory or default_changes
        self._offset = 0
        self._last_emitted: int | None = None
        self.task: asyncio.Task[None] | None = None

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> None:
        """Begin watching. Idempotent; never raises."""
        if self.task is not None and not self.task.done():
            return
        self._offset = self._size()
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Flush, then cancel the watch and release the inotify handle.

        The final drain is not optional. A change event is debounced, and a
        turn can end within milliseconds of the last model call — measured
        live on grok: the reading was appended to the log, the prompt reply
        arrived immediately after, the turn ended, and the watcher was
        cancelled before it ever woke up. The last call of every short turn
        would go unreported. Idempotent; never raises.
        """
        task, self.task = self.task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        with contextlib.suppress(Exception):
            await self._drain()

    # -- internals ------------------------------------------------------
    def _size(self) -> int:
        try:
            return self._path.stat().st_size
        except OSError:
            return 0

    async def _run(self) -> None:
        try:
            async for _change in self._changes_factory(self._path):
                await self._drain()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — telemetry must not kill a turn
            _logger.debug(
                "live-gauge: watcher stopped for %s", self._path, exc_info=True
            )

    async def _drain(self) -> None:
        """Parse only the bytes appended since the last read and report every
        new reading, oldest first.

        Each reading is one model call, so they are all reported rather than
        collapsed: the operator asked to see the context after every call.
        """
        try:
            size = self._size()
            if size < self._offset:
                self._offset = 0  # truncated/rotated underneath us
            if size == self._offset:
                return
            with self._path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return  # missing/unreadable log: no gauge, no noise

        for used in self._parse(chunk):
            if (  # pyright: ignore[reportUnnecessaryIsInstance]
                not isinstance(used, int) or isinstance(used, bool) or used <= 0
            ):
                continue
            if used == self._last_emitted:
                continue
            self._last_emitted = used
            try:
                await self._on_context(used, None)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a broken sink must not stop us
                _logger.debug("live-gauge: sink failed", exc_info=True)
