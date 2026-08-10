"""Filesystem watcher feeding the change detector.

See ``docs/DESIGN_LSP_WATCHER_WIRING.md`` and §9 of
``docs/DESIGN_LSP_SHARED_SERVICE.md``.

Why this module exists
----------------------
``change_detector.py`` decides *whether* a filesystem change is
structural. Until this module existed nothing ever asked it: the
detector had no imports outside its own file and ``notify()`` was
called only from tests. A running service therefore watched nothing —
adding a module to a project went unnoticed until someone issued a
manual ``RefreshProject``.

Division of responsibility (deliberate)
---------------------------------------
This watcher is intentionally **dumb**. It translates batches of
filesystem events into ``detector.notify(paths)`` and nothing else.
Debouncing and the structural-snapshot comparison stay in the
detector so that the §9 provider table remains the single source of
truth about what a refresh means. Duplicating any of that logic here
would create a second place where "should we refresh?" is decided.

Policy notes
------------
* **No polling.** ``watchfiles`` can fall back to a poll loop; the
  project forbids polling outright, so asking for it raises instead of
  silently degrading. If inotify is genuinely unavailable the service
  runs without auto-refresh and says so — manual ``RefreshProject``
  still works. A silent poll loop would satisfy the tests and violate
  the policy.
* **No new threads.** The watcher is an ``asyncio`` task on the
  service's own loop.
* **Noise filtering** is not micro-optimisation: without it a single
  ``git status`` or test run produces thousands of events, and §9
  explicitly requires refreshes to be economical.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Final

from watchfiles import Change, DefaultFilter, awatch

from deep_worker_cli.lsp.service.change_detector import ChangeDetector

_log = logging.getLogger(__name__)

#: ``awatch`` runs its own debounce before yielding a batch. The
#: detector already debounces per §9 (300 ms), so this one is kept
#: short: it exists only to coalesce the burst of inotify events a
#: single ``write()`` produces, not to add a second policy layer.
_BATCH_DEBOUNCE_MS: Final = 50

#: Cap on how long ``start()`` may spend arming the inotify watches
#: before it gives up and returns unready (with a warning). Comfortably
#: above the ~50–200 ms it actually takes in practice, but bounded so a
#: broken host cannot hang startup.
_READY_TIMEOUT_S: Final = 3.0

#: Between probe touches during ``start()``. Short enough that the
#: readiness path adds negligible latency under normal conditions.
_READY_RETRY_S: Final = 0.2

#: Directories that never carry structural meaning but generate huge
#: event volumes. ``DefaultFilter`` already covers ``.git``,
#: ``__pycache__``, ``node_modules`` and ``.venv``; the rest are build
#: outputs common in the languages the §9 table supports.
_EXTRA_IGNORE_DIRS: Final = (
    "build",
    "target",
    "dist",
    "out",
    ".gradle",
    ".idea",
    ".tox",
    ".ruff_cache",
    "htmlcov",
)


class _ProjectFilter(DefaultFilter):
    """``DefaultFilter`` plus build-output directories."""

    def __init__(self) -> None:
        super().__init__(
            ignore_dirs=[*DefaultFilter.ignore_dirs, *_EXTRA_IGNORE_DIRS]
        )


class ProjectWatcher:
    """Watch ``root`` and hand every change batch to ``detector``.

    Lifecycle mirrors the server that owns it: :meth:`start` spawns the
    task, :meth:`stop` asks ``awatch`` to finish via its stop event and
    awaits the task. ``stop`` is idempotent because it is called both
    from ``LSPServiceServer.stop`` and from test teardown.
    """

    def __init__(
        self,
        root: Path,
        detector: ChangeDetector,
        *,
        force_polling: bool = False,
    ) -> None:
        if force_polling:
            # Loud on purpose: this is the exact knob that turns an
            # event-driven design into a polling one by accident.
            raise ValueError(
                "force_polling is refused: polling is forbidden by project "
                "policy. If inotify is unavailable the service runs without "
                "auto-refresh rather than degrading to a poll loop."
            )
        self._root = Path(root).resolve()
        self._detector = detector
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._ready = asyncio.Event()
        self._probe_name = ".dw-watcher-ready"

    # ── introspection (surfaced through ProjectStatus) ───────────────

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def watched_paths(self) -> list[Path]:
        """Roots handed to inotify. One entry today; kept a list so a
        provider needing extra roots (e.g. a sibling gradle home) does
        not change the shape of ``ProjectStatus``."""
        return [self._root]

    @property
    def active_providers(self) -> list[str]:
        return [r.name for r in self._detector.rules]

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bring the watcher fully online before returning.

        ``awatch`` installs its inotify watches lazily inside the
        generator: the task returns to the event loop before any
        watch is armed. A caller that writes a manifest right after
        ``await start()`` loses that write — empirically reproducible,
        and the reason this synchronisation exists.

        We arm a self-probe: touch a marker file **after** the
        watcher task has had a chance to enter ``awatch``. The
        watcher marks itself ready the first time it observes any
        event; the marker guarantees such an event exists as soon as
        the inotify watch is actually armed. To defeat the race
        between "watch not yet armed" and "probe already written",
        we re-touch the marker in a bounded loop with a short sleep
        between attempts — each sleep is a single kernel timer, not
        a poll loop, and the whole loop is capped.
        """
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._ready = asyncio.Event()
        await self._detector.start()
        self._task = asyncio.create_task(self._run(), name="lsp-project-watcher")

        probe = self._root / self._probe_name
        deadline = _READY_TIMEOUT_S
        elapsed = 0.0
        try:
            while not self._ready.is_set() and elapsed < deadline:
                probe.write_text(str(elapsed))
                try:
                    await asyncio.wait_for(
                        self._ready.wait(), timeout=_READY_RETRY_S
                    )
                except asyncio.TimeoutError:
                    elapsed += _READY_RETRY_S
            if not self._ready.is_set():  # pragma: no cover — inotify absent
                _log.warning(
                    "project watcher for %s did not confirm readiness in "
                    "%.1fs; auto-refresh may miss early events",
                    self._root,
                    deadline,
                )
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

    async def stop(self) -> None:
        self._stop_event.set()
        task, self._task = self._task, None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover — defensive
                pass
        await self._detector.stop()

    async def _run(self) -> None:
        try:
            async for batch in awatch(
                self._root,
                watch_filter=_ProjectFilter(),
                debounce=_BATCH_DEBOUNCE_MS,
                stop_event=self._stop_event,
            ):
                # First delivery of any batch proves the inotify
                # watches are installed — that is what ``start()``
                # is waiting for.
                self._ready.set()
                paths = [
                    Path(p)
                    for _change, p in batch
                    if Path(p).name != self._probe_name
                ]
                if paths:
                    await self._detector.notify(paths)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — must not die silently
            # A dead watcher means the model silently goes stale, which
            # is far worse than a loud log line.
            _log.exception(
                "project watcher for %s stopped unexpectedly; auto-refresh "
                "is no longer active",
                self._root,
            )


__all__ = ["ProjectWatcher", "Change"]
