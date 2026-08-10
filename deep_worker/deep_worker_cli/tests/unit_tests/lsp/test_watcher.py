"""Unit tests for the watcher that ties filesystem events to ChangeDetector.

See ``docs/DESIGN_LSP_WATCHER_WIRING.md`` and §9 of the main design.

Why this module was needed at all. ``change_detector.py`` was written
and covered by tests, but used by NOBODY: not a single import outside
the module itself, not a single production ``notify`` call. The detector
made correct decisions in a vacuum, while in the running service nobody
watched the filesystem — an added module would not be noticed until a
manual ``RefreshProject``. Here the missing link is tested.

The tests wait for events EVENT-DRIVEN (``asyncio.Event`` + ``wait_for``
with a hard budget), not by sleeping "long enough": sleeping both slows
the run and masks real races.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.change_detector import ChangeDetector
from deep_worker_cli.lsp.service.watcher import ProjectWatcher

pytestmark = pytest.mark.asyncio

#: Budget for "the event must arrive". inotify+debounce fit within
#: fractions of a second; anything more is broken, not slow.
_ARRIVE_S = 10.0
#: Budget for "there must be NO event". Deliberately larger than the sum
#: of the watcher's and detector's debounce, otherwise the test is green
#: simply because we did not wait long enough.
_SILENCE_S = 2.0


def _py_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
    )
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    return root


class _Recorder:
    """Counts refresh calls and lets you wait for the first one event-driven."""

    def __init__(self) -> None:
        self.reasons: list[str] = []
        self.fired = asyncio.Event()

    def __call__(self, reason: str) -> None:
        self.reasons.append(reason)
        self.fired.set()

    async def wait(self, timeout: float = _ARRIVE_S) -> None:
        await asyncio.wait_for(self.fired.wait(), timeout=timeout)


async def _started(
    root: Path, rec: "_Recorder", **kw
) -> tuple[ProjectWatcher, ChangeDetector]:
    detector = ChangeDetector(root, debounce_s=0.05, on_refresh=rec)
    watcher = ProjectWatcher(root, detector, **kw)
    await watcher.start()
    return watcher, detector


async def test_watcher_triggers_refresh_on_manifest_change(
    tmp_path: Path,
) -> None:
    """A structural edit to the manifest must reach the detector.

    This is exactly the missing link: before the watcher existed, nobody
    noticed such a change.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    watcher, _ = await _started(root, rec)
    try:
        (root / "pyproject.toml").write_text(
            "[project]\nname='demo'\nversion='0'\ndependencies=['httpx']\n"
        )
        await rec.wait()
        assert rec.reasons, "watcher never delivered the manifest change"
    finally:
        await watcher.stop()


async def test_watcher_ignores_edits_inside_existing_sources(
    tmp_path: Path,
) -> None:
    """An edit to the body of an existing ``.py`` is NOT a reason to refresh.

    The flip side of the previous test. Without it, an implementation
    that "fires refresh on any event" passes the first test and destroys
    the whole §9 economy.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    watcher, _ = await _started(root, rec)
    try:
        (root / "pkg" / "mod.py").write_text("def f():\n    return 2\n")
        with pytest.raises(asyncio.TimeoutError):
            await rec.wait(timeout=_SILENCE_S)
        assert rec.reasons == [], f"spurious refresh: {rec.reasons}"
    finally:
        await watcher.stop()


async def test_watcher_ignores_noise_directories(tmp_path: Path) -> None:
    """Noise directories must not wake the detector.

    A single ``git status`` or test run would otherwise generate
    thousands of events; §9 explicitly requires economy.

    The test verifies TWO properties:

    1. `on_refresh` was not called (the user-visible characteristic).
    2. `detector.notify` did not receive a single path from these
       directories (internal, but it is the load-bearing filter check).

    Without point 2 the test is green with ``_ProjectFilter`` disabled,
    because the detector itself discards non-trigger paths via
    ``_is_trigger``. So externally nothing looks broken — while internally
    we already pay for thousands of wake-ups on ``git status``. It is
    exactly this economy that §9 requires.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    detector = ChangeDetector(root, debounce_s=0.05, on_refresh=rec)

    # Trap on notify: instead of intercepting via a subclass — a minimal
    # wrapper, so we do not diverge from the production detector implementation.
    from collections.abc import Sequence

    seen: list[Path] = []
    original = detector.notify

    async def spy(paths: Sequence[Path]) -> None:
        seen.extend(Path(p) for p in paths)
        await original(paths)

    # ty sees a free function vs a bound method and complains about the
    # assignment; this is exactly what we do deliberately (spying on an
    # existing instance), so we suppress it on this specific line.
    detector.notify = spy  # ty: ignore[invalid-assignment]

    watcher = ProjectWatcher(root, detector)
    await watcher.start()
    try:
        for rel in (
            # The default watchfiles filter already covers:
            ".git/objects/ab",
            "__pycache__",
            "node_modules/x",
            ".venv/lib",
            # Our extensions — covered only by _ProjectFilter:
            "build/output",
            "target/debug",
            "dist/wheels",
            ".gradle/caches",
        ):
            d = root / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "junk.json").write_text('{"noise": true}')
        with pytest.raises(asyncio.TimeoutError):
            await rec.wait(timeout=_SILENCE_S)
        assert rec.reasons == [], f"noise woke the detector: {rec.reasons}"
        noisy = [p for p in seen if any(
            part in {".git", "__pycache__", "node_modules", ".venv",
                     "build", "target", "dist", ".gradle"}
            for part in p.parts
        )]
        assert not noisy, (
            f"noise reached ChangeDetector.notify (filter would have "
            f"blocked it in production): {noisy}"
        )
    finally:
        await watcher.stop()


async def test_watcher_stops_cleanly_and_goes_silent(tmp_path: Path) -> None:
    """After ``stop()`` events no longer arrive and the task is finished.

    Otherwise a stopped service would keep writing into closed session
    queues.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    watcher, _ = await _started(root, rec)
    await watcher.stop()

    assert not watcher.running, "watcher still reports itself running"

    (root / "pyproject.toml").write_text("[project]\nname='after-stop'\n")
    with pytest.raises(asyncio.TimeoutError):
        await rec.wait(timeout=_SILENCE_S)
    assert rec.reasons == [], "watcher kept firing after stop()"


async def test_stop_is_idempotent(tmp_path: Path) -> None:
    """A repeated ``stop()`` must not fail: it is called both from the
    test's ``finally`` and from ``LSPServiceServer.stop``."""
    root = _py_project(tmp_path)
    rec = _Recorder()
    watcher, _ = await _started(root, rec)
    await watcher.stop()
    await watcher.stop()
    assert not watcher.running


async def test_watcher_reports_watched_paths_and_providers(
    tmp_path: Path,
) -> None:
    """``ProjectStatus`` must show what exactly we watch.

    The ``active_providers``/``watched_paths`` fields were hardcoded
    empty until now — zero observability.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    watcher, detector = await _started(root, rec)
    try:
        assert "python" in [r.name for r in detector.rules]
        assert watcher.watched_paths, "watcher does not expose watched paths"
        assert str(root) in [str(p) for p in watcher.watched_paths]
    finally:
        await watcher.stop()


async def test_server_starts_and_stops_watcher(tmp_path: Path) -> None:
    """LSPServiceServer must start the watcher in start() and stop it in stop().

    Otherwise the production service becomes "blind" again: the watcher
    exists, but nobody ties it to the server's lifecycle.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )

    root = _py_project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock, project_root=root, idle_shutdown_s=60.0
        )
    )
    await server.start()
    watcher = server.watcher
    assert watcher is not None, "server did not create a watcher"
    assert watcher.running, "server did not start the watcher"
    await server.stop(grace=0.1)
    assert not watcher.running, "server did not stop the watcher on shutdown"
    assert server.watcher is None, "server did not release its watcher reference"


async def test_server_publishes_refresh_events_to_attached_sessions(
    tmp_path: Path,
) -> None:
    """REFRESH_STARTED and REFRESH_FINISHED must reach subscribers.

    Otherwise the client cannot learn that the index was updated — that
    is exactly why SessionEvent was introduced.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

    root = _py_project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock, project_root=root, idle_shutdown_s=60.0
        )
    )
    await server.start()
    try:
        queue = server._register_session()
        try:
            server._trigger_refresh("test")
            events = []
            for _ in range(2):
                ev = await asyncio.wait_for(queue.get(), timeout=2.0)
                events.append(ev.event_type)
            assert pb.SESSION_EVENT_REFRESH_STARTED in events
            assert pb.SESSION_EVENT_REFRESH_FINISHED in events
            assert events.index(pb.SESSION_EVENT_REFRESH_STARTED) < events.index(
                pb.SESSION_EVENT_REFRESH_FINISHED
            ), "FINISHED must not precede STARTED"
        finally:
            server._unregister_session(queue)
    finally:
        await server.stop(grace=0.1)


async def test_status_reports_active_providers_and_watched_paths(
    tmp_path: Path,
) -> None:
    """ProjectStatus must fill the fields, not return stubs.

    Before this commit active_providers/watched_paths were hardcoded
    empty — zero observability.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

    root = _py_project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock, project_root=root, idle_shutdown_s=60.0
        )
    )
    await server.start()
    try:
        reply = await server._servicer.ProjectStatus(pb.StatusRequest(), None)
        assert "python" in list(reply.active_providers), reply.active_providers
        assert any(str(root) in p for p in reply.watched_paths), reply.watched_paths
    finally:
        await server.stop(grace=0.1)


async def test_refresh_is_single_flight(tmp_path: Path) -> None:
    """There are no concurrent refresh cycles.

    If two triggers arrive while the previous backend hook is still
    working, the second must wait for the first and not interleave
    STARTED/FINISHED events. Otherwise a client reading the stream sees
    the sequence STARTED, STARTED, FINISHED, FINISHED — and cannot match
    the pairs. And for an expensive backend (Serena, the next stage)
    parallel `refresh_project` calls are a direct race.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

    root = _py_project(tmp_path)
    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=root,
            idle_shutdown_s=60.0,
            watch_enabled=False,
        )
    )
    await server.start()
    try:
        active = 0
        peak = 0
        gate = asyncio.Event()

        async def slow_backend(reason: str) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await gate.wait()
            active -= 1

        server.set_backend_refresh_handler(slow_backend)
        queue = server._register_session()
        try:
            server._trigger_refresh("a")
            server._trigger_refresh("b")
            server._trigger_refresh("c")

            # Let the first one reach the backend hook and park there.
            await asyncio.sleep(0.05)
            assert peak == 1, (
                f"refresh handler ran in parallel (peak={peak}); the second "
                f"trigger did not wait for the first to finish"
            )

            gate.set()

            events: list[int] = []
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    ev = await asyncio.wait_for(
                        queue.get(), timeout=deadline - time.monotonic()
                    )
                except asyncio.TimeoutError:
                    break
                if ev.event_type in (
                    pb.SESSION_EVENT_REFRESH_STARTED,
                    pb.SESSION_EVENT_REFRESH_FINISHED,
                ):
                    events.append(ev.event_type)
                    if events.count(pb.SESSION_EVENT_REFRESH_FINISHED) >= 3:
                        break

            # We need STRICTLY alternating STARTED, FINISHED, STARTED, ...
            for i, e in enumerate(events):
                expected = (
                    pb.SESSION_EVENT_REFRESH_STARTED
                    if i % 2 == 0
                    else pb.SESSION_EVENT_REFRESH_FINISHED
                )
                assert e == expected, (
                    f"events out of order at position {i}: {events}"
                )
        finally:
            server._unregister_session(queue)
    finally:
        await server.stop(grace=0.1)


async def test_watcher_uses_short_awatch_debounce(tmp_path: Path) -> None:
    """``awatch`` has its own internal debounce; 1600 ms by default.

    Stacking it on top of the detector's §9 debounce (300 ms) means
    keeping structural invalidation delayed for almost two seconds per
    batch of saves. We explicitly pin a short value so that a "reverted
    to the awatch default" regression fails precisely, not through test
    timeouts.
    """
    from deep_worker_cli.lsp.service.watcher import _BATCH_DEBOUNCE_MS

    assert _BATCH_DEBOUNCE_MS <= 100, (
        f"_BATCH_DEBOUNCE_MS={_BATCH_DEBOUNCE_MS} stacks with the "
        f"detector's own debounce; keep it well under 300 ms"
    )


async def test_polling_fallback_is_refused(tmp_path: Path) -> None:
    """Polling is forbidden by project policy.

    ``watchfiles`` supports ``force_polling``, and that is exactly the
    path by which "it still works" quietly turns into a violation.
    Asking for polling is an error, not a silent degradation.
    """
    root = _py_project(tmp_path)
    rec = _Recorder()
    detector = ChangeDetector(root, debounce_s=0.05, on_refresh=rec)
    with pytest.raises(ValueError, match="polling"):
        ProjectWatcher(root, detector, force_polling=True)
