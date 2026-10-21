"""Auto-restart must survive a missing ``pidfd_open``.

``SerenaSupervisor._arm_monitor`` used to catch ``OSError`` and return,
logging "auto-restart disabled".  On a kernel below 5.3, or a CPython
built without ``<sys/pidfd.h>``, that silently turned the supervisor into
a no-op: the Serena child could die and nothing would bring it back.

``docs/REQUIREMENTS.md`` forbids that — a post-4.19 syscall may sharpen a
path, never gate it.  These tests pin the fallback: when no pidfd can be
opened, the supervisor watches the child with ``waitpid`` on the default
executor and still restarts it.

The child here is a **real process** that exits only when we tell it to.
A stub that resolves immediately would pass whether or not the fallback
was wired, because the restart would be driven by the stub rather than by
observed process death.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import subprocess
import sys
from typing import Any, Callable

import pytest

from deep_worker_cli.lsp.service.serena_supervisor import SerenaSupervisor


# A child that lives until its stdin closes — so the test, and only the
# test, decides when the process dies.
_LIVE_CHILD = (
    "import sys; sys.stdin.read(1)"
)


def _enosys(_pid: int, _flags: int = 0) -> int:
    """Stand in for a kernel that does not implement ``pidfd_open``."""
    raise OSError(errno.ENOSYS, "Function not implemented")


def _no_attribute(_pid: int, _flags: int = 0) -> int:
    """Stand in for a CPython built without ``<sys/pidfd.h>``."""
    raise AttributeError("module 'os' has no attribute 'pidfd_open'")


class _StubServer:
    def __init__(self) -> None:
        self.ready = False
        self.attached: list[Any] = []
        self._backend: Any | None = None
        self.restart_notes: list[bool] = []

    def note_restart(self, success: bool) -> None:
        self.restart_notes.append(success)

    async def attach_backend(self, backend: Any) -> bool:
        self.attached.append(backend)
        self._backend = backend
        self.ready = True
        return True

    def mark_backend_not_ready(self) -> None:
        self.ready = False


class _RealChildBackend:
    """Backend whose ``pid`` is a real, living process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _LIVE_CHILD],
            stdin=subprocess.PIPE,
        )
        self.stopped = False

    @property
    def pid(self) -> int | None:
        return self.proc.pid

    def kill_child(self) -> None:
        """Make the real child exit, then let the kernel reap it."""
        self.proc.kill()

    async def stop(self) -> None:
        self.stopped = True
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()

    def set_on_lost(self, _hook: Callable[[], None]) -> None:
        pass

    def _mark_lost(self) -> None:
        pass


@pytest.fixture
def backends() -> list[_RealChildBackend]:
    made: list[_RealChildBackend] = []
    yield made
    for b in made:
        if b.proc.poll() is None:
            b.proc.kill()
        b.proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("opener", "label"),
    [(_enosys, "ENOSYS"), (_no_attribute, "AttributeError")],
)
async def test_auto_restart_works_without_pidfd(
    backends: list[_RealChildBackend],
    opener: Callable[[int, int], int],
    label: str,
) -> None:
    """The child dies, and the supervisor replaces it — with no pidfd.

    Oracle: a SECOND real child exists after the first one is killed.
    That can only happen if the death of a real process was observed and
    drove ``_on_death`` — the pidfd path is unavailable by construction.
    """
    server = _StubServer()

    def _factory() -> _RealChildBackend:
        b = _RealChildBackend()
        backends.append(b)
        return b

    sup = SerenaSupervisor(
        server=server, backend_factory=_factory, pidfd_open_fn=opener
    )

    assert await sup.start() is True
    assert len(backends) == 1, f"[{label}] first backend not created"
    first = backends[0]

    # Kill the real child. Nothing else can set the death event.
    first.kill_child()

    # Wait on the supervisor's own machinery, not on wall-clock polling:
    # the restart is finished when a second backend has been attached.
    await asyncio.wait_for(
        _until(lambda: len(server.attached) >= 2), timeout=10
    )

    assert len(backends) == 2, (
        f"[{label}] supervisor did not create a replacement backend; "
        f"auto-restart was disabled by the missing pidfd"
    )
    assert backends[1].proc.poll() is None, (
        f"[{label}] replacement child is not running"
    )
    assert server.ready is True

    await sup.stop()


@pytest.mark.asyncio
async def test_missing_pidfd_is_logged_loudly(
    backends: list[_RealChildBackend],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback announces itself, and does not claim to be disabled.

    A silent degradation is the defect this guards: the old code logged
    "auto-restart disabled" and meant it.
    """
    server = _StubServer()

    def _factory() -> _RealChildBackend:
        b = _RealChildBackend()
        backends.append(b)
        return b

    sup = SerenaSupervisor(
        server=server, backend_factory=_factory, pidfd_open_fn=_enosys
    )

    with caplog.at_level(logging.WARNING):
        assert await sup.start() is True

    text = caplog.text
    assert "pidfd_open" in text, "the fallback did not announce itself"
    assert "waitpid" in text, "the log does not name the fallback used"
    assert "auto-restart disabled" not in text, (
        "the supervisor still claims auto-restart is disabled"
    )

    await sup.stop()


@pytest.mark.asyncio
async def test_disarming_releases_the_waitpid_watch(
    backends: list[_RealChildBackend],
) -> None:
    """``stop()`` detaches the no-pidfd watch from this supervisor.

    Two things are checked, because the attribute alone would not catch
    the defect. The reference must be dropped — otherwise a stopped
    supervisor pins the Future for the lifetime of the object — and the
    done-callback must no longer reach the death event, or a child
    reaped *after* shutdown would set an event the supervisor has
    already finished with.

    The child is killed only after ``stop()`` returns, so the callback
    fires strictly in the window the fix is about.
    """
    server = _StubServer()

    def _factory() -> _RealChildBackend:
        b = _RealChildBackend()
        backends.append(b)
        return b

    sup = SerenaSupervisor(
        server=server, backend_factory=_factory, pidfd_open_fn=_enosys
    )

    # Declared up front, not conjured by the first armed monitor:
    # reading it on a fresh supervisor must not be an AttributeError.
    assert sup._waitpid_future is None

    assert await sup.start() is True
    assert sup._waitpid_future is not None, (
        "the no-pidfd path did not record its waitpid watch"
    )

    event = sup._death_event
    assert event is not None

    await sup.stop()

    assert sup._waitpid_future is None, (
        "stop() left the waitpid Future attached; a stopped supervisor "
        "must not keep watching a child it has handed back"
    )

    # The child is reaped now — after the supervisor is done with it.
    # ``blocking_waitpid`` returns, its done-callback runs, and nothing
    # it does may touch this supervisor's event any more.
    event.clear()
    backends[0].proc.wait()
    for _ in range(100):
        await asyncio.sleep(0)

    assert not event.is_set(), (
        "a death observed after disarming still set the supervisor's "
        "death event; the done-callback was never detached"
    )


async def _until(pred: Callable[[], bool]) -> None:
    """Await ``pred`` becoming true, driven by the event loop.

    The restart runs as its own task; yielding to the loop is what lets
    it progress. There is no timed backoff here — the enclosing
    ``wait_for`` supplies the only bound.
    """
    while not pred():
        await asyncio.sleep(0)
