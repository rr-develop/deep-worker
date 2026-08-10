"""SerenaSupervisor — unit coverage for design §4/§5/§7 (commit 10).

We do NOT spawn Serena here. The supervisor takes injects for
``pidfd_open`` / ``add_reader`` / ``now`` so the whole restart machine
can be exercised deterministically with a stub backend.

Pinned invariants:

* Successful ``start()`` -> ``server.attach_backend`` succeeded AND
  a pidfd monitor is armed on the backend's pid.
* Failed first attach -> ``start()`` returns False, no monitor, no
  restart accounting.
* Rate-limit: 3 attempts allowed inside the 60s window, 4th blocked;
  attempts (not successes) count — MUT-Q/R.
* Restart calls the FULL ``attach_backend`` on a fresh
  ``SerenaBackend`` produced by the factory, not merely
  ``mark_backend_ready`` — MUT-W.
* Death observed AFTER ``supervisor.stop()`` never spawns a new
  backend — MUT-V (``_stopped`` gate).
* Pidfd callback removes the reader and closes the fd exactly once —
  MUT-T.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest

from deep_worker_cli.lsp.service.serena_supervisor import (
    SerenaSupervisor,
    _MAX_RESTARTS,
    _WINDOW_S,
)


# ── stub server / backend ────────────────────────────────────────────


class _StubServer:
    """Minimal stand-in for LSPServiceServer.

    Records attach_backend calls; mark_backend_ready/not_ready flip a
    boolean. Enough to assert supervisor's wiring without dragging
    the real gRPC server into every test.
    """

    def __init__(self, *, fail_first: bool = False) -> None:
        self.ready = False
        self.attached: list[_StubBackend] = []
        self._fail_first = fail_first
        self._fail_calls = 0
        # Present so SerenaSupervisor can check ``server._backend`` if it
        # ever wants to (mirrors the real server field).
        self._backend: Any | None = None
        #: Commit 12 (observability): every ``note_restart`` outcome
        #: the supervisor reported, in order.
        self.restart_notes: list[bool] = []

    def note_restart(self, success: bool) -> None:
        self.restart_notes.append(success)

    async def attach_backend(self, backend: "_StubBackend") -> bool:
        self.attached.append(backend)
        if self._fail_first and self._fail_calls == 0:
            self._fail_calls += 1
            return False
        self._backend = backend
        self.ready = True
        return True

    def mark_backend_not_ready(self) -> None:
        self.ready = False


class _StubBackend:
    _next_pid = 1000

    def __init__(self) -> None:
        _StubBackend._next_pid += 1
        self._pid: int | None = _StubBackend._next_pid
        self.stopped = False
        self.mark_lost_calls = 0

    async def start(self) -> None:  # not called; attach_backend handles it
        pass

    async def stop(self) -> None:
        self.stopped = True
        self._pid = None

    def _mark_lost(self) -> None:
        self.mark_lost_calls += 1

    def set_on_lost(self, _hook) -> None:
        pass

    @property
    def pid(self) -> int | None:
        return self._pid


# ── pidfd / add_reader stubs ─────────────────────────────────────────


class _FakePidfd:
    """Model a pidfd as an object with a ``fire()`` method.

    The supervisor sees an ``int`` fd (we hand out monotonically
    increasing ints); the fake ``add_reader`` records the callback
    keyed by fd. ``fire(fd)`` invokes it once to simulate the kernel
    marking the fd readable when the process dies.
    """

    def __init__(self) -> None:
        self._next_fd = 100
        self.opened: list[tuple[int, int]] = []  # (fd, pid)
        self.readers: dict[int, Callable[[], None]] = {}
        self.removed: list[int] = []
        self.closed: list[int] = []

    def open(self, pid: int, _flags: int = 0) -> int:
        fd = self._next_fd
        self._next_fd += 1
        self.opened.append((fd, pid))
        return fd

    def add_reader(self, fd: int, cb: Callable[[], None]) -> None:
        self.readers[fd] = cb

    def remove_reader(self, fd: int) -> None:
        self.removed.append(fd)
        self.readers.pop(fd, None)

    def close(self, fd: int) -> None:
        self.closed.append(fd)

    def fire(self, fd: int) -> None:
        cb = self.readers.get(fd)
        if cb is None:
            raise KeyError(f"no reader registered for fd {fd}")
        cb()


# ── tests ────────────────────────────────────────────────────────────


async def _wait_for(pred, timeout=2.0, interval=0.02):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if pred():
            return
        await asyncio.sleep(interval)
    raise AssertionError("predicate did not become true within timeout")


def _make_supervisor(
    *,
    server: _StubServer | None = None,
    pidfd: _FakePidfd | None = None,
    factory: Callable[[], _StubBackend] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> tuple[SerenaSupervisor, _StubServer, _FakePidfd, list[_StubBackend]]:
    server = server or _StubServer()
    pidfd = pidfd or _FakePidfd()
    made: list[_StubBackend] = []

    def _default_factory() -> _StubBackend:
        b = _StubBackend()
        made.append(b)
        return b

    sup = SerenaSupervisor(
        server=server,
        backend_factory=factory or _default_factory,
        pidfd_open_fn=pidfd.open,
        add_reader_fn=pidfd.add_reader,
        remove_reader_fn=pidfd.remove_reader,
        close_fd_fn=pidfd.close,
        now_fn=now_fn,
    )
    return sup, server, pidfd, made


@pytest.mark.asyncio
async def test_start_returns_true_and_arms_pidfd_monitor() -> None:
    sup, server, pidfd, made = _make_supervisor()
    ok = await sup.start()
    try:
        assert ok is True
        assert server.ready is True
        assert len(server.attached) == 1
        # Exactly one pidfd opened for the freshly attached backend.
        assert len(pidfd.opened) == 1
        fd, pid = pidfd.opened[0]
        assert pid == made[0].pid
        assert fd in pidfd.readers
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_start_returns_false_when_first_attach_fails() -> None:
    sup, server, pidfd, _ = _make_supervisor(server=_StubServer(fail_first=True))
    ok = await sup.start()
    try:
        assert ok is False
        assert server.ready is False
        # No monitor when there is no live backend.
        assert pidfd.opened == []
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_stop_cancels_monitor_and_stops_backend() -> None:
    sup, _server, pidfd, made = _make_supervisor()
    await sup.start()
    fd, _pid = pidfd.opened[-1]
    await sup.stop()
    # Monitor gone: fd was removed and closed exactly once.
    assert pidfd.removed.count(fd) == 1
    assert pidfd.closed.count(fd) == 1
    # Backend was stopped.
    assert made[0].stopped is True


@pytest.mark.asyncio
async def test_pidfd_death_triggers_restart_with_new_backend() -> None:
    sup, server, pidfd, made = _make_supervisor()
    await sup.start()
    try:
        first_backend = made[0]
        fd1, _ = pidfd.opened[-1]

        # Simulate kernel marking the fd readable — Serena died.
        pidfd.fire(fd1)

        # Wait for the async _on_death task to attach a fresh backend.
        await _wait_for(lambda: len(server.attached) >= 2)
        # A new backend was manufactured by the factory AND attached.
        assert len(made) == 2
        assert server.attached[-1] is made[1]
        # Old backend was stopped (SerenaBackend.stop is cross-task
        # safe — anyio cancel-scope mismatches are downgraded to a
        # warning rather than propagated).
        assert first_backend.stopped is True
        # New monitor armed on the new pid.
        assert len(pidfd.opened) == 2
        _fd2, pid2 = pidfd.opened[-1]
        assert pid2 == made[1].pid
        # Server is live again.
        assert server.ready is True
        # MUT-T: original fd closed exactly once.
        assert pidfd.closed.count(fd1) == 1
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_rate_limit_blocks_fourth_restart_in_window() -> None:
    """MUT-Q/R: three restarts allowed, fourth stays not-ready."""
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    sup, server, pidfd, made = _make_supervisor(now_fn=_now)
    await sup.start()
    try:
        # 3 real restarts inside the window.
        for t in (1.0, 2.0, 3.0):
            fake_now[0] = t
            fd, _ = pidfd.opened[-1]
            pidfd.fire(fd)
            await _wait_for(lambda t=t: len(server.attached) >= int(t) + 1)
        assert len(server.attached) == _MAX_RESTARTS + 1
        # Fourth death inside the same window — must be REFUSED, so
        # no new attach happens and the server is left not-ready.
        fake_now[0] = 30.0
        fd, _ = pidfd.opened[-1]
        pidfd.fire(fd)
        # Give _on_death time to run and decide "budget exhausted".
        await asyncio.sleep(0.05)
        assert len(server.attached) == _MAX_RESTARTS + 1, (
            "budget exhausted -> no new attach_backend, "
            "supervisor must NOT silently keep trying"
        )
        assert server.ready is False, (
            "budget exhausted -> backend must stay not-ready"
        )
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_rate_limit_window_expires_after_60s() -> None:
    """MUT-R off-by-one: once the oldest attempt slides out of the
    window, a fresh restart is allowed again.

    Concretely: fill the 60s window with 3 restarts, then move the
    clock past ``_WINDOW_S`` from the oldest one. The next death must
    trigger a real restart (attaches jump from 4 -> 5).
    """
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    sup, server, pidfd, _ = _make_supervisor(now_fn=_now)
    await sup.start()
    try:
        # Exhaust the budget at t=1..3 (3 real restarts).
        for t in (1.0, 2.0, 3.0):
            fake_now[0] = t
            fd, _pid = pidfd.opened[-1]
            pidfd.fire(fd)
            await _wait_for(lambda t=t: len(server.attached) >= int(t) + 1)
        assert len(server.attached) == _MAX_RESTARTS + 1
        # Advance the clock past the window (relative to the oldest
        # attempt at t=1).
        fake_now[0] = 1.0 + _WINDOW_S + 0.1
        # The last successful restart armed a fresh pidfd; fire that one.
        fd, _pid = pidfd.opened[-1]
        pidfd.fire(fd)
        # A new restart must be permitted -> attaches climb again.
        await _wait_for(
            lambda: len(server.attached) >= _MAX_RESTARTS + 2,
            timeout=3.0,
        )
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_stop_does_not_race_a_restart() -> None:
    """MUT-V: pidfd firing after ``stop()`` must not spawn a new backend.

    We intentionally poke ``_on_death`` DIRECTLY as if the monitor's
    task had already scheduled it right before ``stop()`` won the
    race — otherwise the ``_stopped`` guard on the OUTER monitor task
    would short-circuit and the mutation would slip past unnoticed.
    """
    sup, server, pidfd, _ = _make_supervisor()
    await sup.start()
    fd, _ = pidfd.opened[-1]
    await sup.stop()
    attaches_before = len(server.attached)
    # Race simulation: on_death task was created between "monitor
    # observed death" and "stop wins".
    death_task = asyncio.create_task(sup._on_death())
    await asyncio.wait_for(death_task, timeout=1.0)
    assert len(server.attached) == attaches_before, (
        "_on_death must respect _stopped and NOT attach a new backend"
    )
    # Even a second death event racing again must stay a noop.
    if fd in pidfd.readers:
        pidfd.fire(fd)
    await asyncio.sleep(0.05)
    assert len(server.attached) == attaches_before


@pytest.mark.asyncio
async def test_restart_installs_handlers_on_new_backend() -> None:
    """MUT-W: restart must go through the full attach_backend, not just
    mark_backend_ready — otherwise tool RPCs keep hitting the old handler."""
    sup, server, pidfd, made = _make_supervisor()
    await sup.start()
    try:
        fd, _ = pidfd.opened[-1]
        pidfd.fire(fd)
        await _wait_for(lambda: len(server.attached) >= 2)
        # ``_StubServer.attach_backend`` records EACH attach; if the
        # supervisor cheated and called only mark_backend_ready, the
        # attached list would still be length 1.
        assert len(server.attached) == 2
        assert server.attached[-1] is made[-1]
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_pidfd_reader_fires_only_once_per_fd() -> None:
    """MUT-T: even if the fake fires twice (rare kernel edge), the fd
    is closed exactly once and no double restart happens."""
    sup, server, pidfd, _ = _make_supervisor()
    await sup.start()
    try:
        fd, _ = pidfd.opened[-1]
        pidfd.fire(fd)
        # Second fire on the same fd — reader was already unregistered.
        with pytest.raises(KeyError):
            pidfd.fire(fd)
        await _wait_for(lambda: len(server.attached) >= 2)
        # Exactly one restart.
        assert len(server.attached) == 2
        assert pidfd.closed.count(fd) == 1
    finally:
        await sup.stop()


# ── commit 12: restart accounting (observability) ────────────────────


@pytest.mark.asyncio
async def test_first_attach_does_not_count_as_restart() -> None:
    """A5: bringing the backend up for the FIRST time is not a restart.

    Without this, every freshly started service would advertise
    ``restarts_total == 1`` and the metric would be useless for
    spotting a genuinely flapping backend.
    """
    sup, server, _pidfd, _made = _make_supervisor()
    ok = await sup.start()
    try:
        assert ok is True
        assert server.restart_notes == [], (
            "first attach must not be reported as a restart"
        )
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_successful_restart_notes_restart_true() -> None:
    """MUT-X: a restart that produced a live backend is counted once."""
    sup, server, pidfd, _made = _make_supervisor()
    await sup.start()
    try:
        fd, _ = pidfd.opened[-1]
        pidfd.fire(fd)
        await _wait_for(lambda: len(server.attached) >= 2)
        await _wait_for(lambda: server.restart_notes == [True])
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_failed_restart_does_not_note_restart() -> None:
    """A3: a restart whose attach failed must NOT bump the counter.

    The failure is already visible as ``alive=False``; counting it as
    a restart would turn ``restarts_total`` into "attempts", which
    reads as recovery that never happened.
    """
    made: list[_StubBackend] = []

    class _FailOnRestartServer(_StubServer):
        async def attach_backend(self, backend: "_StubBackend") -> bool:
            self.attached.append(backend)
            if len(self.attached) == 1:
                self._backend = backend
                self.ready = True
                return True
            self.ready = False
            return False

    def _factory() -> _StubBackend:
        b = _StubBackend()
        made.append(b)
        return b

    server = _FailOnRestartServer()
    sup, _server, pidfd, _ = _make_supervisor(server=server, factory=_factory)
    await sup.start()
    try:
        fd, _ = pidfd.opened[-1]
        pidfd.fire(fd)
        await _wait_for(lambda: len(server.attached) >= 2)
        await asyncio.sleep(0.05)  # let _on_death finish its bookkeeping
        assert server.restart_notes == [], (
            "failed restart must not be counted as a successful restart"
        )
        assert server.ready is False
    finally:
        await sup.stop()
