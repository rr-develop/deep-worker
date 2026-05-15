"""Wave 1 — inotify reader callbacks must not be killed by unexpected
exceptions.

``loop.add_reader(fd, callback)`` invokes ``callback`` synchronously in
the event loop whenever ``fd`` is readable. If ``callback`` raises,
asyncio's default behaviour is to log-and-drop: **the reader is
silently removed** and subsequent inotify events go unnoticed —
registry changes no longer produce agent_discovered / agent_terminated
broadcasts. To the user, the site "dies on the lightest breeze": it
still renders but stops seeing the world.

Contract under test
-------------------
The inotify reader registered inside ``AgentService.monitor_agents``
(``_on_inotify_event``) MUST NOT propagate any exception. Whatever
happens while parsing the inotify buffer — short read, truncated
struct, ``struct.error``, any ``Exception`` — is logged and the reader
survives to process the next event. The debounced reconcile is
triggered on a best-effort basis.

We exercise the callback directly rather than spinning up real
inotify: the goal is to pin the exception-handling contract, not
re-test the kernel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import patch

import pytest

from app.services.agent_service import AgentService


class _FakeRegistry:
    """Registry stub that never changes, never raises."""

    def __init__(self) -> None:
        self.registry_file = "/tmp/_nonexistent_registry.json"

    def list_all(self) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_inotify_reader_survives_os_read_oserror(
    caplog, monkeypatch
) -> None:
    """An ``OSError`` from ``os.read`` on the inotify fd must be logged
    but must not take down the reader. (OSError path is already handled
    today; this test pins the contract so a future refactor cannot
    break it.)"""
    svc = AgentService(registry=_FakeRegistry())

    captured: dict[str, Any] = {}

    real_add_reader = asyncio.get_running_loop().add_reader

    def tracking_add_reader(fd: int, cb: Any, *args: Any) -> None:
        # Grab the callback so we can drive it ourselves.
        captured["fd"] = fd
        captured["cb"] = cb
        real_add_reader(fd, cb, *args)

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_reader", tracking_add_reader)

    task = asyncio.create_task(svc.monitor_agents())
    try:
        # Give monitor_agents time to arm the inotify reader.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if "cb" in captured:
                break
        assert "cb" in captured, "inotify reader was never armed"

        # Make ``os.read`` raise. Invoke the callback manually — this
        # is exactly what the event loop does when the fd signals.
        with patch("os.read", side_effect=OSError(9, "bad fd")):
            with caplog.at_level(logging.ERROR, logger="app.services.agent_service"):
                # Must not propagate — even with a bad os.read behind it.
                captured["cb"]()

        # Reader still installed (we did not remove it).
        # The callback returning normally after an exception IS the
        # survival contract.
    finally:
        svc._shutdown_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_inotify_reader_survives_unexpected_exception(
    monkeypatch, caplog
) -> None:
    """Any ``Exception`` inside the inotify callback — struct.error,
    AttributeError, whatever — must be caught so the reader continues
    to fire on the next event. A ``struct.error`` from a truncated
    buffer was the real-world trigger; any other unexpected exception
    must be handled just as safely."""
    svc = AgentService(registry=_FakeRegistry())

    captured: dict[str, Any] = {}

    real_add_reader = asyncio.get_running_loop().add_reader

    def tracking_add_reader(fd: int, cb: Any, *args: Any) -> None:
        captured["fd"] = fd
        captured["cb"] = cb
        real_add_reader(fd, cb, *args)

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_reader", tracking_add_reader)

    task = asyncio.create_task(svc.monitor_agents())
    try:
        for _ in range(20):
            await asyncio.sleep(0.05)
            if "cb" in captured:
                break
        assert "cb" in captured, "inotify reader was never armed"

        # Inject a synthetic parse failure. The callback loops on
        # ``os.read`` until a BlockingIOError tells it the kernel
        # buffer is drained. We return a bogus, truncated buffer on
        # the first call (struct.unpack_from inside the parser will
        # NOT raise — the ``while offset + _header.size <= len(buf)``
        # guard skips short buffers — so we need something sharper
        # to exercise the catch-all. Raising a surprise
        # AttributeError from the reader body reliably triggers the
        # "anything else" branch we added.
        call_count = {"n": 0}

        def flaky_read(_fd: int, _n: int) -> bytes:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Surprise exception from a syscall the reader makes
                # — exactly the class of bug the catch-all exists
                # to contain.
                raise AttributeError("unexpected reader-side bug")
            raise BlockingIOError  # drain complete

        with patch("os.read", side_effect=flaky_read):
            with caplog.at_level(
                logging.ERROR, logger="app.services.agent_service"
            ):
                captured["cb"]()

        # Nothing to assert on broadcasts (buffer didn't match the
        # registry file). The key invariant: ``captured["cb"]()`` did
        # not propagate, and an ERROR-level log is present explaining
        # what happened so ops knows something went wrong.
        err_logs = [
            rec for rec in caplog.records
            if rec.levelno >= logging.ERROR
        ]
        assert err_logs, (
            "the inotify reader must log when it catches an unexpected "
            "exception, otherwise failures vanish silently"
        )
    finally:
        svc._shutdown_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
