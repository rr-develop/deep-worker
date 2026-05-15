"""Wave 1 — ``app.utils.supervisor.supervise`` must never let an
unhandled exception on a background task vanish silently.

Contract
--------
``supervise(coro, *, name, loop=None, on_exit=None)`` schedules a
background task. If the task raises, ``supervise`` MUST:

1. Log the exception with full traceback under the task's ``name``.
   Logs go to the ``app.utils.supervisor`` logger so callers can
   assert on them via ``caplog``.
2. Invoke ``on_exit(exc)`` if provided. ``on_exit`` receives the
   exception object (``None`` on clean completion). Errors in
   ``on_exit`` itself must be logged, not re-raised, otherwise one
   broken recovery callback could crash the next.
3. Never re-raise — the surrounding loop must keep running.
4. Return the created ``asyncio.Task`` so callers can cancel it.

Rationale
---------
Almost every ``asyncio.create_task(...)`` in the web codebase lacked an
error callback. A monitoring task that died on a transient ``OSError``
froze agent discovery forever and the failure was invisible in the
logs. This helper centralises the "never die silently" pattern so we
can apply it consistently to every background task site.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.utils.supervisor import supervise


@pytest.mark.asyncio
async def test_supervise_logs_exception_and_swallows(caplog) -> None:
    async def boom() -> None:
        raise RuntimeError("synthetic failure")

    with caplog.at_level(logging.ERROR, logger="app.utils.supervisor"):
        task = supervise(boom(), name="test-boom")
        # Let the task run.
        await asyncio.sleep(0.01)

    assert task.done()
    # ``supervise`` must NOT re-raise — caller keeps running.
    # Check the log captured the exception with its traceback.
    matches = [
        rec for rec in caplog.records
        if "test-boom" in rec.getMessage()
        and "synthetic failure" in (rec.exc_text or "")
    ]
    assert matches, (
        "supervise must log the task failure with a traceback under "
        "its name; caplog recorded: "
        + "; ".join(rec.getMessage() for rec in caplog.records)
    )


@pytest.mark.asyncio
async def test_supervise_invokes_on_exit_with_exception() -> None:
    async def boom() -> None:
        raise ValueError("nope")

    seen: list[BaseException | None] = []

    def on_exit(exc: BaseException | None) -> None:
        seen.append(exc)

    supervise(boom(), name="t2", on_exit=on_exit)
    await asyncio.sleep(0.01)

    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)
    assert str(seen[0]) == "nope"


@pytest.mark.asyncio
async def test_supervise_invokes_on_exit_with_none_on_clean_completion() -> None:
    async def fine() -> None:
        return None

    seen: list[BaseException | None] = []

    def on_exit(exc: BaseException | None) -> None:
        seen.append(exc)

    supervise(fine(), name="t3", on_exit=on_exit)
    await asyncio.sleep(0.01)

    assert seen == [None]


@pytest.mark.asyncio
async def test_supervise_swallows_exceptions_from_on_exit(caplog) -> None:
    """A broken recovery callback must not kill the event loop —
    otherwise the very thing supervise is designed to prevent
    (silent task death cascading) would happen inside supervise
    itself.
    """
    async def fine() -> None:
        return None

    def bad_on_exit(_exc: BaseException | None) -> None:
        raise RuntimeError("recovery callback exploded")

    with caplog.at_level(logging.ERROR, logger="app.utils.supervisor"):
        supervise(fine(), name="t4", on_exit=bad_on_exit)
        await asyncio.sleep(0.01)

    # No exception propagated out of the callback — the test
    # reaches this line. Additionally, the error should be logged.
    matches = [
        rec for rec in caplog.records
        if "recovery callback exploded" in (rec.exc_text or "")
    ]
    assert matches, "broken on_exit must be logged with traceback"


@pytest.mark.asyncio
async def test_supervise_ignores_cancelled_error(caplog) -> None:
    """Cancellation is a normal lifecycle event (shutdown), not a
    failure — ``supervise`` must not log it as ERROR, to keep
    shutdown logs clean."""
    async def long_running() -> None:
        await asyncio.sleep(10.0)

    with caplog.at_level(logging.ERROR, logger="app.utils.supervisor"):
        task = supervise(long_running(), name="t5")
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    err_matches = [
        rec for rec in caplog.records
        if rec.levelno >= logging.ERROR
    ]
    assert err_matches == [], (
        "cancellation must not be logged as an ERROR; found: "
        + "; ".join(rec.getMessage() for rec in err_matches)
    )
