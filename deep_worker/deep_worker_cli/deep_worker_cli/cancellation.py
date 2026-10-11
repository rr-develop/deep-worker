"""Telling "we are being shut down" apart from "a task was cancelled inside someone else's scope".

Motivation: an incident (2026-08-29). An unreachable HTTP server of an
external MCP does not reach the caller as a ``ConnectError``: the anyio
TaskGroup inside ``streamablehttp_client`` cancels its own tasks, and a
``CancelledError`` comes out. It inherits from ``BaseException`` rather than
``Exception``, so it passed straight through all three levels of
``except Exception`` (per-server in the pool, the whole pool, agent init) and
killed the process before RPyC started and before registration in the
registry. The agent did not show up in the web UI, and its history could not
be read.

Widening the catch to ``BaseException`` is only allowed together with this
predicate: otherwise a normal shutdown (SIGTERM, turn interrupt) would be
swallowed and the process would stop being stoppable.

The criterion for external cancellation is ``Task.cancelling() > 0``: a
counter that ``Task.cancel()`` increments on the CURRENT task. A cancellation
that happened inside someone else's nested scope (an anyio TaskGroup within
itself) did not cancel our task, so the counter is zero. The idiom is already
used in ``agent_process._deliver_inbox_record`` (distinguishing shutdown from
a user interrupt) -- here it is factored out for reuse.
"""
from __future__ import annotations

import asyncio


def root_cause_text(exc: BaseException, _depth: int = 0) -> str:
    """Readable failure cause, unwrapped through ``ExceptionGroup``.

    anyio surfaces an ``ExceptionGroup`` whose own text is
    "unhandled errors in a TaskGroup (1 sub-exception)". Showing that to the
    owner in the web UI is pointless: the real cause (for example
    ``httpx.ConnectError: All connection attempts failed``) is hidden inside.
    We unwrap the groups and ``__cause__`` so the message states the fact.
    """
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions and _depth < 5:
        inner = [root_cause_text(e, _depth + 1) for e in exc.exceptions]
        unique = list(dict.fromkeys(inner))
        return "; ".join(unique)
    if exc.__cause__ is not None and _depth < 5:
        return root_cause_text(exc.__cause__, _depth + 1)
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def is_external_cancellation(exc: BaseException) -> bool:
    """``True`` if the cancellation was initiated from outside and must be re-raised.

    ``KeyboardInterrupt`` and ``SystemExit`` are always external: they are a
    request to stop the process and must not be caught as a "subsystem failure".

    For ``CancelledError`` the current task's ``cancelling()`` counter decides.
    Outside an event loop (``RuntimeError`` from ``current_task``) the
    cancellation is treated as external -- a conservative choice: better to
    re-raise too much than to swallow a shutdown.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return True
    if not isinstance(exc, asyncio.CancelledError):
        return False
    try:
        current = asyncio.current_task()
    except RuntimeError:  # pragma: no cover -- no running loop
        return True
    return current is not None and current.cancelling() > 0
