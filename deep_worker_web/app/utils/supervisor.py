"""Supervised background tasks.

Every ``asyncio.create_task(...)`` in this codebase is a fire-and-forget
promise. Without a ``done_callback`` an unhandled exception in the
coroutine is logged by asyncio as "Task exception was never retrieved"
only **at GC time** — often long after the damage is done, often
without a traceback if the task reference is still alive. The
operator sees a site that "just stopped reacting" with nothing in
the logs to explain why.

``supervise`` wraps ``create_task`` with the pattern we want
everywhere:

* log the failure immediately under a caller-supplied ``name`` with
  full traceback;
* invoke an optional ``on_exit(exc)`` hook so the caller can
  schedule a respawn (event-driven — via ``loop.call_later`` — not a
  polling loop);
* swallow exceptions from ``on_exit`` itself so a broken recovery
  callback can't poison the supervisor;
* treat ``CancelledError`` as a normal shutdown signal, not a
  failure (no ERROR log, ``on_exit`` still called with the
  exception so subscribers that care can observe it).

This helper is deliberately small and unopinionated: it does NOT
implement the respawn policy itself — each caller decides whether to
restart on failure, wait first, give up after N attempts, etc.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

OnExit = Callable[[Optional[BaseException]], None]


# The silencer targets one very specific rpyc_async leak pattern.
# Pulled out as a module-level constant so tests can assert on it.
_RPYC_DISPATCH_QUALNAME = "Connection._dispatch_request_async"
_PENDING_TASK_MSG = "Task was destroyed but it is pending!"


def install_rpyc_gc_silencer(loop: asyncio.AbstractEventLoop) -> None:
    """Chain an asyncio exception handler on ``loop`` that downgrades
    the ``Task was destroyed but it is pending!`` spam coming out of
    rpyc_async dispatch tasks to DEBUG.

    When an agent process is SIGKILLed, the rpyc_async server side has
    one ``asyncio.Task`` per in-flight inbound RPC
    (``Connection._dispatch_request_async``). The connection is torn
    down before those tasks complete, so when they are GC'd asyncio
    logs one ERROR per task. A batch of 3 SIGKILLs produced ~30k ERROR
    lines in the server log.

    Contract (pinned by
    ``tests/unit/test_wave4_asyncio_exception_handler.py``):

    * ``{message: "Task was destroyed but it is pending!", task: <t>}``
      where ``t.get_coro().__qualname__`` contains
      ``"Connection._dispatch_request_async"`` — FILTERED (DEBUG on
      ``app.utils.supervisor``).
    * Any OTHER context — forwarded unchanged to the previously
      installed handler (or asyncio's default if none).

    Safe to call multiple times: the previously-installed handler is
    captured and re-used, so calling this N times chains N filters (no
    leak — but there is no reason to). It is idempotent in practice.
    """
    previous = loop.get_exception_handler()

    def _silencer(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        msg = context.get("message", "")
        task = context.get("task")
        if msg == _PENDING_TASK_MSG and task is not None:
            try:
                coro_name = getattr(
                    task.get_coro(), "__qualname__", ""
                ) or ""
            except Exception:  # noqa: BLE001 — best-effort probe
                coro_name = ""
            if _RPYC_DISPATCH_QUALNAME in coro_name:
                logger.debug(
                    "asyncio GC'd pending rpyc dispatch task "
                    "(agent peer gone): %s",
                    coro_name,
                )
                return
        if previous is not None:
            previous(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_silencer)


def supervise(
    coro: Coroutine[Any, Any, Any] | Awaitable[Any],
    *,
    name: str,
    on_exit: OnExit | None = None,
) -> asyncio.Task[Any]:
    """Schedule ``coro`` as a task that never dies silently.

    Parameters
    ----------
    coro:
        Coroutine or awaitable to run.
    name:
        Short identifier used in log messages. Pick something that
        makes the log line self-explanatory — e.g. ``"monitor_agents"``
        or ``"stop_agent:demo_agent"``.
    on_exit:
        Optional synchronous callback invoked after the task finishes.
        Receives the exception that killed it, or ``None`` on clean
        exit. Exceptions from ``on_exit`` are logged and swallowed.

    Returns
    -------
    asyncio.Task
        The created task. Callers may hold on to it (e.g. to cancel
        on shutdown) but do not need to — the done-callback we install
        handles the only thing that matters (logging / recovery).
    """
    task: asyncio.Task[Any] = asyncio.create_task(coro)  # type: ignore[arg-type]

    def _done(t: asyncio.Task[Any]) -> None:
        exc: BaseException | None
        if t.cancelled():
            # Cancellation is a normal shutdown signal (e.g. lifespan
            # tearing the monitor down). Do not pollute the ERROR
            # channel; still notify on_exit so subscribers can
            # distinguish "done" from "cancelled".
            exc = asyncio.CancelledError()
        else:
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "supervised task %r crashed: %s",
                    name,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        if on_exit is not None:
            try:
                on_exit(exc)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "supervised task %r: on_exit callback raised; "
                    "swallowing so recovery can't itself be a new bug",
                    name,
                )

    task.add_done_callback(_done)
    return task
