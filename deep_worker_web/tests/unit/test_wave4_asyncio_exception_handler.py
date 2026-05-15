"""Wave 4 — ``install_rpyc_gc_silencer`` must downgrade ONLY the
known-benign post-SIGKILL rpyc dispatch GC noise, and MUST forward
every other exception to the previously-installed handler.

Background
----------
When an agent process is SIGKILLed, rpyc_async's server-side leaves
one ``asyncio.Task`` per in-flight inbound RPC
(``Connection._dispatch_request_async``). Those tasks are still
``pending`` when the connection is GC'd, and asyncio logs one ERROR
per task. A batch of 3 SIGKILLs produced ~30k ERROR lines in the
server log — buried any real error.

If the filter is too loose it will hide real bugs. These tests pin
the contract.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from app.utils.supervisor import install_rpyc_gc_silencer


class _FakeTask:
    """Plain class — NOT a MagicMock — so ``asyncio.call_exception_handler``
    doesn't detect a spurious ``get_context`` attr on it.

    ``call_exception_handler`` does ``hasattr(task, "get_context")`` to
    hop context; a ``MagicMock`` auto-grows every attribute, so the hop
    runs into a mock-``ctx.run`` that silently swallows the handler
    call — the fake wrapper would LOOK like it's being called but
    nothing happens. A real class keeps the check honest.
    """

    def __init__(self, coro_qualname: str) -> None:
        self._coro = SimpleNamespace(__qualname__=coro_qualname)

    def get_coro(self) -> SimpleNamespace:
        return self._coro


def _fake_task(coro_qualname: str) -> "_FakeTask":
    return _FakeTask(coro_qualname)


def _fresh_loop_with_recorder() -> tuple[asyncio.AbstractEventLoop, list[dict]]:
    """Create a throwaway loop, set a recording default handler, install
    the silencer on top, and hand the loop + recorder back to the test.

    Using a real loop (``asyncio.new_event_loop()``) rather than mocks
    keeps the test faithful to the runtime semantics of
    ``loop.call_exception_handler``.
    """
    loop = asyncio.new_event_loop()
    recorded: list[dict] = []

    def _recording_default(_loop, context):
        recorded.append(dict(context))

    loop.set_exception_handler(_recording_default)
    install_rpyc_gc_silencer(loop)
    return loop, recorded


def test_benign_rpyc_gc_pattern_is_filtered_and_debug_logged(
    caplog,
) -> None:
    """The exact benign shape must NOT reach the default handler and
    must be logged at DEBUG under ``app.utils.supervisor``."""
    loop, recorded = _fresh_loop_with_recorder()
    try:
        context = {
            "message": "Task was destroyed but it is pending!",
            "task": _fake_task(
                "Connection._dispatch_request_async"
            ),
        }

        with caplog.at_level(logging.DEBUG, logger="app.utils.supervisor"):
            loop.call_exception_handler(context)

        assert recorded == [], (
            "benign pattern must be filtered; recorder saw: "
            + repr(recorded)
        )
        dbg = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "app.utils.supervisor"
            and "rpyc dispatch task" in r.getMessage()
        ]
        assert dbg, (
            "benign pattern still logged at DEBUG for diagnostic continuity"
        )
    finally:
        loop.close()


def test_unrelated_task_destroyed_still_reaches_default_handler() -> None:
    """'Task was destroyed but it is pending!' from ANY coroutine
    OTHER than the rpyc dispatch one MUST still fall through to the
    previously-installed default handler."""
    loop, recorded = _fresh_loop_with_recorder()
    try:
        context = {
            "message": "Task was destroyed but it is pending!",
            "task": _fake_task("my_module.unrelated_background_task"),
        }

        loop.call_exception_handler(context)

        assert len(recorded) == 1, (
            "unrelated task-destroyed errors MUST fall through to the "
            "default handler; recorded: " + repr(recorded)
        )
    finally:
        loop.close()


def test_arbitrary_exception_falls_through() -> None:
    """Any other exception shape — a synthetic RuntimeError, a
    TimerHandle failure, anything — must reach the default handler."""
    loop, recorded = _fresh_loop_with_recorder()
    try:
        try:
            raise RuntimeError("synthetic background failure")
        except RuntimeError as e:
            exc = e

        context = {
            "message": "Exception in callback",
            "exception": exc,
        }

        loop.call_exception_handler(context)

        assert len(recorded) == 1
        assert "synthetic background failure" in str(
            recorded[0]["exception"]
        )
    finally:
        loop.close()


def test_benign_message_without_rpyc_coro_falls_through() -> None:
    """A ``task`` context with the benign message text but a NON-rpyc
    coroutine must still fall through — we never filter solely on
    message text."""
    loop, recorded = _fresh_loop_with_recorder()
    try:
        context = {
            "message": "Task was destroyed but it is pending!",
            "task": _fake_task("asyncio.selector_events.BaseSelectorEventLoop"),
        }

        loop.call_exception_handler(context)

        assert len(recorded) == 1, (
            "message-only match must not be enough to filter; must "
            "also match the rpyc dispatch qualname"
        )
    finally:
        loop.close()


def test_missing_task_context_falls_through() -> None:
    """Contexts without ``task`` (e.g. write-to-closed-stream errors)
    must pass through — we only gate on ``task.get_coro().__qualname__``."""
    loop, recorded = _fresh_loop_with_recorder()
    try:
        context = {
            "message": "Task was destroyed but it is pending!",
            # no "task" key at all
        }

        loop.call_exception_handler(context)

        assert len(recorded) == 1
    finally:
        loop.close()
