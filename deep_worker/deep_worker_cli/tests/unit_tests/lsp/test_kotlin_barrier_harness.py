"""Focused tests for the Kotlin E2E barrier harness itself.

The Kotlin integration test proves "the language server died while a
request was genuinely in flight". That proof rests entirely on two
properties of the harness in
``tests/integration_tests/lsp/_kotlin_barrier_serve.py``:

1. **Wire-boundary ordering.** ``seen`` must be raised strictly AFTER
   the bytes have been handed to the real transport, never before. If
   the signal could precede the send, the integration test would kill
   Serena before the request existed and pass for the wrong reason —
   and nothing in the E2E run would notice.

2. **No orphaned call task.** The wrapper starts the MCP call as its own
   task and then suspends on the release barrier. A cancellation
   delivered during that wait does not propagate into the task, so
   without an explicit cancel+drain the task would outlive the wrapper
   and keep using the SHARED MCP session concurrently with the next
   call. That is precisely the orphan class the product fix closed in
   ``SerenaBackend._settle``; the test instrumentation must not
   reintroduce it.

Both are cheap to check here and expensive (or impossible) to check in
the full Kotlin E2E, which needs a real language server and ~20 s per
run. Keeping them at this level is what makes the E2E's claim
falsifiable.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.asyncio

_BARRIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "lsp"
    / "_kotlin_barrier_serve.py"
)


def _load_barrier() -> Any:
    """Import the harness module without running the service entrypoint."""
    spec = importlib.util.spec_from_file_location(
        "_kotlin_barrier_serve_under_test", _BARRIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeStdin:
    """Minimal stand-in for the asyncio subprocess stdin transport."""

    def __init__(self, observer: Any = None) -> None:
        self.written: list[bytes] = []
        self._observer = observer

    def write(self, data: bytes) -> None:
        if self._observer is not None:
            self._observer(data)
        self.written.append(data)


async def test_seen_is_raised_only_after_the_bytes_are_written() -> None:
    """``seen`` must be false DURING the write and true only after it.

    This is the ordering the whole in-flight claim depends on. Checking
    it from inside ``original_write`` is what makes the property
    observable: a harness that set ``seen`` first would look identical
    from the outside, yet would let the integration test kill the
    backend before the request was ever sent.
    """
    barrier = _load_barrier()

    seen = asyncio.Event()
    observed_during_write: list[bool] = []

    stdin = _FakeStdin(observer=lambda _data: observed_during_write.append(seen.is_set()))
    barrier._LIVE_STDIN["stream"] = stdin

    undo = barrier._hook_session_write(object(), seen)
    try:
        stdin.write(b'{"jsonrpc":"2.0"}\n')
    finally:
        undo()

    assert stdin.written, "the payload must actually reach the transport"
    assert observed_during_write == [False], (
        "seen was already set while original_write was running: the signal "
        "would then precede the send, the SIGKILL would land on a request "
        "that never left the process, and the in-flight proof would be void"
    )
    assert seen.is_set(), "seen must be raised once the bytes are on the wire"


async def test_every_write_reaches_the_transport_immediately() -> None:
    """No payload may be buffered and released later.

    Holding bytes back would turn the scenario into request-after-death:
    Serena would be killed while the request was still inside this
    process. The E2E would stay green while silently testing something
    else, so the "write through, always" property is pinned here.
    """
    barrier = _load_barrier()

    seen = asyncio.Event()
    stdin = _FakeStdin()
    barrier._LIVE_STDIN["stream"] = stdin

    undo = barrier._hook_session_write(object(), seen)
    try:
        stdin.write(b'{"first":true}\n')
        assert stdin.written == [b'{"first":true}\n'], (
            "the first payload was withheld from the transport"
        )
        stdin.write(b'{"second":true}\n')
        assert stdin.written == [b'{"first":true}\n', b'{"second":true}\n'], (
            "a later payload was withheld from the transport"
        )
    finally:
        undo()


async def test_expired_hold_fails_loudly_instead_of_auto_releasing(
    tmp_path: Path,
) -> None:
    """A hold that times out must raise, not quietly continue.

    ``_wait_for_file`` returns ``False`` on timeout. If the wrapper
    ignored that, a wedged run would silently auto-release: the request
    would go out at an arbitrary moment, the kill would land wherever it
    landed, and the integration test would still report success while
    proving nothing about timing.
    """
    barrier = _load_barrier()

    # Drive the real wrapper: arm it, let the call reach the wire
    # boundary, and never create ``release``. The hold must expire into
    # an exception rather than proceeding as if it had been released.
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    (barrier_dir / "arm").write_text("1")

    stdin = _FakeStdin()
    barrier._LIVE_STDIN["stream"] = stdin

    class _Session:
        async def call_tool(self, _name: str, _payload: dict[str, Any]) -> str:
            # Reaches the boundary, then waits to be released.
            stdin.write(b'{"jsonrpc":"2.0"}\n')
            await asyncio.Event().wait()
            return "unreachable"

    captured: dict[str, Any] = {}

    async def _fake_original(
        _self: Any, session: Any, tool_name: str, args: dict[str, Any], _remaining: float
    ) -> Any:
        captured["wrapped"] = session.call_tool
        return await session.call_tool(tool_name, args)

    # ``_install_barrier`` patches the REAL ``SerenaBackend`` class, and
    # this module is imported fresh but the class object is shared with
    # the rest of the suite. Restoring it is mandatory: leaving the
    # instrumented version in place poisons every later test that goes
    # through ``_call_or_death``.
    real_call_or_death = barrier._sb.SerenaBackend._call_or_death
    real_aenter = barrier._sb._OwnedStdioSession.__aenter__
    try:
        barrier._sb.SerenaBackend._call_or_death = _fake_original
        barrier._install_barrier(barrier_dir, hold_s=0.2)

        instrumented = barrier._sb.SerenaBackend._call_or_death
        with pytest.raises(TimeoutError, match="release"):
            await instrumented(object(), _Session(), "get_symbols_overview", {}, 5.0)
    finally:
        barrier._sb.SerenaBackend._call_or_death = real_call_or_death
        barrier._sb._OwnedStdioSession.__aenter__ = real_aenter


async def test_call_task_is_not_orphaned_when_the_wrapper_is_cancelled() -> None:
    """Cancelling the wrapper must cancel AND drain the in-flight call.

    Regression for the orphan the harness itself introduced: the call
    runs as a separate task, so a cancellation arriving while the
    wrapper waits on the release barrier does not reach it. Left alone,
    that task keeps touching the shared MCP session after the lock is
    released — concurrently with the next call.
    """
    barrier = _load_barrier()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _never_finishes() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.ensure_future(_never_finishes())
    await started.wait()

    await barrier._settle_task(task)

    assert task.done(), "task must be finished, not merely cancelled"
    assert cancelled.is_set(), "the call task must actually observe cancellation"


async def test_settle_task_is_safe_on_an_already_finished_task() -> None:
    """Draining a completed task must be a no-op, never an error.

    The wrapper calls ``_settle_task`` on every failure path, including
    ones where the call already returned; raising there would mask the
    original error.
    """
    barrier = _load_barrier()

    async def _done() -> str:
        return "ok"

    task = asyncio.ensure_future(_done())
    await task

    await barrier._settle_task(task)

    assert task.result() == "ok", "a finished result must survive settling"
