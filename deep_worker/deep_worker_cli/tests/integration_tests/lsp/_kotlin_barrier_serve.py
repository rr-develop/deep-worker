"""Test-only instrumented entrypoint for the shared LSP service.

Runs the REAL service (``deep_worker_cli.lsp.service.__main__``) with one
addition: a barrier around the MCP send inside ``SerenaBackend``, so an
integration test can kill Serena at a *proven* point instead of guessing
at scheduler order.

Why this file exists
--------------------
``test_narrow_call_in_flight_survives_serena_kill`` needs the language
server to die while a request is genuinely in flight. Doing that from the
client side with ``ensure_future`` + ``sleep(0)`` does NOT work: the call
coroutine first enters a nested ``wait_for`` and on its first quantum can
only create the inner ``call_tool`` task before yielding, so the test can
wake up and SIGKILL *before* anything reached Serena. Such a test still
passes, but it pins a scheduling race — not the in-flight path it claims.

So the signal is raised where the truth is known: on the service side,
inside the attempt, immediately around the send to the Serena MCP
session.

Product code is NOT modified. The hook is installed by wrapping the bound
``call_tool`` of the live MCP session object at runtime, from this
test-only module, which the production supervisor never launches.

The ordering is strict and load-bearing: ``original_write(data)`` runs
FIRST and ``sent`` is raised only after it returns. Buffering the payload
and releasing it later would invert the meaning — the SIGKILL would land
on a request that had never left this process, i.e. request-after-death
rather than in-flight — so the E2E would quietly stop testing its own
claim. ``tests/unit_tests/lsp/test_kotlin_barrier_harness.py`` pins that
order directly with a fake stdin.

Protocol (three plain files, atomic ``os.replace``; waits are
event-driven via inotify + ``add_reader``, never polled)::

    <barrier_dir>/arm       created by the test AFTER the warm-up call,
                            so the barrier applies to the observed call
                            and not to whatever ran before it
    <barrier_dir>/sent      created AFTER the request is written to MCP
    <barrier_dir>/release   created by the test to let the attempt finish

``--barrier-hold-s`` bounds how long the attempt waits for ``release`` so
a wedged test can never hang the service forever; an expired hold raises
rather than auto-releasing, because a silent continue would produce a
green run whose timing proves nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from deep_worker_cli.lsp.service import serena_backend as _sb


def _touch_atomic(path: Path) -> None:
    """Create ``path`` so a reader never observes a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("1")
    os.replace(tmp, path)


async def _wait_for_file(path: Path, timeout_s: float) -> bool:
    """Wait until ``path`` exists, event-driven, bounded by ``timeout_s``.

    Uses inotify on the parent directory and drives it through the running
    loop's ``add_reader``; no polling loop, per project policy. Returns
    ``True`` if the file appeared, ``False`` on timeout.
    """
    import ctypes
    import struct

    if path.exists():
        return True

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    try:
        # IN_CREATE | IN_MOVED_TO — os.replace() lands as MOVED_TO.
        wd = libc.inotify_add_watch(fd, str(path.parent).encode(), 0x100 | 0x80)
        if wd < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch failed")

        loop = asyncio.get_running_loop()
        seen: asyncio.Future[bool] = loop.create_future()

        def _on_readable() -> None:
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                return
            offset = 0
            while offset + 16 <= len(data):
                _wd, _mask, _cookie, length = struct.unpack_from("iIII", data, offset)
                name = data[offset + 16 : offset + 16 + length].split(b"\0", 1)[0]
                offset += 16 + length
                if name.decode(errors="ignore") == path.name and not seen.done():
                    seen.set_result(True)

        loop.add_reader(fd, _on_readable)
        try:
            # Re-check after arming: the file may have appeared between
            # the first check and the watch being installed.
            if path.exists():
                return True
            return await asyncio.wait_for(seen, timeout=timeout_s)
        except asyncio.TimeoutError:
            return False
        finally:
            loop.remove_reader(fd)
    finally:
        os.close(fd)


#: Set by the ``__aenter__`` hook below to the stdin transport of the
#: live Serena child. ``_call_or_death`` only ever receives the
#: ``ClientSession``, which does not own the pipe, so the transport is
#: captured where it is created instead of hunted for later.
_LIVE_STDIN: dict[str, Any] = {}


def _install_transport_capture() -> None:
    """Record the Serena child's stdin as each stdio session opens.

    ``_OwnedStdioSession.__aenter__`` spawns the child and wires the
    reader/writer tasks; the writer ends in ``stdin.write(...)``, which
    is the real wire boundary for an MCP request. Capturing the stream
    here keeps the barrier independent of how the backend stores (or
    does not store) the session object.
    """
    owned = _sb._OwnedStdioSession
    original_aenter = owned.__aenter__

    async def _aenter(self: Any) -> Any:
        session = await original_aenter(self)
        proc = getattr(self, "_proc", None)
        stdin = getattr(proc, "stdin", None) if proc is not None else None
        if stdin is not None:
            _LIVE_STDIN["stream"] = stdin
        return session

    owned.__aenter__ = _aenter  # type: ignore[method-assign]


async def _settle_task(task: "asyncio.Future[Any]") -> None:
    """Cancel and DRAIN ``task``; never raise.

    Draining (awaiting after cancel) is the point: a cancelled task is
    not finished until the loop has run it once more, and a
    merely-cancelled-but-unawaited task is still free to touch the
    shared MCP session. Mirrors ``SerenaBackend._settle`` — the harness
    must not leak what the product carefully cleans up.
    """
    if not task.done():
        task.cancel()
    with contextlib.suppress(BaseException):
        await task


def _hook_session_write(session: Any, seen: asyncio.Event) -> "Any":
    """Signal ``seen`` when a request is actually written to the child.

    The MCP session serialises each request and ultimately calls
    ``stdin.write(...)`` on the Serena subprocess (see
    ``_OwnedStdioSession.__aenter__._writer``). That call is the wire
    boundary: before it nothing has left this process, after it the
    request sits in Serena's pipe. Anchoring the barrier there — rather
    than on "a task was created" — is what turns the in-flight claim
    into a fact.

    Returns a callable that removes the hook.
    """
    stdin = _LIVE_STDIN.get("stream")
    if stdin is None or not hasattr(stdin, "write"):
        raise RuntimeError("barrier: MCP stdin transport was never captured")

    original_write = stdin.write

    def _write(data: bytes) -> Any:
        """Send the bytes, THEN report that the request is on the wire.

        The order is the whole point. Buffering the payload instead —
        holding it until the test releases — would mean the request had
        NOT left this process when Serena was killed: that is a
        request-after-death scenario, not an in-flight one, and it would
        silently stop testing the thing the E2E claims to test.

        So ``original_write`` runs first and ``seen`` is raised only
        after it returns. At that instant the request really is in
        Serena's pipe, and the SIGKILL that follows lands on a genuinely
        outstanding request.
        """
        result = original_write(data)
        if not seen.is_set():
            seen.set()
        return result

    stdin.write = _write  # type: ignore[method-assign]

    def _undo() -> None:
        try:
            stdin.write = original_write  # type: ignore[method-assign]
        except Exception:
            pass

    return _undo


def _install_barrier(barrier_dir: Path, hold_s: float) -> None:
    """Wrap ``_call_or_death`` so the send point is observable.

    The wrapper marks ``sent`` only after the underlying MCP request has
    actually been handed to the session, then holds the attempt until the
    test releases it (or ``hold_s`` elapses). Holding is what makes the
    race deterministic in the other direction too: without it, a fast
    ``get_symbols_overview`` could complete before the SIGKILL lands and
    the test would go green without ever exercising the death path.
    """
    _install_transport_capture()
    original = _sb.SerenaBackend._call_or_death
    arm_path = barrier_dir / "arm"
    sent_path = barrier_dir / "sent"
    release_path = barrier_dir / "release"

    async def _instrumented(
        self: Any, session: Any, tool_name: str, args: dict[str, Any], remaining: float
    ) -> Any:
        # The barrier must apply to the call the TEST is watching, not to
        # the warm-up that precedes it. Without this gate, the very first
        # request raises ``sent``, the test kills during warm-up and the
        # observed call never happens — a green run that proves nothing.
        if not arm_path.exists():
            return await original(self, session, tool_name, args, remaining)

        real_call_tool = session.call_tool

        async def _call_tool_marking(name: str, payload: dict[str, Any]) -> Any:
            # ``sent`` means the request HAS LEFT this process: the hook
            # raises ``sent_seen`` only after ``original_write`` returned,
            # so at that moment the bytes are in Serena's pipe. Holding
            # the payload back instead would make the SIGKILL land on a
            # request that had never been sent — a request-after-death
            # scenario, not the in-flight one this test exists to prove.
            sent_seen = asyncio.Event()

            undo_hook = _hook_session_write(session, sent_seen)
            task = asyncio.ensure_future(real_call_tool(name, payload))
            # NOTHING may outlive this wrapper. ``_call_or_death`` can
            # cancel us the moment it sees the backend die, and a
            # cancellation delivered while we await ``release`` does NOT
            # propagate into ``task`` — it would keep using the SHARED
            # MCP session after the lock is released, i.e. concurrently
            # with the next call. That is exactly the orphan class the
            # product fix closed in ``_settle``; test instrumentation
            # must not reintroduce it.
            try:
                # Bounded: if the request never reaches the wire, fail
                # loudly rather than let the test kill blindly and then
                # pass for the wrong reason.
                await asyncio.wait_for(sent_seen.wait(), timeout=hold_s)
                # The request is genuinely on the wire now. This is the
                # window the test kills in.
                _touch_atomic(sent_path)
                # Event-driven wait (inotify via ``add_reader``);
                # polling is forbidden by project policy.
                released = await _wait_for_file(release_path, hold_s)
                if not released:
                    # A silent auto-release would turn a wedged test into
                    # a green one whose timing proves nothing.
                    raise TimeoutError(
                        f"barrier: {release_path.name} not observed within "
                        f"{hold_s}s — the hold expired instead of being released"
                    )
            except BaseException:
                await _settle_task(task)
                raise
            finally:
                # Restore the transport; the request itself already went
                # out on the real write.
                undo_hook()
            return await task

        try:
            session.call_tool = _call_tool_marking  # type: ignore[method-assign]
            return await original(self, session, tool_name, args, remaining)
        finally:
            try:
                session.call_tool = real_call_tool  # type: ignore[method-assign]
            except Exception:
                pass

    _sb.SerenaBackend._call_or_death = _instrumented  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--barrier-dir", required=True)
    parser.add_argument("--barrier-hold-s", type=float, default=30.0)
    known, rest = parser.parse_known_args(argv)

    barrier_dir = Path(known.barrier_dir)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir.chmod(0o700)
    _install_barrier(barrier_dir, known.barrier_hold_s)

    from deep_worker_cli.lsp.service.__main__ import main as _service_main

    return _service_main(rest)


if __name__ == "__main__":
    sys.exit(main())
