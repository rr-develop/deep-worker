"""TDD guard: NO synchronous/blocking RPyC call may sit on the event loop.

Background (regression 2026-06-23, commit ``6ae6b5f``):
``get_agent_dialog_formatted`` introduced

    raw = await asyncio.wait_for(root.get_system_prompt(), timeout=30.0)
    raw = await asyncio.wait_for(root.get_tools(), timeout=30.0)

``root`` is an RPyC **netref**. ``root.get_system_prompt()`` is evaluated
EAGERLY and INLINE on the asyncio event loop: the call triggers
``syncreq -> sync_request -> AsyncResult.value -> wait()`` whose body is
``while self._waiting(): self._conn.serve(...)`` — i.e. it READS THE SOCKET
on the loop thread and busy-spins at 100% CPU. The surrounding
``asyncio.wait_for(..., timeout=30.0)`` is useless: the blocking serve
finishes (or, against a stuck/slow peer, NEVER finishes) BEFORE
``wait_for`` is even entered. Net effect: the whole web server froze —
``/health`` stopped responding, the page would not refresh, and the
restart button hung with a cross (PID 1453559, port 7887, 2026-06-24).

The correct, non-blocking path (used everywhere else in this file via
``fire_and_forget_async``, and for direct results via ``rpyc.async_``):

    raw = await async_(root.get_system_prompt)()

``async_(proxy)()`` issues ``asyncreq`` (does NOT serve inline) and returns
an ``AsyncResult`` whose ``__await__`` builds a ``loop.create_future()`` and
YIELDS to the event loop — cancel-aware, never blocking, no arbitrary
timeout band-aid.

These are STATIC source guards (read the file) so the regression is caught
at pytest-collection time and can never silently come back. See
``app/services/agent_service.py`` module header rule #5 ("NO SYNCHRONOUS
BLOCKING") and POLICIES.md §1/§3/§7.
"""
from __future__ import annotations

import re
from pathlib import Path

_AGENT_SERVICE_PY: Path = (
    Path(__file__).resolve().parents[2] / "app" / "services" / "agent_service.py"
)


def _code_lines() -> list[str]:
    """Source lines with comment-only lines removed (so a comment that
    *mentions* the forbidden pattern as documentation does not trip the
    guard)."""
    out: list[str] = []
    for line in _AGENT_SERVICE_PY.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return out


# Matches an inline netref call being awaited directly, e.g.
#   await root.get_system_prompt()
#   await asyncio.wait_for(root.get_tools(), timeout=30.0)
# but NOT  await async_(root.get_system_prompt)()  (the correct path).
_BLOCKING_NETREF_AWAIT = re.compile(
    r"await\s+(?:asyncio\.wait_for\(\s*)?root\.[A-Za-z_][A-Za-z0-9_]*\s*\("
)


def test_no_inline_awaited_netref_call_on_event_loop() -> None:
    """``await root.<method>()`` (optionally wrapped in ``wait_for``) is a
    synchronous-RPyC-on-the-loop call and is FORBIDDEN. Use
    ``await async_(root.<method>)()`` instead."""
    offenders = [ln.strip() for ln in _code_lines() if _BLOCKING_NETREF_AWAIT.search(ln)]
    assert offenders == [], (
        "Synchronous RPyC call on the event loop detected in "
        f"{_AGENT_SERVICE_PY.name}:\n  "
        + "\n  ".join(offenders)
        + "\n\nEach `await root.<method>()` evaluates the netref EAGERLY and "
        "blocks the loop in AsyncResult.value/serve (100% CPU deadlock vs a "
        "slow peer). Replace with the non-blocking await path:\n"
        "    raw = await async_(root.<method>)()\n"
        "and drop the arbitrary `timeout=` band-aid (the async path yields "
        "to the loop and is cancel-aware). See the module header rule #5 and "
        "test_no_blocking_rpyc_in_handlers.py."
    )


def test_introspection_uses_async_underscore() -> None:
    """Positive guard: the system_prompt / tools introspection RPyC calls
    must go through ``async_(...)`` (the non-blocking await path)."""
    src = _AGENT_SERVICE_PY.read_text(encoding="utf-8")
    for method in ("get_system_prompt", "get_tools"):
        assert re.search(rf"async_\(\s*root\.{method}\s*\)\s*\(", src), (
            f"{method} must be called as `await async_(root.{method})()` "
            f"(non-blocking). A bare `await root.{method}()` blocks the "
            f"event loop — see regression 2026-06-23 (commit 6ae6b5f)."
        )
