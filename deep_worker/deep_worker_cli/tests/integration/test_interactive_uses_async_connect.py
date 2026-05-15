"""TDD guard: interactive.py must use rpyc.async_connect, not rpyc.connect.

Background: deep-worker agents run on an AsyncioServer. The interactive
CLI runs inside an asyncio event loop (Textual). Synchronous
``rpyc.connect(...)`` there blocks the loop on TCP handshake and on
every later ``sync_request``; since the rpyc_async policy update it
also raises ``RuntimeError`` from inside a running loop.

The correct primitive is ``await rpyc.async_connect(host, port, ...)``.
This test is a static guard (reads the source file) so it runs fast
and catches regressions at pytest collection time.
"""
from __future__ import annotations

import re
from pathlib import Path

_INTERACTIVE_PY: Path = (
    Path(__file__).resolve().parents[2]
    / "deep_worker_cli"
    / "cli"
    / "interactive.py"
)


def _read_source() -> str:
    return _INTERACTIVE_PY.read_text(encoding="utf-8")


def test_interactive_file_exists() -> None:
    assert _INTERACTIVE_PY.is_file(), (
        f"Expected interactive CLI at {_INTERACTIVE_PY}, not found."
    )


def test_interactive_does_not_use_sync_rpyc_connect() -> None:
    """interactive.py must NOT call ``rpyc.connect(`` anywhere.

    It runs inside an asyncio event loop (Textual). A sync
    ``rpyc.connect(`` blocks the loop and, after the rpyc_async policy
    update, raises ``RuntimeError`` immediately.
    """
    src = _read_source()
    # Strip single-line comments so a comment that mentions the old API
    # (e.g. "# was rpyc.connect before") doesn't trip the guard.
    # (There is no rpyc.connect string inside a docstring we care
    # about — it would still be wrong to call it from a docstring.)
    code_lines = [
        line for line in src.splitlines()
        if "rpyc.connect(" in line and not line.lstrip().startswith("#")
    ]
    assert code_lines == [], (
        f"{_INTERACTIVE_PY.name} still contains synchronous "
        f"rpyc.connect() calls: {code_lines!r}. Replace each with "
        f"`await rpyc.async_connect(host, port, ...)` — see the "
        f"pattern in deep_worker_cli/client/agent_client.py."
    )


def test_interactive_imports_async_connect() -> None:
    """interactive.py must import async_connect to use it."""
    src = _read_source()
    # Accept either `from rpyc.core.async_connect import async_connect`
    # (the existing style used in agent_client.py) or
    # `rpyc.async_connect` via the top-level package export.
    pattern = re.compile(
        r"(from\s+rpyc\.core\.async_connect\s+import\s+async_connect)"
        r"|(rpyc\.async_connect\()",
    )
    assert pattern.search(src) is not None, (
        f"{_INTERACTIVE_PY.name} does not import or reference "
        f"async_connect. Add `from rpyc.core.async_connect import "
        f"async_connect` and use `await async_connect(...)`."
    )


def test_interactive_does_not_call_enable_asyncio_serving_after_connect() -> None:
    """``async_connect`` already enables asyncio serving internally.

    The old pattern (``rpyc.connect`` + manual ``enable_asyncio_serving(loop)``)
    is redundant with ``async_connect``. If you see both, the async
    migration is half-done.
    """
    src = _read_source()
    uses_async_connect = (
        "async_connect(" in src or "rpyc.async_connect(" in src
    )
    calls_enable = "enable_asyncio_serving(" in src
    if uses_async_connect and calls_enable:
        raise AssertionError(
            f"{_INTERACTIVE_PY.name} calls both async_connect AND "
            f"enable_asyncio_serving. async_connect() already enables "
            f"asyncio serving internally — remove the redundant "
            f"enable_asyncio_serving call."
        )
