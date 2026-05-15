"""TDD guard: agent_service.py must use rpyc.async_connect, not rpyc.connect.

Background: deep-worker agents run on an AsyncioServer. The web
service connects from an asyncio event loop (FastAPI / uvicorn).
Wrapping synchronous ``rpyc.connect`` in ``run_in_executor`` was the
old workaround for "don't block the loop on TCP handshake" — but
``rpyc.async_connect`` does exactly that natively, without a worker
thread, and already auto-enables asyncio serving and eager-fetches
``conn.root``.

This static guard (reads the source file) keeps the migration from
regressing and catches the issue at pytest collection time.
"""
from __future__ import annotations

import re
from pathlib import Path

_AGENT_SERVICE_PY: Path = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "services"
    / "agent_service.py"
)


def _read_source() -> str:
    return _AGENT_SERVICE_PY.read_text(encoding="utf-8")


def test_agent_service_file_exists() -> None:
    assert _AGENT_SERVICE_PY.is_file(), (
        f"Expected agent service at {_AGENT_SERVICE_PY}, not found."
    )


def test_agent_service_does_not_call_sync_rpyc_connect() -> None:
    """agent_service.py must not call ``rpyc.connect(`` anywhere.

    Comments and docstrings that merely mention the old API are fine;
    the check is on actual call expressions in code.
    """
    src = _read_source()
    # Remove comment-only lines.
    non_comment_calls = [
        line for line in src.splitlines()
        if "rpyc.connect(" in line and not line.lstrip().startswith("#")
    ]
    assert non_comment_calls == [], (
        f"{_AGENT_SERVICE_PY.name} still contains synchronous "
        f"rpyc.connect() calls: {non_comment_calls!r}. Replace each "
        f"with `await rpyc.async_connect(host, port, ...)`. "
        f"run_in_executor is not needed — async_connect is natively "
        f"non-blocking."
    )


def test_agent_service_imports_async_connect() -> None:
    src = _read_source()
    pattern = re.compile(
        r"(from\s+rpyc\.core\.async_connect\s+import\s+async_connect)"
        r"|(rpyc\.async_connect\()",
    )
    assert pattern.search(src) is not None, (
        f"{_AGENT_SERVICE_PY.name} does not import or reference "
        f"async_connect. Add `from rpyc.core.async_connect import "
        f"async_connect` and `await async_connect(...)`."
    )


def test_agent_service_drops_run_in_executor_for_connect() -> None:
    """The executor-wrapped connect hack must be gone.

    async_connect is natively non-blocking; wrapping another non-blocking
    primitive in run_in_executor is a code smell that used to be the
    only way to connect without freezing the loop.
    """
    src = _read_source()
    has_executor_connect = re.search(
        r"run_in_executor\([^)]*rpyc\.connect",
        src,
        re.DOTALL,
    )
    assert has_executor_connect is None, (
        f"{_AGENT_SERVICE_PY.name} still wraps rpyc.connect in "
        f"run_in_executor. Drop that hack and `await "
        f"async_connect(...)` directly."
    )


def test_agent_service_does_not_double_enable_asyncio_serving() -> None:
    """async_connect enables asyncio serving internally. Manual
    enable_asyncio_serving on top of it is redundant.
    """
    src = _read_source()
    uses_async_connect = (
        "async_connect(" in src or "rpyc.async_connect(" in src
    )
    calls_enable = "enable_asyncio_serving(" in src
    if uses_async_connect and calls_enable:
        raise AssertionError(
            f"{_AGENT_SERVICE_PY.name} calls both async_connect AND "
            f"enable_asyncio_serving. async_connect enables asyncio "
            f"serving internally — drop the redundant call."
        )
