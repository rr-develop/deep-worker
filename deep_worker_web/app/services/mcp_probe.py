"""One-shot MCP server probe used by the explicit "Test connection" action.

Deliberately isolated in its own module:

* It runs **in the web process**, never in the agent, so a hanging or
  hostile server can never park an agent turn.
* It is **bounded** by an outer ``wait_for`` covering the whole
  connect+initialize+list cycle, not just the socket connect — an MCP
  server that accepts a connection and then goes silent is the realistic
  failure mode, and a transport-level timeout would not catch it.
* It is only ever called from an explicit user action. Nothing in the UI
  probes implicitly, which is both the no-polling rule and what keeps a
  slow server from slowing the tab.
* It returns tool **names only** — descriptions would bloat the rendered
  DOM, which is a measured accessibility problem in this UI.
"""
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any

import anyio

logger = logging.getLogger(__name__)


async def _probe(spec: dict[str, Any]) -> list[str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    declared = spec.get("type")
    is_http = declared in {"http", "sse", "streamable-http"} or (
        declared is None and spec.get("url")
    )

    async with AsyncExitStack() as stack:
        if is_http:
            try:
                from mcp.client.streamable_http import streamablehttp_client
            except ImportError as exc:  # pragma: no cover - old SDK
                raise RuntimeError(
                    "installed mcp SDK has no streamable-http client"
                ) from exc
            headers = spec.get("headers") or {}
            streams = await stack.enter_async_context(
                streamablehttp_client(
                    spec.get("url") or "",
                    headers={str(k): str(v) for k, v in headers.items()} or None,
                )
            )
        else:
            env = spec.get("env") or {}
            params = StdioServerParameters(
                command=spec.get("command") or "",
                args=[str(a) for a in (spec.get("args") or [])],
                env={**os.environ, **{str(k): str(v) for k, v in env.items()}}
                if env
                else None,
                cwd=spec.get("cwd") or None,
            )
            streams = await stack.enter_async_context(stdio_client(params))

        read, write = streams[0], streams[1]
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        return [t.name for t in listed.tools]

    return []  # pragma: no cover - unreachable, satisfies type checkers


async def probe_server(spec: dict[str, Any], *, timeout: float = 10.0) -> list[str]:
    """Connect, initialize, list tools, tear down. Bounded end-to-end.

    ``anyio.fail_after`` is used rather than ``asyncio.wait_for`` because
    the MCP SDK's stdio/http clients are anyio task groups: cancelling
    them from the outside needs the cancel scope to be inside the same
    task, otherwise teardown can leak the child process.
    """
    try:
        with anyio.fail_after(timeout):
            return await _probe(spec)
    except TimeoutError:
        raise RuntimeError(
            f"timed out after {timeout:g}s (no response to initialize/list_tools)"
        ) from None
