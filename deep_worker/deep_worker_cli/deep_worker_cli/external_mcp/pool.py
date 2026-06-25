"""ExternalMCPClientPool — starts MCP clients for every server in the
merged config, reads their tool lists, and returns a list of
``MCPToolAdapter``s to pass into ``create_cli_agent(tools=...)``.

Lifecycle:
    pool = await ExternalMCPClientPool.create(user_path=..., ...)
    tools = pool.tools                  # list[BaseTool] for the graph
    ...
    await pool.stop()                    # closes clients cleanly

Each started server lives as long as the ``pool`` (i.e. for the whole
AgentProcess lifecycle). A crash of one server does not bring down the
pool: it logs a warning and the remaining tools keep working.

See DESIGN_EXTERNAL_MCP_TOOLS.md §3.4.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from deep_worker_cli.external_mcp.adapter import MCPToolAdapter
from deep_worker_cli.external_mcp.config import (
    ServerSpec,
    load_mcp_config,
)
from deep_worker_cli.external_mcp.mangler import MCPToolNameMangler

logger = logging.getLogger(__name__)


class ExternalMCPClientPool:
    """Per-agent pool of external MCP clients."""

    def __init__(self) -> None:
        self._stack: AsyncExitStack = AsyncExitStack()
        self._tools: list[BaseTool] = []
        self._auto_approved_tool_names: set[str] = set()
        # name → (session, declared_spec) — for diagnostics and possible
        # reload in the future.
        self._sessions: dict[str, object] = {}
        self._started: bool = False
        # ONE mangler for the whole pool — this catches collisions between
        # tools of DIFFERENT servers of one agent (see
        # docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md §3.2).
        self._mangler: MCPToolNameMangler = MCPToolNameMangler()

    @property
    def mangler(self) -> MCPToolNameMangler:
        """Per-agent bijection of canonical ↔ provider-safe names. Public
        for UI/introspection (Web Dialog «verbosity=tools» and the like)."""
        return self._mangler

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def auto_approved_tool_names(self) -> set[str]:
        """Names of tools from servers marked ``auto_approve: true``.

        These tools must be skipped by HITL EVEN IF the agent was started
        WITHOUT ``--skip-permissions``. With ``--skip-permissions`` they
        are not interrupted anyway — but a per-server override is useful
        in the general case.
        """
        return set(self._auto_approved_tool_names)

    @classmethod
    async def create(
        cls,
        *,
        user_path: Path | None = None,
        project_path: Path | None = None,
        agent_path: Path | None = None,
        cli_configs: Iterable[Path] = (),
        cli_servers: Iterable[str] = (),
    ) -> ExternalMCPClientPool:
        """Collect the merged config and start all enabled servers.

        ``UnresolvedEnvError`` from ``load_mcp_config`` is propagated
        outward — this is an intentional fail-loud (see §3.1 of the design).
        """
        merged = load_mcp_config(
            user_path=user_path,
            project_path=project_path,
            agent_path=agent_path,
            cli_configs=list(cli_configs),
            cli_servers=list(cli_servers),
        )
        pool = cls()
        await pool._start_all(merged)
        return pool

    async def _start_all(self, merged: dict[str, ServerSpec]) -> None:
        if self._started:
            raise RuntimeError("pool already started")
        self._started = True

        # Import the mcp SDK only on an actual start — so unit tests that
        # do not use the network/processes do not pay for the import.
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            streamablehttp_client: Any
            try:
                from mcp.client.streamable_http import (
                    streamablehttp_client as _streamablehttp_client,
                )

                streamablehttp_client = _streamablehttp_client
            except Exception:  # pragma: no cover — older mcp
                streamablehttp_client = None
        except Exception as e:  # pragma: no cover
            logger.warning("mcp SDK not available, external MCP disabled: %s", e)
            return

        for name, spec in merged.items():
            server_stack = AsyncExitStack()
            try:
                if spec.type == "stdio":
                    params = StdioServerParameters(
                        command=spec.command or "",
                        args=list(spec.args),
                        env={**os.environ, **dict(spec.env)} if spec.env else None,
                        cwd=spec.cwd,
                    )
                    streams_cm = stdio_client(params)
                elif spec.type == "http":
                    if streamablehttp_client is None:
                        logger.warning(
                            "mcp server %r: http transport not available "
                            "in installed mcp SDK; skipping", name
                        )
                        continue
                    streams_cm = streamablehttp_client(
                        spec.url or "",
                        headers=dict(spec.headers) if spec.headers else None,
                    )
                else:  # pragma: no cover
                    logger.warning(
                        "unknown mcp server type for %r: %s", name, spec.type
                    )
                    continue

                streams = await server_stack.enter_async_context(streams_cm)
                # stdio returns (read, write); HTTP returns
                # (read, write, session-id callback).
                read, write = streams[0], streams[1]
                session = await server_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                listed = await session.list_tools()
                # All adapters of this server are registered in the SHARED
                # per-agent mangler — it also catches collisions between
                # different servers (see
                # docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md §3.2).
                # So that a collision on the N-th tool does not leave dirty
                # partial registrations in the main mangler, we build the
                # adapters on a TEMPORARY mangler, then — only-on-success —
                # move the issued pairs into the main one. A collision →
                # MCPToolNameCollisionError, handled in the common
                # `except` below as "server failed to start" (fail-loud +
                # degrade): the server is skipped, the rest keep working.
                tentative_mangler = MCPToolNameMangler()
                # Replay already-registered canonical names from the main
                # mangler so a collision with previous servers also fires
                # at the build stage.
                for prior_canonical in self._mangler.all_canonical_names():
                    tentative_mangler.add(prior_canonical)
                adapters = [
                    MCPToolAdapter(
                        session=session,
                        server_name=name,
                        tool_name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema
                        or {"type": "object", "properties": {}},
                        mangler=tentative_mangler,
                    )
                    for tool in listed.tools
                ]
                # All-or-nothing commit: move the new pairs into the main
                # mangler. Every add() here is guaranteed collision-free
                # because tentative_mangler already contained the prior state.
                for adapter in adapters:
                    self._mangler.add(adapter.canonical_name)

                await self._stack.enter_async_context(server_stack)
                self._tools.extend(adapters)
                if spec.auto_approve:
                    self._auto_approved_tool_names.update(a.name for a in adapters)
                self._sessions[name] = session
                logger.info(
                    "external MCP server %r: %d tools loaded", name, len(adapters)
                )
            except Exception as e:
                await server_stack.aclose()
                logger.warning(
                    "external MCP server %r failed to start: %s",
                    name, e, exc_info=True,
                )

    async def stop(self) -> None:
        if not self._started:
            return
        # AsyncExitStack guarantees a correct teardown of all resources in
        # reverse order (sessions → streams), including killing the child
        # processes of stdio servers.
        try:
            await self._stack.aclose()
        except Exception as e:  # noqa: BLE001
            logger.warning("ExternalMCPClientPool.stop encountered: %s", e)
        finally:
            self._started = False
            self._tools.clear()
            self._sessions.clear()
            self._auto_approved_tool_names.clear()
