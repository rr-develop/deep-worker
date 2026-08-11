"""A2A HTTP server bootstrap for a deep-worker agent.

Builds a Starlette ASGI app from the SDK route factories (a2a-sdk 1.1.0 has no
``A2AStarletteApplication`` class — §5.1) and runs it via
``uvicorn.Server.serve()`` as an asyncio task ON THE AGENT'S EXISTING EVENT
LOOP. No new threads are created (project policy).

localhost-only by default (``A2AConfig`` refuses non-loopback bind without an
explicit opt-in). Disabled by default; a missing ``a2a-sdk`` import is handled
by :func:`maybe_start_a2a_server` (logged, non-fatal).

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§3.1, §5.1).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .config import A2AConfig

logger = logging.getLogger(__name__)


@dataclass
class A2AServerHandle:
    """Handle to a running A2A server."""

    server: Any  # uvicorn.Server
    task: asyncio.Task
    host: str
    port: int
    url: str

    async def shutdown(self) -> None:
        """Ask uvicorn to stop and await the serving task."""
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self.task.cancel()


def build_app(agent_process: Any, agent_card: Any, *, stream_executor: bool = True) -> Any:
    """Build the Starlette ASGI app exposing this agent over A2A.

    Routes: the agent card (``/.well-known/agent-card.json``) plus the JSON-RPC
    endpoint (``message/send``, ``message/stream``, ``tasks/get``,
    ``tasks/cancel``) driven by :class:`DeepWorkerAgentExecutor`.
    """
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.server.routes import (
        create_agent_card_routes,
        create_jsonrpc_routes,
    )
    from a2a.utils.constants import DEFAULT_RPC_URL
    from starlette.applications import Starlette

    from .executor import DeepWorkerAgentExecutor

    executor = DeepWorkerAgentExecutor(agent_process, stream=stream_executor)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = create_agent_card_routes(agent_card) + create_jsonrpc_routes(
        handler, DEFAULT_RPC_URL
    )
    return Starlette(routes=routes)


async def start_a2a_server(
    agent_process: Any,
    config: A2AConfig,
    *,
    agent_name: str,
    version: str = "1.0.0",
) -> A2AServerHandle:
    """Start the A2A server as a task on the current event loop.

    The actual bound port is read back from the uvicorn server's sockets (so
    ``config.port == 0`` resolves to the OS-assigned port). The AgentCard is
    built AFTER the port is known so its advertised URL is correct.

    Returns:
        A :class:`A2AServerHandle` with the resolved host/port/url.
    """
    import uvicorn

    from .card import build_agent_card

    # Card URL needs the real port. With port 0 we must bind first, read the
    # port, then we can advertise. uvicorn binds during ``serve()``, so we
    # construct the app with a placeholder card and patch the served card's
    # URL once bound. To keep it simple and correct we pre-create the socket.
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.host, config.port))
    bound_host, bound_port = sock.getsockname()[:2]
    url = config.advertised_url(host=bound_host, port=bound_port)

    agent_card = build_agent_card(agent_name=agent_name, url=url, version=version)
    app = build_app(agent_process, agent_card, stream_executor=True)

    uv_config = uvicorn.Config(
        app,
        host=config.host,
        port=bound_port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(uv_config)
    # Hand our pre-bound socket to uvicorn so the advertised port is the one
    # actually served (no race, no second bind).
    task = asyncio.ensure_future(server.serve(sockets=[sock]))

    handle = A2AServerHandle(
        server=server, task=task, host=bound_host, port=bound_port, url=url
    )
    logger.info("A2A server started for %s at %s", agent_name, url)
    return handle


async def maybe_start_a2a_server(
    agent_process: Any,
    config: A2AConfig,
    *,
    agent_name: str,
    version: str = "1.0.0",
) -> A2AServerHandle | None:
    """Start the A2A server only if enabled; never fatal on failure (RC-7).

    Returns the handle on success, or None if disabled or if the optional
    ``a2a-sdk`` dependency is missing / startup failed (logged, agent
    continues without A2A).
    """
    if not config.enabled:
        return None
    try:
        return await start_a2a_server(
            agent_process, config, agent_name=agent_name, version=version
        )
    except ImportError:
        logger.warning(
            "A2A enabled but a2a-sdk is not importable; continuing without A2A. "
            "Install with: pip install 'a2a-sdk[http-server]'"
        )
        return None
    except Exception:  # noqa: BLE001 — A2A must never crash the agent
        logger.exception("Failed to start A2A server; continuing without A2A")
        return None
