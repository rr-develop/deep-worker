"""Main FastAPI application with lifespan management.

CRITICAL SECURITY:
==================
This application MUST ONLY run on 127.0.0.1. Validation happens at
config import and at startup (see app/config.py). The port defaults
per checkout — 7887 for /path/to/production, 7888 for
/path/to/dev, 7887 anywhere else — but an explicit port is
honoured; only the loopback binding is enforced.

ARCHITECTURE:
=============
- Single event loop for all async operations
- AgentService discovers and manages agent connections
- WebSocket broadcasts agent events to all clients
- No threads, no multiple event loops
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from . import paths


class NoStaleStaticFiles(StaticFiles):
    """``StaticFiles`` that forces the browser to revalidate every asset.

    A bare ``StaticFiles`` mount serves ``index.html`` / ``main.js`` with an
    ETag but no ``Cache-Control``, so browsers apply heuristic caching and can
    keep running the PREVIOUS UI without revalidating. That is the root cause
    of the Full Dialog introspection bug: a stale 3-option verbosity dropdown
    cannot emit ``system_prompt``/``tools``, so clicking Full Dialog shows the
    conversation instead. See ``docs/DESIGN_STATIC_NO_STALE_CACHE.md``.

    ``no-cache`` means "cache, but revalidate before every reuse" — combined
    with the existing ETag the browser gets cheap ``304``s while never serving
    a stale asset. On localhost the revalidation cost is negligible.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

from app.config import settings
from app.routers import agents_router, mcp_router, websocket_router
from app.routers.server import router as server_router
from app.services.agent_service import AgentService
from app.utils import uptime
from app.utils.supervisor import install_rpyc_gc_silencer, supervise

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# Wait, in seconds, between a monitor task crash and the respawn.
# Event-driven via ``loop.call_later`` — NOT a polling loop.
# A small delay avoids a tight crash-restart cycle if the failure is
# deterministic (e.g. a bug in _reconcile_from_registry).
_MONITOR_RESPAWN_DELAY = 1.0


def _start_monitor_task(app: FastAPI) -> asyncio.Task[None]:
    """Schedule ``AgentService.monitor_agents`` under the supervisor.

    Declared here (module scope) because both the lifespan startup
    path and the respawn path call it.
    """
    service: AgentService = app.state.agent_service
    loop = asyncio.get_running_loop()

    def _on_monitor_exit(exc: BaseException | None) -> None:
        # Shutdown is signalled by cancellation OR by the service's
        # own shutdown event — in either case do not respawn.
        if isinstance(exc, asyncio.CancelledError):
            return
        if service._shutdown_event.is_set():
            return
        if exc is None:
            # Monitor returned cleanly without shutdown being asked
            # for — unusual, but not a reason to spin up again.
            logger.info("monitor_agents returned cleanly; not respawning")
            return
        logger.error(
            "monitor_agents crashed (%s); respawning in %.1fs",
            exc,
            _MONITOR_RESPAWN_DELAY,
        )

        def _respawn() -> None:
            if service._shutdown_event.is_set():
                return
            app.state.monitor_task = _start_monitor_task(app)

        loop.call_later(_MONITOR_RESPAWN_DELAY, _respawn)

    return supervise(
        service.monitor_agents(),
        name="monitor_agents",
        on_exit=_on_monitor_exit,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown:
    - Startup: Initialize AgentService, start monitoring
    - Shutdown: Cleanup connections, stop tasks

    CRITICAL: All operations run in the SAME event loop!

    Startup is deliberately resilient: a failure in
    ``AgentService.initialize()`` or in spawning the monitor task
    MUST NOT prevent the app from serving. Users should see a
    running app with an error banner / empty agent list, not a
    server that refuses to accept connections. Each failure path is
    logged as ERROR so operators can diagnose.

    Yields:
        None (context manager protocol)
    """
    # === STARTUP ===
    logger.critical(
        f"SECURITY CHECK: Server bound to {settings.host}:{settings.port} ONLY"
    )
    # One-line startup diagnostic with git commit SHA + launch configuration.
    import os as _os
    from app.config import _CHECKOUT_ROOT, get_git_commit_info

    # None when installed from a wheel: there is no checkout and no git
    # tree to describe. ``str(None)`` would log the literal "None" as if it
    # were a path.
    _tree = str(_CHECKOUT_ROOT) if _CHECKOUT_ROOT is not None else "installed"
    _commit, _describe = get_git_commit_info(
        str(_CHECKOUT_ROOT) if _CHECKOUT_ROOT is not None else None
    )
    _env_web = {k: v for k, v in sorted(_os.environ.items())
                if k.startswith("DEEP_WORKER_")}
    _env_str = " ".join(f"{k}={v}" for k, v in _env_web.items())
    logger.info(
        "[STARTUP] component=deep_worker_web "
        f"host={settings.host} port={settings.port} "
        f"commit={_commit} describe={_describe} "
        f"tree={_tree} "
        f"dev_tree={'true' if _os.environ.get('DEEP_WORKER_DEV_TREE') == '1' else 'false'} "
        f"pid={_os.getpid()} "
        f"debug={str(settings.debug).lower()} "
        f"reload={str(settings.reload).lower()} "
        f"{_env_str}"
    )

    logger.info("Starting deep-worker-web application")

    # Initialize uptime tracking
    import time
    uptime.set_server_start_time(time.time())

    # Get the main event loop
    loop = asyncio.get_running_loop()
    logger.info(f"Running in event loop: {id(loop)}")

    # Silence the asyncio GC-time spam for a specific known-benign
    # pattern. See ``make_rpyc_gc_silencer`` for the filter contract
    # and ``test_wave4_asyncio_exception_handler.py`` for the tests.
    install_rpyc_gc_silencer(loop)

    # Always construct the service — so routes that touch
    # ``app.state.agent_service`` don't blow up with AttributeError
    # even if initialize() later fails.
    agent_service = AgentService()
    app.state.agent_service = agent_service
    app.state.monitor_task = None

    logger.info("Initializing AgentService")
    try:
        await agent_service.initialize()
        logger.info(
            "AgentService initialized with %d agents",
            len(await agent_service.list_agents()),
        )
    except Exception as exc:  # noqa: BLE001
        # Resilience: log and continue. The app comes up in degraded
        # mode — agent list will be empty until the registry recovers
        # and the monitor task reconciles.
        logger.error(
            "AgentService.initialize() failed (%s); app starting in "
            "degraded mode — registry scan will retry via the monitor "
            "task",
            exc,
            exc_info=True,
        )

    # Start background monitoring task under supervision so a crash
    # gets logged + respawned event-driven via loop.call_later (no
    # polling). If spawning itself raises, we still want the app to
    # come up and serve the static UI — operators can restart later.
    try:
        app.state.monitor_task = _start_monitor_task(app)
        logger.info("Started background agent monitoring task")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "failed to spawn monitor_agents task (%s); the app is up "
            "but will not react to registry changes until restarted",
            exc,
            exc_info=True,
        )

    logger.info("Startup complete")

    yield

    # === SHUTDOWN ===
    logger.info("Shutting down deep-worker-web application")

    # Shutdown AgentService (best-effort — do not let an error here
    # mask downstream shutdown steps).
    try:
        await agent_service.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "AgentService.shutdown() raised: %s", exc, exc_info=True
        )

    logger.info("Shutdown complete")


def install_exception_handlers(app: FastAPI) -> None:
    """Wire a catch-all exception handler onto ``app``.

    Any unhandled exception raised inside a route handler is turned
    into a structured 500 JSON payload:

        {"error": str(exc), "error_type": exc.__class__.__name__}

    and logged under the ``app.main`` logger with full traceback.
    This is the last line of defence — individual endpoints SHOULD
    still translate known errors into 4xx with clearer messages;
    this handler's job is to make sure the worker stays up and the
    client gets a machine-readable failure rather than a bare HTML
    debug page or a silent connection reset.
    """
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(  # noqa: D401
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled exception in %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            },
        )


# Create FastAPI application
app = FastAPI(
    title="Deep-Worker-Web",
    description="Web UI for managing deep-worker agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Wire the catch-all exception handler before any router so that
# errors originating inside routers still flow through it.
install_exception_handlers(app)

# Include routers
app.include_router(agents_router)
app.include_router(websocket_router)
app.include_router(server_router)
app.include_router(mcp_router)

# Mount static files. The assets are package data: they ship inside the
# wheel next to this package, and the shared resolver finds them in both
# layouts (see app/paths.py). Locating them with a parent chain tuned for
# the checkout would, under the installed layout, point at a directory that
# exists but holds no assets — a silent 404 for every page.
static_dir = paths.web_asset_dir()
if static_dir.is_dir():
    app.mount("/static", NoStaleStaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static files from {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")


@app.get("/")
async def root():
    """Root endpoint - redirect to main page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "host": settings.host,
        "port": settings.port,
    }


if __name__ == "__main__":
    # This should normally be run via run.py, but we support direct execution
    import uvicorn
    
    logger.critical(
        f"Running directly via python3 -m app.main\n"
        f"SECURITY: Bound to {settings.host}:{settings.port} ONLY"
    )
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
        reload=settings.debug,
    )
