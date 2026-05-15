"""Main FastAPI application with lifespan management.

CRITICAL SECURITY:
==================
This application MUST ONLY run on 127.0.0.1, on the port that matches
the active checkout — 7887 for the production checkout, 7888 for the
development checkout. Security validation happens at config import
and startup (see app/config.py).

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
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routers import agents_router, websocket_router
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

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static files from {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Setup templates
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    logger.info(f"Configured templates from {templates_dir}")
else:
    logger.warning(f"Templates directory not found: {templates_dir}")


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
