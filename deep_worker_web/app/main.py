"""Main FastAPI application with lifespan management.

CRITICAL SECURITY:
==================
This application MUST ONLY run on 127.0.0.1:7887!
Security validation happens at config import and startup.

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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routers import agents_router, websocket_router
from app.routers.server import router as server_router
from app.services.agent_service import AgentService
from app.utils import uptime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.
    
    Handles startup and shutdown:
    - Startup: Initialize AgentService, start monitoring
    - Shutdown: Cleanup connections, stop tasks
    
    CRITICAL: All operations run in the SAME event loop!
    
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

    # Initialize AgentService
    logger.info("Initializing AgentService")
    agent_service = AgentService()
    await agent_service.initialize()
    
    # Store in app state for dependency injection
    app.state.agent_service = agent_service
    
    logger.info(f"AgentService initialized with {len(await agent_service.list_agents())} agents")
    
    # Start background monitoring task
    # This periodically checks ProcessRegistry for new/removed agents
    # and maintains connections
    monitor_task = asyncio.create_task(agent_service.monitor_agents())
    app.state.monitor_task = monitor_task
    logger.info("Started background agent monitoring task")
    
    # Note: Connections to agents happen on-demand or via monitoring task
    # We don't connect at startup to avoid blocking tests
    
    logger.info("Startup complete")

    yield

    # === SHUTDOWN ===
    logger.info("Shutting down deep-worker-web application")
    
    # Shutdown AgentService
    await agent_service.shutdown()
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Deep-Worker-Web",
    description="Web UI for managing deep-worker agents",
    version="1.0.0",
    lifespan=lifespan,
)

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
