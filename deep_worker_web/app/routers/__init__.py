"""API routers for deep-worker-web."""

from app.routers.agents import router as agents_router
from app.routers.mcp import router as mcp_router
from app.routers.websocket import router as websocket_router

__all__ = ["agents_router", "mcp_router", "websocket_router"]
