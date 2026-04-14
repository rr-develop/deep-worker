"""API routers for deep-worker-web."""

from app.routers.agents import router as agents_router
from app.routers.websocket import router as websocket_router

__all__ = ["agents_router", "websocket_router"]
