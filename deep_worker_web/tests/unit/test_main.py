"""Tests for main application."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class TestMainApplication:
    """Test main FastAPI application."""

    def test_app_creation(self) -> None:
        """Test that app is created successfully."""
        assert app is not None
        assert app.title == "Deep-Worker-Web"

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["host"] == "127.0.0.1"
            # The bound port is tree-dependent by design (7887 for the prod
            # checkout, 7888 for the dev checkout — see app/config.py
            # _APPROVED_BINDINGS).
            # Assert against the configured value rather than a hard-coded
            # number so the test passes in whichever tree it runs from.
            assert data["port"] == settings.port

    def test_root_redirect(self) -> None:
        """Test that root redirects to index.html."""
        with TestClient(app) as client:
            response = client.get("/", follow_redirects=False)
            assert response.status_code == 307  # Temporary redirect
            assert response.headers["location"] == "/static/index.html"

    def test_routers_included(self) -> None:
        """Test that all routers are included."""
        routes = [route.path for route in app.routes]
        
        # Check for API routes
        assert "/api/agents" in routes
        assert "/api/agents/{agent_name}" in routes
        
        # Check for WebSocket route
        assert "/ws/agents" in routes
        
        # Check for utility routes
        assert "/health" in routes
