"""Tests for interactive mode (dwcli <agent-name>)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add rpyc_async to path
import rpyc

from deep_worker_cli.cli.interactive import InteractiveApp
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo

class TestInteractiveMode:
    """Tests for interactive mode functionality."""

    @pytest.mark.asyncio
    async def test_interactive_app_connects_to_agent(self):
        """Test that InteractiveApp connects to running agent via RPyC."""
        agent_name = "test-agent"

        with patch("deep_worker_cli.cli.interactive.ProcessRegistry") as mock_registry_class:
            # Mock registry to return agent info with RPyC port
            mock_registry = MagicMock(spec=ProcessRegistry)
            mock_agent_info = MagicMock(spec=AgentProcessInfo)
            mock_agent_info.name = agent_name
            mock_agent_info.rpyc_port = 18861
            mock_registry.lookup.return_value = mock_agent_info
            mock_registry_class.return_value = mock_registry

            app = InteractiveApp(agent_name=agent_name)

            # App should store agent name
            assert app.agent_name == agent_name
            assert app._rpyc_conn is None  # Not connected yet

    @pytest.mark.asyncio
    async def test_interactive_app_sends_message(self):
        """Test sending message to agent via RPyC."""
        agent_name = "test-agent"
        user_message = "Hello, agent!"

        with patch("deep_worker_cli.cli.interactive.rpyc") as mock_rpyc_module:
            # Create a mock root object
            mock_root = AsyncMock()
            mock_root.invoke_agent = AsyncMock(return_value={
                "success": True,
                "response": {
                    "role": "assistant",
                    "content": "Hello, user!"
                }
            })

            app = InteractiveApp(agent_name=agent_name)
            app._rpyc_root = mock_root

            # Send message through app
            result = await app.send_user_message(user_message)

            # Should return success (fire-and-forget, so response is generic)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_interactive_app_disconnects_gracefully(self):
        """Test graceful disconnect from agent via RPyC.

        `disconnect_from_agent` now awaits `self._rpyc_conn.aclose()`
        (the async-native close path from rpyc_async); the legacy
        sync `.close()` / `disable_asyncio_serving()` pair is no
        longer called directly — aclose handles both internally.
        """
        agent_name = "test-agent"

        # Create mock connection with async aclose.
        mock_conn = MagicMock()
        mock_conn.aclose = AsyncMock()

        app = InteractiveApp(agent_name=agent_name)
        app._rpyc_conn = mock_conn

        # Disconnect
        await app.disconnect_from_agent()

        # aclose must be awaited exactly once and the conn reference
        # must be cleared.
        mock_conn.aclose.assert_awaited_once()
        assert app._rpyc_conn is None

    def test_interactive_app_requires_agent_name(self):
        """Test that agent name is required."""
        with pytest.raises(ValueError, match="agent_name"):
            InteractiveApp(agent_name="")

    @pytest.mark.asyncio
    async def test_load_more_history_pagination(self):
        """Test that _load_more_history() loads additional message pages with offset via RPyC."""
        import os
        from textual.containers import Container

        agent_name = "test-agent"
        pagination_log = "/tmp/dwcli_pagination_unit_test.log"

        # Clean up log file
        if os.path.exists(pagination_log):
            os.remove(pagination_log)

        # Create mock root object with get_history method
        mock_root = AsyncMock()

        # _load_more_history does `json.loads(response_json_str)` on
        # the reply, so the mock must return a JSON STRING — see
        # interactive.py ~line 1094 and RPYC_SERIALIZATION_DESIGN.md.
        async def mock_get_history(limit, offset):
            """Mock paginated history responses via RPyC (JSON-encoded)."""
            import json as _json
            if offset == 20:
                with open(pagination_log, "a") as f:
                    f.write(f"PAGINATION: offset={offset}, total=100\n")
                return _json.dumps({
                    "success": True,
                    "total": 100,
                    "history": [
                        {"type": "HumanMessage", "content": f"Query {i}"}
                        for i in range(20, 40)
                    ]
                })
            elif offset == 40:
                with open(pagination_log, "a") as f:
                    f.write(f"PAGINATION: offset={offset}, total=100\n")
                return _json.dumps({
                    "success": True,
                    "total": 100,
                    "history": [
                        {"type": "HumanMessage", "content": f"Query {i}"}
                        for i in range(40, 60)
                    ]
                })
            else:
                return _json.dumps({"success": True, "total": 100, "history": []})

        mock_root.get_history = AsyncMock(side_effect=mock_get_history)

        # Create app
        app = InteractiveApp(agent_name=agent_name)
        app._rpyc_root = mock_root
        app._connected = True

        # Mock Textual components that _load_more_history needs
        mock_container = MagicMock(spec=Container)
        mock_container.mount = AsyncMock()

        # Mock VerticalScroll for chat scroll tracking
        mock_chat_scroll = MagicMock()
        mock_chat_scroll.scroll_y = 100
        mock_chat_scroll.max_scroll_y = 500

        def mock_query_one(selector, *args):
            """Mock query_one to return different objects based on selector."""
            if selector == "#chat":
                return mock_chat_scroll
            elif selector == "#messages":
                return mock_container
            return MagicMock()

        with patch.object(app, 'query_one', side_effect=mock_query_one):
            # Simulate initial state: 20 messages loaded
            app._history_offset = 20
            app._history_total = 100
            app._loading_history = False

            # Call _load_more_history() - should load offset=20
            await app._load_more_history()

            # Verify: offset should increase to 40
            assert app._history_offset == 40, f"Expected offset 40, got {app._history_offset}"

            # Call again - should load offset=40
            await app._load_more_history()

            # Verify: offset should increase to 60
            assert app._history_offset == 60, f"Expected offset 60, got {app._history_offset}"

            # Verify get_history was called twice (for offset=20 and offset=40)
            assert mock_root.get_history.call_count == 2

            # Verify pagination log exists and has events
            assert os.path.exists(pagination_log), "Pagination log file should exist"

            with open(pagination_log, "r") as f:
                log_content = f.read()
                assert "PAGINATION: offset=20" in log_content, "Should log first pagination"
                assert "PAGINATION: offset=40" in log_content, "Should log second pagination"
                pagination_count = log_content.count("PAGINATION:")
                assert pagination_count >= 2, f"Expected at least 2 pagination events, got {pagination_count}"

        # Clean up
        if os.path.exists(pagination_log):
            os.remove(pagination_log)

class TestInteractiveCommand:
    """Tests for interactive command entry point."""

    @pytest.mark.asyncio
    async def test_connect_interactive_with_valid_agent(self):
        """Test connecting to valid running agent."""
        from deep_worker_cli.cli.client_command import connect_interactive

        agent_name = "test-agent"

        # Patch where InteractiveApp is imported (inside the function)
        with patch("deep_worker_cli.cli.interactive.InteractiveApp") as mock_app_class, \
             patch("deep_worker_cli.cli.client_command.ProcessRegistry") as mock_registry:

            # Mock registry to show agent exists
            mock_registry_instance = MagicMock()
            mock_registry_instance.lookup.return_value = MagicMock(name=agent_name, pid=12345, rpyc_port=18861)
            mock_registry.return_value = mock_registry_instance

            # Mock app
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            mock_app_class.return_value = mock_app

            # Connect (async)
            result = await connect_interactive(agent_name)

            # Should create app with agent name
            mock_app_class.assert_called_once_with(agent_name=agent_name)

            # Should run app
            mock_app.run_async.assert_called_once()

            # Should return success
            assert result == 0

    @pytest.mark.asyncio
    async def test_connect_interactive_with_missing_agent(self):
        """Test connecting to non-existent agent."""
        from deep_worker_cli.cli.client_command import connect_interactive

        agent_name = "missing-agent"

        with patch("deep_worker_cli.cli.client_command.ProcessRegistry") as mock_registry:
            # Mock registry to show agent doesn't exist
            mock_registry_instance = MagicMock()
            mock_registry_instance.lookup.return_value = None
            mock_registry.return_value = mock_registry_instance

            # Connect should fail (async)
            result = await connect_interactive(agent_name)

            # Should return error code
            assert result == 1
