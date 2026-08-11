"""Tests for ClientManager - manages ClaudeSDKClient instances per thread_id."""

from typing import Any
import pytest
from unittest.mock import AsyncMock, Mock, patch

from deep_worker.providers.claude_code_provider.client_manager import ClientManager
from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata


class TestClientManagerInit:
    """Tests for ClientManager initialization."""

    def test_init_with_base_options(self) -> None:
        """ClientManager should initialize with base options."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        assert manager.base_options == options
        assert manager.clients == {}

    def test_init_without_options(self) -> None:
        """ClientManager should work with None options."""
        manager = ClientManager(base_options=None)

        assert manager.base_options is None
        assert manager.clients == {}


class TestClientManagerGetOrCreate:
    """Tests for get_or_create_client functionality."""

    @pytest.mark.asyncio
    async def test_create_new_client(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Should create new client for new thread_id."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Mock ClaudeSDKClient constructor
        with patch('claude_code_provider.client_manager.ClaudeSDKClient', return_value=mock_claude_sdk_client):
            client, proxy_mcp = await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

        # Should have created client
        assert client == mock_claude_sdk_client
        assert proxy_mcp is not None

        # Should have called connect
        mock_claude_sdk_client.connect.assert_called_once()

        # Should be cached
        assert "session-1" in manager.clients

    @pytest.mark.asyncio
    async def test_return_cached_client(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Should return cached client for existing thread_id."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Mock ClaudeSDKClient
        with patch('claude_code_provider.client_manager.ClaudeSDKClient', return_value=mock_claude_sdk_client):
            # First call - creates client
            client1, proxy1 = await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

            # Second call - should return cached
            client2, proxy2 = await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

        # Should be same instances
        assert client1 is client2
        assert proxy1 is proxy2

        # Connect should only be called once
        assert mock_claude_sdk_client.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_different_threads_different_clients(
        self,
        sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Different thread_ids should get different clients."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Create two different mock clients
        client1_mock = Mock()
        client1_mock.connect = AsyncMock()
        client2_mock = Mock()
        client2_mock.connect = AsyncMock()

        call_count = [0]
        def client_factory(*args: Any, **kwargs: Any) -> Mock:
            call_count[0] += 1
            return client1_mock if call_count[0] == 1 else client2_mock

        with patch('claude_code_provider.client_manager.ClaudeSDKClient', side_effect=client_factory):
            client1, proxy1 = await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

            client2, proxy2 = await manager.get_or_create_client(
                thread_id="session-2",
                tools_metadata=sample_tool_metadata
            )

        # Should be different instances
        assert client1 is not client2
        assert proxy1 is not proxy2

        # Both should be cached
        assert "session-1" in manager.clients
        assert "session-2" in manager.clients

    @pytest.mark.asyncio
    async def test_disallowed_tools_configured(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Should configure disallowed_tools to block all built-in tools."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        created_options = None
        def capture_options(*args: Any, **kwargs: Any) -> Mock:
            nonlocal created_options
            created_options = kwargs.get('options')
            return mock_claude_sdk_client

        with patch('claude_code_provider.client_manager.ClaudeSDKClient', side_effect=capture_options):
            await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

        # Should have disallowed all built-in tools
        assert created_options is not None
        assert "disallowed_tools" in created_options.__dict__
        disallowed = created_options.disallowed_tools

        # Check for key built-in tools
        assert "Read" in disallowed
        assert "Write" in disallowed
        assert "Bash" in disallowed
        assert "Task" in disallowed

    @pytest.mark.asyncio
    async def test_mcp_server_configured(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Should configure MCP server with ProxyMCPServer."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        created_options_mcp = None
        def capture_options_mcp(*args: Any, **kwargs: Any) -> Mock:
            nonlocal created_options_mcp
            created_options_mcp = kwargs.get('options')
            return mock_claude_sdk_client

        with patch('claude_code_provider.client_manager.ClaudeSDKClient', side_effect=capture_options_mcp):
            await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

        # Should have MCP server configured
        created_options = created_options_mcp
        assert created_options is not None
        assert "deepworker" in created_options.mcp_servers
        assert created_options.mcp_servers["deepworker"]["type"] == "sdk"


class TestClientManagerCleanup:
    """Tests for cleanup_client functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_existing_client(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Should disconnect and remove client on cleanup."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Create client
        with patch('claude_code_provider.client_manager.ClaudeSDKClient', return_value=mock_claude_sdk_client):
            await manager.get_or_create_client(
                thread_id="session-1",
                tools_metadata=sample_tool_metadata
            )

        # Verify cached
        assert "session-1" in manager.clients

        # Cleanup
        await manager.cleanup_client("session-1")

        # Should have called disconnect
        mock_claude_sdk_client.disconnect.assert_called_once()

        # Should be removed from cache
        assert "session-1" not in manager.clients

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_client(self) -> None:
        """Cleanup of non-existent client should not raise error."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Should not raise
        await manager.cleanup_client("nonexistent")

    @pytest.mark.asyncio
    async def test_cleanup_all_clients(
        self,
        sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Should be able to cleanup all clients."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        # Create multiple clients
        client1 = Mock()
        client1.connect = AsyncMock()
        client1.disconnect = AsyncMock()

        client2 = Mock()
        client2.connect = AsyncMock()
        client2.disconnect = AsyncMock()

        call_count = [0]
        def client_factory(*args: Any, **kwargs: Any) -> Mock:
            call_count[0] += 1
            return client1 if call_count[0] == 1 else client2

        with patch('claude_code_provider.client_manager.ClaudeSDKClient', side_effect=client_factory):
            await manager.get_or_create_client("session-1", sample_tool_metadata)
            await manager.get_or_create_client("session-2", sample_tool_metadata)

        # Cleanup all
        await manager.cleanup_client("session-1")
        await manager.cleanup_client("session-2")

        # Both should be disconnected
        client1.disconnect.assert_called_once()
        client2.disconnect.assert_called_once()

        # Cache should be empty
        assert len(manager.clients) == 0


class TestClientManagerConcurrency:
    """Tests for concurrent access to ClientManager."""

    @pytest.mark.asyncio
    async def test_concurrent_get_or_create_same_thread(
        self,
        sample_tool_metadata: list[ToolMetadata],
        mock_claude_sdk_client: Mock
    ) -> None:
        """Concurrent calls for same thread_id should return same client."""
        import asyncio
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        manager = ClientManager(base_options=options)

        with patch('claude_code_provider.client_manager.ClaudeSDKClient', return_value=mock_claude_sdk_client):
            # Start multiple concurrent calls
            results = await asyncio.gather(
                manager.get_or_create_client("session-1", sample_tool_metadata),
                manager.get_or_create_client("session-1", sample_tool_metadata),
                manager.get_or_create_client("session-1", sample_tool_metadata),
            )

        # All should return same client
        clients = [r[0] for r in results]
        assert clients[0] is clients[1]
        assert clients[1] is clients[2]

        # Connect should only be called once
        assert mock_claude_sdk_client.connect.call_count == 1
