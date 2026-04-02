"""Tests for permission delegation architecture in claude-code provider.

ClaudeCodeChatModel uses ClaudeSDKClient internally. To avoid double layer
of permission prompts (one in Claude SDK, one in deep-worker), we MUST
configure ClaudeSDKClient to bypass its internal permission checks.

All tool approval decisions should be made by deep-worker's AgentProcess,
NOT by Claude SDK.

Architecture:
    User → deep-worker (AgentProcess - ONLY approval layer)
          → ClaudeSDKClient (NO permission prompts)
          → ProxyMCPServer → tool execution

This test verifies that ClaudeSDKClient is always created with
permission_mode='bypassPermissions' to delegate all approval to deep-worker.
"""

import pytest

from claude_code_provider.client_manager import ClientManager


class TestPermissionDelegation:
    """Test that ClaudeSDKClient bypasses internal permission checks."""

    @pytest.mark.asyncio
    async def test_client_created_with_bypass_permissions(self):
        """ClaudeSDKClient MUST be created with bypassPermissions mode.

        This ensures:
        - NO double layer of permission prompts
        - ALL approval decisions made by deep-worker (AgentProcess)
        - Claude SDK does NOT prompt for tool execution
        """
        # Create ClientManager with default options (no explicit permission_mode)
        manager = ClientManager(base_options=None)

        # Get or create a client
        client, _proxy_mcp = await manager.get_or_create_client(
            thread_id="test_permissions",
            tools_metadata=[
                {
                    "name": "dangerous_tool",
                    "description": "A dangerous tool that requires approval",
                    "input_schema": {},
                }
            ],
        )

        # CRITICAL CHECK: Verify that client was created with bypassPermissions
        # This ensures Claude SDK will NOT show its own permission prompts
        # and all approval is delegated to deep-worker
        assert hasattr(client, "options"), "Client should have options attribute"

        options = client.options
        assert hasattr(
            options, "permission_mode"
        ), "Options should have permission_mode"

        # THE KEY ASSERTION: permission_mode MUST be 'bypassPermissions'
        assert (
            options.permission_mode == "bypassPermissions"
        ), f"Expected permission_mode='bypassPermissions', got '{options.permission_mode}'. Claude SDK MUST NOT show its own permission prompts!"

        # Cleanup
        await manager.cleanup_client("test_permissions")

    @pytest.mark.asyncio
    async def test_explicit_base_options_still_bypass_permissions(self):
        """Even with explicit base_options, permission_mode should be bypassPermissions.

        If user provides ClaudeAgentOptions with a different permission_mode,
        ClientManager should override it to ensure consistent architecture.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        # User provides options with 'default' permission mode (wrong!)
        user_options = ClaudeAgentOptions(
            model="claude-sonnet-4-5-20250929",
            permission_mode="default",  # This should be overridden!
        )

        manager = ClientManager(base_options=user_options)

        client, _proxy_mcp = await manager.get_or_create_client(
            thread_id="test_override", tools_metadata=[]
        )

        # Even though user specified 'default', it should be overridden to 'bypassPermissions'
        options = client.options
        assert (
            options.permission_mode == "bypassPermissions"
        ), "ClientManager MUST override permission_mode to ensure approval is delegated to deep-worker"

        # Cleanup
        await manager.cleanup_client("test_override")

    @pytest.mark.asyncio
    async def test_multiple_clients_all_bypass_permissions(self):
        """All clients created by ClientManager should bypass permissions.

        This tests that the permission delegation is consistent across
        multiple thread_ids and client instances.
        """
        manager = ClientManager(base_options=None)

        thread_ids = ["thread1", "thread2", "thread3"]

        try:
            for thread_id in thread_ids:
                client, _proxy_mcp = await manager.get_or_create_client(
                    thread_id=thread_id, tools_metadata=[]
                )

                options = client.options
                assert (
                    options.permission_mode == "bypassPermissions"
                ), f"Client for {thread_id} should have bypassPermissions mode"
        finally:
            # Cleanup all (ignore errors during cleanup)
            for thread_id in thread_ids:
                try:
                    await manager.cleanup_client(thread_id)
                except BaseException:
                    # Ignore cleanup errors (including CancelledError)
                    # The test assertions already passed
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
