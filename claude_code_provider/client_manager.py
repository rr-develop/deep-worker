"""ClientManager - manages ClaudeSDKClient instances for different thread_ids.

This module provides thread-safe management of Claude SDK client instances, ensuring that
each thread_id (session) gets its own isolated client and ProxyMCPServer.
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402

from claude_code_provider.proxy_mcp_server import ProxyMCPServer  # noqa: E402
from claude_code_provider.custom_types import ToolMetadata  # noqa: E402

# All built-in Claude Code tools that should be disabled
# We want all tools to execute in deep-worker, not in Claude CLI
BUILT_IN_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Glob",
    "Grep",
    "Task",
    "TodoWrite",
    "Skill",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "BashOutput",
    "KillShell",
    "AskUserQuestion",
    "SlashCommand",
    "LS",
]


class ClientManager:
    """Manages ClaudeSDKClient instances for different thread_ids.

    Each thread_id (LangGraph thread) gets its own ClaudeSDKClient and ProxyMCPServer.
    This ensures session isolation while allowing parallel execution of multiple threads.

    The manager:
    1. Creates new clients on demand with configured ProxyMCPServer
    2. Caches clients by thread_id to reuse across multiple calls
    3. Disables all built-in Claude Code tools (execution happens in deep-worker)
    4. Provides cleanup to disconnect and remove clients

    Thread-safety: Uses asyncio lock to prevent race conditions when creating clients.

    Attributes:
        base_options: Base ClaudeAgentOptions to use for all clients
        clients: Cache mapping thread_id to (client, proxy_mcp) tuples
        _creation_locks: Locks per thread_id to prevent concurrent creation

    Example:
        >>> from claude_agent_sdk import ClaudeAgentOptions
        >>> options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        >>> manager = ClientManager(base_options=options)
        >>>
        >>> # Get client for a thread
        >>> client, proxy = await manager.get_or_create_client(
        ...     thread_id="session-1",
        ...     tools_metadata=[...]
        ... )
        >>>
        >>> # Cleanup when done
        >>> await manager.cleanup_client("session-1")
    """

    def __init__(self, base_options: ClaudeAgentOptions | None = None) -> None:
        """Initialize ClientManager with base options.

        Args:
            base_options: Base options to use for all clients. If None, defaults will be used.
        """
        self.base_options = base_options
        self.clients: dict[str, tuple[ClaudeSDKClient, ProxyMCPServer]] = {}
        self._creation_locks: dict[str, asyncio.Lock] = {}

    async def get_or_create_client(
        self, thread_id: str, tools_metadata: list[ToolMetadata]
    ) -> tuple[ClaudeSDKClient, ProxyMCPServer]:
        """Get or create ClaudeSDKClient for a thread_id.

        If a client already exists for this thread_id, returns the cached instance.
        Otherwise, creates a new client with ProxyMCPServer and caches it.

        This method is thread-safe: concurrent calls for the same thread_id will
        wait for the first call to complete and then return the same client.

        Args:
            thread_id: Unique identifier for this thread/session
            tools_metadata: Metadata for tools to expose through ProxyMCPServer

        Returns:
            Tuple of (ClaudeSDKClient, ProxyMCPServer)
        """
        # If already cached, return immediately
        if thread_id in self.clients:
            return self.clients[thread_id]

        # Get or create lock for this thread_id
        if thread_id not in self._creation_locks:
            self._creation_locks[thread_id] = asyncio.Lock()

        async with self._creation_locks[thread_id]:
            # Double-check after acquiring lock (another coroutine might have created it)
            if thread_id in self.clients:
                return self.clients[thread_id]

            # Create ProxyMCPServer for this thread
            proxy_mcp = ProxyMCPServer(tools_metadata)

            # Build options for this client
            if self.base_options is not None:
                # Use replace to create new instance with updated fields
                options = replace(
                    self.base_options,
                    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
                    disallowed_tools=BUILT_IN_TOOLS,
                )
            else:
                # Create default options
                options = ClaudeAgentOptions(
                    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
                    disallowed_tools=BUILT_IN_TOOLS,
                )

            # Create and connect client
            client = ClaudeSDKClient(options=options)
            await client.connect()

            # Cache
            self.clients[thread_id] = (client, proxy_mcp)

            return client, proxy_mcp

    async def cleanup_client(self, thread_id: str) -> None:
        """Cleanup client for a thread_id.

        Disconnects the client and removes it from cache.
        Safe to call even if thread_id doesn't exist.

        Args:
            thread_id: Thread ID to cleanup
        """
        if thread_id in self.clients:
            client, _ = self.clients[thread_id]
            await client.disconnect()
            del self.clients[thread_id]

            # Also remove lock
            if thread_id in self._creation_locks:
                del self._creation_locks[thread_id]
