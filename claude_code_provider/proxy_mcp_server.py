"""ProxyMCPServer - In-process MCP server that delegates tool execution to deep-worker.

This module provides a critical component for integrating Claude Code with deep-worker.
The ProxyMCPServer registers tools from deep-worker as MCP tools, but does NOT execute them.
Instead, it blocks on asyncio.Future until deep-worker executes the tool and provides the result.

This allows Claude Code CLI to call tools through MCP, but execution happens in deep-worker,
maintaining full transparency and compatibility with LangGraph.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# Add claude_sdk from source to sys.path
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: E402

from claude_code_provider.custom_types import (  # noqa: E402
    MCPResult,
    PendingCall,
    PendingCallInternal,
    ToolMetadata,
)


class ProxyMCPServer:
    """In-process MCP server that proxies tool calls to deep-worker.

    This server:
    1. Registers tools from deep-worker as MCP tools
    2. When Claude calls a tool, creates an asyncio.Future and BLOCKS
    3. Returns the tool call info to ClaudeCodeChatModel
    4. Waits for deep-worker to execute the tool
    5. When resolve_call() is called with results, unblocks and returns to Claude

    This design allows Claude Code CLI to see tools and call them through MCP,
    while actual execution happens in deep-worker's LangGraph workflow.

    Attributes:
        tool_handlers: List of decorated tool handlers
        mcp_server_config: MCP server configuration dict
        pending_calls: Dict mapping call_id to PendingCallInternal
        timeout: Timeout in seconds for tool execution

    Example:
        >>> tools_metadata = [
        ...     {"name": "read_file", "description": "Read a file", "input_schema": {...}}
        ... ]
        >>> proxy = ProxyMCPServer(tools_metadata)
        >>> # In background: handler = await proxy.tool_handlers[0].handler({"path": "/tmp/file.txt"})
        >>> # Handler blocks here...
        >>> pending = proxy.get_pending_calls()
        >>> # Execute tool in deep-worker
        >>> result = execute_tool(pending[0]["name"], pending[0]["args"])
        >>> proxy.resolve_call(pending[0]["id"], result)
        >>> # Handler unblocks and returns result to Claude
    """

    def __init__(self, tools_metadata: list[ToolMetadata], timeout: float = 60.0) -> None:
        """Initialize ProxyMCPServer with tool metadata.

        Args:
            tools_metadata: List of tool metadata dicts with name, description, input_schema
            timeout: Timeout in seconds for tool execution (default: 60.0)
        """
        self.pending_calls: dict[str, PendingCallInternal] = {}
        self.tool_handlers: list[Any] = []
        self.timeout = timeout
        self._call_counter = 0

        # Create handler for each tool
        for tool_meta in tools_metadata:
            handler = self._create_tool_handler(tool_meta["name"])

            # Decorate as MCP tool
            decorated = tool(
                tool_meta["name"], tool_meta["description"], tool_meta["input_schema"]
            )(handler)

            self.tool_handlers.append(decorated)

        # Create MCP server
        # Type: McpSdkServerConfig but we declare as dict for compatibility
        self.mcp_server_config: Any = create_sdk_mcp_server(
            name="deepworker", version="1.0.0", tools=self.tool_handlers
        )

    def _create_tool_handler(self, tool_name: str) -> Any:
        """Create async handler for a tool.

        The handler:
        1. Creates asyncio.Future
        2. Stores it in pending_calls
        3. BLOCKS waiting for Future to be resolved
        4. Returns result when resolve_call() is called

        Args:
            tool_name: Name of the tool

        Returns:
            Async handler function
        """

        async def handler(args: dict[str, Any]) -> MCPResult:
            # Generate unique call ID
            self._call_counter += 1
            call_id = f"{tool_name}_{self._call_counter}_{id(args)}"

            # Create Future for this call
            future: asyncio.Future[MCPResult] = asyncio.Future()
            self.pending_calls[call_id] = {"future": future, "name": tool_name, "args": args}

            try:
                # BLOCK here waiting for result (with timeout)
                result = await asyncio.wait_for(future, timeout=self.timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout - return error
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool execution timeout after {self.timeout} seconds",
                        }
                    ],
                    "is_error": True,
                }
            finally:
                # Clean up pending call
                self.pending_calls.pop(call_id, None)

        return handler

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls.

        Returns:
            True if there are pending calls, False otherwise
        """
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        """Get list of pending tool calls.

        Returns:
            List of PendingCall dicts with id, name, and args
        """
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Resolve a pending tool call with its result.

        This unblocks the handler waiting on the Future, causing it to return
        the result to Claude Code CLI.

        Args:
            call_id: ID of the pending call (from get_pending_calls)
            result: Tool execution result in MCP format
        """
        if call_id in self.pending_calls:
            future = self.pending_calls[call_id]["future"]
            if not future.done():
                future.set_result(result)
