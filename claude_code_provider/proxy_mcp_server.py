"""ProxyMCPServer - In-process MCP server that delegates tool execution to LangGraph.

⚠️ ARCHITECTURAL INVARIANT: ALL tool calls MUST go through LangGraph ToolNode.
═══════════════════════════════════════════════════════════════════════════════
ProxyMCPServer registers deep-worker tools as MCP tools for Claude Code CLI.
When Claude calls a tool, the handler blocks on asyncio.Future. This causes:

  1. ainvoke() detects pending_calls → returns AIMessage(tool_calls=[...])
  2. LangGraph routes to ToolNode → ToolNode executes the tool
  3. ToolMessage result → next ainvoke() → resolve_call() → Future unblocks
  4. MCP handler returns result to Claude CLI → Claude continues

This ensures tool calls are VISIBLE in LangGraph state (result.messages),
which is REQUIRED for:
  - dwcli TUI broadcasting tool calls to subscribers
  - Checkpointer saving tool call history
  - interrupt_on (HITL approval) before tool execution
  - TaskManager seeing tool activity

DO NOT replace with direct execution in MCP handler!
Direct execution (calling tool.ainvoke inside handler) makes tools INVISIBLE
to LangGraph — dwcli shows nothing, checkpointer saves nothing, HITL breaks.
This was the bug in commits a8cbdcd/01872d7 that broke dwcli.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
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
    """In-process MCP server that proxies tool calls to LangGraph via Futures.

    Flow:
    1. Claude calls tool → handler creates Future and BLOCKS
    2. ainvoke() detects pending calls → returns AIMessage(tool_calls)
    3. LangGraph executes tool via ToolNode
    4. Next ainvoke() receives ToolMessage → resolve_call() → Future unblocks
    5. Handler returns result to Claude CLI

    Attributes:
        tool_handlers: List of decorated MCP tool handlers
        mcp_server_config: MCP server configuration for Claude SDK
        pending_calls: Dict mapping call_id to pending call info with Future
        timeout: Timeout in seconds for tool execution (default: 120.0)
    """

    def __init__(
        self,
        tools_metadata: list[ToolMetadata],
        timeout: float = 120.0,
        working_dir: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize ProxyMCPServer with tool metadata.

        Args:
            tools_metadata: List of tool metadata dicts with name, description, input_schema
            timeout: Timeout in seconds for tool execution (default: 120.0)
            working_dir: Agent working directory (preserved for future use)
            agent_name: Agent thread_id (used for subagent thread isolation)
        """
        self.pending_calls: dict[str, PendingCallInternal] = {}
        self.tool_handlers: list[Any] = []
        self.timeout = timeout
        self._call_counter = 0
        self._working_dir = working_dir
        self._agent_name = agent_name

        # Create Future-based handler for each tool
        for tool_meta in tools_metadata:
            handler = self._create_tool_handler(tool_meta["name"])

            decorated = tool(
                tool_meta["name"], tool_meta["description"], tool_meta["input_schema"]
            )(handler)

            self.tool_handlers.append(decorated)

        # Create MCP server
        self.mcp_server_config: Any = create_sdk_mcp_server(
            name="deepworker", version="1.0.0", tools=self.tool_handlers
        )

    def _create_tool_handler(self, tool_name: str) -> Any:
        """Create async handler that blocks on Future until LangGraph resolves it.

        ⚠️ ARCHITECTURAL RULE: handler MUST block on Future!
        This ensures tool calls go through LangGraph ToolNode, making them
        visible in result.messages for dwcli broadcast, checkpointing, and HITL.

        DO NOT add direct execution here! See module docstring for details.
        """

        async def handler(args: dict[str, Any]) -> MCPResult:
            self._call_counter += 1
            call_id = f"{tool_name}_{self._call_counter}_{id(args)}"

            future: asyncio.Future[MCPResult] = asyncio.Future()
            self.pending_calls[call_id] = {
                "future": future, "name": tool_name, "args": args,
            }

            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
                return result
            except asyncio.TimeoutError:
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
                self.pending_calls.pop(call_id, None)

        return handler

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls."""
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        """Get list of pending tool calls for LangGraph to execute."""
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Resolve a pending tool call with result from LangGraph ToolNode.

        Unblocks the Future handler, returning result to Claude CLI.
        """
        if call_id in self.pending_calls:
            future = self.pending_calls[call_id]["future"]
            if not future.done():
                future.set_result(result)
