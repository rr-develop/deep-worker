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
import contextlib
import sys
from collections.abc import Sequence
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
        self._tool_names = [m["name"] for m in tools_metadata]
        self.timeout = timeout
        self._call_counter = 0
        self._pending_waiters: list[
            tuple[asyncio.Future[None], str | None, dict[str, Any] | None]
        ] = []
        self._pending_change_waiters: set[asyncio.Future[None]] = set()
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
            self._wake_matching_pending_waiters(tool_name, args)

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

    @property
    def tool_names(self) -> list[str]:
        """Tool names exposed through this proxy MCP server."""
        return list(self._tool_names)

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls."""
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        """Get list of pending tool calls for LangGraph to execute."""
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    async def wait_for_pending_calls(
        self,
        *,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> list[PendingCall]:
        """Wait until a matching pending call exists, without polling.

        The Claude SDK may deliver ``ToolUseBlock`` before the spawned MCP
        handler task has registered its pending call. This method handles both
        possible orderings: return an existing match immediately, or park on a
        one-shot Future that the handler wakes when it creates a matching call.
        """
        existing = self._matching_pending_calls(tool_name, arguments)
        if existing:
            return existing

        fut = asyncio.get_running_loop().create_future()
        waiter = (fut, tool_name, arguments)
        self._pending_waiters.append(waiter)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return self._matching_pending_calls(tool_name, arguments)
        finally:
            with contextlib.suppress(ValueError):
                self._pending_waiters.remove(waiter)

    async def wait_for_any_pending_call(
        self,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Wait until at least one pending call exists, without polling.

        Returns as soon as any handler registers a pending call (the
        ``_pending_change_waiters`` Future is woken from
        :meth:`_wake_matching_pending_waiters`, which is called from inside
        ``_create_tool_handler`` right after ``self.pending_calls[call_id] =
        ...``). Returns immediately if a pending call already exists.

        Use this from ``ainvoke`` to know when it is safe to surface the
        currently-registered pending calls to LangGraph. Note that Claude
        CLI dispatches MCP control_requests SEQUENTIALLY (the next one is
        only sent after the previous tool_result), so even when multiple
        ``ToolUseBlock`` values arrive in one ``AssistantMessage`` only
        the first handler is actually invoked. ``ainvoke`` must therefore
        return after the first pending call registers, let LangGraph run
        that tool, ``resolve_call`` it, and pick up the next handler on
        the next iteration — NOT try to wait for all of them at once
        (that would deadlock).

        Raises:
            asyncio.TimeoutError: if no handler registers within ``timeout``
                seconds. ``timeout`` only bounds the registration race
                between the ``ToolUseBlock`` arriving in the message stream
                and the corresponding ``control_request`` (and handler
                spawn) being received via the SDK control channel; it does
                NOT bound the actual tool execution (handlers block on
                ``self.timeout`` once registered).
        """
        if self.pending_calls:
            return

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._pending_change_waiters.add(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_change_waiters.discard(fut)

    async def wait_for_matching_pending_calls(
        self,
        *,
        expected_calls: Sequence[tuple[str, dict[str, Any]]],
        timeout: float = 30.0,
    ) -> list[PendingCall]:
        """Wait until all expected pending calls exist, without polling.

        Multiple ``ToolUseBlock`` values can arrive in one
        ``AssistantMessage``. The SDK may register the corresponding MCP
        handler Futures in separate tasks, so ``ainvoke`` must not return
        after the first one. This method waits on pending-call creation events
        and returns only when every expected ``(tool_name, args)`` pair has a
        distinct active pending call.
        """
        if not expected_calls:
            return []

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            existing = self._matching_pending_call_batch(expected_calls)
            if existing:
                return existing

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError

            fut = loop.create_future()
            self._pending_change_waiters.add(fut)
            try:
                await asyncio.wait_for(fut, timeout=remaining)
            finally:
                self._pending_change_waiters.discard(fut)

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Resolve a pending tool call with result from LangGraph ToolNode.

        Unblocks the Future handler, returning result to Claude CLI.
        """
        if call_id in self.pending_calls:
            future = self.pending_calls[call_id]["future"]
            if not future.done():
                future.set_result(result)

    def _matching_pending_calls(
        self,
        tool_name: str | None,
        arguments: dict[str, Any] | None,
    ) -> list[PendingCall]:
        out: list[PendingCall] = []
        for call_id, info in self.pending_calls.items():
            if tool_name is not None and info["name"] != tool_name:
                continue
            if arguments is not None and info["args"] != arguments:
                continue
            out.append({"id": call_id, "name": info["name"], "args": info["args"]})
        return out

    def _matching_pending_call_batch(
        self,
        expected_calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> list[PendingCall]:
        available = list(self.pending_calls.items())
        used_indexes: set[int] = set()
        matched: list[PendingCall] = []

        for expected_name, expected_args in expected_calls:
            found_index: int | None = None
            for idx, (call_id, info) in enumerate(available):
                if idx in used_indexes:
                    continue
                if info["name"] == expected_name and info["args"] == expected_args:
                    found_index = idx
                    matched.append({
                        "id": call_id,
                        "name": info["name"],
                        "args": info["args"],
                    })
                    break
            if found_index is None:
                return []
            used_indexes.add(found_index)

        return matched

    def _wake_matching_pending_waiters(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        for fut, wanted_tool, wanted_args in list(self._pending_waiters):
            if fut.done():
                continue
            if wanted_tool is not None and wanted_tool != tool_name:
                continue
            if wanted_args is not None and wanted_args != arguments:
                continue
            fut.set_result(None)
        for fut in list(self._pending_change_waiters):
            if not fut.done():
                fut.set_result(None)
