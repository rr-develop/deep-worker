"""ClientManager - manages ClaudeSDKClient instances for different thread_ids.

This module provides thread-safe management of Claude SDK client instances, ensuring that
each thread_id (session) gets its own isolated client and ProxyMCPServer.
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any  # type: ignore[reportUnusedImport]

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402

from claude_code_provider.proxy_mcp_server import ProxyMCPServer  # noqa: E402
from claude_code_provider.custom_types import ToolMetadata  # noqa: E402

# All built-in Claude Code tools that should be disabled.
# We want all tools to execute in deep-worker, not in Claude CLI.
#
# ``ToolSearch`` is in this list for a SUBTLER reason than the others:
# it is a Claude Code beta feature (tool_search_tool_bm25_20251119,
# rolled out to opus/sonnet in @anthropic-ai/claude-code 2.1.140) that,
# when active, instructs the CLI to mark every MCP tool with
# ``defer_loading: true``. Models then see only tool *names* in the
# initial system prompt; the full schemas have to be fetched on demand
# via the built-in ``ToolSearch`` tool. That breaks our contract — a
# deep-worker agent has a small, fixed, LangGraph-mediated tool set
# and MUST know all of those schemas at turn 1.
#
# The Claude Code CLI itself documents the opt-out path (visible as a
# string in the binary):
#
#     "Tool search disabled: ToolSearchTool is not available
#      (may have been disallowed via disallowedTools)."
#
# Putting ``"ToolSearch"`` here makes the CLI's ``isToolSearchEnabled``
# short-circuit to ``false``, which sets ``deferLoading=false`` on every
# tool. Schemas come back in the initial prompt. Critically, this
# entry travels through ``ClaudeAgentOptions.disallowed_tools`` ->
# ``subprocess_cli.py: cmd.extend(["--disallowedTools", ...])``, so the
# opt-out is a *local CLI argument of the spawned subprocess* — it does
# NOT affect any other ``claude`` invocation on this host.
#
# If we ever onboard a large set of optional/lazy MCP servers, the
# right answer is to remove ``"ToolSearch"`` from this list AND mark
# the deep-worker MCP server itself with ``disable_deferred_loading:
# true`` so the *base* toolset stays eager while pluggable MCPs go
# lazy. See docs/DESIGN_DEV_FLAG.md and the bin/claude.exe strings
# table for the contract.
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
    "ToolSearch",
    # ─────────────────────────────────────────────────────────────────────
    # "Ready while waiting" built-ins.
    # ─────────────────────────────────────────────────────────────────────
    # These Claude Code built-ins all schedule work that fires *between
    # turns*. In interactive ``claude`` that is exactly the design: the
    # CLI parks the session in IDLE state and re-activates it on the
    # trigger (cron fire time, new monitor output line, remote routine
    # callback, push reply). The official docs spell this out:
    #
    #   "Tasks only fire while Claude Code is running AND IDLE. A
    #   scheduled prompt fires BETWEEN your turns, not while Claude
    #   is mid-response."
    #   — docs.claude.com/.../scheduled-tasks#how-scheduled-tasks-run
    #
    # In deep-worker we have NO such idle-between-turns state. LangGraph
    # observes a final AIMessage and considers the turn complete, while
    # the side effect the model promised never reaches the agent. The
    # symptom is "agent reports ready, but it actually expected to be
    # resumed later".
    #
    # Names from the official tools-reference:
    #   https://docs.claude.com/en/docs/claude-code/tools-reference
    #
    # If/when deep-worker grows its own external-event channel, the
    # right answer is to bridge it through LangGraph state, NOT to
    # re-enable these CLI built-ins. The CLI's "park the session" model
    # is fundamentally incompatible with the LangGraph turn loop.
    "CronCreate",
    "CronDelete",
    "CronList",
    "Monitor",
    "PushNotification",
    "RemoteTrigger",
    # ─────────────────────────────────────────────────────────────────────
    # 2026-05-14 audit additions.
    # ─────────────────────────────────────────────────────────────────────
    # See docs/CLAUDE_CODE_BUILTIN_TOOLS_AUDIT.md for the full per-tool
    # rationale. Three categories:
    #
    # (A) Modern names for stale entries we already block. The CLI
    #     renamed several tools between releases; without the new
    #     names the model can bypass our block by calling the same
    #     functionality under its current name.
    "Agent",       # superseded "Task" as the subagent-spawn tool
    "TaskStop",    # superseded "KillShell"
    "TaskOutput",  # superseded "BashOutput" (then deprecated in 2.0.64)
    "PowerShell",  # parallel to Bash on Windows / opt-in on Linux
    # (B) Task-list family (Claude Code 2.1.20+, 2026-01-27) replacing
    #     TodoWrite. We block TodoWrite so the model plans via
    #     LangGraph state; block these for the same reason.
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    # (C) Agent-teams tools (Claude Code 2.1.32+, 2026-02-05). Gated on
    #     CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 which we never set,
    #     but we defend in depth: SendMessage parks the session in
    #     idle waiting for the teammate's reply, TeamCreate spawns
    #     coordinated peer processes outside LangGraph.
    "SendMessage",
    "TeamCreate",
    "TeamDelete",
]


# Global singleton instance - shared across all ClaudeCodeChatModel instances
# This ensures _sent_count and clients persist even when LangGraph creates
# new model instances during deserialization/pickling
_global_client_manager: "ClientManager | None" = None


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
        # Track sent message count per thread for proper message mirroring
        # This is stored in ClientManager (not ClaudeCodeChatModel) because
        # LangGraph may create new model instances, but ClientManager is stable
        self._sent_count: dict[str, int] = {}
        # ⚠️ CRITICAL: Session epoch for Claude CLI session rotation.
        # When SummarizationMiddleware resets LangGraph message history,
        # we MUST rotate to a new Claude CLI session_id. Otherwise the CLI
        # session accumulates unbounded history and eventually hits
        # "Prompt is too long". DO NOT remove this mechanism.
        self._session_epoch: dict[str, int] = {}

    async def get_or_create_client(
        self,
        thread_id: str,
        tools_metadata: list[ToolMetadata],
        working_dir: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[ClaudeSDKClient, ProxyMCPServer]:
        """Get or create ClaudeSDKClient for a thread_id.

        If a client already exists for this thread_id, returns the cached instance.
        Otherwise, creates a new client with ProxyMCPServer and caches it.

        This method is thread-safe: concurrent calls for the same thread_id will
        wait for the first call to complete and then return the same client.

        Args:
            thread_id: Unique identifier for this thread/session
            tools_metadata: Metadata for tools to expose through ProxyMCPServer
            working_dir: Working directory for the agent
            system_prompt: System prompt from LangGraph middleware (e.g.
                ProjectMemoryMiddleware cognitive hints). Passed to Claude SDK
                session via ClaudeAgentOptions.system_prompt.

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

            # Create ProxyMCPServer for this thread.
            # ⚠️ ProxyMCPServer uses Future-based proxy — NO direct execution.
            # Tools are executed by LangGraph ToolNode, not by MCP handler.
            proxy_mcp = ProxyMCPServer(
                tools_metadata,
                working_dir=working_dir,
                agent_name=thread_id,
            )

            # Build options for this client
            # ⚠️ CRITICAL: Permission delegation architecture
            # ALL tool approval decisions MUST be made by deep-worker's AgentProcess,
            # NOT by Claude SDK's internal permission system. This prevents double-layer
            # prompts and ensures consistent approval flow:
            #   User → deep-worker (ONLY approval layer) → ClaudeSDKClient (NO prompts) → tools
            if self.base_options is not None:
                # Use replace to create new instance with updated fields
                # ALWAYS override permission_mode to ensure bypass, even if user specified otherwise
                options = replace(
                    self.base_options,
                    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
                    disallowed_tools=BUILT_IN_TOOLS,
                    permission_mode="bypassPermissions",  # Delegate all approval to deep-worker
                    # ⚠️ CRITICAL: Forward system prompt from LangGraph middleware.
                    # This carries cognitive hints (e.g. project_memory instructions).
                    # DO NOT remove — agents lose tool awareness without it.
                    system_prompt=system_prompt,
                )
            else:
                # Create default options
                options = ClaudeAgentOptions(
                    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
                    disallowed_tools=BUILT_IN_TOOLS,
                    permission_mode="bypassPermissions",  # Delegate all approval to deep-worker
                    system_prompt=system_prompt,
                )

            # Create and connect client
            client = ClaudeSDKClient(options=options)
            await client.connect()

            # Cache
            self.clients[thread_id] = (client, proxy_mcp)

            return client, proxy_mcp

    def get_sent_count(self, thread_id: str) -> int:
        """Get number of messages sent for a thread.

        Args:
            thread_id: Thread ID to check

        Returns:
            Number of messages sent to Claude for this thread
        """
        return self._sent_count.get(thread_id, 0)

    def update_sent_count(self, thread_id: str, count: int) -> None:
        """Update sent message count for a thread.

        Args:
            thread_id: Thread ID to update
            count: New sent count value
        """
        self._sent_count[thread_id] = count

    def get_session_epoch(self, thread_id: str) -> int:
        """Get current session epoch for a thread.

        Args:
            thread_id: Thread ID to check.

        Returns:
            Current epoch number (0 = original session).
        """
        return self._session_epoch.get(thread_id, 0)

    def get_effective_session_id(self, thread_id: str) -> str:
        """Get the effective session_id for Claude CLI queries.

        Epoch 0 returns the plain thread_id for backward compatibility.
        Epoch N>0 returns ``"{thread_id}__epoch_{N}"``.

        Args:
            thread_id: LangGraph thread ID.

        Returns:
            Session ID string to pass to ``client.query(session_id=...)``.
        """
        epoch = self._session_epoch.get(thread_id, 0)
        if epoch == 0:
            return thread_id
        return f"{thread_id}__epoch_{epoch}"

    def rotate_session(self, thread_id: str) -> str:
        """Rotate to a new Claude CLI session for this thread.

        Called when LangGraph summarization resets the message history
        (detected by ``sent_count > len(messages)`` in ainvoke).
        Increments epoch, resets sent_count, and returns the new session_id.

        The old Claude CLI session is simply abandoned — Claude CLI
        handles its own garbage collection.

        Args:
            thread_id: Thread ID to rotate.

        Returns:
            New effective session_id.
        """
        current = self._session_epoch.get(thread_id, 0)
        self._session_epoch[thread_id] = current + 1
        self._sent_count[thread_id] = 0
        return self.get_effective_session_id(thread_id)

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

        # Remove sent_count and epoch tracking
        self._sent_count.pop(thread_id, None)
        self._session_epoch.pop(thread_id, None)


def get_global_client_manager() -> ClientManager:
    """Get or create global ClientManager singleton.

    This ensures all ClaudeCodeChatModel instances share the same ClientManager,
    which is crucial for maintaining _sent_count across model instance changes
    (e.g., when LangGraph deserializes models).

    Returns:
        Global ClientManager singleton
    """
    global _global_client_manager
    if _global_client_manager is None:
        _global_client_manager = ClientManager()
    return _global_client_manager
