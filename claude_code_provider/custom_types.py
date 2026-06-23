"""Type definitions for claude_code_provider package."""

from typing import Any, TypedDict

from typing_extensions import NotRequired


class ToolMetadata(TypedDict):
    """Metadata for a tool to be exposed through MCP.

    Attributes:
        name: Unique tool name (e.g., "read_file", "execute_command")
        description: Human-readable description for Claude
        input_schema: JSON schema for tool parameters
        long_running: TRI-STATE wait policy (HARD POLICY, 2026-06-16).
            * unset / ``None`` (the DEFAULT for every tool) AND ``True`` →
              the **job-handle** mechanism applies: the handler waits the
              configured ``long_running_wait_seconds`` and, if the tool has
              not finished, parks the live Future as a job and returns a
              non-error ``{status: in_progress, job_id, actions:
              wait_for_job/kill_job}`` so the agent can decide. The work is
              NOT cancelled. NO tool ever blocks forever.
            * ``False`` → an EXPLICIT opt-out of the job-handle mechanism.
              This is allowed ONLY together with an explicit ``timeout``
              (see below); opting out without a timeout is FORBIDDEN and
              raises at construction (it would reintroduce the eternal
              hang). On timeout the handler returns a non-error
              "timed out, still running" notice — NEVER ``is_error=true``,
              the agent just continues / retries.
        timeout: REQUIRED iff ``long_running`` is explicitly ``False``.
            Seconds to wait before returning the timeout notice. Ignored
            for the default/job-handle path. Provider-agnostic; applies to
            existing and future tools, no exceptions. See POLICIES.md §7.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    long_running: NotRequired[bool]
    timeout: NotRequired[float]


class JobHandle(TypedDict):
    """An in-flight long_running tool call parked as a job.

    The Future is NOT cancelled — the real work continues on the
    LangGraph ToolNode side. Claude has been handed an intermediate
    handle and can call ``wait_for_job`` / ``kill_job`` against the
    same ``job_id`` (which equals the original Claude ``tool_use_id``).
    """

    future: Any  # asyncio.Future
    tool_name: str
    args: dict[str, Any]
    agent_name: NotRequired[str | None]


class PendingCall(TypedDict):
    """Information about a pending tool call waiting for execution.

    Attributes:
        id: Unique identifier for this call (used for resolving)
        name: Tool name that was called
        args: Arguments passed to the tool
    """

    id: str
    name: str
    args: dict[str, Any]


class MCPResult(TypedDict):
    """Result from MCP tool execution in Claude's format.

    Attributes:
        content: List of content blocks (text, image, etc.)
        is_error: Optional flag indicating if this is an error result
    """

    content: list[dict[str, Any]]
    is_error: NotRequired[bool]


class PendingCallInternal(TypedDict):
    """Internal structure for tracking pending calls.

    This is used internally by ProxyMCPServer to track futures and metadata.

    Attributes:
        future: asyncio.Future that will be resolved with the result
        name: Tool name
        args: Tool arguments
    """

    future: Any  # asyncio.Future, but we avoid the import here for type checking
    name: str
    args: dict[str, Any]
