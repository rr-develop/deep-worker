"""Type definitions for claude_code_provider package."""

from typing import Any, TypedDict

from typing_extensions import NotRequired


class ToolMetadata(TypedDict):
    """Metadata for a tool to be exposed through MCP.

    Attributes:
        name: Unique tool name (e.g., "read_file", "execute_command")
        description: Human-readable description for Claude
        input_schema: JSON schema for tool parameters
    """

    name: str
    description: str
    input_schema: dict[str, Any]


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
