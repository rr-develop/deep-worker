"""PreToolUse hooks for WebSearch parameter enforcement.

This module provides hook factories for enforcing original query parameters
in Claude Code SDK WebSearch delegations. The hooks ensure that Claude doesn't
modify the search query or parameters that were specified by the calling tool.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from claude_agent_sdk.types import (
    HookContext,
    HookInput,
    HookJSONOutput,
)


def create_enforcement_hook(
    original_params: dict[str, Any]
) -> Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]:
    """Create PreToolUse hook that enforces original query parameters.

    This hook ensures that Claude Code's WebSearch tool receives EXACTLY
    the parameters we specified, preventing Claude from modifying the query.

    The hook is critical for maintaining query integrity when delegating
    web searches to Claude Code SDK. Without this hook, Claude might:
    - Rephrase the search query
    - Add or remove domain filters
    - Modify search parameters in unexpected ways

    Args:
        original_params: The original parameters to enforce
            {
                "query": "search string",
                "allowed_domains": [...],  # optional
                "blocked_domains": [...],  # optional
            }

    Returns:
        Async hook function for PreToolUse event that enforces the parameters

    Example:
        >>> hook = create_enforcement_hook({
        ...     "query": "Python tutorials",
        ...     "allowed_domains": ["python.org"]
        ... })
        >>> options = ClaudeAgentOptions(
        ...     hooks={
        ...         "PreToolUse": [
        ...             HookMatcher(matcher="WebSearch", hooks=[hook])
        ...         ]
        ...     }
        ... )
        >>> async with ClaudeSDKClient(options) as client:
        ...     await client.query("Search for Python tutorials")
        ...     # Hook will enforce original parameters

    Implementation Details:
        The returned hook function is a closure that captures `original_params`.
        When Claude attempts to call WebSearch, the hook:
        1. Checks if the tool being called is "WebSearch"
        2. Logs Claude's attempted parameters
        3. Returns the original parameters via `updatedInput`
        4. Allows the tool call with `permissionDecision: "allow"`
    """

    async def enforcement_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Enforce original WebSearch parameters.

        This is the actual hook function that gets called by Claude SDK
        when a tool is about to be invoked.

        If Claude tries to call WebSearch with modified parameters,
        we replace them with the original parameters from the tool invocation.

        Args:
            input_data: Hook input containing tool_name, tool_input, session info
            tool_use_id: Unique ID for this tool use (may be None)
            context: Hook context with signal for cancellation (future feature)

        Returns:
            Hook output with updated parameters if tool is WebSearch,
            empty dict otherwise (no intervention)
        """
        tool_name = input_data.get("tool_name", "")

        if tool_name == "WebSearch":
            tool_input = input_data.get("tool_input", {})

            # Log what Claude wanted to do vs what we're enforcing
            print(f"[PreToolUse Hook] Claude's WebSearch parameters: {tool_input}")
            print(f"[PreToolUse Hook] Enforcing original parameters: {original_params}")

            # Return original parameters (override Claude's modifications)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "updatedInput": original_params,  # EXACT original params
                    "permissionDecision": "allow",
                    "permissionDecisionReason": f"Enforcing original query: {original_params.get('query', 'N/A')}"
                }
            }

        # Other tools: no intervention
        return {}

    return enforcement_hook


def create_logging_hook(
    verbose: bool = True
) -> Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]:
    """Create PreToolUse hook that logs all WebSearch calls without modification.

    This hook is useful for debugging and monitoring what Claude is trying
    to do with WebSearch, without enforcing any parameter changes.

    Args:
        verbose: Whether to print detailed logs (default: True)

    Returns:
        Async hook function that logs WebSearch calls

    Example:
        >>> hook = create_logging_hook(verbose=True)
        >>> options = ClaudeAgentOptions(
        ...     hooks={
        ...         "PreToolUse": [
        ...             HookMatcher(matcher="WebSearch", hooks=[hook])
        ...         ]
        ...     }
        ... )
    """

    async def logging_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Log WebSearch parameters without modification."""

        if input_data.get("tool_name") == "WebSearch" and verbose:
            tool_input = input_data.get("tool_input", {})
            print(f"[WebSearch Log] Tool call detected")
            print(f"[WebSearch Log] Parameters: {tool_input}")
            print(f"[WebSearch Log] Session: {input_data.get('session_id', 'unknown')}")

        # Allow without modification
        return {}

    return logging_hook


def create_validation_hook(
    max_query_length: int = 500,
    allowed_domains_only: list[str] | None = None,
) -> Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]:
    """Create PreToolUse hook that validates WebSearch parameters.

    This hook checks that WebSearch parameters meet certain criteria
    and can deny the tool call if validation fails.

    Args:
        max_query_length: Maximum allowed query length (default: 500 chars)
        allowed_domains_only: If set, only allow searches restricted to these domains

    Returns:
        Async hook function that validates WebSearch parameters

    Example:
        >>> hook = create_validation_hook(
        ...     max_query_length=200,
        ...     allowed_domains_only=["python.org", "github.com"]
        ... )
        >>> # This will deny searches that are too long or unrestricted
    """

    async def validation_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Validate WebSearch parameters and deny if invalid."""

        if input_data.get("tool_name") != "WebSearch":
            return {}

        tool_input = input_data.get("tool_input", {})
        query = tool_input.get("query", "")

        # Validate query length
        if len(query) > max_query_length:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Query too long ({len(query)} > {max_query_length} chars)"
                }
            }

        # Validate domain restrictions
        if allowed_domains_only:
            search_domains = tool_input.get("allowed_domains", [])
            if not search_domains or not all(d in allowed_domains_only for d in search_domains):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Search must be restricted to: {allowed_domains_only}"
                    }
                }

        # Valid - allow
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Parameters validated"
            }
        }

    return validation_hook
