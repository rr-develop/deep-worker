# pyright: reportReturnType=false
"""WebSearch delegation tool - universal web search via Claude Code SDK.

This module provides a LangChain tool that enables ANY AI provider
(Gemini, OpenAI, DeepSeek, Claude, etc.) to perform web searches by
delegating to Claude Code's built-in WebSearch capability.

The tool creates an isolated Claude SDK session, enforces query parameters
via PreToolUse hooks, and extracts search results from the response.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (  # noqa: E402
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    ToolResultBlock,
    TextBlock,
)
from claude_agent_sdk.types import HookMatcher  # noqa: E402

from claude_code_provider.web_search.web_search_hooks import create_enforcement_hook  # noqa: E402


class WebSearchInput(BaseModel):
    """Input schema for web search delegation.

    This schema defines the parameters that can be passed to the WebSearch tool.
    All parameters are optional except for the query itself.

    Attributes:
        query: The search query string (required)
        allowed_domains: List of domains to restrict results to (optional)
        blocked_domains: List of domains to exclude from results (optional)
    """

    query: str = Field(description="The search query to execute")
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Only include results from these domains (e.g., ['python.org', 'github.com'])"
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Exclude results from these domains (e.g., ['spam.com'])"
    )


class WebSearchDelegationTool(BaseTool):
    """Universal web search tool that delegates to Claude Code SDK.

    This tool allows ANY AI provider (Gemini, OpenAI, DeepSeek, Claude) to
    perform web searches by delegating to Claude Code's built-in WebSearch tool.

    Architecture:
    1. Creates isolated ClaudeSDKClient session with unique session_id
    2. Configures PreToolUse hook to enforce original query parameters
    3. Sends search prompt to Claude
    4. Intercepts WebSearch results from Claude's response messages
    5. Parses and formats results as JSON
    6. Returns to calling provider

    Key Features:
    - Works with any LangChain-compatible LLM provider
    - Parameter enforcement prevents Claude from modifying queries
    - Session isolation prevents cross-contamination
    - Automatic timeout and error handling
    - Verbose logging for debugging

    Example:
        >>> from claude_code_provider.web_search_delegation import WebSearchDelegationTool
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>>
        >>> # Create tool
        >>> tool = WebSearchDelegationTool(verbose=True)
        >>>
        >>> # Use with Gemini
        >>> model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        >>> model_with_search = model.bind_tools([tool])
        >>>
        >>> # Perform search
        >>> result = await tool.arun(
        ...     query="Python 3.13 features",
        ...     allowed_domains=["python.org"]
        ... )
        >>> print(result)  # JSON with search results

    Attributes:
        name: Tool name for LangChain registration ("web_search")
        description: Human-readable description of tool capabilities
        args_schema: Pydantic schema for input validation
        timeout: Timeout in seconds for SDK session (default: 30.0)
        verbose: Enable debug logging (default: False)
        model: Claude model to use for search (default: "claude-sonnet-4-5-20250929")
    """

    name: str = "web_search"
    description: str = """Search the web for current information and facts.

Use this tool when you need:
- Recent events or breaking news
- Current facts not in your training data
- Real-time information (prices, weather, etc.)
- Web resources and documentation
- Latest versions or releases

Input:
- query: Search query string (required)
- allowed_domains: Restrict to specific domains (optional)
- blocked_domains: Exclude specific domains (optional)

Output:
JSON array of search results with:
- title: Page title
- url: Page URL
- snippet: Brief description

Example:
{
  "query": "Python 3.13 new features",
  "allowed_domains": ["python.org"]
}
"""
    args_schema: type[BaseModel] | dict[str, Any] = WebSearchInput  # type: ignore[assignment]

    # Configuration parameters
    timeout: float = 30.0  # SDK session timeout in seconds
    verbose: bool = False  # Debug logging
    model: str = "claude-sonnet-4-5-20250929"  # Claude model to use

    async def _arun(  # type: ignore[reportReturnType]
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        """Execute web search via Claude Code SDK delegation.

        This is the main entry point for the tool. It orchestrates the entire
        search process: creating the SDK client, sending the query, extracting
        results, and formatting the response.

        Args:
            query: Search query string (e.g., "Python tutorials")
            allowed_domains: Optional list of domains to restrict results
                           (e.g., ["python.org", "stackoverflow.com"])
            blocked_domains: Optional list of domains to exclude from results
                           (e.g., ["spam.com", "ads.example.com"])

        Returns:
            JSON string with search results:
            ```json
            [
                {
                    "title": "Result title",
                    "url": "https://example.com/page",
                    "snippet": "Brief description of the page..."
                },
                ...
            ]
            ```

            On error, returns JSON with error field:
            ```json
            {
                "error": "Error message",
                "query": "original query"
            }
            ```

        Raises:
            Does not raise exceptions - all errors are caught and returned
            as JSON error objects.

        Implementation Details:
            1. Builds original_params dict from input arguments
            2. Creates enforcement hook with these parameters
            3. Configures ClaudeAgentOptions with hook and restrictions
            4. Creates isolated SDK session with unique session_id
            5. Sends search prompt to Claude
            6. Extracts results from response messages
            7. Formats and returns as JSON
        """
        # 1. Prepare original parameters for hook enforcement
        original_params: dict[str, Any] = {
            "query": query,
        }
        if allowed_domains:
            original_params["allowed_domains"] = allowed_domains
        if blocked_domains:
            original_params["blocked_domains"] = blocked_domains

        # 2. Create enforcement hook with captured parameters
        enforcement_hook = create_enforcement_hook(original_params)

        # 3. Generate unique session ID for isolation
        session_id = f"websearch-delegation-{uuid.uuid4().hex[:8]}"

        # 4. Configure Claude SDK client options
        options = ClaudeAgentOptions(
            model=self.model,
            permission_mode="bypassPermissions",  # No prompts for WebSearch
            allowed_tools=["WebSearch"],  # ONLY allow WebSearch tool
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="WebSearch",  # Only intercept WebSearch calls
                        hooks=[enforcement_hook]
                    )
                ]
            },
        )

        # 5. Execute search in isolated SDK session
        try:
            async with ClaudeSDKClient(options) as client:
                # Build search prompt
                prompt = self._build_search_prompt(query, allowed_domains, blocked_domains)

                if self.verbose:
                    print(f"[WebSearchDelegation] Session: {session_id}")
                    print(f"[WebSearchDelegation] Original params: {original_params}")
                    print(f"[WebSearchDelegation] Prompt: {prompt}")

                # Send search request
                await client.query(prompt, session_id=session_id)

                # 6. Extract results from messages
                results = await asyncio.wait_for(
                    self._extract_results(client),
                    timeout=self.timeout
                )

                # 7. Format and return
                return json.dumps(results, indent=2, ensure_ascii=False)

        except asyncio.TimeoutError:
            error_result = {
                "error": f"Web search timeout after {self.timeout}s",
                "query": query
            }
            if self.verbose:
                print(f"[WebSearchDelegation] ERROR: {error_result}")
            return json.dumps(error_result, indent=2)

        except Exception as e:
            error_result = {
                "error": f"Web search failed: {str(e)}",
                "query": query
            }
            if self.verbose:
                print(f"[WebSearchDelegation] ERROR: {error_result}")
                import traceback
                traceback.print_exc()
            return json.dumps(error_result, indent=2)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous version not supported.

        WebSearchDelegationTool requires async/await due to the async nature
        of ClaudeSDKClient. Use arun() or ainvoke() instead.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "WebSearchDelegationTool only supports async operations. "
            "Use arun() or ainvoke() instead of run() or invoke()."
        )

    def _build_search_prompt(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        """Build prompt for Claude to perform web search.

        We construct a very specific prompt to ensure Claude calls WebSearch
        with the parameters we want. The prompt explicitly instructs Claude
        to use the WebSearch tool.

        Args:
            query: Search query string
            allowed_domains: Optional domain whitelist
            blocked_domains: Optional domain blacklist

        Returns:
            Formatted prompt string that will trigger WebSearch

        Example:
            >>> tool = WebSearchDelegationTool()
            >>> prompt = tool._build_search_prompt(
            ...     "Python tutorials",
            ...     allowed_domains=["python.org"],
            ...     blocked_domains=None
            ... )
            >>> print(prompt)
            Search the web for: Python tutorials

            Only include results from these domains: python.org

            Use the WebSearch tool to find this information.
        """
        prompt = f"Search the web for: {query}\n\n"

        if allowed_domains:
            domains_str = ", ".join(allowed_domains)
            prompt += f"Only include results from these domains: {domains_str}\n"

        if blocked_domains:
            domains_str = ", ".join(blocked_domains)
            prompt += f"Exclude results from these domains: {domains_str}\n"

        prompt += "\nUse the WebSearch tool to find this information."

        return prompt

    async def _extract_results(
        self,
        client: ClaudeSDKClient
    ) -> list[dict[str, str]]:
        """Extract search results from Claude SDK messages.

        This method listens to the message stream from Claude and extracts
        WebSearch tool results. It handles various message types:
        - AssistantMessage with ToolResultBlock (contains search results)
        - AssistantMessage with TextBlock (Claude's commentary)
        - ResultMessage (end of response)

        Args:
            client: Connected ClaudeSDKClient instance

        Returns:
            List of search results, each with title, url, snippet:
            ```python
            [
                {
                    "title": "Python Tutorial",
                    "url": "https://python.org/tutorial",
                    "snippet": "Learn Python basics..."
                }
            ]
            ```

        Implementation Details:
            We collect ALL messages until ResultMessage, then parse the
            ToolResultBlock content to extract structured search results.
            The parsing handles various result formats that WebSearch might return.
        """
        search_results: list[dict[str, str]] = []
        websearch_tool_result: str | list[dict[str, Any]] | None = None

        async for message in client.receive_response():
            # =================================================================
            # CRITICAL: ToolResultBlock appears in UserMessage, NOT AssistantMessage!
            # =================================================================
            # Claude SDK returns WebSearch results in this message flow:
            # 1. AssistantMessage with ToolUseBlock (Claude calls WebSearch)
            # 2. UserMessage with ToolResultBlock (WebSearch results) <-- HERE!
            # 3. AssistantMessage with TextBlock (Claude formats results)
            #
            # Previous code only checked AssistantMessage, causing empty results.
            # This regression occurred when Claude SDK changed message format.
            # =================================================================
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    # Claude's text response (may contain formatted results)
                    if isinstance(block, TextBlock):
                        if self.verbose:
                            text_preview = block.text[:200] + "..." if len(block.text) > 200 else block.text
                            print(f"[WebSearchDelegation] Claude text: {text_preview}")

            elif hasattr(message, "content") and isinstance(message.content, list):
                # Check ANY message type that has content blocks
                # This includes UserMessage which contains ToolResultBlock
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        # This is the result from WebSearch tool
                        websearch_tool_result = block.content

                        if self.verbose:
                            print(f"[WebSearchDelegation] Found tool result block in {type(message).__name__}")
                            print(f"[WebSearchDelegation] Tool use ID: {block.tool_use_id}")
                            print(f"[WebSearchDelegation] Content type: {type(block.content)}")
                            if isinstance(block.content, str):
                                preview = block.content[:200] + "..." if len(block.content) > 200 else block.content
                                print(f"[WebSearchDelegation] Content preview: {preview}")

            elif isinstance(message, ResultMessage):
                # End of response
                if self.verbose:
                    print(f"[WebSearchDelegation] Search completed")
                    print(f"[WebSearchDelegation] Duration: {message.duration_ms}ms")
                    if message.total_cost_usd:
                        print(f"[WebSearchDelegation] Cost: ${message.total_cost_usd:.4f}")
                break

        # Parse results from tool output
        if websearch_tool_result:
            search_results = self._parse_websearch_result(websearch_tool_result)
            if self.verbose:
                print(f"[WebSearchDelegation] Parsed {len(search_results)} results")
        else:
            if self.verbose:
                print("[WebSearchDelegation] WARNING: No tool result found")

        return search_results

    def _parse_websearch_result(
        self,
        tool_result: str | list[dict[str, Any]] | None
    ) -> list[dict[str, str]]:
        """Parse WebSearch tool result into structured format.

        WebSearch returns results in various formats depending on the search.
        This method normalizes all formats to a consistent structure:
        ```python
        [
            {"title": "...", "url": "...", "snippet": "..."},
            ...
        ]
        ```

        Args:
            tool_result: Raw result from WebSearch tool (string or list)

        Returns:
            Normalized list of search results

        Handles:
        - List of dicts with "type": "text" blocks
        - JSON string containing results array
        - Plain text (wrapped as single result)
        - None/empty (returns empty list)
        """
        if not tool_result:
            return []

        # Case 1: Already structured as list
        if isinstance(tool_result, list):
            normalized: list[dict[str, str]] = []
            for item in tool_result:
                if isinstance(item, dict):
                    # Check if it's a text block from MCP
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        # Try to parse text as JSON
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, list):
                                normalized.extend(parsed)
                            elif isinstance(parsed, dict):
                                normalized.append(parsed)
                        except json.JSONDecodeError:
                            # Plain text - wrap it
                            normalized.append({
                                "title": "Search Result",
                                "url": "",
                                "snippet": text
                            })
                    # Already a result dict
                    elif "title" in item or "url" in item or "snippet" in item:
                        normalized.append(item)
            return normalized

        # Case 2: String - try to parse as JSON
        if isinstance(tool_result, str):
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

            # Fallback: return as single text result
            return [{
                "title": "Search Result",
                "url": "",
                "snippet": tool_result
            }]

        # Unknown format
        return []
