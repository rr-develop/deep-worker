# WebSearch Delegation Tool Design

**Status:** Design Document
**Created:** 2026-03-12
**Purpose:** Universal web search tool that delegates to Claude Code SDK, working with any LLM provider

---

## 1. Overview

### 1.1 Problem Statement

Current situation:
- Claude Code CLI has built-in `WebSearch` tool with access to web search capabilities
- Other AI providers (Gemini, OpenAI, DeepSeek) lack web search functionality
- Need a **universal web search tool** that works with ANY provider by delegating to Claude Code SDK

### 1.2 Solution Concept

Create a LangChain tool that:
1. Accepts web search query from any AI provider
2. Creates a **separate Claude Code SDK session** to perform the search
3. Uses `PreToolUse` hook to **enforce original query parameters** (prevent Claude distortion)
4. Intercepts the search results from Claude's WebSearch tool
5. Returns results to the original AI provider

### 1.3 Key Benefits

- ✅ **Universal:** Works with Gemini, OpenAI, DeepSeek, Claude - any LangChain provider
- ✅ **Transparent:** Delegates to proven Claude Code WebSearch implementation
- ✅ **Controlled:** Hook ensures query integrity (no parameter distortion)
- ✅ **Isolated:** Separate SDK session prevents cross-contamination

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Provider (Gemini/OpenAI/DeepSeek/Claude)                         │
│                                                                      │
│  "Search the web for Python tutorials"                              │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ WebSearchDelegationTool (LangChain BaseTool)                        │
│                                                                      │
│  1. Extract query parameters (query, allowed_domains, etc.)         │
│  2. Create ClaudeSDKClient with PreToolUse hook                     │
│  3. Send prompt: "Search the web for: {query}"                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ClaudeSDKClient (Separate Session)                                  │
│                                                                      │
│  • Session ID: "websearch-delegation-{uuid}"                        │
│  • Permission mode: "bypassPermissions"                             │
│  • Hooks: PreToolUse for WebSearch                                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PreToolUse Hook (Parameter Enforcement)                             │
│                                                                      │
│  IF tool_name == "WebSearch":                                       │
│    - Replace tool_input with ORIGINAL parameters                    │
│    - Log: "Enforcing original query: {original_query}"              │
│    - Return: updatedInput = original_parameters                     │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Claude Code WebSearch Tool (Built-in)                               │
│                                                                      │
│  • Performs actual web search                                       │
│  • Returns search results                                           │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Message Interception (receive_response)                             │
│                                                                      │
│  • Collect ALL messages (UserMessage, AssistantMessage, etc.)       │
│  • Extract ToolResultBlock for WebSearch                            │
│  • Parse search results                                             │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Return Results to Original Provider                                 │
│                                                                      │
│  • Format: JSON with search results                                 │
│  • Structure: List[{title, url, snippet}]                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 WebSearchDelegationTool (Main Class)

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(description="Search query")
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Only include results from these domains"
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Never include results from these domains"
    )

class WebSearchDelegationTool(BaseTool):
    """Universal web search tool that delegates to Claude Code SDK."""

    name: str = "web_search"
    description: str = """Search the web for information.

    Use this tool when you need to find current information, recent events,
    or any data not in your training set.

    Input: A search query string and optional domain filters.
    Output: List of search results with titles, URLs, and snippets.
    """
    args_schema: type[BaseModel] = WebSearchInput

    async def _arun(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        """Execute web search via Claude Code SDK delegation."""
        # Implementation in Section 3
```

#### 2.2.2 PreToolUse Hook (Parameter Enforcement)

```python
async def enforce_original_query_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    original_params: dict[str, Any]  # Captured from tool input
) -> HookJSONOutput:
    """Enforce original query parameters to prevent Claude distortion."""

    if input_data["tool_name"] == "WebSearch":
        # Claude may have modified the query - restore original
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": original_params,  # EXACT original parameters
                "permissionDecision": "allow",
                "permissionDecisionReason": "Enforcing original query parameters"
            }
        }

    return {}
```

#### 2.2.3 Result Extraction

```python
async def extract_websearch_results(
    client: ClaudeSDKClient
) -> list[dict[str, str]]:
    """Extract WebSearch results from Claude SDK messages."""

    results = []

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    if block.tool_use_id.startswith("websearch"):
                        # Parse search results from content
                        results = parse_search_results(block.content)

        elif isinstance(message, ResultMessage):
            # End of response
            break

    return results
```

---

## 3. Detailed Implementation

### 3.1 File Structure

```
claude_code_provider/
├── __init__.py
├── web_search_delegation.py  # NEW: Main tool implementation
├── web_search_hooks.py        # NEW: PreToolUse hook logic
└── tests/
    └── test_web_search_delegation.py  # NEW: Tests
```

### 3.2 Core Implementation

#### File: `claude_code_provider/web_search_delegation.py`

```python
"""WebSearch delegation tool - universal web search via Claude Code SDK."""

import asyncio
import json
import uuid
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    ToolResultBlock,
    TextBlock,
)
from claude_agent_sdk.types import HookMatcher

from claude_code_provider.web_search_hooks import create_enforcement_hook


class WebSearchInput(BaseModel):
    """Input schema for web search delegation."""

    query: str = Field(description="The search query")
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Only include results from these domains"
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Exclude results from these domains"
    )


class WebSearchDelegationTool(BaseTool):
    """Universal web search tool that delegates to Claude Code SDK.

    This tool allows ANY AI provider (Gemini, OpenAI, DeepSeek, Claude) to
    perform web searches by delegating to Claude Code's built-in WebSearch tool.

    Architecture:
    1. Creates isolated ClaudeSDKClient session
    2. Uses PreToolUse hook to enforce original query parameters
    3. Intercepts WebSearch results from Claude's response
    4. Returns formatted results to calling provider

    Example:
        >>> tool = WebSearchDelegationTool()
        >>> result = await tool.arun(
        ...     query="Python tutorials",
        ...     allowed_domains=["python.org", "stackoverflow.com"]
        ... )
        >>> print(result)  # JSON with search results
    """

    name: str = "web_search"
    description: str = """Search the web for current information.

Use this when you need:
- Recent events or news
- Current facts not in your training data
- Real-time information
- Web resources and documentation

Input: Search query and optional domain filters.
Output: List of search results (title, URL, snippet).
"""
    args_schema: type[BaseModel] = WebSearchInput

    # Configuration
    timeout: float = 30.0  # Timeout for SDK session
    verbose: bool = False  # Debug logging

    async def _arun(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        """Execute web search via Claude Code SDK delegation.

        Args:
            query: Search query string
            allowed_domains: Optional list of allowed domains
            blocked_domains: Optional list of blocked domains

        Returns:
            JSON string with search results:
            [
                {
                    "title": "Result title",
                    "url": "https://example.com",
                    "snippet": "Result description..."
                },
                ...
            ]
        """
        # 1. Prepare original parameters (for hook enforcement)
        original_params = {
            "query": query,
        }
        if allowed_domains:
            original_params["allowed_domains"] = allowed_domains
        if blocked_domains:
            original_params["blocked_domains"] = blocked_domains

        # 2. Create enforcement hook with captured parameters
        enforcement_hook = create_enforcement_hook(original_params)

        # 3. Configure Claude SDK client
        session_id = f"websearch-delegation-{uuid.uuid4().hex[:8]}"

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-5-20250929",
            permission_mode="bypassPermissions",  # No prompts for WebSearch
            allowed_tools=["WebSearch"],  # ONLY WebSearch
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="WebSearch",  # Only intercept WebSearch
                        hooks=[enforcement_hook]
                    )
                ]
            },
        )

        # 4. Execute search in isolated session
        try:
            async with ClaudeSDKClient(options) as client:
                # Send search request
                prompt = self._build_search_prompt(query, allowed_domains, blocked_domains)

                if self.verbose:
                    print(f"[WebSearchDelegation] Session: {session_id}")
                    print(f"[WebSearchDelegation] Prompt: {prompt}")

                await client.query(prompt, session_id=session_id)

                # 5. Extract results from messages
                results = await self._extract_results(client)

                # 6. Format and return
                return json.dumps(results, indent=2, ensure_ascii=False)

        except asyncio.TimeoutError:
            return json.dumps({
                "error": f"Web search timeout after {self.timeout}s",
                "query": query
            })
        except Exception as e:
            return json.dumps({
                "error": f"Web search failed: {str(e)}",
                "query": query
            })

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous version not supported."""
        raise NotImplementedError(
            "WebSearchDelegationTool only supports async operations. Use arun()."
        )

    def _build_search_prompt(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        """Build prompt for Claude to perform web search.

        We use a VERY specific prompt to ensure Claude calls WebSearch
        with the exact parameters we want.
        """
        prompt = f"Search the web for: {query}\n\n"

        if allowed_domains:
            prompt += f"Only include results from these domains: {', '.join(allowed_domains)}\n"

        if blocked_domains:
            prompt += f"Exclude results from these domains: {', '.join(blocked_domains)}\n"

        prompt += "\nUse the WebSearch tool to find this information."

        return prompt

    async def _extract_results(
        self,
        client: ClaudeSDKClient
    ) -> list[dict[str, str]]:
        """Extract search results from Claude SDK messages.

        Listens for:
        - AssistantMessage with ToolResultBlock (WebSearch results)
        - AssistantMessage with TextBlock (Claude's summary)
        - ResultMessage (end of response)

        Returns:
            List of search results with title, url, snippet
        """
        search_results: list[dict[str, str]] = []
        websearch_tool_result = None

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    # Check for WebSearch tool result
                    if isinstance(block, ToolResultBlock):
                        # This is the result from WebSearch tool
                        websearch_tool_result = block.content

                        if self.verbose:
                            print(f"[WebSearchDelegation] Raw tool result: {websearch_tool_result}")

                    # Claude's text response (may contain formatted results)
                    elif isinstance(block, TextBlock):
                        if self.verbose:
                            print(f"[WebSearchDelegation] Claude response: {block.text[:200]}...")

            elif isinstance(message, ResultMessage):
                # End of response
                if self.verbose:
                    print(f"[WebSearchDelegation] Search completed in {message.duration_ms}ms")
                break

        # Parse results from tool output
        if websearch_tool_result:
            search_results = self._parse_websearch_result(websearch_tool_result)

        return search_results

    def _parse_websearch_result(
        self,
        tool_result: str | list[dict[str, Any]] | None
    ) -> list[dict[str, str]]:
        """Parse WebSearch tool result into structured format.

        WebSearch returns results in various formats. We normalize to:
        [
            {"title": "...", "url": "...", "snippet": "..."},
            ...
        ]
        """
        if not tool_result:
            return []

        # If already structured
        if isinstance(tool_result, list):
            normalized = []
            for item in tool_result:
                if isinstance(item, dict) and item.get("type") == "text":
                    # Extract from text block
                    text = item.get("text", "")
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            normalized.extend(parsed)
                    except json.JSONDecodeError:
                        # Plain text - wrap it
                        normalized.append({
                            "title": "Search Result",
                            "url": "",
                            "snippet": text
                        })
            return normalized

        # If string, try to parse
        if isinstance(tool_result, str):
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

            # Fallback: return as single result
            return [{
                "title": "Search Result",
                "url": "",
                "snippet": tool_result
            }]

        return []
```

#### File: `claude_code_provider/web_search_hooks.py`

```python
"""PreToolUse hooks for WebSearch parameter enforcement."""

from typing import Any, Callable

from claude_agent_sdk.types import (
    HookContext,
    HookInput,
    HookJSONOutput,
)


def create_enforcement_hook(
    original_params: dict[str, Any]
) -> Callable[[HookInput, str | None, HookContext], HookJSONOutput]:
    """Create PreToolUse hook that enforces original query parameters.

    This hook ensures that Claude Code's WebSearch tool receives EXACTLY
    the parameters we specified, preventing Claude from modifying the query.

    Args:
        original_params: The original parameters to enforce
            {
                "query": "search string",
                "allowed_domains": [...],  # optional
                "blocked_domains": [...],  # optional
            }

    Returns:
        Async hook function for PreToolUse event

    Example:
        >>> hook = create_enforcement_hook({"query": "Python", "allowed_domains": ["python.org"]})
        >>> options = ClaudeAgentOptions(
        ...     hooks={"PreToolUse": [HookMatcher(matcher="WebSearch", hooks=[hook])]}
        ... )
    """

    async def enforcement_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Enforce original WebSearch parameters.

        If Claude tries to call WebSearch with modified parameters,
        we replace them with the original parameters from the tool invocation.
        """
        tool_name = input_data.get("tool_name", "")

        if tool_name == "WebSearch":
            tool_input = input_data.get("tool_input", {})

            # Log what Claude wanted to do
            print(f"[PreToolUse Hook] Claude's parameters: {tool_input}")
            print(f"[PreToolUse Hook] Enforcing original: {original_params}")

            # Return original parameters (override Claude's modifications)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "updatedInput": original_params,  # EXACT original params
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "Enforcing original query parameters"
                }
            }

        # Other tools: no intervention
        return {}

    return enforcement_hook
```

---

## 4. Testing Strategy

### 4.1 Unit Tests

```python
# File: claude_code_provider/tests/test_web_search_delegation.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_code_provider.web_search_delegation import (
    WebSearchDelegationTool,
    WebSearchInput,
)


class TestWebSearchDelegationTool:
    """Unit tests for WebSearchDelegationTool."""

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """Test basic web search without domain filters."""
        tool = WebSearchDelegationTool(verbose=True)

        # Mock ClaudeSDKClient
        with patch("claude_code_provider.web_search_delegation.ClaudeSDKClient") as mock_client:
            # Setup mock response
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            # Mock search results
            from claude_agent_sdk import AssistantMessage, ResultMessage, ToolResultBlock

            mock_result = ToolResultBlock(
                tool_use_id="websearch_1",
                content=json.dumps([
                    {
                        "title": "Python Tutorial",
                        "url": "https://python.org/tutorial",
                        "snippet": "Learn Python programming"
                    }
                ])
            )

            mock_instance.receive_response.return_value = [
                AssistantMessage(content=[mock_result], model="claude-sonnet-4-5-20250929"),
                ResultMessage(
                    subtype="result",
                    duration_ms=1000,
                    duration_api_ms=900,
                    is_error=False,
                    num_turns=1,
                    session_id="test",
                )
            ].__iter__()

            # Execute
            result = await tool.arun(query="Python tutorials")

            # Verify
            assert "Python Tutorial" in result
            assert "python.org" in result
            mock_instance.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_domain_filtering(self):
        """Test web search with domain filters."""
        tool = WebSearchDelegationTool()

        result = await tool.arun(
            query="Python",
            allowed_domains=["python.org"],
            blocked_domains=["spam.com"]
        )

        # Result should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_input_schema(self):
        """Test input schema validation."""
        # Valid input
        valid = WebSearchInput(query="test")
        assert valid.query == "test"
        assert valid.allowed_domains is None

        # With domains
        with_domains = WebSearchInput(
            query="test",
            allowed_domains=["example.com"],
            blocked_domains=["spam.com"]
        )
        assert with_domains.allowed_domains == ["example.com"]
        assert with_domains.blocked_domains == ["spam.com"]
```

### 4.2 Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_websearch_delegation():
    """Integration test with real Claude Code SDK.

    This test requires:
    - Claude CLI installed and authenticated
    - Internet connection
    """
    tool = WebSearchDelegationTool(verbose=True)

    # Perform real search
    result = await tool.arun(
        query="Python 3.12 release date",
        allowed_domains=["python.org"]
    )

    # Parse result
    parsed = json.loads(result)

    # Verify structure
    assert isinstance(parsed, list)
    assert len(parsed) > 0

    # Check first result has required fields
    first = parsed[0]
    assert "title" in first
    assert "url" in first
    assert "snippet" in first

    # Verify domain filtering worked
    assert "python.org" in first["url"]

    print(f"Search returned {len(parsed)} results")
    print(f"First result: {first['title']}")
```

### 4.3 Hook Testing

```python
@pytest.mark.asyncio
async def test_parameter_enforcement_hook():
    """Test that PreToolUse hook enforces original parameters."""
    from claude_code_provider.web_search_hooks import create_enforcement_hook

    # Create hook with original params
    original_params = {
        "query": "original query",
        "allowed_domains": ["original.com"]
    }

    hook = create_enforcement_hook(original_params)

    # Simulate Claude modifying the query
    hook_input = {
        "tool_name": "WebSearch",
        "tool_input": {
            "query": "MODIFIED query",
            "allowed_domains": ["modified.com"]
        },
        "session_id": "test",
        "transcript_path": "/tmp/test",
        "cwd": "/tmp"
    }

    # Call hook
    result = await hook(hook_input, "tool_123", {"signal": None})

    # Verify original params were enforced
    assert result["hookSpecificOutput"]["updatedInput"] == original_params
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
```

---

## 5. Usage Examples

### 5.1 Basic Usage with Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from claude_code_provider.web_search_delegation import WebSearchDelegationTool

# Create Gemini model with web search capability
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tool = WebSearchDelegationTool()

# Bind tool to model
model_with_search = model.bind_tools([tool])

# Use it
response = await model_with_search.ainvoke([
    HumanMessage(content="What are the latest Python 3.13 features? Use web search.")
])

print(response.content)
```

### 5.2 Usage with OpenAI

```python
from langchain_openai import ChatOpenAI
from claude_code_provider.web_search_delegation import WebSearchDelegationTool

model = ChatOpenAI(model="gpt-4o")
tool = WebSearchDelegationTool()

model_with_search = model.bind_tools([tool])

response = await model_with_search.ainvoke([
    HumanMessage(content="Search for recent AI research papers on transformers")
])
```

### 5.3 Direct Tool Usage

```python
from claude_code_provider.web_search_delegation import WebSearchDelegationTool
import json

tool = WebSearchDelegationTool(verbose=True)

# Perform search with domain filtering
result = await tool.arun(
    query="LangChain documentation",
    allowed_domains=["python.langchain.com"],
)

# Parse results
results = json.loads(result)
for item in results:
    print(f"📄 {item['title']}")
    print(f"🔗 {item['url']}")
    print(f"📝 {item['snippet']}\n")
```

---

## 6. Configuration Options

### 6.1 Tool Parameters

```python
tool = WebSearchDelegationTool(
    timeout=30.0,      # SDK session timeout (seconds)
    verbose=True,      # Enable debug logging
)
```

### 6.2 SDK Options

Modify `ClaudeAgentOptions` in `_arun()` for custom behavior:

```python
options = ClaudeAgentOptions(
    model="claude-sonnet-4-5-20250929",  # Model for search
    permission_mode="bypassPermissions",  # No prompts
    allowed_tools=["WebSearch"],          # Only WebSearch
    max_budget_usd=0.10,                  # Cost limit
    hooks={...},                          # PreToolUse enforcement
)
```

---

## 7. Error Handling

### 7.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `TimeoutError` | Search took too long | Increase `timeout` parameter |
| `CLIConnectionError` | Claude CLI not available | Install/authenticate Claude CLI |
| `Permission denied` | WebSearch blocked | Check `permission_mode` setting |
| `Empty results` | No matching results | Broaden search query or domains |

### 7.2 Error Response Format

```json
{
  "error": "Web search timeout after 30s",
  "query": "original search query"
}
```

---

## 8. Security Considerations

### 8.1 Session Isolation

- Each search uses unique `session_id` → No cross-contamination
- Sessions are ephemeral → No persistent history
- `bypassPermissions` mode → Only for WebSearch, not file operations

### 8.2 Parameter Validation

- Hook enforces EXACT original parameters
- Prevents Claude from adding unexpected filters
- Logs all parameter modifications for audit

### 8.3 Cost Control

```python
options = ClaudeAgentOptions(
    max_budget_usd=0.05,  # Limit cost per search
    max_turns=1,          # Single search operation
)
```

---

## 9. Performance Optimization

### 9.1 Caching Strategy

For repeated queries, implement caching:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
async def cached_search(query: str, domains_hash: str) -> str:
    """Cached search results."""
    tool = WebSearchDelegationTool()
    return await tool.arun(query=query)

# Usage
query = "Python tutorials"
domains_hash = hashlib.md5(str(allowed_domains).encode()).hexdigest()
result = await cached_search(query, domains_hash)
```

### 9.2 Timeout Tuning

- Short queries: `timeout=10.0`
- Complex searches: `timeout=60.0`
- Batch operations: Use asyncio.gather with timeout

---

## 10. Future Enhancements

### 10.1 Planned Features

- [ ] Result caching with TTL
- [ ] Retry logic with exponential backoff
- [ ] Streaming results (progressive updates)
- [ ] Multiple search providers (fallback)
- [ ] Custom result parsing strategies

### 10.2 Advanced Hooks

```python
# PostToolUse hook for result validation
async def validate_results_hook(input_data, tool_use_id, context):
    """Validate search results quality."""
    tool_response = input_data.get("tool_response", "")

    # Check for minimum result count
    results = parse_results(tool_response)
    if len(results) < 3:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "Warning: Few results found. Consider broadening query."
            }
        }

    return {}
```

---

## 11. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `web_search_delegation.py` with `WebSearchDelegationTool`
- [ ] Create `web_search_hooks.py` with parameter enforcement
- [ ] Implement `_extract_results()` message parsing
- [ ] Add error handling and timeout logic

### Phase 2: Testing
- [ ] Write unit tests for tool logic
- [ ] Write tests for hook enforcement
- [ ] Write integration tests with real SDK
- [ ] Test with all providers (Gemini, OpenAI, DeepSeek)

### Phase 3: Documentation
- [ ] Add docstrings to all classes/methods
- [ ] Create usage examples
- [ ] Document error scenarios
- [ ] Add to project README

### Phase 4: Integration
- [ ] Export tool in `claude_code_provider/__init__.py`
- [ ] Add to provider configuration
- [ ] Create example scripts
- [ ] Update integration tests

---

## 12. Related Documents

- [Claude Code Provider Design](claude_code_provider_design.md)
- [Claude Code Provider Implementation Plan](claude_code_provider_implementation_plan.md)
- [Claude SDK Documentation](https://github.com/anthropics/claude-sdk-python)

---

## 13. Appendix

### A. WebSearch Tool Result Format

Claude Code's WebSearch returns results in this structure:

```json
[
  {
    "title": "Page title",
    "url": "https://example.com/page",
    "snippet": "Brief description of the page content...",
    "publishedDate": "2025-01-15",  // Optional
    "author": "Author name"         // Optional
  }
]
```

### B. Complete Hook Example

```python
from claude_agent_sdk.types import HookInput, HookContext, HookJSONOutput

async def comprehensive_websearch_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    original_params: dict,
) -> HookJSONOutput:
    """Comprehensive WebSearch hook with logging and validation."""

    if input_data["tool_name"] != "WebSearch":
        return {}

    claude_params = input_data["tool_input"]

    # Log differences
    if claude_params != original_params:
        print(f"⚠️ Parameter mismatch detected!")
        print(f"   Original: {original_params}")
        print(f"   Claude's: {claude_params}")

    # Enforce original
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": original_params,
            "permissionDecision": "allow",
            "permissionDecisionReason": f"Enforced original query: {original_params.get('query')}"
        }
    }
```

---

**Document Status:** Ready for Implementation
**Next Steps:** Begin Phase 1 implementation of core tool class
