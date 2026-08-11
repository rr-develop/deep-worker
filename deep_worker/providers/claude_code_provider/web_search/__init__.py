"""WebSearch delegation tool for universal web search via Claude Code SDK.

This package provides a LangChain tool that enables ANY AI provider
(Gemini, OpenAI, DeepSeek, Claude, etc.) to perform web searches by
delegating to Claude Code's built-in WebSearch capability.

Key components:
- WebSearchDelegationTool: Main LangChain tool for web search
- web_search_hooks: PreToolUse hooks for parameter enforcement

Usage:
    from . import WebSearchDelegationTool

    # Create tool
    tool = WebSearchDelegationTool(verbose=True)

    # Use with any LLM provider
    result = await tool.ainvoke({
        "query": "Python tutorials",
        "allowed_domains": ["python.org"]
    })

Architecture:
    AI Provider → WebSearchDelegationTool → ClaudeSDKClient (isolated session)
                  ↓ PreToolUse Hook enforces params
                  ↓ Claude WebSearch executes
                  ↓ Results extracted
                  → Formatted JSON returned
"""

from .web_search_delegation import (
    WebSearchDelegationTool,
    WebSearchInput,
)
from .web_search_hooks import (
    create_enforcement_hook,
    create_logging_hook,
    create_validation_hook,
)

__all__ = [
    "WebSearchDelegationTool",
    "WebSearchInput",
    "create_enforcement_hook",
    "create_logging_hook",
    "create_validation_hook",
]

__version__ = "1.0.0"
