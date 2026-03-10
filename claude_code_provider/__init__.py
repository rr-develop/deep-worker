"""Claude Code Provider for deep-worker.

This package provides a LangChain-compatible AI provider that uses Claude Code
(via ClaudeSDKClient) as the backend, enabling deep-worker to leverage Claude Code's
pricing model while maintaining full transparency at the LangGraph level.

Key components:
- ClaudeCodeChatModel: LangChain BaseChatModel wrapper around ClaudeSDKClient
- ProxyMCPServer: In-process MCP server that delegates tool execution to deep-worker
- ClientManager: Manages ClaudeSDKClient instances for different thread_ids

Usage:
    from claude_code_provider import ClaudeCodeChatModel
    from claude_agent_sdk import ClaudeAgentOptions

    # Create the model
    model = ClaudeCodeChatModel(
        base_options=ClaudeAgentOptions(
            model="claude-sonnet-4-5-20250929"
        )
    )

    # Bind tools
    model = model.bind_tools([your_tools])

    # Use with LangGraph
    response = await model.ainvoke(messages, config={"configurable": {"thread_id": "session-1"}})
"""

from claude_code_provider.chat_model import ClaudeCodeChatModel
from claude_code_provider.client_manager import ClientManager
from claude_code_provider.proxy_mcp_server import ProxyMCPServer

__all__ = [
    "ClaudeCodeChatModel",
    "ClientManager",
    "ProxyMCPServer",
]

__version__ = "0.1.0"
