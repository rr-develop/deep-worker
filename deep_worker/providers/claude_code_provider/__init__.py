"""Claude Code Provider for deep-worker.

This package provides a LangChain-compatible AI provider that uses Claude Code
(via ClaudeSDKClient) as the backend, enabling deep-worker to leverage Claude Code's
pricing model while maintaining full transparency at the LangGraph level.

Key components:
- ClaudeCodeChatModel: LangChain BaseChatModel wrapper around ClaudeSDKClient
- ProxyMCPServer: In-process MCP server that delegates tool execution to deep-worker
- ClientManager: Manages ClaudeSDKClient instances for different thread_ids
- WebSearchDelegationTool: Universal web search tool that works with any LLM provider

Usage:
    from deep_worker.providers.claude_code_provider import (
        ClaudeCodeChatModel, WebSearchDelegationTool,
    )

    # Create the model
    model = ClaudeCodeChatModel(base_options=...)

    # Bind tools (including web search)
    web_search = WebSearchDelegationTool()
    model = model.bind_tools([web_search, ...other_tools])

    # Use with LangGraph
    response = await model.ainvoke(messages, config={"configurable": {"thread_id": "session-1"}})

LAZY SURFACE (DESIGN_PROVIDERS_FROM_WHEEL §3.2): importing this root package
never imports the heavy submodules (``chat_model``, ``client_manager``,
``proxy_mcp_server``, ``web_search``) and therefore never pulls
``claude_agent_sdk``. Submodules are loaded lazily on first attribute access
via PEP 562 ``__getattr__``. Capability classification happens at the call
sites (``deep_worker_cli.provider_paths.ensure_provider_importable``); a
missing SDK surfaces as a classified error, not an eager import crash.
"""
from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_model import ClaudeCodeChatModel as ClaudeCodeChatModel
    from .client_manager import ClientManager as ClientManager
    from .proxy_mcp_server import ProxyMCPServer as ProxyMCPServer
    from .web_search import WebSearchDelegationTool as WebSearchDelegationTool

# name → submodule (relative to this package). Kept as a plain mapping so the
# lazy ``__getattr__`` below stays import-free and side-effect-free. Keep in
# sync with ``__all__`` below (pyright requires ``__all__`` to be a literal).
_LAZY = {
    "ClaudeCodeChatModel": "chat_model",
    "ClientManager": "client_manager",
    "ProxyMCPServer": "proxy_mcp_server",
    "WebSearchDelegationTool": "web_search",
}

__all__ = [
    "ClaudeCodeChatModel",
    "ClientManager",
    "ProxyMCPServer",
    "WebSearchDelegationTool",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(f".{sub}", __name__)
    value = getattr(module, name)
    # Cache on the package so repeated access does not re-import.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

