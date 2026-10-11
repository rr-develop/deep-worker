"""claude-code provider for deep-worker — Claude Code over ACP.

Public surface re-exports ClaudeCodeAcpChatModel for
``deep_worker_cli/config.py``.

The provider drives Claude Code through the pinned
``@agentclientprotocol/claude-agent-acp`` adapter in stdio ACP mode (Agent
Client Protocol v1 over JSON-RPC). Authn is Claude Code's own credential
store (``~/.claude.json`` subscription); no separate API key is needed.
This removes the hard dependency on a private Claude Agent SDK fork
that the legacy ``claude-code-sdk`` provider relies on.
See docs/providers/acp/docs/DESIGN_ACP_PROVIDER_LAYER.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_model import ClaudeCodeAcpChatModel  # noqa: F401
    from .profile import AcpAdapterError, build_profile, describe_preflight  # noqa: F401


def __getattr__(name: str):
    if name == "ClaudeCodeAcpChatModel":
        from .chat_model import ClaudeCodeAcpChatModel

        return ClaudeCodeAcpChatModel
    if name == "AcpAdapterError":
        from .profile import AcpAdapterError

        return AcpAdapterError
    if name == "build_profile":
        from .profile import build_profile

        return build_profile
    if name == "describe_preflight":
        from .profile import describe_preflight

        return describe_preflight
    raise AttributeError(
        f"module 'claude_code_provider' has no attribute {name!r}"
    )


__all__ = [
    "ClaudeCodeAcpChatModel",
    "AcpAdapterError",
    "build_profile",
    "describe_preflight",
]
__version__ = "0.1.0"
