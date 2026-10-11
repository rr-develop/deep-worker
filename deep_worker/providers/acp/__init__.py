"""Shared ACP provider layer — vendor-agnostic.

The ACP layer drives any CLI that speaks the Agent Client Protocol v1 over
stdio (Claude Code via ``@agentclientprotocol/claude-agent-acp``, Grok via
``grok agent stdio``, Codex, Gemini CLI, …). The public surface re-exports
:class:`AcpChatModel` for the provider packages that build on it.

Per-CLI differences live in :class:`~.profile.AcpProfile`; the concrete
provider packages (e.g. ``claude_code_provider``) construct a profile
and subclass or configure ``AcpChatModel`` accordingly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_model import AcpChatModel  # noqa: F401


def __getattr__(name: str):
    if name == "AcpChatModel":
        from .chat_model import AcpChatModel

        return AcpChatModel
    raise AttributeError(f"module 'acp' has no attribute {name!r}")


__all__ = ["AcpChatModel"]
__version__ = "0.1.0"
