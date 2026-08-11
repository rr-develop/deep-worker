"""Grok provider for deep-worker — sibling of codex_provider.

Public surface re-exports GrokChatModel for ``deep_worker_cli/config.py``.

The provider drives the xAI Grok CLI in its ``grok agent stdio`` mode,
which speaks the Agent Client Protocol (ACP) v1 over JSON-RPC. Authn is
by the user's supergrok subscription cache (``~/.grok/auth.json``); no
separate API key is needed. See docs/DESIGN_GROK_PROVIDER.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_model import GrokChatModel  # noqa: F401


def __getattr__(name: str):
    if name == "GrokChatModel":
        from .chat_model import GrokChatModel

        return GrokChatModel
    raise AttributeError(f"module 'grok_provider' has no attribute {name!r}")


__all__ = ["GrokChatModel"]
__version__ = "0.1.0"
