"""Codex provider for deep-worker — sibling of claude_code_provider.

Public surface re-exports CodexChatModel for ``deep_worker_cli/config.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_model import CodexChatModel  # noqa: F401


def __getattr__(name: str):
    if name == "CodexChatModel":
        from .chat_model import CodexChatModel
        return CodexChatModel
    raise AttributeError(f"module 'codex_provider' has no attribute {name!r}")


__all__ = ["CodexChatModel"]
__version__ = "0.1.0"
