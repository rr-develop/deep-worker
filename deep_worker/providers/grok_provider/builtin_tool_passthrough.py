"""grok built-in passthrough — a thin binding of the shared ACP implementation.

The mechanism (record grok's local built-ins into LangGraph history from the
ACP ``tool_call`` / ``tool_call_update`` stream) is entirely provider-neutral
and lives in ``..acp.builtin_tool_passthrough``. Grok's fork of it differed
only in prose and in the marker key names, so only the grok-specific DATA — the
built-in name set and the generic wildcard name — remains here.

This module stays as a named entry point because production code imports it
directly by path:

  * ``deep_worker_cli/process/agent_process.py``
  * ``deep_worker_cli/task_manager/task_manager.py``

both of which call the zero-argument ``build_passthrough_tools()`` below.
"""
from __future__ import annotations

from ..acp.builtin_tool_passthrough import (
    BUILTIN_MARKER,
    META_KEY,
    build_passthrough_tools as _build_passthrough_tools,
    is_builtin_passthrough_tool,
    resolve_builtin_future,
)
from .acp_types import KNOWN_GROK_BUILTINS
from .profile import GENERIC_BUILTIN_NAME

# Re-exported under the fork's historical names so existing imports keep
# working; the values are the shared layer's.
GROK_BUILTIN_MARKER = BUILTIN_MARKER
GROK_META_KEY = META_KEY

__all__ = [
    "GROK_BUILTIN_MARKER",
    "GROK_META_KEY",
    "build_passthrough_tools",
    "is_builtin_passthrough_tool",
    "resolve_builtin_future",
]

# The fork also exported ``is_grok_builtin_passthrough_tool`` and
# ``get_generic_passthrough_tool``. Both were consumed only by the fork's own
# chat model (verified: no references anywhere else in the tree), so they are
# dropped rather than aliased.


def build_passthrough_tools() -> tuple:
    """One passthrough tool per known grok built-in, plus the generic wildcard.

    Kept zero-argument: the two production call sites above take no arguments,
    and the grok-specific data they would otherwise have to pass lives here.
    ``CallMcpTool`` is deliberately excluded — it dispatches our proxy MCP tools
    (channel A), never a local built-in.
    """
    return _build_passthrough_tools(KNOWN_GROK_BUILTINS, GENERIC_BUILTIN_NAME)
