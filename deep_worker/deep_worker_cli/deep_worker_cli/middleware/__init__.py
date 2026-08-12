"""Agent middlewares shipped with deep_worker_cli.

See also ``deep_worker_cli/local_context.py`` which predates this package
and still lives at the top level for historical reasons.
"""

from deep_worker_cli.middleware.history_access import HistoryAccessMiddleware
from deep_worker_cli.middleware.history_memory import HistoryMemoryMiddleware
from deep_worker_cli.middleware.inter_agent_reply import InterAgentReplyMiddleware
from deep_worker_cli.middleware.safe_tool_call import SafeToolCallMiddleware
from deep_worker_cli.middleware.send_message_completion import (
    TOOL_NAME,
    SendMessageCompletionMiddleware,
)
from deep_worker_cli.middleware.tool_completion import (
    DuplicateTurnEndingToolError,
    ToolCompletionMiddleware,
    TurnEndingTool,
)

__all__ = [
    "TOOL_NAME",
    "DuplicateTurnEndingToolError",
    "HistoryAccessMiddleware",
    "HistoryMemoryMiddleware",
    "InterAgentReplyMiddleware",
    "SafeToolCallMiddleware",
    "SendMessageCompletionMiddleware",
    "ToolCompletionMiddleware",
    "TurnEndingTool",
]
