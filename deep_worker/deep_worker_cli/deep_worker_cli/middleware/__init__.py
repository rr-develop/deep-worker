"""Agent middlewares shipped with deep_worker_cli.

See also ``deep_worker_cli/local_context.py`` which predates this package
and still lives at the top level for historical reasons.
"""

from deep_worker_cli.middleware.send_message_completion import (
    TOOL_NAME,
    SendMessageCompletionMiddleware,
)

__all__ = [
    "TOOL_NAME",
    "SendMessageCompletionMiddleware",
]
