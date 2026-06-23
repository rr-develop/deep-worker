"""Middleware for unconditional history access tool registration.

Registers ``history_range``, ``history_at``, and ``history_slice`` tools for
ALL agents, regardless of the ``--history-memory`` flag. These tools give the
agent direct access to pre-summarization history stored in markdown\nsection files.

See ``docs/DESIGN_HISTORY_ACCESS_TOOLS.md`` for the full design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from typing_extensions import NotRequired

from deep_worker_cli.history_access_tool import history_at, history_range, history_slice


class HistoryAccessState(AgentState):
    """State schema for HistoryAccessMiddleware.

    Attributes:
        history_access_assistant_id: Agent name for resolving summary paths.
        history_access_home: Home directory override for resolving summary paths.
    """
    history_access_assistant_id: str
    history_access_home: NotRequired[str]


class HistoryAccessMiddleware(AgentMiddleware):
    """Unconditionally registers ``history_range``, ``history_at``, ``history_slice``.

    Unlike ``HistoryMemoryMiddleware``, there is no ``enabled`` guard — these
    tools are always available to every agent.
    """

    state_schema = HistoryAccessState

    def __init__(self, assistant_id: str, home: Path | None = None) -> None:
        self.assistant_id = assistant_id
        self._home = home or Path.home()
        self.tools = [history_range, history_at, history_slice]

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, str] | None:
        """Set assistant_id and home in state for the history access tools."""
        return {
            "history_access_assistant_id": self.assistant_id,
            "history_access_home": str(self._home),
        }
