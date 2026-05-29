"""Regression: a usage/session-limit error from the manager agent must
propagate out of the TaskManager, not be swallowed into a silent ``None``.

Incident (observed 2026-05-21): when the Claude account hit its usage cap,
the manager agent's ``ainvoke`` produced a degenerate response and the
provider now raises ``ClaudeUsageLimitError``. ``_evaluate_with_manager``
and ``on_subordinate_response`` both had broad ``except Exception: return
None`` clauses, which turned the limit into "evaluation done/unable" — the
TaskManager loop ended silently, the subordinate's valid answer was lost,
and the agent settled to idle with no error shown to the user.

Contract under test: when the manager agent raises an exception whose class
name is ``ClaudeUsageLimitError``, ``on_subordinate_response`` re-raises it
(so the invoke path can transition the agent to ERROR), while any OTHER
exception is still swallowed to ``None`` (unchanged behavior).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_worker_cli.task_manager.task_manager import TaskManager  # noqa: E402


class ClaudeUsageLimitError(RuntimeError):
    """Stand-in matching the provider's class NAME (the code keys on
    ``type(e).__name__`` to stay provider-agnostic)."""


def _make_task_manager() -> TaskManager:
    cfg = MagicMock()
    cfg.provider = "claude-code"
    cfg.model = "opus"
    cfg.working_dir = Path("/tmp")
    tm = TaskManager(
        agent_name="testagent",
        config=cfg,
        session_manager=MagicMock(),
    )
    # Pretend it has been started and is mid-task awaiting a response.
    tm._running = True
    tm.state = "waiting"
    tm.initial_request = "run the backend tests"
    tm.is_first_evaluation = True
    return tm


@pytest.mark.asyncio
async def test_session_limit_propagates_not_swallowed() -> None:
    tm = _make_task_manager()

    manager = MagicMock()
    manager.ainvoke = AsyncMock(
        side_effect=ClaudeUsageLimitError(
            "You've hit your session limit · resets 5:40pm (Europe/Amsterdam)"
        )
    )
    tm.manager_agent = manager

    with pytest.raises(ClaudeUsageLimitError):
        await tm.on_subordinate_response(
            response="459 passed, 1 skipped, 0 failed",
            send_to_subordinate_callback=None,
        )


@pytest.mark.asyncio
async def test_other_errors_still_swallowed_to_none() -> None:
    """A non-limit manager failure keeps the old resilient behavior: the
    TaskManager loop ends gracefully (returns None) rather than crashing the
    whole turn."""
    tm = _make_task_manager()

    manager = MagicMock()
    manager.ainvoke = AsyncMock(side_effect=ValueError("transient parse glitch"))
    tm.manager_agent = manager

    result = await tm.on_subordinate_response(
        response="some response",
        send_to_subordinate_callback=None,
    )
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
