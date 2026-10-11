"""Tests for the operator-flag plumbing of auto_approve into the TaskManager
(review R3): the manager must inherit the operator's ``-sp`` decision rather
than unconditionally force ``auto_approve=True`` on the ACP provider.

Covers:
- TaskManager(auto_approve=False) → create_model(auto_approve=False) — the
  fail-closed default (deny-all permissions, never approve anything).
- TaskManager(auto_approve=True) → create_model(auto_approve=True) — the
  explicit operator ``-sp`` opt-in.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make provider packages importable from the worktree root.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_worker_cli.task_manager.task_manager import TaskManager  # noqa: E402


def _make_tm(*, auto_approve: bool) -> TaskManager:
    cfg = MagicMock()
    cfg.provider = "claude-code"
    cfg.model = None
    cfg.working_dir = Path("/tmp")
    cfg.web_search_policy = "auto"
    return TaskManager(
        agent_name="t1",
        config=cfg,
        session_manager=MagicMock(),
        auto_approve=auto_approve,
    )


class TestTaskManagerAutoApprovePlumbing:
    def test_default_is_fail_closed(self) -> None:
        tm = _make_tm(auto_approve=False)
        assert tm.auto_approve is False

    def test_operator_flag_forwarded_to_create_model(self) -> None:
        for flag in (False, True):
            tm = _make_tm(auto_approve=flag)
            with patch(
                "deep_worker_cli.config.create_model"
            ) as create_model, patch(
                "deep_worker_cli.agent.create_cli_agent",
                return_value=(MagicMock(), MagicMock()),
            ), patch(
                "deep_worker_cli.tools.resolve_web_search_tool",
                return_value=MagicMock(tool=None),
            ), patch(
                "deep_worker.providers.grok_provider.builtin_tool_passthrough.build_passthrough_tools",
                return_value=[],
            ):
                create_model.return_value = MagicMock()
                asyncio.run(tm._initialize_manager())

            _, kwargs = create_model.call_args
            assert kwargs["auto_approve"] is flag, (
                "TaskManager must forward the operator flag verbatim — never "
                "force auto_approve=True"
            )
