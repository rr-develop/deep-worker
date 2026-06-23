"""Unit tests for deep_worker_cli.middleware.history_memory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.middleware.history_memory import (
    COGNITIVE_HINT,
    HistoryMemoryMiddleware,
)


class TestHistoryMemoryMiddleware:
    def test_disabled_by_default(self) -> None:
        """Middleware should register no tools when disabled."""
        mw = HistoryMemoryMiddleware(assistant_id="test_agent", enabled=False)
        assert mw.tools == []
        assert mw._cognitive_hint is None

    def test_no_tools_when_no_index(self) -> None:
        """Middleware should register no tools when no index exists."""
        with patch(
            "deep_worker_cli.middleware.history_memory._history_index_exists",
            return_value=False,
        ):
            mw = HistoryMemoryMiddleware(assistant_id="test_agent", enabled=True)
            assert mw.tools == []
            assert mw._cognitive_hint is None

    def test_registers_tool_when_index_exists(self) -> None:
        """Middleware should register history_memory tool when index exists."""
        with patch(
            "deep_worker_cli.middleware.history_memory._history_index_exists",
            return_value=True,
        ):
            mw = HistoryMemoryMiddleware(assistant_id="test_agent", enabled=True)
            assert len(mw.tools) == 1
            assert mw.tools[0].name == "history_memory"
            assert mw._cognitive_hint == COGNITIVE_HINT

    def test_before_agent_sets_summaries_dir(self) -> None:
        """before_agent should set summaries_dir in state when enabled."""
        with patch(
            "deep_worker_cli.middleware.history_memory._history_index_exists",
            return_value=True,
        ):
            mw = HistoryMemoryMiddleware(assistant_id="test_agent", enabled=True)
            state = {}
            result = mw.before_agent(state, None)
            assert result is not None
            assert "summaries_dir" in result
            assert "test_agent" in result["summaries_dir"]

    def test_before_agent_returns_none_when_disabled(self) -> None:
        """before_agent should return None when middleware is disabled."""
        mw = HistoryMemoryMiddleware(assistant_id="test_agent", enabled=False)
        result = mw.before_agent({}, None)
        assert result is None


class TestCognitiveHint:
    def test_hint_contains_critical(self) -> None:
        assert "CRITICAL" in COGNITIVE_HINT
        assert "history_memory" in COGNITIVE_HINT

    def test_hint_contains_examples(self) -> None:
        assert "database schema" in COGNITIVE_HINT
        assert "error message" in COGNITIVE_HINT
