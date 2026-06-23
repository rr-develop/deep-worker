"""Tests for DatabaseError handling in _handle_get_last_response and _handle_get_history.

TDD: Verifies that sqlite3.DatabaseError and aiosqlite.DatabaseError
are caught specifically with a clear "Database corrupted" message,
while other exceptions fall through to the generic handler.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess


def _make_agent_process(error_cls: type[Exception]) -> AgentProcess:
    """Create an AgentProcess whose aget_state raises a specific exception."""
    process = AgentProcess.__new__(AgentProcess)
    process.name = "test-agent"

    mock_agent = AsyncMock()
    mock_agent.aget_state = AsyncMock(side_effect=error_cls("database disk image is malformed"))
    process.agent = mock_agent

    return process


def _make_agent_process_for_history(error_cls: type[Exception]) -> AgentProcess:
    """Create an AgentProcess whose aget_state raises for history handler."""
    from deep_worker_cli.process.message_dict_cache import MessageDictCache

    process = AgentProcess.__new__(AgentProcess)
    process.name = "test-agent"

    mock_agent = MagicMock()
    mock_agent.aget_state = AsyncMock(side_effect=error_cls("database disk image is malformed"))
    process.agent = mock_agent
    process._message_dict_cache = MessageDictCache()
    return process


class TestHandleGetLastResponseDatabaseError:
    """DatabaseError in _handle_get_last_response → specific error message."""

    @pytest.mark.asyncio
    async def test_sqlite3_database_error_returns_corruption_message(self):
        """sqlite3.DatabaseError should return 'Database corrupted' error."""
        import sqlite3

        process = _make_agent_process(sqlite3.DatabaseError)
        result = await process._handle_get_last_response()

        assert result["success"] is False
        assert "Database corrupted" in result["error"]

    @pytest.mark.asyncio
    async def test_aiosqlite_database_error_returns_corruption_message(self):
        """aiosqlite.DatabaseError should return 'Database corrupted' error."""
        import aiosqlite

        process = _make_agent_process(aiosqlite.DatabaseError)
        result = await process._handle_get_last_response()

        assert result["success"] is False
        assert "Database corrupted" in result["error"]

    @pytest.mark.asyncio
    async def test_generic_exception_returns_generic_message(self):
        """Non-database exception should use generic error message (regression)."""
        process = _make_agent_process(RuntimeError)
        result = await process._handle_get_last_response()

        assert result["success"] is False
        assert "Database corrupted" not in result["error"]
        assert "Failed to get last response" in result["error"]


class TestHandleGetHistoryDatabaseError:
    """DatabaseError in _handle_get_history → specific error message."""

    @pytest.mark.asyncio
    async def test_sqlite3_database_error_returns_corruption_message(self):
        """sqlite3.DatabaseError in get_history should return 'Database corrupted'."""
        import sqlite3

        process = _make_agent_process_for_history(sqlite3.DatabaseError)
        result = await process._handle_get_history({})

        assert result["success"] is False
        assert "database corrupted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_aiosqlite_database_error_returns_corruption_message(self):
        """aiosqlite.DatabaseError in get_history should return 'Database corrupted'."""
        import aiosqlite

        process = _make_agent_process_for_history(aiosqlite.DatabaseError)
        result = await process._handle_get_history({})

        assert result["success"] is False
        assert "database corrupted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_generic_exception_returns_generic_message(self):
        """Non-database exception in get_history should use generic error (regression)."""
        process = _make_agent_process_for_history(RuntimeError)
        result = await process._handle_get_history({})

        assert result["success"] is False
        assert "Database corrupted" not in result["error"]
        assert "Failed to retrieve history" in result["error"]
