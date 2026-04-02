"""Tests for list_all_threads functionality."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deep_worker_cli import sessions


class TestListAllThreads:
    """Tests for list_all_threads function."""

    @pytest.fixture
    def temp_db_with_threads(self, tmp_path):
        """Create a temporary database with multiple test threads."""
        db_path = tmp_path / "test_sessions.db"

        # Create tables
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)

        # Insert test data with various thread_ids and agent_names
        test_data = [
            # New format: thread_id = agent name
            ("my_agent", "agent1", "2026-03-29T10:00:00+00:00", 5),
            ("another_agent", "agent2", "2026-03-29T11:00:00+00:00", 10),
            ("test_agent_tm", "tm_agent", "2026-03-29T12:00:00+00:00", 3),
            # Old format: hex thread_ids with agent_name in metadata
            ("abc12345", "oldagent", "2026-03-28T10:00:00+00:00", 20),
            ("def67890", "oldagent", "2026-03-28T11:00:00+00:00", 15),
            # Mixed: thread without agent_name
            ("orphan01", None, "2026-03-27T10:00:00+00:00", 2),
        ]

        serde = JsonPlusSerializer()

        for thread_id, agent_name, updated_at, msg_count in test_data:
            # Create checkpoint blob with messages
            messages = [{"role": "user", "content": f"msg {i}"} for i in range(msg_count)]
            checkpoint_data = {
                "v": 1,
                "id": f"checkpoint_{thread_id}",
                "ts": updated_at,
                "channel_values": {"messages": messages},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }

            checkpoint_tuple = serde.dumps_typed(checkpoint_data)
            checkpoint_type = checkpoint_tuple[0]
            checkpoint_blob = checkpoint_tuple[1]

            # Create metadata blob
            metadata = {
                "updated_at": updated_at,
            }
            if agent_name:
                metadata["agent_name"] = agent_name

            metadata_blob = json.dumps(metadata).encode("utf-8")

            conn.execute(
                """
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)
                VALUES (?, '', ?, ?, ?, ?)
                """,
                (thread_id, f"checkpoint_{thread_id}", checkpoint_type, checkpoint_blob, metadata_blob)
            )

        conn.commit()
        conn.close()

        return db_path

    @pytest.mark.asyncio
    async def test_list_all_threads_basic(self, temp_db_with_threads):
        """Test basic listing of all threads."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads()

            assert len(result) == 6
            # Verify all thread_ids are present
            thread_ids = [t["thread_id"] for t in result]
            assert "my_agent" in thread_ids
            assert "another_agent" in thread_ids
            assert "test_agent_tm" in thread_ids
            assert "abc12345" in thread_ids
            assert "def67890" in thread_ids
            assert "orphan01" in thread_ids

    @pytest.mark.asyncio
    async def test_list_all_threads_sorted_by_thread_id(self, temp_db_with_threads):
        """Test threads are sorted alphabetically by thread_id (default)."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(sort_by="thread_id")

            thread_ids = [t["thread_id"] for t in result]
            assert thread_ids == sorted(thread_ids, key=str.lower)
            # Verify first few in alphabetical order
            assert thread_ids[0] == "abc12345"
            assert thread_ids[1] == "another_agent"

    @pytest.mark.asyncio
    async def test_list_all_threads_sorted_by_agent(self, temp_db_with_threads):
        """Test threads sorted by agent name."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(sort_by="agent")

            # NULL agent_names should appear first, then sorted alphabetically
            agent_names = [t["agent_name"] for t in result]
            # First should be None
            assert agent_names[0] is None
            # Rest should be sorted (ignoring None)
            non_none_agents = [a for a in agent_names if a is not None]
            assert non_none_agents == sorted(non_none_agents, key=lambda x: (x or "").lower())

    @pytest.mark.asyncio
    async def test_list_all_threads_sorted_by_updated(self, temp_db_with_threads):
        """Test threads sorted by updated timestamp."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(sort_by="updated")

            # Most recent first (descending)
            updated_times = [t["updated_at"] for t in result]
            assert updated_times == sorted(updated_times, reverse=True)
            # Most recent should be test_agent_tm (2026-03-29T12:00:00)
            assert result[0]["thread_id"] == "test_agent_tm"

    @pytest.mark.asyncio
    async def test_list_all_threads_reverse_sort(self, temp_db_with_threads):
        """Test reverse sort order."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(sort_by="thread_id", reverse=True)

            thread_ids = [t["thread_id"] for t in result]
            # Should be reverse alphabetical
            assert thread_ids == sorted(thread_ids, key=str.lower, reverse=True)
            # Last alphabetically should be first
            assert thread_ids[0] == "test_agent_tm"

    @pytest.mark.asyncio
    async def test_list_all_threads_filter_by_agent(self, temp_db_with_threads):
        """Test filtering by agent name."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(agent_name="oldagent")

            assert len(result) == 2
            thread_ids = [t["thread_id"] for t in result]
            assert "abc12345" in thread_ids
            assert "def67890" in thread_ids

    @pytest.mark.asyncio
    async def test_list_all_threads_filter_nonexistent_agent(self, temp_db_with_threads):
        """Test filtering by non-existent agent returns empty list."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(agent_name="nonexistent")

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_all_threads_without_counts(self, temp_db_with_threads):
        """Test listing without message counts (fast mode, default)."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(include_message_count=False)

            # Should not have message_count field
            for thread in result:
                assert "message_count" not in thread
                assert "thread_id" in thread
                assert "updated_at" in thread

    @pytest.mark.asyncio
    async def test_list_all_threads_with_counts(self, temp_db_with_threads):
        """Test listing with message counts (slow mode)."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads(include_message_count=True)

            # Should have message_count field
            for thread in result:
                assert "message_count" in thread
                assert isinstance(thread["message_count"], int)
                assert thread["message_count"] >= 0

            # Verify specific counts
            my_agent = next(t for t in result if t["thread_id"] == "my_agent")
            assert my_agent["message_count"] == 5

            another_agent = next(t for t in result if t["thread_id"] == "another_agent")
            assert another_agent["message_count"] == 10

    @pytest.mark.asyncio
    async def test_list_all_threads_empty_database(self, tmp_path):
        """Test listing threads from empty database."""
        db_path = tmp_path / "empty.db"

        # Create empty database with table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        conn.close()

        with patch.object(sessions, "get_db_path", return_value=db_path):
            result = await sessions.list_all_threads()
            assert result == []

    @pytest.mark.asyncio
    async def test_list_all_threads_no_checkpoints_table(self, tmp_path):
        """Test listing threads when checkpoints table doesn't exist."""
        db_path = tmp_path / "no_table.db"

        # Create database without checkpoints table
        conn = sqlite3.connect(str(db_path))
        conn.close()

        with patch.object(sessions, "get_db_path", return_value=db_path):
            result = await sessions.list_all_threads()
            assert result == []

    @pytest.mark.asyncio
    async def test_list_all_threads_return_structure(self, temp_db_with_threads):
        """Test the structure of returned thread dictionaries."""
        with patch.object(sessions, "get_db_path", return_value=temp_db_with_threads):
            result = await sessions.list_all_threads()

            # Verify structure of first thread
            thread = result[0]
            assert "thread_id" in thread
            assert "agent_name" in thread  # Can be None
            assert "updated_at" in thread

            # Verify types
            assert isinstance(thread["thread_id"], str)
            assert thread["agent_name"] is None or isinstance(thread["agent_name"], str)
            assert isinstance(thread["updated_at"], str)

            # Verify updated_at is ISO format
            datetime.fromisoformat(thread["updated_at"])  # Should not raise
