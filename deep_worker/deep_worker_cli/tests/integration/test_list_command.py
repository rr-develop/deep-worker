"""Integration tests for deep-worker --list command."""

import json
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deep_worker_cli import sessions


@pytest.fixture
def temp_db_for_integration(tmp_path):
    """Create a temporary database with test data for integration tests."""
    db_path = tmp_path / "sessions.db"

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

    # Insert test data
    test_data = [
        ("agent_alpha", "user1", "2026-03-29T10:00:00+00:00", 5),
        ("agent_beta", "user2", "2026-03-29T11:00:00+00:00", 10),
        ("agent_gamma", "user3", "2026-03-29T12:00:00+00:00", 3),
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
            "agent_name": agent_name,
            "updated_at": updated_at,
        }
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


class TestListCommandIntegration:
    """Integration tests for --list command."""

    @pytest.mark.asyncio
    async def test_list_basic_text_output(self, temp_db_for_integration):
        """Test basic --list command with text output."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse

        args = argparse.Namespace(
            list_threads=True,
            format="text",
            with_counts=False,
            agent=None,
            sort="thread_id",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=temp_db_for_integration):
            exit_code = await list_threads_command(args)
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_list_json_output(self, temp_db_for_integration):
        """Test --list command with JSON output."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse
        from unittest.mock import patch
        from io import StringIO

        args = argparse.Namespace(
            list_threads=True,
            format="json",
            with_counts=False,
            agent=None,
            sort="thread_id",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=temp_db_for_integration):
            # Capture output
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                exit_code = await list_threads_command(args)
                output = mock_stdout.getvalue()

            assert exit_code == 0
            # Verify JSON is valid
            data = json.loads(output)
            assert "threads" in data
            assert "total" in data
            assert data["total"] == 3

    @pytest.mark.asyncio
    async def test_list_with_counts(self, temp_db_for_integration):
        """Test --list command with message counts."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse
        from unittest.mock import patch
        from io import StringIO

        args = argparse.Namespace(
            list_threads=True,
            format="json",
            with_counts=True,
            agent=None,
            sort="thread_id",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=temp_db_for_integration):
            # Capture output
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                exit_code = await list_threads_command(args)
                output = mock_stdout.getvalue()

            assert exit_code == 0
            data = json.loads(output)
            # Verify message counts are present
            for thread in data["threads"]:
                assert "message_count" in thread

    @pytest.mark.asyncio
    async def test_list_filtered_by_agent(self, temp_db_for_integration):
        """Test --list command with agent filter."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse
        from unittest.mock import patch
        from io import StringIO

        args = argparse.Namespace(
            list_threads=True,
            format="json",
            with_counts=False,
            agent="user1",
            sort="thread_id",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=temp_db_for_integration):
            # Capture output
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                exit_code = await list_threads_command(args)
                output = mock_stdout.getvalue()

            assert exit_code == 0
            data = json.loads(output)
            assert data["total"] == 1
            assert data["threads"][0]["agent_name"] == "user1"

    @pytest.mark.asyncio
    async def test_list_sorted_by_updated(self, temp_db_for_integration):
        """Test --list command sorted by updated timestamp."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse
        from unittest.mock import patch
        from io import StringIO

        args = argparse.Namespace(
            list_threads=True,
            format="json",
            with_counts=False,
            agent=None,
            sort="updated",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=temp_db_for_integration):
            # Capture output
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                exit_code = await list_threads_command(args)
                output = mock_stdout.getvalue()

            assert exit_code == 0
            data = json.loads(output)
            # Most recent should be first (agent_gamma at 12:00:00)
            assert data["threads"][0]["thread_id"] == "agent_gamma"

    @pytest.mark.asyncio
    async def test_list_empty_database(self, tmp_path):
        """Test --list command with empty database."""
        from deep_worker_cli.cli.agent_command import list_threads_command
        import argparse
        from unittest.mock import patch
        from io import StringIO

        db_path = tmp_path / "empty.db"
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

        args = argparse.Namespace(
            list_threads=True,
            format="text",
            with_counts=False,
            agent=None,
            sort="thread_id",
            reverse=False,
        )

        with patch.object(sessions, "get_db_path", return_value=db_path):
            # Capture output
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                exit_code = await list_threads_command(args)
                output = mock_stdout.getvalue()

            assert exit_code == 0
            assert "No threads found" in output
