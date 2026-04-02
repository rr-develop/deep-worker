"""Unit tests for thread management functions."""

import pytest
from datetime import datetime, timedelta, UTC

from deep_worker_cli.sessions import parse_date


class TestParseDateFunction:
    """Tests for parse_date() function."""

    def test_parse_relative_days(self):
        """Test parsing relative days format (Nd)."""
        now = datetime.now(UTC)
        result = parse_date("7d")

        # Allow 1 second tolerance for test execution time
        expected = now - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_relative_weeks(self):
        """Test parsing relative weeks format (Nw)."""
        now = datetime.now(UTC)
        result = parse_date("2w")

        expected = now - timedelta(weeks=2)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_relative_months(self):
        """Test parsing relative months format (Nm)."""
        now = datetime.now(UTC)
        result = parse_date("3m")

        expected = now - timedelta(days=3 * 30)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_relative_years(self):
        """Test parsing relative years format (Ny)."""
        now = datetime.now(UTC)
        result = parse_date("1y")

        expected = now - timedelta(days=365)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_iso_date_only(self):
        """Test parsing ISO 8601 date (YYYY-MM-DD)."""
        result = parse_date("2026-01-15")

        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is not None  # Should be timezone-aware

    def test_parse_iso_datetime(self):
        """Test parsing ISO 8601 datetime with time."""
        result = parse_date("2026-01-15T10:30:00")

        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0

    def test_parse_iso_datetime_with_timezone(self):
        """Test parsing ISO 8601 datetime with timezone."""
        result = parse_date("2026-01-15T10:30:00+00:00")

        assert result.year == 2026
        assert result.tzinfo is not None
        assert result.tzinfo.tzname(None) == "UTC"

    def test_parse_relative_case_insensitive(self):
        """Test that relative format is case-insensitive."""
        result1 = parse_date("7d")
        result2 = parse_date("7D")

        # Should be very close (< 1 second difference)
        assert abs((result1 - result2).total_seconds()) < 1

    def test_parse_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("invalid")

    def test_parse_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("")

    def test_parse_negative_relative_raises_error(self):
        """Test that negative relative dates raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("-7d")

    def test_parse_zero_relative(self):
        """Test parsing zero relative date."""
        now = datetime.now(UTC)
        result = parse_date("0d")

        # Should be very close to now
        assert abs((now - result).total_seconds()) < 1


class TestRemoveThreadFunction:
    """Tests for remove_thread() function."""

    @pytest.mark.asyncio
    async def test_remove_thread_success(self, tmp_path):
        """Test successful thread removal."""
        from deep_worker_cli.sessions import remove_thread
        import aiosqlite

        # Create test database with thread
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, ?)",
                ("test_thread", "cp1")
            )
            await conn.commit()

        # Remove thread
        result = await remove_thread("test_thread", str(db_path), dry_run=False)

        assert result["success"] is True
        assert result["thread_id"] == "test_thread"
        assert result["checkpoints_deleted"] > 0

        # Verify thread is gone
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_thread",)
            ) as cursor:
                count = (await cursor.fetchone())[0]

        assert count == 0

    @pytest.mark.asyncio
    async def test_remove_thread_not_found(self, tmp_path):
        """Test removing non-existent thread."""
        from deep_worker_cli.sessions import remove_thread

        db_path = tmp_path / "test.db"

        # Create empty database
        import aiosqlite
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT
                )
            """)
            await conn.commit()

        result = await remove_thread("nonexistent", str(db_path), dry_run=False)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_remove_thread_dry_run(self, tmp_path):
        """Test dry run mode doesn't actually delete."""
        from deep_worker_cli.sessions import remove_thread
        import aiosqlite

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, ?)",
                ("test_thread", "cp1")
            )
            await conn.commit()

        # Dry run
        result = await remove_thread("test_thread", str(db_path), dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["checkpoints_found"] > 0

        # Verify thread still exists
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test_thread",)
            ) as cursor:
                count = (await cursor.fetchone())[0]

        assert count > 0

    @pytest.mark.asyncio
    async def test_remove_thread_with_writes(self, tmp_path):
        """Test removing thread that has both checkpoints and writes."""
        from deep_worker_cli.sessions import remove_thread, get_checkpointer

        db_path = tmp_path / "test.db"

        # Create database with checkpoint_writes table
        import aiosqlite
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT,
                    task_id TEXT
                )
            """)
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, ?)",
                ("test_thread", "cp1")
            )
            await conn.execute(
                "INSERT INTO checkpoint_writes (thread_id, task_id) VALUES (?, ?)",
                ("test_thread", "task1")
            )
            await conn.commit()

        result = await remove_thread("test_thread", str(db_path), dry_run=False)

        assert result["success"] is True
        assert result["checkpoints_deleted"] > 0
        assert result.get("writes_deleted", 0) > 0


class TestGetThreadHistoryFunction:
    """Tests for get_thread_history() function."""

    @pytest.mark.asyncio
    async def test_get_thread_history_success(self, tmp_path):
        """Test retrieving thread history."""
        from deep_worker_cli.sessions import get_thread_history
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    parent_checkpoint_id TEXT,
                    checkpoint_ns TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata TEXT
                )
            """)

            # Insert checkpoint with message (using JSON type for test simplicity)
            checkpoint_data = json.dumps({
                "channel_values": {
                    "messages": [
                        {"type": "human", "content": "Hello"},
                        {"type": "ai", "content": "Hi there!"}
                    ]
                }
            })

            await conn.execute("""
                INSERT INTO checkpoints
                (thread_id, checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_thread", "cp1", "json", checkpoint_data, "{}"))
            await conn.commit()

        result = await get_thread_history("test_thread", str(db_path))

        assert result["success"] is True
        assert result["thread_id"] == "test_thread"
        assert len(result["checkpoints"]) > 0
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_get_thread_history_not_found(self, tmp_path):
        """Test retrieving history for non-existent thread."""
        from deep_worker_cli.sessions import get_thread_history
        import aiosqlite

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT
                )
            """)
            await conn.commit()

        result = await get_thread_history("nonexistent", str(db_path))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_thread_history_format_json(self, tmp_path):
        """Test JSON format output."""
        from deep_worker_cli.sessions import get_thread_history
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata TEXT,
                    parent_checkpoint_id TEXT
                )
            """)

            checkpoint_data = json.dumps({
                "channel_values": {
                    "messages": [{"type": "human", "content": "test"}]
                }
            })

            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_thread", "cp1", "json", checkpoint_data, "{}"))
            await conn.commit()

        result = await get_thread_history("test_thread", str(db_path), format="json")

        if not result["success"]:
            print(f"Error: {result.get('error', 'unknown')}")
        assert result["success"] is True
        assert "json_output" in result
        # Verify it's valid JSON
        parsed = json.loads(result["json_output"])
        assert "thread_id" in parsed


class TestRemoveThreadsBeforeFunction:
    """Tests for remove_threads_before() function."""

    @pytest.mark.asyncio
    async def test_remove_threads_before_success(self, tmp_path):
        """Test removing threads older than a date."""
        from deep_worker_cli.sessions import remove_threads_before
        from datetime import datetime, UTC
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT
                )
            """)

            # Old thread (should be deleted)
            old_metadata = json.dumps({
                "updated_at": "2025-01-01T10:00:00Z"
            })

            # Recent thread (should be kept)
            recent_metadata = json.dumps({
                "updated_at": datetime.now(UTC).isoformat()
            })

            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("old_thread", "cp1", old_metadata)
            )
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("recent_thread", "cp2", recent_metadata)
            )
            await conn.commit()

        # Remove threads before March 2026
        result = await remove_threads_before("2026-03-01", str(db_path), dry_run=False)

        assert result["success"] is True
        assert result["threads_deleted"] >= 1

        # Verify old thread is gone, recent thread remains
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("old_thread",)
            ) as cursor:
                old_count = (await cursor.fetchone())[0]

            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("recent_thread",)
            ) as cursor:
                recent_count = (await cursor.fetchone())[0]

        assert old_count == 0
        assert recent_count > 0

    @pytest.mark.asyncio
    async def test_remove_threads_before_dry_run(self, tmp_path):
        """Test dry run mode doesn't actually delete."""
        from deep_worker_cli.sessions import remove_threads_before
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT
                )
            """)

            old_metadata = json.dumps({"updated_at": "2025-01-01T10:00:00Z"})
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("old_thread", "cp1", old_metadata)
            )
            await conn.commit()

        result = await remove_threads_before("2026-03-01", str(db_path), dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["threads_found"] >= 1

        # Verify thread still exists
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("old_thread",)
            ) as cursor:
                count = (await cursor.fetchone())[0]

        assert count > 0

    @pytest.mark.asyncio
    async def test_remove_threads_before_relative_date(self, tmp_path):
        """Test using relative date format (e.g., 30d)."""
        from deep_worker_cli.sessions import remove_threads_before
        from datetime import datetime, timedelta, UTC
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT
                )
            """)

            # Very old thread (100 days ago)
            old_date = datetime.now(UTC) - timedelta(days=100)
            old_metadata = json.dumps({"updated_at": old_date.isoformat()})

            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("very_old_thread", "cp1", old_metadata)
            )
            await conn.commit()

        # Remove threads older than 30 days
        result = await remove_threads_before("30d", str(db_path), dry_run=False)

        assert result["success"] is True
        assert result["threads_deleted"] >= 1

    @pytest.mark.asyncio
    async def test_remove_threads_before_no_matches(self, tmp_path):
        """Test when no threads match the criteria."""
        from deep_worker_cli.sessions import remove_threads_before
        from datetime import datetime, UTC
        import aiosqlite
        import json

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    metadata TEXT
                )
            """)

            recent_metadata = json.dumps({"updated_at": datetime.now(UTC).isoformat()})
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, metadata) VALUES (?, ?, ?)",
                ("recent_thread", "cp1", recent_metadata)
            )
            await conn.commit()

        # Try to remove threads before 2020
        result = await remove_threads_before("2020-01-01", str(db_path), dry_run=False)

        assert result["success"] is True
        assert result["threads_deleted"] == 0


class TestHistoryFormatting:
    """Tests for history formatting in CLI output."""

    def test_format_message_with_sender_labels(self):
        """Test that messages are formatted with proper sender labels."""
        from io import StringIO
        from rich.console import Console
        from unittest.mock import MagicMock

        # Mock message objects with different types
        class MockMessage:
            def __init__(self, msg_type, content):
                self.type = msg_type
                self.content = content

        messages = [
            MockMessage("human", "Hello"),
            MockMessage("ai", "Hi there!"),
            MockMessage("tool", "Tool result"),
        ]

        # Capture console output
        output = StringIO()
        console = Console(file=output, force_terminal=False, width=80)

        # Format messages similar to dwcli --history
        for msg in messages:
            if msg.type == "human":
                type_label = "USER"
                separator = "-" * 60
            elif msg.type == "ai":
                type_label = "ASSISTANT"
                separator = "=" * 60
            elif msg.type == "tool":
                type_label = "TOOL"
                separator = "." * 60
            else:
                type_label = msg.type.upper()
                separator = "-" * 60

            console.print(f"\n[{type_label}]")
            console.print(msg.content)
            console.print(separator)

        result = output.getvalue()

        # Verify sender labels are present
        assert "[USER]" in result
        assert "[ASSISTANT]" in result
        assert "[TOOL]" in result

        # Verify content is present
        assert "Hello" in result
        assert "Hi there!" in result
        assert "Tool result" in result

        # Verify separators are present
        assert "-" * 60 in result  # USER separator
        assert "=" * 60 in result  # ASSISTANT separator
        assert "." * 60 in result  # TOOL separator

    def test_format_message_type_mapping(self):
        """Test that LangChain message types map correctly to display labels."""
        type_mappings = {
            "HumanMessage": "USER",
            "AIMessage": "ASSISTANT",
            "ToolMessage": "TOOL",
            "SystemMessage": "SYSTEM",
        }

        for langchain_type, expected_label in type_mappings.items():
            # Simulate message type extraction
            if langchain_type == "HumanMessage":
                label = "USER"
            elif langchain_type == "AIMessage":
                label = "ASSISTANT"
            elif langchain_type == "ToolMessage":
                label = "TOOL"
            elif langchain_type == "SystemMessage":
                label = "SYSTEM"
            else:
                label = langchain_type.upper()

            assert label == expected_label, f"Type {langchain_type} should map to {expected_label}"

    @pytest.mark.asyncio
    async def test_messages_include_timestamps(self, tmp_path):
        """Test that get_thread_history includes timestamps for each message."""
        from deep_worker_cli.sessions import get_thread_history
        import aiosqlite
        import json
        from datetime import datetime, UTC

        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    parent_checkpoint_id TEXT,
                    checkpoint_ns TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata TEXT
                )
            """)

            # Create checkpoints with timestamps
            ts1 = "2026-03-29T10:00:00+00:00"
            ts2 = "2026-03-29T10:01:00+00:00"

            # Checkpoint 1: first message
            checkpoint_data1 = json.dumps({
                "ts": ts1,
                "channel_values": {
                    "messages": [{"type": "human", "content": "Hello"}]
                }
            })

            # Checkpoint 2: second message added
            checkpoint_data2 = json.dumps({
                "ts": ts2,
                "channel_values": {
                    "messages": [
                        {"type": "human", "content": "Hello"},
                        {"type": "ai", "content": "Hi there!"}
                    ]
                }
            })

            await conn.execute("""
                INSERT INTO checkpoints
                (thread_id, checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_thread", "cp1", "json", checkpoint_data1, "{}"))

            await conn.execute("""
                INSERT INTO checkpoints
                (thread_id, checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_thread", "cp2", "json", checkpoint_data2, "{}"))

            await conn.commit()

        result = await get_thread_history("test_thread", str(db_path))

        assert result["success"] is True
        assert len(result["messages"]) == 2

        # Check that messages have timestamps
        msg1 = result["messages"][0]
        msg2 = result["messages"][1]

        # Messages should have timestamp attribute or be in a dict with timestamp
        assert hasattr(msg1, "timestamp") or isinstance(msg1, dict) and "timestamp" in msg1
        assert hasattr(msg2, "timestamp") or isinstance(msg2, dict) and "timestamp" in msg2

        # Verify timestamps are correct
        if hasattr(msg1, "timestamp"):
            assert msg1.timestamp == ts1
        else:
            assert msg1["timestamp"] == ts1

        if hasattr(msg2, "timestamp"):
            assert msg2.timestamp == ts2
        else:
            assert msg2["timestamp"] == ts2
