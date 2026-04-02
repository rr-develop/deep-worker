"""Tests for thread list formatting functions."""

import json
from datetime import datetime

import pytest

from deep_worker_cli.cli import agent_command


class TestFormatThreadsText:
    """Tests for format_threads_text function."""

    def test_format_empty_list(self):
        """Test formatting empty thread list."""
        result = agent_command.format_threads_text([], with_counts=False)
        assert "No threads found" in result

    def test_format_single_thread_without_counts(self):
        """Test formatting single thread without message counts."""
        threads = [
            {
                "thread_id": "my_agent",
                "agent_name": None,
                "updated_at": "2026-03-29T17:30:15.123456+00:00"
            }
        ]
        result = agent_command.format_threads_text(threads, with_counts=False)

        # Should contain thread_id in first column
        assert "my_agent" in result
        # Should show (self) for agent_name when NULL
        assert "(self)" in result
        # Should not contain message count column header
        assert "Messages" not in result
        # Should contain updated timestamp
        assert "Mar 29" in result

    def test_format_single_thread_with_counts(self):
        """Test formatting single thread with message counts."""
        threads = [
            {
                "thread_id": "my_agent",
                "agent_name": None,
                "message_count": 42,
                "updated_at": "2026-03-29T17:30:15.123456+00:00"
            }
        ]
        result = agent_command.format_threads_text(threads, with_counts=True)

        # Should contain message count
        assert "42" in result
        # Should contain "Messages" header
        assert "Messages" in result

    def test_format_multiple_threads(self):
        """Test formatting multiple threads."""
        threads = [
            {
                "thread_id": "agent_a",
                "agent_name": None,
                "updated_at": "2026-03-29T10:00:00+00:00"
            },
            {
                "thread_id": "agent_b",
                "agent_name": "oldagent",
                "updated_at": "2026-03-28T15:30:00+00:00"
            },
            {
                "thread_id": "abc12345",
                "agent_name": "hexagent",
                "updated_at": "2026-03-27T08:45:00+00:00"
            }
        ]
        result = agent_command.format_threads_text(threads, with_counts=False)

        # All thread_ids should be present
        assert "agent_a" in result
        assert "agent_b" in result
        assert "abc12345" in result

        # Agent names
        assert "(self)" in result  # For agent_a (NULL agent_name)
        assert "oldagent" in result
        assert "hexagent" in result

        # Should be multiple lines (one header + separator + 3 threads)
        lines = result.strip().split("\n")
        assert len(lines) >= 5  # Header + separator + 3 threads

    def test_format_thread_with_long_id(self):
        """Test formatting thread with very long ID."""
        threads = [
            {
                "thread_id": "very_long_thread_id_name",
                "agent_name": None,
                "updated_at": "2026-03-29T10:00:00+00:00"
            }
        ]
        result = agent_command.format_threads_text(threads, with_counts=False)
        assert "very_long_thread_id_name" in result

    def test_format_preserves_alphabetical_order(self):
        """Test that formatting preserves thread order."""
        threads = [
            {"thread_id": "aaa", "agent_name": None, "updated_at": "2026-03-29T10:00:00+00:00"},
            {"thread_id": "bbb", "agent_name": None, "updated_at": "2026-03-29T10:00:00+00:00"},
            {"thread_id": "ccc", "agent_name": None, "updated_at": "2026-03-29T10:00:00+00:00"},
        ]
        result = agent_command.format_threads_text(threads, with_counts=False)

        # Check that they appear in order
        aaa_pos = result.find("aaa")
        bbb_pos = result.find("bbb")
        ccc_pos = result.find("ccc")
        assert aaa_pos < bbb_pos < ccc_pos


class TestFormatThreadsJson:
    """Tests for format_threads_json function."""

    def test_format_empty_list(self):
        """Test formatting empty thread list as JSON."""
        result = agent_command.format_threads_json([], sort_by="thread_id", filter_agent=None)
        data = json.loads(result)

        assert data["threads"] == []
        assert data["total"] == 0
        assert data["sort_by"] == "thread_id"
        assert data["filter_agent"] is None

    def test_format_single_thread(self):
        """Test formatting single thread as JSON."""
        threads = [
            {
                "thread_id": "my_agent",
                "agent_name": None,
                "updated_at": "2026-03-29T17:30:15.123456+00:00"
            }
        ]
        result = agent_command.format_threads_json(threads, sort_by="thread_id", filter_agent=None)
        data = json.loads(result)

        assert len(data["threads"]) == 1
        assert data["threads"][0]["thread_id"] == "my_agent"
        assert data["threads"][0]["agent_name"] is None
        assert data["threads"][0]["updated_at"] == "2026-03-29T17:30:15.123456+00:00"
        assert data["total"] == 1

    def test_format_with_message_counts(self):
        """Test formatting threads with message counts."""
        threads = [
            {
                "thread_id": "my_agent",
                "agent_name": None,
                "message_count": 42,
                "updated_at": "2026-03-29T17:30:15.123456+00:00"
            }
        ]
        result = agent_command.format_threads_json(threads, sort_by="thread_id", filter_agent=None)
        data = json.loads(result)

        assert data["threads"][0]["message_count"] == 42

    def test_format_multiple_threads(self):
        """Test formatting multiple threads as JSON."""
        threads = [
            {"thread_id": "agent_a", "agent_name": None, "updated_at": "2026-03-29T10:00:00+00:00"},
            {"thread_id": "agent_b", "agent_name": "oldagent", "updated_at": "2026-03-28T15:30:00+00:00"},
        ]
        result = agent_command.format_threads_json(threads, sort_by="updated", filter_agent="oldagent")
        data = json.loads(result)

        assert len(data["threads"]) == 2
        assert data["total"] == 2
        assert data["sort_by"] == "updated"
        assert data["filter_agent"] == "oldagent"

    def test_json_is_valid(self):
        """Test that output is valid JSON."""
        threads = [
            {"thread_id": "test", "agent_name": "agent", "updated_at": "2026-03-29T10:00:00+00:00"}
        ]
        result = agent_command.format_threads_json(threads, sort_by="thread_id", filter_agent=None)

        # Should not raise
        json.loads(result)

    def test_json_pretty_printed(self):
        """Test that JSON output is pretty-printed."""
        threads = [
            {"thread_id": "test", "agent_name": "agent", "updated_at": "2026-03-29T10:00:00+00:00"}
        ]
        result = agent_command.format_threads_json(threads, sort_by="thread_id", filter_agent=None)

        # Pretty-printed JSON should have newlines and indentation
        assert "\n" in result
        assert "  " in result  # Indentation
