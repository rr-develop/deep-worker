# pyright: reportPrivateUsage=false, reportUnusedImport=false
"""Tests for session epoch rotation on LangGraph summarization.

When SummarizationMiddleware resets message history, the Claude CLI session
must rotate to a new session_id to avoid "Prompt is too long" errors from
accumulated history on the CLI side.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deep_worker.providers.claude_code_provider.client_manager import ClientManager


class TestClientManagerEpoch:
    """Tests for session epoch tracking in ClientManager."""

    def test_initial_epoch_is_zero(self) -> None:
        """New thread_id should have epoch 0."""
        manager = ClientManager()
        assert manager.get_session_epoch("thread-1") == 0

    def test_effective_session_id_epoch_zero(self) -> None:
        """Epoch 0 should return plain thread_id (backward compat)."""
        manager = ClientManager()
        assert manager.get_effective_session_id("agent-a") == "agent-a"

    @pytest.mark.asyncio
    async def test_rotate_session_increments_epoch(self) -> None:
        """rotate_session should increment epoch."""
        manager = ClientManager()
        await manager.rotate_session("agent-a")
        assert manager.get_session_epoch("agent-a") == 1

    @pytest.mark.asyncio
    async def test_effective_session_id_after_rotation(self) -> None:
        """After rotation, effective session_id includes epoch suffix."""
        manager = ClientManager()
        await manager.rotate_session("agent-a")
        assert manager.get_effective_session_id("agent-a") == "agent-a__epoch_1"

    @pytest.mark.asyncio
    async def test_multiple_rotations(self) -> None:
        """Multiple rotations increment epoch each time."""
        manager = ClientManager()
        await manager.rotate_session("agent-a")
        await manager.rotate_session("agent-a")
        await manager.rotate_session("agent-a")
        assert manager.get_session_epoch("agent-a") == 3
        assert manager.get_effective_session_id("agent-a") == "agent-a__epoch_3"

    @pytest.mark.asyncio
    async def test_rotation_resets_sent_count(self) -> None:
        """rotate_session should reset sent_count to 0."""
        manager = ClientManager()
        manager._sent_count["agent-a"] = 500
        await manager.rotate_session("agent-a")
        assert manager.get_sent_count("agent-a") == 0

    @pytest.mark.asyncio
    async def test_rotation_independent_per_thread(self) -> None:
        """Rotation of one thread should not affect another."""
        manager = ClientManager()
        await manager.rotate_session("thread-a")
        await manager.rotate_session("thread-a")
        assert manager.get_session_epoch("thread-a") == 2
        assert manager.get_session_epoch("thread-b") == 0

    def test_cleanup_removes_epoch(self) -> None:
        """cleanup_client should remove epoch tracking."""
        manager = ClientManager()
        manager._session_epoch["agent-a"] = 3
        manager._sent_count["agent-a"] = 42

        # Mock client for cleanup
        mock_client = Mock()
        mock_client.disconnect = AsyncMock()
        manager.clients["agent-a"] = (mock_client, Mock())

        import asyncio
        asyncio.get_event_loop().run_until_complete(manager.cleanup_client("agent-a"))

        assert "agent-a" not in manager._session_epoch
        assert "agent-a" not in manager._sent_count


class TestChatModelRotationDetection:
    """Tests verifying chat_model.py detects summarization and rotates."""

    def test_ainvoke_uses_effective_session_id(self) -> None:
        """chat_model.py must use get_effective_session_id, not raw thread_id."""
        source = (Path(__file__).parent.parent / "chat_model.py").read_text()
        assert "get_effective_session_id" in source, (
            "chat_model.py MUST use get_effective_session_id() for Claude CLI queries. "
            "Using raw thread_id causes session overflow after summarization."
        )

    def test_ainvoke_calls_rotate_session_on_history_reset(self) -> None:
        """chat_model.py must call rotate_session when sent_count > len(messages)."""
        source = (Path(__file__).parent.parent / "chat_model.py").read_text()
        assert "rotate_session" in source, (
            "chat_model.py MUST call rotate_session() when summarization resets "
            "message history (sent_count > len(messages)). Without this, the Claude "
            "CLI session accumulates history indefinitely and hits 'Prompt is too long'."
        )
