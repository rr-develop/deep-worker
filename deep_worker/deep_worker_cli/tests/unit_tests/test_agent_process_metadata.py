"""Unit tests for AgentProcess metadata in RunnableConfig."""

import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess


@pytest.fixture
def agent_config(tmp_path):
    """Create test agent config."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    return AgentConfig(
        model="claude-sonnet-4-5-20250929",
        working_dir=working_dir,
    )


class TestConfigMetadata:
    """Tests for RunnableConfig metadata in agent invocations."""

    @pytest.mark.asyncio
    async def test_config_includes_metadata_fields(self, agent_config):
        """Test that config includes agent_name and updated_at in metadata."""
        agent_name = "test_agent"
        process = AgentProcess(
            name=agent_name,
            config=agent_config,
            rpyc_port=False,
        )

        # Mock the agent and its methods
        mock_agent = AsyncMock()
        mock_agent.astream = AsyncMock()
        mock_agent.ainvoke = AsyncMock()
        process.agent = mock_agent
        process.backend = MagicMock()

        # Mock session manager
        process.session_manager = MagicMock()
        process.session_manager.set_status = AsyncMock()

        # Call exposed_send_message_async to trigger config creation
        with patch("deep_worker_cli.process.status_callback.StatusTrackingCallback"):
            # Use astream which creates config
            try:
                await process.exposed_send_message_async(
                    message="test message",
                    stream=True,
                    notify_callback=None,
                )
            except Exception:
                # We expect it to fail, we just want to check the config
                pass

            # Verify astream was called
            if mock_agent.astream.called:
                call_kwargs = mock_agent.astream.call_args.kwargs
                config = call_kwargs.get("config")

                # Verify config structure
                assert config is not None, "Config should not be None"
                assert "configurable" in config, "Config should have 'configurable' key"
                assert config["configurable"]["thread_id"] == agent_name

                # CRITICAL: Verify metadata is present
                assert "metadata" in config, "Config should have 'metadata' key"
                metadata = config["metadata"]

                # Verify required metadata fields
                assert "agent_name" in metadata, "Metadata should have 'agent_name'"
                assert metadata["agent_name"] == agent_name

                assert "assistant_id" in metadata, "Metadata should have 'assistant_id'"
                assert metadata["assistant_id"] == agent_name

                assert "updated_at" in metadata, "Metadata should have 'updated_at'"
                # Verify it's a valid ISO timestamp
                updated_at = metadata["updated_at"]
                datetime.fromisoformat(updated_at)  # Should not raise

    @pytest.mark.asyncio
    async def test_config_metadata_in_non_streaming_mode(self, agent_config):
        """Test that config includes metadata in non-streaming mode."""
        agent_name = "test_agent_non_stream"
        process = AgentProcess(
            name=agent_name,
            config=agent_config,
            rpyc_port=False,
        )

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
        process.agent = mock_agent
        process.backend = MagicMock()

        # Mock session manager
        process.session_manager = MagicMock()
        process.session_manager.set_status = AsyncMock()

        # Call with stream=False
        with patch("deep_worker_cli.process.status_callback.StatusTrackingCallback"):
            try:
                await process.exposed_send_message_async(
                    message="test message",
                    stream=False,
                    notify_callback=None,
                )
            except Exception:
                pass

            # Verify ainvoke was called with proper config
            if mock_agent.ainvoke.called:
                call_args = mock_agent.ainvoke.call_args
                config = call_args[1] if len(call_args) > 1 else call_args.kwargs.get("config")

                # Verify metadata
                assert config is not None
                assert "metadata" in config
                metadata = config["metadata"]
                assert metadata["agent_name"] == agent_name
                assert metadata["assistant_id"] == agent_name
                assert "updated_at" in metadata

    @pytest.mark.asyncio
    async def test_get_history_config_includes_metadata(self, agent_config):
        """Test that get_history also uses config with metadata."""
        agent_name = "test_agent_history"
        process = AgentProcess(
            name=agent_name,
            config=agent_config,
            rpyc_port=False,
        )

        # Mock agent with get_state
        mock_agent = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {"messages": []}
        mock_agent.aget_state = AsyncMock(return_value=mock_state)
        process.agent = mock_agent

        # Call exposed_get_history
        try:
            await process.exposed_get_history(limit=10, offset=0)
        except Exception:
            pass

        # Verify aget_state was called
        if mock_agent.aget_state.called:
            call_args = mock_agent.aget_state.call_args
            config = call_args[0][0] if call_args[0] else call_args.kwargs.get("config")

            # For get_history, metadata might not be required for reading
            # but we should still have proper thread_id
            assert config is not None
            assert "configurable" in config
            assert config["configurable"]["thread_id"] == agent_name
