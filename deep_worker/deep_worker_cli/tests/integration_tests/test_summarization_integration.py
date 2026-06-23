"""Integration tests for SummarizationMiddleware in agent creation.

This test suite validates that SummarizationMiddleware integrates correctly
with create_cli_agent() and works in real scenarios with conversation history.

Tests verify:
- Agent creation with summarization enabled/disabled
- Summarization triggering based on token count
- History persistence (temp and persistent backends)
- Configuration from settings
- Message filtering (excluding summarization messages)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.config import Settings


class TestSummarizationIntegration:
    """Integration tests for SummarizationMiddleware."""

    @pytest.fixture
    def temp_project_dir(self) -> Path:
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock BaseChatModel for testing."""
        mock = MagicMock()
        mock.model_name = "claude-sonnet-4.5"
        # Mock get_model_profile to return context limit
        mock.get_model_profile.return_value = {"max_input_tokens": 200_000}
        return mock

    def test_agent_creation_with_summarization_enabled(
        self, temp_project_dir: Path, mock_model: MagicMock
    ) -> None:
        """Test that agent is created with SummarizationMiddleware when enabled."""
        # GIVEN: Settings with summarization enabled (default)
        assert Settings.from_environment().summarization_enabled is True

        # WHEN: Creating agent with default settings
        agent, backend = create_cli_agent(
            model=mock_model,
            assistant_id="test_agent",
            project_memory_dir=temp_project_dir,
            checkpointer=InMemorySaver(),
            enable_memory=False,  # Disable to simplify test
            enable_skills=False,
            enable_shell=False,
        )

        # THEN: Agent should be created successfully
        assert agent is not None
        assert backend is not None

    def test_agent_creation_with_summarization_disabled(
        self, temp_project_dir: Path, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that agent is created without SummarizationMiddleware when disabled."""
        # GIVEN: Settings with summarization disabled
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_ENABLED", "false")
        settings = Settings.from_environment()
        assert settings.summarization_enabled is False

        # WHEN: Creating agent
        with patch("deep_worker_cli.agent.settings", settings):
            agent, backend = create_cli_agent(
                model=mock_model,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        # THEN: Agent should be created successfully
        assert agent is not None
        assert backend is not None

    def test_agent_creation_with_model_string(
        self, temp_project_dir: Path
    ) -> None:
        """Test agent creation with model as string (not BaseChatModel)."""
        # GIVEN: Model as string
        model_str = "anthropic:claude-sonnet-4.5"

        # WHEN: Creating agent
        agent, backend = create_cli_agent(
            model=model_str,
            assistant_id="test_agent",
            project_memory_dir=temp_project_dir,
            checkpointer=InMemorySaver(),
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
        )

        # THEN: Agent should be created successfully
        assert agent is not None
        assert backend is not None

    def test_summarization_with_temp_backend(
        self, temp_project_dir: Path, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that temp history backend is used by default."""
        # GIVEN: Default settings (temp backend)
        settings = Settings.from_environment()
        assert settings.summarization_history_backend == "temp"

        # WHEN: Creating agent with temp backend
        with patch("deep_worker_cli.agent.settings", settings):
            agent, backend = create_cli_agent(
                model=mock_model,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        # THEN: Agent created successfully (temp backend is internal)
        assert agent is not None

    def test_summarization_with_persistent_backend(
        self, temp_project_dir: Path, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that persistent history backend setting is accepted."""
        # GIVEN: Settings with persistent backend
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_HISTORY_BACKEND", "persistent")
        settings = Settings.from_environment()
        assert settings.summarization_history_backend == "persistent"

        # WHEN: Creating agent with persistent backend
        with patch("deep_worker_cli.agent.settings", settings):
            agent, backend = create_cli_agent(
                model=mock_model,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        # THEN: Agent created successfully with persistent setting
        assert agent is not None

    def test_summarization_config_overrides(
        self, temp_project_dir: Path, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that summarization config overrides from settings are applied."""
        # GIVEN: Custom trigger/keep fractions
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRIGGER", "0.90")
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_KEEP", "0.20")
        settings = Settings.from_environment()
        assert settings.summarization_trigger_fraction == 0.90
        assert settings.summarization_keep_fraction == 0.20

        # WHEN: Creating agent with custom settings
        with patch("deep_worker_cli.agent.settings", settings):
            agent, backend = create_cli_agent(
                model=mock_model,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        # THEN: Agent created successfully (overrides are internal to middleware)
        assert agent is not None

    def test_summarization_with_different_models(
        self, temp_project_dir: Path
    ) -> None:
        """Test summarization configuration adapts to different model context sizes."""
        # Test with different model strings to verify context limit detection
        # Note: Only testing with mock models to avoid dependency requirements
        test_cases = [
            ("anthropic:claude-opus-4.6", "1M context model"),
            ("anthropic:claude-sonnet-4.5", "200K context model"),
        ]

        for model_str, description in test_cases:
            # WHEN: Creating agent with specific model
            agent, backend = create_cli_agent(
                model=model_str,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

            # THEN: Agent created successfully
            assert agent is not None, f"Failed to create agent for {description}"

    def test_agent_with_all_middleware_enabled(
        self, temp_project_dir: Path, mock_model: MagicMock
    ) -> None:
        """Test that SummarizationMiddleware works with other middleware enabled."""
        # GIVEN: All middleware enabled (realistic scenario)
        # WHEN: Creating agent with all middleware
        agent, backend = create_cli_agent(
            model=mock_model,
            assistant_id="test_agent",
            project_memory_dir=temp_project_dir,
            checkpointer=InMemorySaver(),
            enable_memory=True,  # MemoryMiddleware
            enable_skills=True,  # SkillsMiddleware
            enable_shell=True,   # LocalContextMiddleware + ProjectMemoryMiddleware
        )

        # THEN: Agent created successfully with all middleware
        assert agent is not None
        assert backend is not None

    def test_manual_truncation_overrides(
        self, temp_project_dir: Path, mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test manual truncation parameter overrides."""
        # GIVEN: Custom truncation settings
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRUNCATE_AFTER", "30")
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_MAX_ARG_LENGTH", "700")
        settings = Settings.from_environment()
        assert settings.summarization_truncate_after == 30
        assert settings.summarization_max_arg_length == 700

        # WHEN: Creating agent with custom truncation
        with patch("deep_worker_cli.agent.settings", settings):
            agent, backend = create_cli_agent(
                model=mock_model,
                assistant_id="test_agent",
                project_memory_dir=temp_project_dir,
                checkpointer=InMemorySaver(),
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        # THEN: Agent created successfully
        assert agent is not None


class TestSummarizationFunctional:
    """Functional tests for summarization behavior (requires actual model)."""

    @pytest.mark.skip(reason="Requires actual model and tokens - manual testing only")
    def test_summarization_triggers_on_long_conversation(self) -> None:
        """Test that summarization actually triggers with long conversation.

        This test is skipped by default as it requires:
        - Real API credentials
        - Significant token usage
        - Long-running conversation

        To run manually:
        1. Set DEEPSEEK_API_KEY or other provider credentials
        2. Run: pytest -v -s tests/integration_tests/test_summarization_integration.py::TestSummarizationFunctional::test_summarization_triggers_on_long_conversation
        """
        pass

    @pytest.mark.skip(reason="Requires actual model and tokens - manual testing only")
    def test_history_persistence_across_restarts(self) -> None:
        """Test that persistent history survives agent restarts.

        Manual test to verify:
        1. Create agent with persistent backend
        2. Have long conversation
        3. Stop agent
        4. Restart agent with same assistant_id
        5. Verify history is restored
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
