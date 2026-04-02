"""Unit tests for Settings summarization configuration fields.

This test suite validates that Settings class correctly stores and provides
summarization configuration fields for controlling SummarizationMiddleware.

Following TDD approach: tests written first (RED), then implementation (GREEN).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from deep_worker_cli.config import Settings


class TestSettingsSummarizationFields:
    """Tests for summarization-related fields in Settings class."""

    def test_default_summarization_enabled(self) -> None:
        """Test that summarization is enabled by default."""
        settings = Settings.from_environment()
        assert settings.summarization_enabled is True

    def test_default_trigger_fraction_none(self) -> None:
        """Test that trigger fraction is None by default (auto-detect)."""
        settings = Settings.from_environment()
        assert settings.summarization_trigger_fraction is None

    def test_default_keep_fraction_none(self) -> None:
        """Test that keep fraction is None by default (auto-detect)."""
        settings = Settings.from_environment()
        assert settings.summarization_keep_fraction is None

    def test_default_truncate_after_none(self) -> None:
        """Test that truncate_after is None by default (auto-detect)."""
        settings = Settings.from_environment()
        assert settings.summarization_truncate_after is None

    def test_default_max_arg_length_none(self) -> None:
        """Test that max_arg_length is None by default (auto-detect)."""
        settings = Settings.from_environment()
        assert settings.summarization_max_arg_length is None

    def test_default_history_backend_temp(self) -> None:
        """Test that history backend defaults to 'temp'."""
        settings = Settings.from_environment()
        assert settings.summarization_history_backend == "temp"

    def test_env_disable_summarization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test disabling summarization via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_ENABLED", "false")
        settings = Settings.from_environment()
        assert settings.summarization_enabled is False

    def test_env_enable_summarization_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test explicitly enabling summarization via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_ENABLED", "true")
        settings = Settings.from_environment()
        assert settings.summarization_enabled is True

    def test_env_trigger_fraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting trigger fraction via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRIGGER", "0.85")
        settings = Settings.from_environment()
        assert settings.summarization_trigger_fraction == 0.85

    def test_env_keep_fraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting keep fraction via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_KEEP", "0.20")
        settings = Settings.from_environment()
        assert settings.summarization_keep_fraction == 0.20

    def test_env_truncate_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting truncate_after via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRUNCATE_AFTER", "25")
        settings = Settings.from_environment()
        assert settings.summarization_truncate_after == 25

    def test_env_max_arg_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting max_arg_length via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_MAX_ARG_LENGTH", "600")
        settings = Settings.from_environment()
        assert settings.summarization_max_arg_length == 600

    def test_env_history_backend_persistent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test setting history backend to persistent via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_HISTORY_BACKEND", "persistent")
        settings = Settings.from_environment()
        assert settings.summarization_history_backend == "persistent"

    def test_env_history_backend_temp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting history backend to temp via environment variable."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_HISTORY_BACKEND", "temp")
        settings = Settings.from_environment()
        assert settings.summarization_history_backend == "temp"

    def test_invalid_trigger_fraction_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid trigger fraction is ignored (falls back to None)."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRIGGER", "invalid")
        settings = Settings.from_environment()
        # Should fall back to None rather than crashing
        assert settings.summarization_trigger_fraction is None

    def test_invalid_keep_fraction_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid keep fraction is ignored (falls back to None)."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_KEEP", "not_a_number")
        settings = Settings.from_environment()
        assert settings.summarization_keep_fraction is None

    def test_invalid_truncate_after_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid truncate_after is ignored (falls back to None)."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_TRUNCATE_AFTER", "abc")
        settings = Settings.from_environment()
        assert settings.summarization_truncate_after is None

    def test_invalid_max_arg_length_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid max_arg_length is ignored (falls back to None)."""
        monkeypatch.setenv("DEEPAGENTS_SUMMARIZATION_MAX_ARG_LENGTH", "xyz")
        settings = Settings.from_environment()
        assert settings.summarization_max_arg_length is None

    def test_direct_instantiation(self) -> None:
        """Test that Settings can be instantiated with summarization fields."""
        settings = Settings(
            deepseek_api_key=None,
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            tavily_api_key=None,
            google_cloud_project=None,
            deepagents_langchain_project=None,
            user_langchain_project=None,
            project_root=None,
            shell_allow_list=None,
            summarization_enabled=False,
            summarization_trigger_fraction=0.90,
            summarization_keep_fraction=0.15,
            summarization_truncate_after=30,
            summarization_max_arg_length=700,
            summarization_history_backend="persistent",
        )
        assert settings.summarization_enabled is False
        assert settings.summarization_trigger_fraction == 0.90
        assert settings.summarization_keep_fraction == 0.15
        assert settings.summarization_truncate_after == 30
        assert settings.summarization_max_arg_length == 700
        assert settings.summarization_history_backend == "persistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
