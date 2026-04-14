"""Test that summarization prompt doesn't exceed context limits.

This test verifies the fix for the bug where the summary prompt itself
(~600 tokens) + trimmed messages (4000 tokens default) could exceed
the model's context limit, causing empty summaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware


class TestSummarizationPromptFix:
    """Tests for the summarization prompt size fix."""

    def test_trim_messages_accounts_for_prompt_size(self) -> None:
        """Test that _trim_messages_for_summary accounts for prompt token count."""
        # GIVEN: Mock model and middleware with default settings
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 200000}

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 10000),
            keep=("messages", 5),
            trim_tokens_to_summarize=4000,  # Default
        )

        # Create test messages that would be ~4500 tokens if not properly trimmed
        test_messages = [
            HumanMessage(content="This is a test message " * 100)
            for _ in range(50)
        ]

        # WHEN: Trimming messages for summary
        trimmed = middleware._trim_messages_for_summary(test_messages)

        # THEN: Should return trimmed messages
        assert trimmed is not None
        assert len(trimmed) < len(test_messages)

        # Calculate actual token counts
        prompt_tokens = middleware.token_counter([HumanMessage(content=middleware.summary_prompt)])
        trimmed_tokens = middleware.token_counter(trimmed)

        # THEN: Total (prompt + trimmed messages) should be <= trim_tokens_to_summarize
        total_tokens = prompt_tokens + trimmed_tokens
        assert total_tokens <= middleware.trim_tokens_to_summarize, (
            f"Total tokens ({total_tokens}) exceeds limit ({middleware.trim_tokens_to_summarize}). "
            f"Prompt: {prompt_tokens}, Trimmed messages: {trimmed_tokens}"
        )

    def test_trim_messages_with_none_limit(self) -> None:
        """Test that trim_tokens_to_summarize=None returns all messages."""
        # GIVEN: Middleware with no trim limit
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 200000}

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 10000),
            keep=("messages", 5),
            trim_tokens_to_summarize=None,  # No limit
        )

        test_messages = [
            HumanMessage(content=f"Message {i}") for i in range(10)
        ]

        # WHEN: Trimming messages
        trimmed = middleware._trim_messages_for_summary(test_messages)

        # THEN: Should return all messages
        assert trimmed == test_messages

    def test_trim_messages_minimum_safety_buffer(self) -> None:
        """Test that minimum 1000 token safety buffer is enforced."""
        # GIVEN: Middleware with very small trim limit
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 200000}

        # Set trim limit smaller than prompt size (should use 1000 token minimum)
        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 10000),
            keep=("messages", 5),
            trim_tokens_to_summarize=500,  # Smaller than prompt (~600 tokens)
        )

        test_messages = [
            HumanMessage(content="Test message " * 50) for _ in range(20)
        ]

        # WHEN: Trimming messages
        trimmed = middleware._trim_messages_for_summary(test_messages)

        # THEN: Should still return some messages (at least 1000 token buffer)
        assert trimmed is not None
        assert len(trimmed) > 0
        # Trimmed messages should be <= 1000 tokens (the safety minimum)
        trimmed_tokens = middleware.token_counter(trimmed)
        assert trimmed_tokens <= 1000 + 100  # Allow small buffer for approximation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
