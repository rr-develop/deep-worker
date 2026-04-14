"""End-to-end integration test for summarization with context overflow scenario.

This test simulates the exact bug scenario where an agent hits context limits
and verifies that summarization now works correctly with the fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware


class TestSummarizationE2E:
    """End-to-end test for summarization with context overflow."""

    def test_summarization_with_near_context_limit(self) -> None:
        """Test that summarization works when history is near context limit.

        This simulates the scenario where:
        1. Agent has long conversation history (near context limit)
        2. Summarization triggers
        3. Summary is created WITHOUT "Prompt is too long" error
        4. Summary is NOT empty
        """
        # GIVEN: Mock model with small context limit for testing
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 10000}  # Small limit for testing

        # Mock model.invoke to return a summary (not "Prompt is too long")
        mock_summary_response = MagicMock()
        mock_summary_response.text = """<session_intent>
Testing summarization functionality with near-context-limit scenario.
</session_intent>

<summary>
User sent multiple test messages to verify summarization behavior.
Agent responded to each message appropriately.
</summary>

<artifacts>
None
</artifacts>

<next_steps>
Continue testing summarization.
</next_steps>"""
        mock_model.invoke.return_value = mock_summary_response

        # Create middleware with LOW trigger to force summarization
        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 5000),  # Trigger at 5K tokens (50% of 10K limit)
            keep=("tokens", 1000),     # Keep 1K tokens
            trim_tokens_to_summarize=4000,  # Default trim limit
        )

        # Create many large messages to exceed trigger threshold
        messages = []
        for i in range(50):
            # Each message ~200 tokens
            content = f"Test message {i}: " + "This is a long test message with lots of content. " * 20
            messages.append(HumanMessage(content=content))
            messages.append(AIMessage(content=f"Response to message {i}: Acknowledged."))

        # Calculate current token count
        total_tokens = middleware.token_counter(messages)
        print(f"\nTotal tokens in history: {total_tokens}")

        # THEN: Should exceed trigger threshold
        assert total_tokens > 5000, f"Expected >5000 tokens, got {total_tokens}"

        # WHEN: Creating summary
        summary = middleware._create_summary(messages)

        # THEN: Summary should NOT be empty
        assert summary is not None
        assert len(summary) > 0
        assert summary != "No previous conversation history."
        assert summary != "Previous conversation was too long to summarize."
        assert "Prompt is too long" not in summary
        assert "Error generating summary" not in summary

        # Verify summary contains expected sections
        assert "<session_intent>" in summary
        assert "<summary>" in summary

        print(f"\n✅ Summary created successfully ({len(summary)} chars)")
        print(f"Summary preview: {summary[:200]}...")

    def test_summarization_respects_adjusted_token_limit(self) -> None:
        """Test that _trim_messages_for_summary respects adjusted token limit.

        This verifies the fix: prompt_tokens + trimmed_tokens <= trim_limit
        """
        # GIVEN: Mock model
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 200000}

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 10000),
            keep=("messages", 5),
            trim_tokens_to_summarize=4000,
        )

        # Create messages that would exceed trim limit without adjustment
        messages = [
            HumanMessage(content="Test message " * 100)
            for _ in range(100)
        ]

        # WHEN: Trimming messages for summary
        trimmed = middleware._trim_messages_for_summary(messages)

        # THEN: Verify token counts
        prompt_tokens = middleware.token_counter([HumanMessage(content=middleware.summary_prompt)])
        trimmed_tokens = middleware.token_counter(trimmed)
        total_tokens = prompt_tokens + trimmed_tokens

        print(f"\n📊 Token breakdown:")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Trimmed message tokens: {trimmed_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Limit: {middleware.trim_tokens_to_summarize}")

        # CRITICAL: Total must not exceed limit
        assert total_tokens <= middleware.trim_tokens_to_summarize, (
            f"BUG: Total tokens ({total_tokens}) exceeds limit ({middleware.trim_tokens_to_summarize})"
        )

        print(f"✅ Token limit respected: {total_tokens} <= {middleware.trim_tokens_to_summarize}")

    @pytest.mark.asyncio
    async def test_async_summarization_with_context_overflow(self) -> None:
        """Test async summarization path with context overflow scenario."""
        # GIVEN: Mock model
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 10000}

        # Mock async invoke
        mock_summary_response = MagicMock()
        mock_summary_response.text = """<session_intent>
Async summarization test.
</session_intent>

<summary>
Testing async summarization path.
</summary>

<artifacts>
None
</artifacts>

<next_steps>
Continue testing.
</next_steps>"""

        async def mock_ainvoke(*args, **kwargs):
            return mock_summary_response

        mock_model.ainvoke = mock_ainvoke

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 5000),
            keep=("tokens", 1000),
            trim_tokens_to_summarize=4000,
        )

        # Create large message history
        messages = [
            HumanMessage(content=f"Async test {i}: " + "content " * 50)
            for i in range(50)
        ]

        # WHEN: Creating summary asynchronously
        summary = await middleware._acreate_summary(messages)

        # THEN: Summary should be valid
        assert summary is not None
        assert len(summary) > 0
        assert "<summary>" in summary
        assert "Prompt is too long" not in summary

        print(f"\n✅ Async summary created successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
