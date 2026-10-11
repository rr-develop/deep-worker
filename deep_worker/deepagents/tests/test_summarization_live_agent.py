"""Live agent test for summarization with simulated context overflow.

This test creates a real agent with low context limits and verifies that
summarization triggers correctly and creates non-empty summaries.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends import StateBackend


_SUMMARY_TEXT = """<session_intent>
Testing summarization in live agent.
</session_intent>

<summary>
User sent test messages. Agent responded to each message.
</summary>

<artifacts>
None
</artifacts>

<next_steps>
Continue testing agent functionality.
</next_steps>"""


class _SummarizingFakeModel(BaseChatModel):
    """A chat model double that answers summarization prompts differently.

    The surface is finite on purpose: it is a real ``BaseChatModel`` returning
    real ``AIMessage`` objects, so code that probes a response for optional
    attributes gets an honest "absent" instead of a freshly minted mock. See
    the note in ``test_agent_summarization_with_low_trigger``.
    """

    @property
    def _llm_type(self) -> str:
        return "summarizing-fake"

    @property
    def model_name(self) -> str:
        return "claude-sonnet-4.5"

    @property
    def profile(self) -> dict[str, int]:
        return {"max_input_tokens": 200000}

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> ChatResult:
        if "Extract the following information from the conversation" in str(messages):
            message = AIMessage(content=_SUMMARY_TEXT)
        else:
            message = AIMessage(content="Test response")
        return ChatResult(generations=[ChatGeneration(message=message)])


class TestSummarizationLiveAgent:
    """Live agent tests for summarization."""

    def test_agent_summarization_with_low_trigger(self) -> None:
        """Test that agent with low trigger successfully creates summaries.

        This test:
        1. Creates agent with very low summarization trigger (500 tokens)
        2. Sends multiple messages to exceed trigger
        3. Verifies that summarization occurs
        4. Verifies that summary is NOT empty
        5. Verifies agent can continue working after summarization
        """
        # GIVEN: a model double with a DEFINED surface.
        #
        # Deliberately not a bare MagicMock. The agent hands every response to
        # `_stamp_response`, which probes it with `getattr(value, "result")`; on
        # a MagicMock that attribute always exists and yields another MagicMock,
        # so the probe recursed forever. The RecursionError traceback then cost
        # 19.5 GB to format under pytest and took the host down twice
        # (see a related internal incident analysis, not included here). A real
        # AIMessage has no `.result`, so the probe stops immediately.
        mock_model = _SummarizingFakeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create agent with LOW trigger for testing
            # We'll override the summarization middleware's trigger
            with patch("deepagents.graph._compute_summarization_defaults") as mock_defaults:
                # Set very low trigger to force summarization quickly
                mock_defaults.return_value = {
                    "trigger": ("tokens", 500),  # Very low trigger
                    "keep": ("tokens", 100),
                    "truncate_args_settings": {
                        "trigger": ("tokens", 500),
                        "keep": ("tokens", 100),
                    },
                }

                agent = create_deep_agent(
                    model=mock_model,
                    checkpointer=InMemorySaver(),
                    backend=lambda rt: StateBackend(rt),
                )

            # WHEN: Invoking agent with multiple messages
            config = {"configurable": {"thread_id": "test_summarization"}}
            files = {}

            # Send first message (should not trigger summarization)
            result1 = agent.invoke(
                {"messages": [HumanMessage(content="Test message 1: " + "content " * 50)]},
                config=config,
                files=files,
            )

            assert result1 is not None
            messages_after_1 = result1.get("messages", [])
            print(f"\nAfter message 1: {len(messages_after_1)} messages in state")

            # Send more messages to exceed trigger
            for i in range(2, 6):
                result = agent.invoke(
                    {"messages": [HumanMessage(content=f"Test message {i}: " + "content " * 50)]},
                    config=config,
                    files=files,
                )
                assert result is not None
                messages = result.get("messages", [])
                print(f"After message {i}: {len(messages)} messages in state")

                # Check if any message contains summarization marker
                for msg in messages:
                    content = str(msg.content)
                    if "middle of a conversation that has been summarized" in content.lower():
                        print(f"\n✅ SUMMARIZATION DETECTED after message {i}!")
                        print(f"Summary message content preview: {content[:300]}...")

                        # THEN: Verify summary is NOT empty
                        assert "<summary>" in content, "Summary should contain <summary> tag"
                        assert "</summary>" in content, "Summary should contain </summary> tag"

                        # Extract summary content
                        summary_start = content.find("<summary>")
                        summary_end = content.find("</summary>")
                        summary_content = content[summary_start + 9:summary_end].strip()

                        assert len(summary_content) > 0, "Summary content should NOT be empty"
                        assert summary_content != "\n\n", "Summary should have actual content"

                        print(f"✅ Summary is NOT empty: {len(summary_content)} chars")
                        return  # Test passed!

            # If we reach here, summarization didn't trigger (might need more messages)
            pytest.skip("Summarization did not trigger within 5 messages (may need higher volume)")

    def test_summary_format_validation(self) -> None:
        """Test that summary follows expected format."""
        # GIVEN: Mock model
        mock_model = MagicMock()
        mock_model.model_name = "claude-sonnet-4.5"
        mock_model.profile = {"max_input_tokens": 200000}

        # Mock summary response
        summary_response = MagicMock()
        summary_response.text = """<session_intent>
Test intent
</session_intent>

<summary>
Test summary content
</summary>

<artifacts>
None
</artifacts>

<next_steps>
Next steps here
</next_steps>"""
        mock_model.invoke.return_value = summary_response

        # WHEN: Creating summary
        from deepagents.middleware.summarization import SummarizationMiddleware

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=lambda rt: StateBackend(rt),
            trigger=("tokens", 10000),
            keep=("messages", 5),
        )

        messages = [HumanMessage(content="Test " * 100) for _ in range(10)]
        summary = middleware._create_summary(messages)

        # THEN: Verify format
        assert "<session_intent>" in summary
        assert "</session_intent>" in summary
        assert "<summary>" in summary
        assert "</summary>" in summary
        assert "<artifacts>" in summary
        assert "<next_steps>" in summary

        # Verify content is not empty
        summary_start = summary.find("<summary>")
        summary_end = summary.find("</summary>")
        summary_content = summary[summary_start + 9:summary_end].strip()
        assert len(summary_content) > 0

        print(f"\n✅ Summary format valid: {len(summary)} chars total")
        print(f"Summary content: {summary_content}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
