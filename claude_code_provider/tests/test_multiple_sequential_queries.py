"""Test for multiple sequential queries with same thread_id."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from claude_code_provider.chat_model import ClaudeCodeChatModel


class TestMultipleSequentialQueries:
    """Test multiple sequential queries to same thread."""

    @pytest.mark.asyncio
    async def test_two_sequential_queries_different_prompts(self) -> None:
        """Should handle two sequential queries with different prompts correctly.

        This is the core bug test: Second query should NOT return first answer.
        """
        model = ClaudeCodeChatModel()
        thread_id = "test-sequential"

        # First query: 2+2
        result1 = await model.ainvoke(
            [HumanMessage("2+2")],
            config={"configurable": {"thread_id": thread_id}}
        )

        print(f"\n[TEST] First query result: {result1.content}")
        assert "4" in result1.content, f"First query should contain '4', got: {result1.content}"

        # Second query: 3+3 (with full history including previous exchange)
        result2 = await model.ainvoke(
            [
                HumanMessage("2+2"),
                AIMessage(result1.content),
                HumanMessage("3+3")
            ],
            config={"configurable": {"thread_id": thread_id}}
        )

        print(f"\n[TEST] Second query result: {result2.content}")

        # CRITICAL CHECK: Second query should NOT return first answer
        assert "6" in result2.content, (
            f"Second query should contain '6', got: {result2.content}\n"
            f"This indicates the bug - returning first answer instead of processing second query"
        )

        # Cleanup
        await model.cleanup_thread(thread_id)
