#!/usr/bin/env python3
"""Direct test for multiple queries issue WITHOUT agent process."""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from chat_model import ClaudeCodeChatModel


async def test_multiple_queries():
    """Test multiple sequential queries with same thread_id."""
    print("\n" + "=" * 80)
    print("DIRECT TEST: Multiple Sequential Queries")
    print("=" * 80)

    model = ClaudeCodeChatModel()
    thread_id = "test-thread"

    # First query
    print("\n>>> First Query: 2+2")
    result1 = await model.ainvoke(
        [HumanMessage("2+2")],
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"<<< First Response: {result1.content}")

    # Wait a bit
    await asyncio.sleep(1)

    # Second query (with history)
    print("\n>>> Second Query: 3+3 (with history)")
    result2 = await model.ainvoke(
        [
            HumanMessage("2+2"),
            AIMessage(result1.content),
            HumanMessage("3+3")
        ],
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"<<< Second Response: {result2.content}")

    # Check if second response is correct
    print("\n" + "=" * 80)
    if "6" in result2.content:
        print("✅ SUCCESS: Second query returned correct answer!")
    elif "4" in result2.content:
        print("❌ FAILURE: Second query returned FIRST answer (bug detected)!")
    else:
        print("⚠️  UNKNOWN: Response doesn't contain expected numbers")
    print("=" * 80)

    # Cleanup
    await model.cleanup_thread(thread_id)


if __name__ == "__main__":
    asyncio.run(test_multiple_queries())
