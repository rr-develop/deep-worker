"""ARPyC client with streaming example."""

import asyncio

from arpyc import RpycAsyncSession, StreamingCallback


async def main() -> None:
    """Demonstrate ARPyC streaming."""
    print("=== ARPyC Streaming Example ===\n")

    async with RpycAsyncSession("localhost", 18861) as session:
        print("✓ Connected to server\n")

        loop = asyncio.get_running_loop()

        # Example 1: Stream numbers
        print("Example 1: Streaming numbers")
        callback = StreamingCallback[int](loop)

        # Start streaming (non-blocking)
        session.root.stream_numbers(callback, count=5, delay=0.3)

        # Consume stream
        async for number in callback:
            print(f"  Received: {number}")

        print("  ✓ Stream complete\n")

        # Example 2: Stream messages
        print("Example 2: Streaming messages")
        msg_callback = StreamingCallback[str](loop)

        session.root.stream_messages(msg_callback)

        async for message in msg_callback:
            print(f"  📨 {message}")

        print("  ✓ Stream complete\n")

        # Example 3: Handle streaming errors
        print("Example 3: Streaming with error handling")
        error_callback = StreamingCallback[str](loop)

        session.root.stream_with_error(error_callback)

        try:
            async for chunk in error_callback:
                print(f"  Chunk: {chunk}")
        except RuntimeError as e:
            print(f"  ⚠ Caught streaming error: {e}\n")

    print("✓ Disconnected from server")


async def concurrent_streams_example() -> None:
    """Demonstrate concurrent streaming."""
    print("=== Concurrent Streaming Example ===\n")

    async with RpycAsyncSession("localhost", 18861) as session:
        loop = asyncio.get_running_loop()

        # Start multiple streams concurrently
        callback1 = StreamingCallback[int](loop)
        callback2 = StreamingCallback[int](loop)

        session.root.stream_numbers(callback1, count=3, delay=0.5)
        session.root.stream_numbers(callback2, count=3, delay=0.7)

        # Consume both streams concurrently
        async def consume_stream(name: str, callback):
            async for item in callback:
                print(f"[{name}] {item}")

        await asyncio.gather(
            consume_stream("Stream1", callback1),
            consume_stream("Stream2", callback2),
        )

        print("\n✓ Both streams complete")


if __name__ == "__main__":
    asyncio.run(main())
    # Or try: asyncio.run(concurrent_streams_example())
