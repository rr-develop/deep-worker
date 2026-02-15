"""ARPyC client with event callbacks example."""

import asyncio

from arpyc import AsyncioEventCallback, RpycAsyncSession


async def main() -> None:
    """Demonstrate ARPyC event callbacks."""
    print("=== ARPyC Callbacks Example ===\n")

    async with RpycAsyncSession("localhost", 18861) as session:
        print("✓ Connected to server\n")

        loop = asyncio.get_running_loop()

        # Sync event handler
        def on_event(name: str, payload: dict) -> None:
            print(f"Event: {name}")
            print(f"  Payload: {payload}\n")

        # Create callback adapter
        callback = AsyncioEventCallback(loop, on_event)

        # Subscribe to events
        session.root.subscribe(callback)
        print("✓ Subscribed to events\n")

        # Emit test event
        print("Emitting test event...")
        session.root.emit_test_event("Hello from client!")

        await asyncio.sleep(0.5)

        # Start counter
        print("Starting counter (will emit events every second)...")
        session.root.start_counter(1.0)

        # Listen for events
        print("Listening for 5 seconds...\n")
        await asyncio.sleep(5)

        # Stop counter
        print("\nStopping counter...")
        session.root.stop_counter()

        # Unsubscribe
        session.root.unsubscribe(callback)
        print("✓ Unsubscribed from events\n")

    print("✓ Disconnected from server")


async def async_handler_example() -> None:
    """Demonstrate async event handler."""
    print("=== ARPyC Async Handler Example ===\n")

    async with RpycAsyncSession("localhost", 18861) as session:
        loop = asyncio.get_running_loop()

        # Async event handler
        async def on_event_async(name: str, payload: dict) -> None:
            # Simulate async processing
            await asyncio.sleep(0.1)
            print(f"[Async] Event: {name} - {payload}")

        callback = AsyncioEventCallback(loop, on_event_async)
        session.root.subscribe(callback)

        # Emit some events
        for i in range(3):
            session.root.emit_test_event(f"Message {i + 1}")
            await asyncio.sleep(0.5)

        session.root.unsubscribe(callback)


if __name__ == "__main__":
    asyncio.run(main())
    # Or try: asyncio.run(async_handler_example())
