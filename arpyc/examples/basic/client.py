"""Basic ARPyC client example."""

import asyncio

from arpyc import RpycAsyncSession


async def main() -> None:
    """Demonstrate basic ARPyC usage."""
    print("=== ARPyC Basic Example ===\n")

    # Connect to server using context manager
    async with RpycAsyncSession("localhost", 18861) as session:
        print("✓ Connected to server\n")

        # Method 1: Direct async_call
        print("Method 1: Using async_call")
        result = await session.async_call(session.root.add, 10, 20)
        print(f"  10 + 20 = {result}")

        result = await session.async_call(session.root.multiply, 5, 7)
        print(f"  5 * 7 = {result}\n")

        # Method 2: Transparent proxy (recommended)
        print("Method 2: Using transparent proxy")
        proxy = session.proxy(session.root)

        result = await proxy.subtract(100, 42)
        print(f"  100 - 42 = {result}")

        result = await proxy.divide(22, 7)
        print(f"  22 / 7 = {result:.4f}\n")

        # Method 3: Sync calls when needed
        print("Method 3: Synchronous calls")
        greeting = proxy.greet.sync("World")
        print(f"  {greeting}\n")

        # Error handling
        print("Error handling:")
        try:
            await proxy.divide(10, 0)
        except ValueError as e:
            print(f"  Caught error: {e}\n")

        # Multiple concurrent calls
        print("Concurrent calls:")
        results = await asyncio.gather(
            proxy.add(1, 2),
            proxy.add(3, 4),
            proxy.add(5, 6),
        )
        print(f"  Concurrent results: {results}\n")

    print("✓ Disconnected from server")


if __name__ == "__main__":
    asyncio.run(main())
