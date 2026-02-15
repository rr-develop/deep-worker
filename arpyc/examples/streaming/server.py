"""RPyC server with streaming example."""

import time

import rpyc
from rpyc.utils.server import ThreadedServer


class StreamingService(rpyc.Service):
    """RPyC service with streaming support."""

    def exposed_stream_numbers(self, callback, count: int = 10, delay: float = 0.5) -> None:
        """Stream numbers to client."""
        try:
            for i in range(count):
                callback.on_chunk(i)
                time.sleep(delay)
            callback.on_done()
        except Exception as e:
            callback.on_error(e)

    def exposed_stream_messages(self, callback) -> None:
        """Stream messages to client."""
        messages = [
            "Hello from server!",
            "This is message 2",
            "Processing data...",
            "Almost done...",
            "Goodbye!",
        ]

        try:
            for msg in messages:
                callback.on_chunk(msg)
                time.sleep(0.5)
            callback.on_done()
        except Exception as e:
            callback.on_error(e)

    def exposed_stream_with_error(self, callback) -> None:
        """Stream that encounters an error."""
        try:
            callback.on_chunk("Chunk 1")
            time.sleep(0.3)
            callback.on_chunk("Chunk 2")
            time.sleep(0.3)
            # Simulate error
            raise RuntimeError("Simulated streaming error")
        except Exception as e:
            callback.on_error(e)


def main() -> None:
    """Start the streaming server."""
    print("Starting Streaming RPyC server on port 18861...")

    server = ThreadedServer(
        StreamingService,
        hostname="127.0.0.1",
        port=18861,
        protocol_config={"allow_public_attrs": True},
    )

    print("Server ready! Press Ctrl+C to stop.")
    server.start()


if __name__ == "__main__":
    main()
