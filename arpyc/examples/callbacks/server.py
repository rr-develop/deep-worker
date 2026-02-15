"""RPyC server with event pub/sub example."""

import threading
import time

import rpyc
from rpyc.utils.server import ThreadedServer


class EventBus:
    """Simple event bus for pub/sub."""

    def __init__(self) -> None:
        self._subscribers: set = set()
        self._lock = threading.Lock()

    def subscribe(self, callback) -> None:
        """Subscribe to events."""
        with self._lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback) -> None:
        """Unsubscribe from events."""
        with self._lock:
            self._subscribers.discard(callback)

    def emit(self, name: str, payload: dict) -> None:
        """Emit event to all subscribers."""
        with self._lock:
            subscribers = list(self._subscribers)

        for sub in subscribers:
            try:
                sub.on_event(name, payload)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")


class EventService(rpyc.Service):
    """RPyC service with event support."""

    def on_connect(self, conn) -> None:
        """Called when client connects."""
        self._bus = EventBus()
        self._counter = 0
        self._emitting = False

    def exposed_subscribe(self, callback) -> bool:
        """Subscribe to events."""
        self._bus.subscribe(callback)
        return True

    def exposed_unsubscribe(self, callback) -> bool:
        """Unsubscribe from events."""
        self._bus.unsubscribe(callback)
        return True

    def exposed_emit_test_event(self, message: str) -> None:
        """Emit a test event."""
        self._bus.emit("test_event", {"message": message, "timestamp": time.time()})

    def exposed_start_counter(self, interval: float = 1.0) -> None:
        """Start emitting counter events periodically."""
        if self._emitting:
            return

        self._emitting = True

        def emit_loop():
            while self._emitting:
                self._counter += 1
                self._bus.emit(
                    "counter",
                    {
                        "count": self._counter,
                        "timestamp": time.time(),
                    },
                )
                time.sleep(interval)

        thread = threading.Thread(target=emit_loop, daemon=True)
        thread.start()

    def exposed_stop_counter(self) -> None:
        """Stop emitting counter events."""
        self._emitting = False


def main() -> None:
    """Start the event server."""
    print("Starting Event RPyC server on port 18861...")

    server = ThreadedServer(
        EventService,
        hostname="127.0.0.1",
        port=18861,
        protocol_config={"allow_public_attrs": True},
    )

    print("Server ready! Press Ctrl+C to stop.")
    server.start()


if __name__ == "__main__":
    main()
