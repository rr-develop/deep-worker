"""Basic RPyC server example."""

import rpyc
from rpyc.utils.server import ThreadedServer


class CalculatorService(rpyc.Service):
    """Simple calculator service."""

    def exposed_add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def exposed_subtract(self, a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b

    def exposed_multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def exposed_divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    def exposed_greet(self, name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"


def main() -> None:
    """Start the RPyC server."""
    print("Starting Calculator RPyC server on port 18861...")

    server = ThreadedServer(
        CalculatorService,
        hostname="127.0.0.1",
        port=18861,
        protocol_config={"allow_public_attrs": True},
    )

    print("Server ready! Press Ctrl+C to stop.")
    server.start()


if __name__ == "__main__":
    main()
