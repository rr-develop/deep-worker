"""Test cross-process recursion using SINGLE bidirectional connection.

Key insight from RPyC docs:
"While the client is waiting for the result (a synchronous request),
it will SERVE all incoming requests, meaning the server can invoke
the callback it had received on the client."

This means we should use the SAME connection for both directions!
"""

from __future__ import annotations

import asyncio
import multiprocessing
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import rpyc
from rpyc.utils.server import ThreadedServer

_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))


def _find_free_port() -> int:
    """Find a free port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BidirectionalService(rpyc.Service):
    """Service that can both call and be called.

    CRITICAL: This service is used by BOTH A and B processes.
    The distinction between A and B is only in the role, not the service class.
    """

    # Class variables shared across all instances in this process
    call_log: list[str] = []
    log_lock = threading.Lock()
    role = "unknown"  # Will be set to "A" or "B"

    def on_connect(self, conn: Any) -> None:
        """Called when a client connects."""
        print(f"[{self.role}] Client connected")

    def exposed_set_role(self, role: str) -> None:
        """Set the role of this service instance (A or B)."""
        BidirectionalService.role = role
        print(f"[Service] Role set to: {role}")

    def exposed_get_call_log(self) -> list[str]:
        """Get call log."""
        with BidirectionalService.log_lock:
            return list(BidirectionalService.call_log)

    def exposed_clear_log(self) -> None:
        """Clear log."""
        with BidirectionalService.log_lock:
            BidirectionalService.call_log.clear()

    def exposed_process_and_callback(
        self,
        depth: int,
        message: str,
        caller_proxy: Any
    ) -> str:
        """Process request and possibly call back recursively.

        Args:
            depth: Recursion depth
            message: Message from caller
            caller_proxy: Proxy to the caller (for callback)

        Returns:
            Result string
        """
        log_entry = f"{self.role}.process_and_callback(depth={depth})"
        with BidirectionalService.log_lock:
            BidirectionalService.call_log.append(log_entry)

        print(f"[{self.role}] {log_entry}, msg='{message}'")

        if depth > 0:
            # Recursive case: call back to the caller
            print(f"[{self.role}] Calling back with depth={depth-1}")

            # CRITICAL: Use the caller_proxy to call back
            # The caller is already serving this connection, so it can handle
            # the callback while waiting for us to return
            # We pass `self` which is the current service instance
            result = caller_proxy.process_and_callback(
                depth - 1,
                f"{self.role}->callback depth {depth}",
                self  # Pass ourselves (current service) for next callback
            )

            return f"{self.role} received: {result}"
        else:
            # Base case
            return f"{self.role}: Base case (depth=0)"


def _run_server(port: int, role: str, ready: multiprocessing.Event) -> None:
    """Run server for bidirectional service."""
    try:
        server = ThreadedServer(
            BidirectionalService,
            hostname="127.0.0.1",
            port=port,
            protocol_config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
            },
        )

        # Set role after service creation but before starting
        # This doesn't work because Service instances are created per-connection
        # We'll set it via exposed_set_role instead

        ready.set()
        print(f"[Server] {role} ready on port {port}")
        server.start()

    except Exception as e:
        print(f"[Server] Error: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_bidirectional_single_connection():
    """Test recursion using single bidirectional connection.

    Process A starts a server.
    Process B starts a server.
    Test connects to A, A serves the request and connects to B for callback.
    B serves the callback and connects back to A via the SAME connection.

    Actually, let's simplify: Only B has a server.
    A is just a client that can also receive calls.
    """
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"Test: Single Bidirectional Connection Recursion")
    print(f"Process B port: {port_b}")
    print(f"{'='*80}\n")

    ready_b = multiprocessing.Event()

    # Only start B as a server
    proc_b = multiprocessing.Process(
        target=_run_server,
        args=(port_b, "B", ready_b),
        daemon=True,
    )
    proc_b.start()

    try:
        assert ready_b.wait(10), "B not ready"
        await asyncio.sleep(1.0)

        print("[Test] Connecting to B...")

        # Connect to B
        conn = rpyc.connect(
            "127.0.0.1",
            port_b,
            config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
            }
        )

        print("[Test] Connected, setting up roles...")

        # B's role is already set in the server
        # But we need to set it via the connection
        conn.root.set_role("B")

        # Clear logs
        conn.root.clear_log()

        # Now, make a call to B, passing a proxy to ourselves (the test/client)
        # But wait - we don't have a service on the client side!
        # We need to use ClassicService or create a service

        # Let's try a different approach: Use Callback object
        depth = 3

        print(f"\n[Test] Starting recursion with depth={depth}")
        print("[Test] Test will act as 'A', B is the server")
        start_time = time.time()

        # Create a local callback object
        class ClientCallback:
            """Callback handler on client side (acts as 'A')."""
            call_log: list[str] = []

            def process_and_callback(self, depth: int, message: str, caller_proxy: Any) -> str:
                log_entry = f"A.process_and_callback(depth={depth})"
                ClientCallback.call_log.append(log_entry)
                print(f"[Test/A] {log_entry}, msg='{message}'")

                if depth > 0:
                    # Call back to B
                    print(f"[Test/A] Calling back to B with depth={depth-1}")
                    result = caller_proxy.process_and_callback(
                        depth - 1,
                        f"A->B depth {depth}",
                        self  # Pass self as callback
                    )
                    return f"A received: {result}"
                else:
                    return "A: Base case (depth=0)"

        callback = ClientCallback()

        try:
            # Start the recursion
            result = conn.root.process_and_callback(
                depth,
                "Initial call from test",
                callback  # Pass callback object
            )

            elapsed = time.time() - start_time
            print(f"\n[Test] ✅ Completed in {elapsed:.2f}s")
            print(f"[Test] Result: {result}")

            log_a = callback.call_log
            log_b = conn.root.get_call_log()

            print(f"\nClient/A log ({len(log_a)} calls):")
            for entry in log_a:
                print(f"  - {entry}")

            print(f"\nServer/B log ({len(log_b)} calls):")
            for entry in log_b:
                print(f"  - {entry}")

            # Verify recursion
            # depth=3: B(3)->A(3)->B(2)->A(2)->B(1)->A(1)->B(0)->A(0)
            # B called with: 3, 2, 1, 0 = 4 times
            # A called with: 3, 2, 1, 0 = 4 times
            expected = depth + 1
            assert len(log_b) == expected, f"Expected {expected} B calls, got {len(log_b)}"
            assert len(log_a) == expected, f"Expected {expected} A calls, got {len(log_a)}"

            print(f"\n✅ All assertions passed!")
            print(f"   B called {len(log_b)} times (expected {expected})")
            print(f"   A called {len(log_a)} times (expected {expected})")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ FAILURE after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            raise

        conn.close()

    finally:
        proc_b.terminate()
        proc_b.join(5)
        if proc_b.is_alive():
            proc_b.kill()


if __name__ == "__main__":
    asyncio.run(test_bidirectional_single_connection())
