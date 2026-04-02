"""Test cross-process recursion over RPyC using BgServingThread.

This test demonstrates how to enable reentrant/recursive calls in RPyC:
1. Use BgServingThread to handle incoming callbacks asynchronously
2. Enable bind_threads=True to prevent thread conflicts
3. Make connections bidirectional with proper serving

Key insight: RPyC can support recursion if BOTH sides serve requests
concurrently using BgServingThread.
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

# ============================================================================
# CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))


def _find_free_port() -> int:
    """Find a free port dynamically using OS auto-assignment."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        return port


class RecursiveServiceA(rpyc.Service):
    """RPyC service for Process A - initiates and participates in recursion.

    CRITICAL: Use class variables to share state between Service instances!
    ThreadedServer creates a NEW Service instance for each connection,
    so instance variables don't work. Class variables are shared across
    all instances in the same process.
    """

    # Class variables - shared across all Service instances in this process
    peer_connection = None
    bg_thread = None
    call_log: list[str] = []
    log_lock = threading.Lock()

    def on_connect(self, conn: Any) -> None:
        """Called when a client connects."""
        print(f"[Process A] Client connected: {conn}")

    def exposed_set_peer(self, host: str, port: int) -> None:
        """Connect to peer Process B with BgServingThread enabled."""
        if RecursiveServiceA.peer_connection is None:
            print(f"[Process A] Connecting to peer at {host}:{port}")
            RecursiveServiceA.peer_connection = rpyc.connect(
                host, port,
                config={
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,
                    "bind_threads": True,  # CRITICAL: Enable thread binding
                }
            )
            # CRITICAL: Start BgServingThread to handle incoming callbacks
            RecursiveServiceA.bg_thread = rpyc.BgServingThread(
                RecursiveServiceA.peer_connection
            )
            print(f"[Process A] Connected to peer, BgServingThread started")

    def exposed_get_call_log(self) -> list[str]:
        """Get the log of all calls."""
        with RecursiveServiceA.log_lock:
            return list(RecursiveServiceA.call_log)

    def exposed_clear_log(self) -> None:
        """Clear the call log."""
        with RecursiveServiceA.log_lock:
            RecursiveServiceA.call_log.clear()

    def exposed_callback_from_b(self, depth: int, message: str) -> str:
        """Called by Process B via callback during recursion.

        This is where the recursion happens:
        - B calls this method on A (callback)
        - A then calls B again if depth > 0 (recursive call)
        """
        log_entry = f"A.callback_from_b(depth={depth}, msg='{message}')"
        with RecursiveServiceA.log_lock:
            RecursiveServiceA.call_log.append(log_entry)

        print(f"[Process A] {log_entry}")

        if depth > 0:
            # Recurse: A calls B again
            if RecursiveServiceA.peer_connection is None:
                return "A: ERROR - no peer connection"

            print(f"[Process A] Recursing: calling B.process_with_callback(depth={depth-1})")
            result = RecursiveServiceA.peer_connection.root.process_with_callback(
                depth - 1,
                f"A->B recursion depth {depth}"
            )
            return f"A received from B: {result}"
        else:
            return "A: Recursion base case reached"


class RecursiveServiceB(rpyc.Service):
    """RPyC service for Process B - processes and calls back to A.

    CRITICAL: Use class variables to share state between Service instances!
    """

    # Class variables - shared across all Service instances in this process
    peer_connection = None
    bg_thread = None
    call_log: list[str] = []
    log_lock = threading.Lock()

    def on_connect(self, conn: Any) -> None:
        """Called when a client connects."""
        print(f"[Process B] Client connected: {conn}")

    def exposed_set_peer(self, host: str, port: int) -> None:
        """Connect to peer Process A with BgServingThread enabled."""
        if RecursiveServiceB.peer_connection is None:
            print(f"[Process B] Connecting to peer at {host}:{port}")
            RecursiveServiceB.peer_connection = rpyc.connect(
                host, port,
                config={
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,
                    "bind_threads": True,  # CRITICAL: Enable thread binding
                }
            )
            # CRITICAL: Start BgServingThread to handle incoming callbacks
            RecursiveServiceB.bg_thread = rpyc.BgServingThread(
                RecursiveServiceB.peer_connection
            )
            print(f"[Process B] Connected to peer, BgServingThread started")

    def exposed_get_call_log(self) -> list[str]:
        """Get the log of all calls."""
        with RecursiveServiceB.log_lock:
            return list(RecursiveServiceB.call_log)

    def exposed_clear_log(self) -> None:
        """Clear the call log."""
        with RecursiveServiceB.log_lock:
            RecursiveServiceB.call_log.clear()

    def exposed_process_with_callback(self, depth: int, message: str) -> str:
        """Process a request and call back to Process A.

        Args:
            depth: Current recursion depth
            message: Message from caller

        Returns:
            Result string
        """
        log_entry = f"B.process_with_callback(depth={depth}, msg='{message}')"
        with RecursiveServiceB.log_lock:
            RecursiveServiceB.call_log.append(log_entry)

        print(f"[Process B] {log_entry}")

        # Call back to Process A
        if RecursiveServiceB.peer_connection is None:
            return "B: ERROR - no peer connection"

        print(f"[Process B] Calling back to A.callback_from_b(depth={depth})")
        result = RecursiveServiceB.peer_connection.root.callback_from_b(
            depth,
            f"B->A callback depth {depth}"
        )

        return f"B processed and got: {result}"


def _run_server_process(
    service_class: type,
    port: int,
    ready_event: multiprocessing.Event
) -> None:
    """Run RPyC server in a separate process with bind_threads enabled."""
    try:
        print(f"[Server] Starting {service_class.__name__} on port {port}")

        server = ThreadedServer(
            service_class,
            hostname="127.0.0.1",
            port=port,
            protocol_config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
                "bind_threads": True,  # CRITICAL: Enable thread binding on server
            },
        )

        # Signal that server is ready
        ready_event.set()
        print(f"[Server] {service_class.__name__} ready on port {port}")

        # Start serving (blocks until closed)
        server.start()

    except Exception as e:
        print(f"[Server] Error in {service_class.__name__}: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_recursive_calls_with_bgthread():
    """Test that cross-process recursion works with BgServingThread.

    Key technique:
    1. Both processes establish bidirectional RPyC connections
    2. Both use BgServingThread to serve incoming calls asynchronously
    3. bind_threads=True prevents thread conflicts
    4. Result: Recursion works without deadlock!
    """
    # Allocate ports dynamically
    port_a = _find_free_port()
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"Test: Cross-Process Recursion with BgServingThread")
    print(f"Process A port: {port_a}")
    print(f"Process B port: {port_b}")
    print(f"{'='*80}\n")

    ready_a = multiprocessing.Event()
    ready_b = multiprocessing.Event()

    # Start Process A server
    process_a = multiprocessing.Process(
        target=_run_server_process,
        args=(RecursiveServiceA, port_a, ready_a),
        daemon=True,
    )
    process_a.start()

    # Start Process B server
    process_b = multiprocessing.Process(
        target=_run_server_process,
        args=(RecursiveServiceB, port_b, ready_b),
        daemon=True,
    )
    process_b.start()

    try:
        # Wait for servers to be ready
        assert ready_a.wait(timeout=10), "Process A server did not start in time"
        assert ready_b.wait(timeout=10), "Process B server did not start in time"

        await asyncio.sleep(1.0)

        print("[Test] Both servers are ready, connecting from test...")

        # Connect to both processes from test
        # CRITICAL: Use bind_threads=True and BgServingThread for test connections too
        conn_a = rpyc.connect(
            "127.0.0.1",
            port_a,
            config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
                "bind_threads": True,
            }
        )

        conn_b = rpyc.connect(
            "127.0.0.1",
            port_b,
            config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
                "bind_threads": True,
            }
        )

        # Start BgServingThread for test connections
        bg_a = rpyc.BgServingThread(conn_a)
        bg_b = rpyc.BgServingThread(conn_b)

        print("[Test] Connected to both servers with BgServingThread")

        service_a = conn_a.root
        service_b = conn_b.root

        # Set up bidirectional connections between A and B
        print("[Test] Setting up peer connections...")
        service_a.set_peer("127.0.0.1", port_b)
        service_b.set_peer("127.0.0.1", port_a)

        # Give time for peer connections to establish
        await asyncio.sleep(0.5)

        print("[Test] Peer connections established")

        # Clear logs
        service_a.clear_log()
        service_b.clear_log()

        # Start the recursion with depth=5
        recursion_depth = 5

        print(f"\n[Test] Starting recursion with depth={recursion_depth}")
        start_time = time.time()

        try:
            result = service_b.process_with_callback(
                recursion_depth,
                "Initial call from test"
            )

            elapsed = time.time() - start_time
            print(f"\n[Test] ✅ Recursion completed in {elapsed:.2f}s")
            print(f"[Test] Result: {result}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[Test] ❌ FAILURE: Exception after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Get final call logs
        final_log_a = service_a.get_call_log()
        final_log_b = service_b.get_call_log()

        print(f"\n[Test] Final call logs:")
        print(f"Process A ({len(final_log_a)} calls):")
        for entry in final_log_a:
            print(f"  - {entry}")

        print(f"\nProcess B ({len(final_log_b)} calls):")
        for entry in final_log_b:
            print(f"  - {entry}")

        # Verify recursion happened correctly
        # With depth=5:
        # B(5) -> A(5) -> B(4) -> A(4) -> B(3) -> A(3) -> B(2) -> A(2) -> B(1) -> A(1) -> B(0) -> A(0)
        # B is called with depths: 5, 4, 3, 2, 1, 0 = 6 times
        # A is called with depths: 5, 4, 3, 2, 1, 0 = 6 times
        expected_calls = recursion_depth + 1

        assert len(final_log_b) == expected_calls, \
            f"Expected {expected_calls} calls to B, got {len(final_log_b)}"
        assert len(final_log_a) == expected_calls, \
            f"Expected {expected_calls} calls to A, got {len(final_log_a)}"

        print(f"\n✅ All assertions passed!")
        print(f"   - Process B was called {len(final_log_b)} times (expected {expected_calls})")
        print(f"   - Process A was called {len(final_log_a)} times (expected {expected_calls})")

        # Stop BgServingThreads
        bg_a.stop()
        bg_b.stop()

        # Clean up connections
        conn_a.close()
        conn_b.close()

    finally:
        # Terminate processes
        print("\n[Test] Cleaning up processes...")
        process_a.terminate()
        process_b.terminate()
        process_a.join(timeout=5)
        process_b.join(timeout=5)

        if process_a.is_alive():
            process_a.kill()
        if process_b.is_alive():
            process_b.kill()

        print("[Test] Cleanup complete")


@pytest.mark.asyncio
async def test_simple_callback_without_recursion():
    """Test simple callback without recursion to verify BgServingThread basics."""
    port_a = _find_free_port()
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"Test: Simple Callback (No Recursion)")
    print(f"Process A port: {port_a}")
    print(f"Process B port: {port_b}")
    print(f"{'='*80}\n")

    ready_a = multiprocessing.Event()
    ready_b = multiprocessing.Event()

    process_a = multiprocessing.Process(
        target=_run_server_process,
        args=(RecursiveServiceA, port_a, ready_a),
        daemon=True,
    )
    process_a.start()

    process_b = multiprocessing.Process(
        target=_run_server_process,
        args=(RecursiveServiceB, port_b, ready_b),
        daemon=True,
    )
    process_b.start()

    try:
        assert ready_a.wait(timeout=10)
        assert ready_b.wait(timeout=10)
        await asyncio.sleep(1.0)

        conn_a = rpyc.connect("127.0.0.1", port_a, config={"allow_public_attrs": True, "bind_threads": True})
        conn_b = rpyc.connect("127.0.0.1", port_b, config={"allow_public_attrs": True, "bind_threads": True})

        bg_a = rpyc.BgServingThread(conn_a)
        bg_b = rpyc.BgServingThread(conn_b)

        service_a = conn_a.root
        service_b = conn_b.root

        service_a.set_peer("127.0.0.1", port_b)
        service_b.set_peer("127.0.0.1", port_a)

        await asyncio.sleep(0.5)

        service_a.clear_log()
        service_b.clear_log()

        # Call with depth=0 (no recursion, just one callback)
        print("[Test] Calling B with depth=0 (simple callback)")
        result = service_b.process_with_callback(0, "Simple test")

        print(f"[Test] Result: {result}")

        log_a = service_a.get_call_log()
        log_b = service_b.get_call_log()

        print(f"Process A log: {log_a}")
        print(f"Process B log: {log_b}")

        assert len(log_b) == 1, "B should be called once"
        assert len(log_a) == 1, "A should be called once (callback)"

        print("✅ Simple callback test passed!")

        bg_a.stop()
        bg_b.stop()
        conn_a.close()
        conn_b.close()

    finally:
        process_a.terminate()
        process_b.terminate()
        process_a.join(timeout=5)
        process_b.join(timeout=5)
        if process_a.is_alive():
            process_a.kill()
        if process_b.is_alive():
            process_b.kill()


if __name__ == "__main__":
    """Run tests directly for debugging."""
    print("Running BgServingThread recursion tests...")
    print("=" * 80)

    # Run simple test first
    asyncio.run(test_simple_callback_without_recursion())

    print("\n" + "=" * 80)
    print("Simple test completed, starting recursion test...")
    print("=" * 80 + "\n")

    # Run recursion test
    asyncio.run(test_recursive_calls_with_bgthread())

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
