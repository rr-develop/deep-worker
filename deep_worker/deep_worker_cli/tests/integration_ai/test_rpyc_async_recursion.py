"""Test truly asynchronous recursion using rpyc.async_() and AsyncResult.

This demonstrates how to make RPyC calls asynchronously (non-blocking)
while still supporting recursion through callbacks.

KEY DIFFERENCE from sync version:
- Sync: call blocks until result arrives, but serves incoming requests
- Async: call returns AsyncResult immediately, can do other work while waiting
"""

from __future__ import annotations

import asyncio
import multiprocessing
import socket
import sys
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
    """Find free port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AsyncRecursiveServer(rpyc.Service):
    """Server that supports async recursive callbacks."""

    def on_connect(self, conn: Any) -> None:
        """Store connection reference."""
        self._conn = conn
        self.call_log: list[str] = []
        self.role = "Server"

    def exposed_set_role(self, role: str) -> None:
        """Set role for logging."""
        self.role = role

    def exposed_get_call_log(self) -> list[str]:
        """Get call log."""
        return list(self.call_log)

    def exposed_clear_log(self) -> None:
        """Clear call log."""
        self.call_log.clear()

    def exposed_async_recursive_call(self, depth: int, message: str, callback: Any) -> str:
        """Async recursive RPC call with callback.

        This method is called SYNCHRONOUSLY by RPyC, but it can make
        ASYNCHRONOUS calls to the callback using rpyc.async_().
        """
        log_entry = f"{self.role}.async_recursive_call(depth={depth})"
        self.call_log.append(log_entry)
        print(f"[{self.role}] {log_entry}, msg='{message}'")

        if depth > 0:
            # ASYNC CALL: Use rpyc.async_() to call callback asynchronously
            print(f"[{self.role}] Async calling back with depth={depth-1}")

            # This returns AsyncResult immediately (non-blocking)
            async_result = rpyc.async_(callback)(
                depth - 1,
                f"{self.role}->async_callback",
                self.exposed_async_recursive_call
            )

            # Now we can do other work while waiting...
            print(f"[{self.role}] AsyncResult created, can do other work...")

            # When we need the result, we wait for it
            # This will serve incoming requests while waiting
            print(f"[{self.role}] Waiting for async result...")
            result = async_result.value  # Blocks here, but serves requests

            return f"{self.role} received (async): {result}"
        else:
            return f"{self.role}: Base case (depth={depth})"


def _run_server(port: int, role: str, ready: multiprocessing.Event) -> None:
    """Run server process."""
    try:
        server = ThreadedServer(
            AsyncRecursiveServer,
            hostname="127.0.0.1",
            port=port,
            protocol_config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
            },
        )
        ready.set()
        print(f"[Server] {role} ready on port {port}")
        server.start()
    except Exception as e:
        print(f"[Server] Error: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_async_recursion_with_async_results():
    """Test async recursion using rpyc.async_() and AsyncResult.

    Demonstrates:
    1. Making async RPC calls that return immediately
    2. Doing work while waiting for async results
    3. Recursion still works with async calls
    """
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"✅ TEST: Async Recursion with rpyc.async_() and AsyncResult")
    print(f"Server B port: {port_b}")
    print(f"{'='*80}\n")

    ready_b = multiprocessing.Event()

    proc_b = multiprocessing.Process(
        target=_run_server,
        args=(port_b, "B", ready_b),
        daemon=True,
    )
    proc_b.start()

    try:
        assert ready_b.wait(10), "Server B did not start"
        await asyncio.sleep(1.0)

        print("[Test] Connecting to Server B...")

        conn = rpyc.connect(
            "127.0.0.1",
            port_b,
            config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
            }
        )

        print("[Test] Connected successfully")

        conn.root.set_role("B")
        conn.root.clear_log()

        # Client callback
        call_log_a: list[str] = []

        def client_async_callback(depth: int, message: str, server_func: Any) -> str:
            """Client callback that can make async calls to server."""
            log_entry = f"A.async_callback(depth={depth})"
            call_log_a.append(log_entry)
            print(f"[A] {log_entry}, msg='{message}'")

            if depth > 0:
                print(f"[A] Making ASYNC call to server with depth={depth-1}")

                # Make async call to server
                async_result = rpyc.async_(server_func)(
                    depth - 1,
                    f"A->async_recursive",
                    client_async_callback
                )

                # Can do work here while waiting...
                print(f"[A] AsyncResult created, doing other work...")
                time.sleep(0.001)  # Simulate some work

                # Wait for result
                print(f"[A] Waiting for async result...")
                result = async_result.value

                return f"A received (async): {result}"
            else:
                return f"A: Base case (depth={depth})"

        # Start async recursion
        recursion_depth = 3

        print(f"\n[Test] Starting ASYNC recursion with depth={recursion_depth}")
        start_time = time.time()

        try:
            # We can also make the initial call async
            print("[Test] Making initial ASYNC call...")
            async_result = rpyc.async_(conn.root.async_recursive_call)(
                recursion_depth,
                "Initial async call",
                client_async_callback
            )

            print("[Test] Initial call returned AsyncResult, can do other work...")
            print("[Test] Checking if result is ready...")
            print(f"[Test] ready={async_result.ready}")

            # Wait for final result
            print("[Test] Waiting for final async result...")
            result = async_result.value

            elapsed = time.time() - start_time
            print(f"\n✅ SUCCESS: Async recursion completed in {elapsed:.2f}s")
            print(f"[Test] Final result: {result}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ FAILURE after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Get call logs
        log_a = call_log_a
        log_b = conn.root.get_call_log()

        print(f"\n[Test] Call logs:")
        print(f"\nClient A log ({len(log_a)} calls):")
        for i, entry in enumerate(log_a, 1):
            print(f"  {i}. {entry}")

        print(f"\nServer B log ({len(log_b)} calls):")
        for i, entry in enumerate(log_b, 1):
            print(f"  {i}. {entry}")

        # Verify
        expected_b_calls = (recursion_depth // 2) + 1
        expected_a_calls = recursion_depth - expected_b_calls + 1

        assert len(log_b) == expected_b_calls, \
            f"Expected {expected_b_calls} B calls, got {len(log_b)}"
        assert len(log_a) == expected_a_calls, \
            f"Expected {expected_a_calls} A calls, got {len(log_a)}"

        print(f"\n✅ All assertions passed!")
        print(f"   - Server B called {len(log_b)} times")
        print(f"   - Client A called {len(log_a)} times")
        print(f"   - NO DEADLOCK with ASYNC calls!")

        conn.close()

    finally:
        print("\n[Test] Cleaning up...")
        proc_b.terminate()
        proc_b.join(5)
        if proc_b.is_alive():
            proc_b.kill()
        print("[Test] Cleanup complete")


@pytest.mark.asyncio
async def test_comparison_sync_vs_async():
    """Compare sync vs async RPyC calls to show the difference."""
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: Sync vs Async RPyC Calls")
    print(f"{'='*80}\n")

    ready_b = multiprocessing.Event()
    proc_b = multiprocessing.Process(
        target=_run_server,
        args=(port_b, "B", ready_b),
        daemon=True,
    )
    proc_b.start()

    try:
        assert ready_b.wait(10)
        await asyncio.sleep(1.0)

        conn = rpyc.connect("127.0.0.1", port_b, config={"allow_public_attrs": True})
        conn.root.set_role("B")
        conn.root.clear_log()

        def simple_callback(depth, msg, server_func):
            if depth > 0:
                return server_func(depth - 1, msg, simple_callback)
            return "done"

        print("=" * 40)
        print("1. SYNCHRONOUS CALL (blocking)")
        print("=" * 40)

        start = time.time()
        # Sync call - blocks until complete
        result_sync = conn.root.async_recursive_call(3, "sync test", simple_callback)
        elapsed_sync = time.time() - start

        print(f"✓ Sync call completed in {elapsed_sync:.4f}s")
        print(f"  Result: {result_sync[:50]}...")
        print(f"  Behavior: Called and WAITED for result")

        conn.root.clear_log()

        print("\n" + "=" * 40)
        print("2. ASYNCHRONOUS CALL (non-blocking)")
        print("=" * 40)

        start = time.time()
        # Async call - returns immediately
        async_result = rpyc.async_(conn.root.async_recursive_call)(
            3, "async test", simple_callback
        )
        time_to_get_async_result = time.time() - start

        print(f"✓ AsyncResult received in {time_to_get_async_result:.4f}s")
        print(f"  Type: {type(async_result)}")
        print(f"  Ready: {async_result.ready}")
        print(f"  Behavior: Called and got AsyncResult IMMEDIATELY")

        # Simulate doing other work
        print("\n  → Simulating other work while RPC executes...")
        time.sleep(0.01)
        print("  → Other work done!")

        # Now wait for result
        print("\n  → Waiting for async result.value...")
        result_async = async_result.value
        elapsed_async = time.time() - start

        print(f"✓ Async call completed in {elapsed_async:.4f}s (total)")
        print(f"  Result: {result_async[:50]}...")

        print("\n" + "=" * 40)
        print("SUMMARY:")
        print("=" * 40)
        print(f"Sync:  Blocks immediately, waits for result")
        print(f"Async: Returns AsyncResult, can do work, then wait")
        print(f"\nBoth support RECURSION without deadlock! ✅")
        print("=" * 40)

        conn.close()

    finally:
        proc_b.terminate()
        proc_b.join(5)
        if proc_b.is_alive():
            proc_b.kill()


if __name__ == "__main__":
    print("="*80)
    print("Running Async RPyC Recursion Tests")
    print("="*80)

    asyncio.run(test_async_recursion_with_async_results())
    asyncio.run(test_comparison_sync_vs_async())

    print("\n" + "="*80)
    print("✅ All async tests passed!")
    print("="*80)
