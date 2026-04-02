"""✅ WORKING SOLUTION: Cross-process recursion via RPyC using single bidirectional connection.

This test demonstrates the CORRECT way to implement recursive RPC calls:
1. Use ONE bidirectional connection (not two separate connections)
2. Server stores reference to connection in on_connect()
3. Server uses conn.root to pass proxy for recursive callbacks
4. RPyC automatically serves incoming requests while waiting for response

KEY INSIGHT: RPyC is designed to handle reentrant calls through connection symmetry.
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


class RecursiveServer(rpyc.Service):
    """Server that supports recursive callbacks.

    CRITICAL: This service works for BOTH processes in recursion.
    The server stores a reference to the connection to get root proxy.
    """

    def on_connect(self, conn: Any) -> None:
        """Store connection reference - needed to get root proxy."""
        self._conn = conn
        self.call_log: list[str] = []
        self.role = "Server"
        print(f"[{self.role}] Client connected")

    def exposed_set_role(self, role: str) -> None:
        """Set role for logging purposes."""
        self.role = role
        print(f"[{self.role}] Role set")

    def exposed_get_call_log(self) -> list[str]:
        """Get call log."""
        return list(self.call_log)

    def exposed_clear_log(self) -> None:
        """Clear call log."""
        self.call_log.clear()

    def exposed_recursive_call(self, depth: int, message: str, callback: Any) -> str:
        """Recursive RPC call with callback.

        Args:
            depth: Current recursion depth
            message: Message from caller
            callback: Callback function to call back

        Returns:
            Result string
        """
        log_entry = f"{self.role}.recursive_call(depth={depth})"
        self.call_log.append(log_entry)
        print(f"[{self.role}] {log_entry}, msg='{message}'")

        if depth > 0:
            # CRITICAL: Pass self.exposed_recursive_call (this method)
            # so callback can recursively call the server
            print(f"[{self.role}] Calling back with depth={depth-1}")
            result = callback(depth - 1, f"{self.role}->callback", self.exposed_recursive_call)
            return f"{self.role} received: {result}"
        else:
            return f"{self.role}: Base case (depth={depth})"


def _run_server(port: int, role: str, ready: multiprocessing.Event) -> None:
    """Run server process."""
    try:
        server = ThreadedServer(
            RecursiveServer,
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
async def test_rpyc_recursion_single_bidirectional_connection():
    """✅ TEST: Recursive RPC calls using single bidirectional connection.

    Scenario:
    - Server B runs on port_b
    - Test (acting as Client A) connects to B
    - Test calls B.recursive_call(depth=5, callback=ClientCallback)
    - B calls callback.on_callback() passing B's root proxy
    - Callback calls B.recursive_call() again through the SAME connection
    - Recursion continues: B->A->B->A->B->...

    Expected: All calls complete successfully without deadlock.
    """
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"✅ TEST: Cross-Process Recursion (Single Bidirectional Connection)")
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

        # Define client callback
        call_log_a: list[str] = []

        def client_callback(depth: int, message: str, server_func: Any) -> str:
            """Client-side callback function (acts as 'A').

            Args:
                depth: Current recursion depth
                message: Message from server
                server_func: Server function to call recursively

            Returns:
                Result string
            """
            log_entry = f"A.callback(depth={depth})"
            call_log_a.append(log_entry)
            print(f"[A] {log_entry}, msg='{message}'")

            if depth > 0:
                # Recursively call server function through the SAME connection
                print(f"[A] Recursing: calling server with depth={depth-1}")
                result = server_func(depth - 1, f"A->recursive", client_callback)
                return f"A received: {result}"
            else:
                return "A: Base case (depth={depth})"

        # Start recursion
        recursion_depth = 5

        print(f"\n[Test] Starting recursion with depth={recursion_depth}")
        print(f"[Test] Flow: B({recursion_depth})->A({recursion_depth-1})->B({recursion_depth-2})->...")
        start_time = time.time()

        try:
            result = conn.root.recursive_call(
                recursion_depth,
                "Initial call from test",
                client_callback
            )

            elapsed = time.time() - start_time
            print(f"\n✅ SUCCESS: Recursion completed in {elapsed:.2f}s")
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

        # Verify recursion depth
        # With depth=5:
        # B(5) -> A callback(4) -> B(4) -> A callback(3) -> B(3) -> A callback(2) -> B(2) -> A callback(1) -> B(1) -> A callback(0) -> B(0)
        # Wait, let's trace more carefully:
        # Test calls B.recursive_call(5)
        #   B calls callback.on_callback(4, ..., B_proxy)
        #     A calls B_proxy.recursive_call(3, ..., callback)
        #       B calls callback.on_callback(2, ..., B_proxy)
        #         A calls B_proxy.recursive_call(1, ..., callback)
        #           B calls callback.on_callback(0, ..., B_proxy)
        #             A returns base case
        #           B returns
        #         A returns
        #       B returns
        #     A returns
        #   B returns
        #
        # So: B called with 5, 3, 1 = 3 times
        #     A called with 4, 2, 0 = 3 times
        # Total = 6 calls for depth=5

        # Actually, let's recalculate: depth=5
        # B.recursive_call(5) -> calls A.on_callback(4)
        #   A.on_callback(4) -> calls B.recursive_call(3)
        #     B.recursive_call(3) -> calls A.on_callback(2)
        #       A.on_callback(2) -> calls B.recursive_call(1)
        #         B.recursive_call(1) -> calls A.on_callback(0)
        #           A.on_callback(0) -> returns base case (no more calls)
        #
        # B called: 5, 3, 1 (3 times)
        # A called: 4, 2, 0 (3 times)
        # Total: 6 calls

        expected_b_calls = (recursion_depth // 2) + 1  # For depth=5: 3
        expected_a_calls = recursion_depth - expected_b_calls + 1  # For depth=5: 3

        print(f"\n[Test] Verification:")
        print(f"  Expected B calls: {expected_b_calls}, actual: {len(log_b)}")
        print(f"  Expected A calls: {expected_a_calls}, actual: {len(log_a)}")

        assert len(log_b) == expected_b_calls, \
            f"Expected {expected_b_calls} B calls, got {len(log_b)}"
        assert len(log_a) == expected_a_calls, \
            f"Expected {expected_a_calls} A calls, got {len(log_a)}"

        print(f"\n✅ All assertions passed!")
        print(f"   - Server B called {len(log_b)} times")
        print(f"   - Client A called {len(log_a)} times")
        print(f"   - Total: {len(log_a) + len(log_b)} recursive calls")
        print(f"   - NO DEADLOCK!")

        conn.close()

    finally:
        print("\n[Test] Cleaning up...")
        proc_b.terminate()
        proc_b.join(5)
        if proc_b.is_alive():
            proc_b.kill()
        print("[Test] Cleanup complete")


@pytest.mark.asyncio
async def test_deep_recursion():
    """Test deeper recursion to verify no deadlock at depth=10."""
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"✅ TEST: Deep Recursion (depth=10)")
    print(f"{'='*80}\n")

    ready_b = multiprocessing.Event()
    proc_b = multiprocessing.Process(
        target=_run_server, args=(port_b, "B", ready_b), daemon=True
    )
    proc_b.start()

    try:
        assert ready_b.wait(10)
        await asyncio.sleep(1.0)

        conn = rpyc.connect("127.0.0.1", port_b, config={"allow_public_attrs": True, "sync_request_timeout": 300})
        conn.root.set_role("B")
        conn.root.clear_log()

        call_log_a: list[str] = []

        def client_callback(depth: int, message: str, server_func: Any) -> str:
            call_log_a.append(f"A.callback(depth={depth})")
            if depth > 0:
                return server_func(depth - 1, "A->B", client_callback)
            return "A: Base"

        depth = 10

        print(f"[Test] Starting deep recursion with depth={depth}")
        start = time.time()

        result = conn.root.recursive_call(depth, "Initial", client_callback)

        elapsed = time.time() - start
        print(f"✅ Deep recursion completed in {elapsed:.2f}s")
        print(f"Result: {result}")

        log_a = call_log_a
        log_b = conn.root.get_call_log()

        print(f"\nA calls: {len(log_a)}, B calls: {len(log_b)}, Total: {len(log_a) + len(log_b)}")
        print(f"✅ NO DEADLOCK at depth={depth}!")

        conn.close()

    finally:
        proc_b.terminate()
        proc_b.join(5)
        if proc_b.is_alive():
            proc_b.kill()


if __name__ == "__main__":
    print("="*80)
    print("Running RPyC Recursion Tests")
    print("="*80)

    asyncio.run(test_rpyc_recursion_single_bidirectional_connection())

    print("\n" + "="*80)
    asyncio.run(test_deep_recursion())

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
