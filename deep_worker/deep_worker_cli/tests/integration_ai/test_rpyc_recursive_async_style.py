"""Test cross-process recursion using async_() for callbacks.

Strategy: Use rpyc.async_() to make callbacks fire-and-forget,
then use a different pattern to return results.
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


class RecursiveServiceA(rpyc.Service):
    """Service A with async callbacks."""

    peer_connection = None
    bg_thread = None
    call_log: list[str] = []
    log_lock = threading.Lock()
    # For storing async results
    pending_results: dict[int, Any] = {}
    results_lock = threading.Lock()

    def exposed_set_peer(self, host: str, port: int) -> None:
        """Connect to peer."""
        if RecursiveServiceA.peer_connection is None:
            print(f"[Process A] Connecting to peer at {host}:{port}")
            RecursiveServiceA.peer_connection = rpyc.connect(
                host, port,
                config={
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,
                    "bind_threads": True,
                }
            )
            RecursiveServiceA.bg_thread = rpyc.BgServingThread(
                RecursiveServiceA.peer_connection
            )
            print(f"[Process A] Connected with BgServingThread")

    def exposed_get_call_log(self) -> list[str]:
        """Get call log."""
        with RecursiveServiceA.log_lock:
            return list(RecursiveServiceA.call_log)

    def exposed_clear_log(self) -> None:
        """Clear log."""
        with RecursiveServiceA.log_lock:
            RecursiveServiceA.call_log.clear()
            RecursiveServiceA.pending_results.clear()

    def exposed_callback_from_b(self, depth: int, message: str, call_id: int) -> None:
        """Async callback from B - does not return a value directly.

        Instead, stores result and B will query for it.
        """
        log_entry = f"A.callback_from_b(depth={depth}, call_id={call_id})"
        with RecursiveServiceA.log_lock:
            RecursiveServiceA.call_log.append(log_entry)

        print(f"[Process A] {log_entry}")

        result = None
        if depth > 0 and RecursiveServiceA.peer_connection:
            # Recursively call B
            print(f"[Process A] Recursing to B with depth={depth-1}")
            result = RecursiveServiceA.peer_connection.root.process_with_callback(
                depth - 1,
                f"A->B depth {depth}",
                call_id
            )
        else:
            result = f"A: Base case (depth={depth})"

        # Store result for B to retrieve
        with RecursiveServiceA.results_lock:
            RecursiveServiceA.pending_results[call_id] = result

    def exposed_get_result(self, call_id: int, timeout: float = 10.0) -> Any:
        """Get result of async callback."""
        start = time.time()
        while time.time() - start < timeout:
            with RecursiveServiceA.results_lock:
                if call_id in RecursiveServiceA.pending_results:
                    return RecursiveServiceA.pending_results.pop(call_id)
            time.sleep(0.01)
        return f"ERROR: Timeout waiting for result {call_id}"


class RecursiveServiceB(rpyc.Service):
    """Service B that uses async callbacks."""

    peer_connection = None
    bg_thread = None
    call_log: list[str] = []
    log_lock = threading.Lock()
    call_id_counter = 0
    counter_lock = threading.Lock()

    def exposed_set_peer(self, host: str, port: int) -> None:
        """Connect to peer."""
        if RecursiveServiceB.peer_connection is None:
            print(f"[Process B] Connecting to peer at {host}:{port}")
            RecursiveServiceB.peer_connection = rpyc.connect(
                host, port,
                config={
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,
                    "bind_threads": True,
                }
            )
            RecursiveServiceB.bg_thread = rpyc.BgServingThread(
                RecursiveServiceB.peer_connection
            )
            print(f"[Process B] Connected with BgServingThread")

    def exposed_get_call_log(self) -> list[str]:
        """Get call log."""
        with RecursiveServiceB.log_lock:
            return list(RecursiveServiceB.call_log)

    def exposed_clear_log(self) -> None:
        """Clear log."""
        with RecursiveServiceB.log_lock:
            RecursiveServiceB.call_log.clear()

    def exposed_process_with_callback(self, depth: int, message: str, call_id: int | None = None) -> str:
        """Process request and call back to A using async pattern."""
        # Generate call_id if not provided
        if call_id is None:
            with RecursiveServiceB.counter_lock:
                RecursiveServiceB.call_id_counter += 1
                call_id = RecursiveServiceB.call_id_counter

        log_entry = f"B.process_with_callback(depth={depth}, call_id={call_id})"
        with RecursiveServiceB.log_lock:
            RecursiveServiceB.call_log.append(log_entry)

        print(f"[Process B] {log_entry}")

        if RecursiveServiceB.peer_connection is None:
            return "B: ERROR - no peer connection"

        # Call A asynchronously using rpyc.async_()
        print(f"[Process B] Async callback to A with depth={depth}, call_id={call_id}")
        async_result = rpyc.async_(RecursiveServiceB.peer_connection.root.callback_from_b)(
            depth, f"B->A callback depth {depth}", call_id
        )

        # Don't wait for async_result.value - that would block!
        # Instead, poll for the result
        print(f"[Process B] Waiting for A to complete callback...")
        result = RecursiveServiceB.peer_connection.root.get_result(call_id, timeout=30.0)

        return f"B got from A: {result}"


def _run_server(service_class: type, port: int, ready: multiprocessing.Event) -> None:
    """Run server."""
    try:
        server = ThreadedServer(
            service_class,
            hostname="127.0.0.1",
            port=port,
            protocol_config={
                "allow_public_attrs": True,
                "sync_request_timeout": 300,
                "bind_threads": True,
            },
        )
        ready.set()
        print(f"[Server] {service_class.__name__} ready on {port}")
        server.start()
    except Exception as e:
        print(f"[Server] Error: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.asyncio
async def test_async_callback_recursion():
    """Test recursion using async_() for callbacks."""
    port_a = _find_free_port()
    port_b = _find_free_port()

    print(f"\n{'='*80}")
    print(f"Test: Async Callback Recursion")
    print(f"Process A: {port_a}, Process B: {port_b}")
    print(f"{'='*80}\n")

    ready_a = multiprocessing.Event()
    ready_b = multiprocessing.Event()

    proc_a = multiprocessing.Process(
        target=_run_server, args=(RecursiveServiceA, port_a, ready_a), daemon=True
    )
    proc_b = multiprocessing.Process(
        target=_run_server, args=(RecursiveServiceB, port_b, ready_b), daemon=True
    )

    proc_a.start()
    proc_b.start()

    try:
        assert ready_a.wait(10), "A not ready"
        assert ready_b.wait(10), "B not ready"
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

        # Test with recursion depth
        depth = 3
        print(f"\n[Test] Starting recursion with depth={depth}")
        start_time = time.time()

        try:
            result = service_b.process_with_callback(depth, "Initial call")
            elapsed = time.time() - start_time

            print(f"\n[Test] ✅ Completed in {elapsed:.2f}s")
            print(f"[Test] Result: {result}")

            log_a = service_a.get_call_log()
            log_b = service_b.get_call_log()

            print(f"\nProcess A log ({len(log_a)} calls):")
            for entry in log_a:
                print(f"  - {entry}")

            print(f"\nProcess B log ({len(log_b)} calls):")
            for entry in log_b:
                print(f"  - {entry}")

            # Verify recursion depth
            expected = depth + 1
            assert len(log_b) == expected, f"Expected {expected} B calls, got {len(log_b)}"
            assert len(log_a) == expected, f"Expected {expected} A calls, got {len(log_a)}"

            print(f"\n✅ All assertions passed!")

        except Exception as e:
            print(f"\n❌ FAILURE: {e}")
            import traceback
            traceback.print_exc()
            raise

        bg_a.stop()
        bg_b.stop()
        conn_a.close()
        conn_b.close()

    finally:
        proc_a.terminate()
        proc_b.terminate()
        proc_a.join(5)
        proc_b.join(5)
        if proc_a.is_alive():
            proc_a.kill()
        if proc_b.is_alive():
            proc_b.kill()


if __name__ == "__main__":
    asyncio.run(test_async_callback_recursion())
