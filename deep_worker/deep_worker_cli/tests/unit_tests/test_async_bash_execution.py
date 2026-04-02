"""Tests for async bash execution to prevent event loop blocking.

This test suite verifies that bash command execution doesn't block the event loop,
allowing concurrent RPC calls (like get_status) to be processed while long-running
commands execute.
"""

import asyncio
import time
from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIShellBackend


class TestAsyncBashExecution:
    """Test that bash execution is truly async and doesn't block event loop."""

    @pytest.mark.asyncio
    async def test_execute_is_async(self):
        """Test that execute() is an async method."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        # execute() should be a coroutine function
        import inspect
        assert inspect.iscoroutinefunction(backend.execute), \
            "execute() must be async to avoid blocking event loop"

    @pytest.mark.asyncio
    async def test_long_command_does_not_block_event_loop(self):
        """Test that long-running commands don't block concurrent tasks.

        This is the critical test: while a 2-second sleep is running,
        we should be able to execute other async operations concurrently.
        """
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        # Track execution order
        events = []

        async def long_command():
            """Execute a 2-second sleep command."""
            events.append("long_start")
            result = await backend.execute("sleep 2")
            events.append("long_end")
            assert result.exit_code == 0
            return result

        async def quick_task():
            """Quick task that should run while sleep is executing."""
            await asyncio.sleep(0.5)  # Wait a bit to ensure sleep started
            events.append("quick_task")
            return "done"

        # Run both concurrently
        start = time.time()
        results = await asyncio.gather(
            long_command(),
            quick_task()
        )
        duration = time.time() - start

        # Verify:
        # 1. Total time should be ~2 seconds (dominated by sleep), not 2.5+
        assert duration < 2.5, \
            f"Concurrent execution took {duration}s, expected <2.5s (would be >2.5s if blocking)"

        # 2. Events should interleave (quick_task runs WHILE sleep executes)
        assert events == ["long_start", "quick_task", "long_end"], \
            f"Expected interleaved execution, got {events}"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_commands(self):
        """Test that multiple bash commands can run concurrently."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        async def run_sleep(duration: float, label: str):
            """Run sleep command and return timing info."""
            start = time.time()
            result = await backend.execute(f"sleep {duration}")
            elapsed = time.time() - start
            return label, elapsed, result.exit_code

        # Run 3 sleep commands concurrently
        start = time.time()
        results = await asyncio.gather(
            run_sleep(1.0, "task1"),
            run_sleep(1.0, "task2"),
            run_sleep(1.0, "task3"),
        )
        total_duration = time.time() - start

        # If truly concurrent, should take ~1 second (all run in parallel)
        # If blocking, would take ~3 seconds (sequential)
        assert total_duration < 1.5, \
            f"Concurrent execution took {total_duration}s, expected <1.5s (would be ~3s if blocking)"

        # All tasks should succeed
        for label, elapsed, exit_code in results:
            assert exit_code == 0, f"{label} failed with exit code {exit_code}"

    @pytest.mark.asyncio
    async def test_timeout_still_works(self):
        """Test that timeout mechanism still works with async execution."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        # Command that sleeps longer than timeout
        start = time.time()
        result = await backend.execute("sleep 10", timeout=1)
        duration = time.time() - start

        # Should timeout after ~1-2 seconds (timeout + graceful termination)
        assert duration < 3.0, f"Timeout took {duration}s, expected <3s"
        assert result.exit_code == 124, f"Expected timeout exit code 124, got {result.exit_code}"
        assert "timed out" in result.output.lower()

    @pytest.mark.asyncio
    async def test_command_output_captured(self):
        """Test that stdout and stderr are still captured correctly."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        # Test stdout
        result = await backend.execute("echo 'Hello from stdout'")
        assert result.exit_code == 0
        assert "Hello from stdout" in result.output

        # Test stderr
        result = await backend.execute("echo 'Error message' >&2")
        assert result.exit_code == 0
        assert "Error message" in result.output
        assert "[stderr]" in result.output

    @pytest.mark.asyncio
    async def test_exit_code_captured(self):
        """Test that exit codes are captured correctly."""
        backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

        # Success
        result = await backend.execute("exit 0")
        assert result.exit_code == 0

        # Failure
        result = await backend.execute("exit 42")
        assert result.exit_code == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
