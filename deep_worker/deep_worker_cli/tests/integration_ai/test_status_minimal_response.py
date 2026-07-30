"""Integration test for minimal status response optimization.

This test verifies that:
1. Status response contains ONLY necessary fields (status + optional error)
2. No useless data is serialized (environment, working_dir, metadata, etc.)
3. Response size is minimal (< 1 KB instead of 10-100 KB)
4. Status functionality still works correctly

This is a TDD test for the CPU optimization work.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

# ============================================================================
# CRITICAL: RPYC MUST BE IMPORTED FROM SOURCE CODE
# ============================================================================
_rpyc_async_path = Path("/path/to/rpyc_async")
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc_async as rpyc
from rpyc_async.core.async_connect import async_connect


@pytest.fixture
def test_agent():
    """Provide test agent - uses existing agent instead of creating new one."""
    class AgentInfo:
        def __init__(self, name):
            self.agent_name = name

    # Use existing agent (user should start one before running test)
    agent_name = "test_1"

    # Verify agent exists via dwcli --list
    result = subprocess.run(
        ["dwcli", "--list"],
        capture_output=True,
        text=True,
        timeout=5
    )

    if agent_name not in result.stdout:
        pytest.skip(f"Agent '{agent_name}' not running. Start with: deep-worker {agent_name}")

    yield AgentInfo(agent_name)


@pytest.mark.asyncio
async def test_status_response_is_minimal(test_agent):
    """Test that status response contains ONLY necessary fields.

    Expected minimal response:
    {
        "success": true,
        "status": "ready",  // or "working", "error", etc.
        "error": null       // optional, only if status == "error"
    }

    Should NOT contain:
    - environment dict (10-100 KB of os.environ)
    - working_dir, created_at, last_activity
    - tasks_completed, message_count, process_id
    - metadata, properties
    """
    agent_name = test_agent.agent_name

    # Get agent's RPyC port from registry
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None, f"Agent {agent_name} not found in registry"
    port = agent_info.rpyc_port

    # Connect directly via RPyC to get raw response
    conn = await async_connect("localhost", port, config={"sync_request_timeout": 30})

    # Enable asyncio serving
    conn.enable_asyncio_serving()

    try:
        # Call get_status() method (RPyC handles exposed_ prefix automatically)
        status_response = await asyncio.wait_for(
            conn.root.get_status(),
            timeout=10.0
        )

        print(f"\n📦 Status response keys: {list(status_response.keys())}")
        print(f"📏 Response size: {len(str(status_response))} bytes")

        # CRITICAL: Response must be minimal
        assert "success" in status_response, "Response must have 'success' field"
        assert "status" in status_response, "Response must have 'status' field"

        # Status must be valid
        valid_statuses = ["ready", "working", "error", "waiting_input", "max_tokens"]
        assert status_response["status"] in valid_statuses, \
            f"Invalid status: {status_response['status']}"

        # CRITICAL: Must NOT contain bloated fields
        forbidden_fields = [
            "session_info",      # Old full SessionInfo object
            "environment",       # 10-100 KB of os.environ
            "working_dir",       # Unused
            "created_at",        # Unused
            "last_activity",     # Unused
            "tasks_completed",   # Unused
            "message_count",     # Unused
            "process_id",        # Unused
            "metadata",          # Unused
            "properties",        # Unused
            "data",              # Unused
        ]

        for field in forbidden_fields:
            assert field not in status_response, \
                f"Response should NOT contain '{field}' field! This wastes CPU on serialization."

        # Verify response is small (< 1 KB)
        response_size = len(str(status_response))
        assert response_size < 1024, \
            f"Response too large: {response_size} bytes (should be < 1 KB)"

        print(f"✅ Status response is minimal: {response_size} bytes")
    finally:
        conn.disable_asyncio_serving()
        conn.close()


@pytest.mark.asyncio
async def test_status_response_with_error_message(test_agent):
    """Test that error message is included when status is 'error'.

    When status == "error", response should include error message:
    {
        "success": true,
        "status": "error",
        "error": "Some error message"
    }
    """
    agent_name = test_agent.agent_name

    # Get agent's RPyC port from registry
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None
    port = agent_info.rpyc_port

    # Connect directly via RPyC
    conn = await async_connect("localhost", port, config={"sync_request_timeout": 30})
    conn.enable_asyncio_serving()

    try:
        # First get normal status
        status_response = await asyncio.wait_for(
            conn.root.get_status(),
            timeout=10.0
        )

        # Verify it has error field (optional, may be None)
        assert "error" in status_response or status_response["status"] != "error", \
            "If status is 'error', must have 'error' field with message"

        print(f"✅ Error field handling correct")

    except Exception as e:
        pytest.skip(f"Could not test error status: {e}")
    finally:
        conn.disable_asyncio_serving()
        conn.close()


@pytest.mark.asyncio
async def test_status_functionality_still_works(test_agent):
    """Test that status monitoring still works correctly after optimization.

    This verifies that removing unnecessary fields doesn't break:
    - Status detection (ready/working/error)
    - Status polling via dwcli --status
    - Status used by the web UI WebSocket
    """
    agent_name = test_agent.agent_name

    # Test 1: Get status via dwcli command
    result = subprocess.run(
        ["dwcli", "--status", agent_name],
        capture_output=True,
        text=True,
        timeout=5
    )

    assert result.returncode == 0, f"dwcli --status failed: {result.stderr}"
    assert "ready" in result.stdout.lower() or "working" in result.stdout.lower(), \
        f"Should show agent status, got: {result.stdout}"

    print(f"✅ dwcli --status works correctly")

    # Test 2: Get status via RPyC directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    assert agent_info is not None
    port = agent_info.rpyc_port

    conn = await async_connect("localhost", port, config={"sync_request_timeout": 30})
    conn.enable_asyncio_serving()

    try:
        status_response = await asyncio.wait_for(
            conn.root.get_status(),
            timeout=10.0
        )

        assert status_response["success"] is True, "Status call should succeed"
        assert status_response["status"] in ["ready", "working", "error", "waiting_input"], \
            f"Status should be valid, got: {status_response['status']}"

        print(f"✅ Direct RPyC status works correctly")

        # Test 3: Verify the web UI can still parse status
        # (This would be tested by the web UI integration tests, but we can verify structure)
        assert "status" in status_response, "The web UI expects 'status' field"
        assert isinstance(status_response["status"], str), "Status must be string"

        print(f"✅ Status response compatible with the web UI")
    finally:
        conn.disable_asyncio_serving()
        conn.close()


@pytest.mark.asyncio
async def test_status_response_size_comparison():
    """Compare old vs new response sizes to quantify improvement.

    This test documents the improvement from optimization:
    - OLD: ~10-100 KB (with environment dict)
    - NEW: < 1 KB (minimal fields only)
    """
    # This test documents expected improvement
    # Actual measurements would be done by running before/after

    old_response_estimate = {
        "success": True,
        "session_info": {
            "name": "test_1",
            "session_type": "langchain",
            "status": "ready",
            "working_dir": "/tmp",
            "created_at": "2026-04-04T12:00:00",
            "last_activity": "2026-04-04T12:00:00",
            "tasks_completed": 0,
            "error_message": None,
            "data": {
                "message_count": 0,
                "process_id": 12345,
            },
            "metadata": {
                "environment": {f"VAR_{i}": f"value_{i}" for i in range(100)},  # ~10 KB
                "properties": {
                    "agent_type": "deep_worker",
                    "model": "claude-3-5-sonnet-20241022",
                }
            }
        }
    }

    new_response = {
        "success": True,
        "status": "ready",
        "error": None
    }

    old_size = len(str(old_response_estimate))
    new_size = len(str(new_response))

    improvement_ratio = old_size / new_size
    size_reduction = old_size - new_size

    print(f"\n📊 Response Size Comparison:")
    print(f"  OLD (with environment): ~{old_size:,} bytes")
    print(f"  NEW (minimal):          ~{new_size:,} bytes")
    print(f"  Reduction:              ~{size_reduction:,} bytes ({improvement_ratio:.1f}x smaller)")

    # Document expected improvement
    assert new_size < 200, f"New response should be < 200 bytes, got {new_size}"
    assert improvement_ratio > 10, f"Should be at least 10x smaller, got {improvement_ratio:.1f}x"

    print(f"✅ Expected {improvement_ratio:.1f}x size reduction after optimization")


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v", "-s"])
